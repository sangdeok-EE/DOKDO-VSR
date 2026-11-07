"""
model.py - Korean VSR model wrapping the Auto-AVSR / Chaplin E2E architecture.

Loads pretrained LRS3 visual-only weights (Frontend + Conformer Encoder), then
replaces the English output heads (5,049 units) with Korean SentencePiece Unigram
heads (vocab_size = 8,001: 8,000 subword units + 1 CTC blank at index 8000).

Supports:
  1. Hybrid CTC + Attention Decoder Cross-Entropy (Label Smoothing) Loss:
     L = ctc_weight * L_CTC + (1 - ctc_weight) * L_CE   (default ctc_weight = 0.3)
  2. CTC-only forward pass for CTC Greedy inference or CTC-only ablation.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPLIN_ROOT = os.environ.get("CHAPLIN_ROOT", PROJECT_ROOT)
if CHAPLIN_ROOT not in sys.path:
    sys.path.insert(0, CHAPLIN_ROOT)

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class KoreanVSRModel(nn.Module):
    """Wrap Auto-AVSR / Chaplin E2E model for Korean VSR transfer learning.

    1. Load LRS3 visual-only pretrained backbone (3D Conv + ResNet-18 + 12-layer Conformer)
    2. Replace CTC linear head + Transformer Decoder embedding & output layer with Korean vocab_size (8,001)
    3. Support 2-stage transfer learning (Phase 1: Head/Decoder warm-up -> Phase 2: Full fine-tuning)
    """

    def __init__(
        self,
        chaplin_checkpoint,
        model_conf,
        vocab_size=8001,
        device="cpu",
        sos_id=1,
        eos_id=2,
        ignore_id=-1,
        lsm_weight=0.1,
    ):
        """
        Args:
            chaplin_checkpoint: Path to pretrained weight file (LRS3_V_WER19.1/model.pth)
                                or None if loading a fine-tuned Korean checkpoint directly.
            model_conf: Path to model.json.
            vocab_size: tokenizer.vocab_size (default: 8001).
            device: torch device string.
            sos_id: Start-of-sequence token ID for decoder teacher forcing (default: 1 for <s>).
            eos_id: End-of-sequence token ID for decoder teacher forcing (default: 2 for </s>).
            ignore_id: Padding ignore ID for decoder target sequences (default: -1).
            lsm_weight: Label smoothing weight for decoder CE loss (default: 0.1).
        """
        super().__init__()

        with open(model_conf, "rb") as f:
            confs = json.load(f)
        args = confs if isinstance(confs, dict) else confs[2]
        train_args = argparse.Namespace(**args)
        self.train_args = train_args

        orig_token_file = os.path.join(
            CHAPLIN_ROOT, "pipelines", "tokens", "unigram5000_units.txt"
        )
        if os.path.exists(orig_token_file):
            with open(orig_token_file, "r", encoding="utf-8-sig") as f:
                words = [line.split()[0] for line in f.read().splitlines() if line.strip()]
            orig_vocab_size = len(["<blank>"] + words + ["<eos>"])
        else:
            orig_vocab_size = 5049

        self.model = E2E(orig_vocab_size, train_args)
        if chaplin_checkpoint and os.path.isfile(chaplin_checkpoint):
            state_dict = torch.load(chaplin_checkpoint, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            # Check if checkpoint already has Korean vocab_size (8001) or LRS3 vocab_size (5049)
            ctc_w = state_dict.get("ctc.ctc_lo.weight", state_dict.get("model.ctc.ctc_lo.weight"))
            if ctc_w is not None and ctc_w.shape[0] == vocab_size:
                self._replace_heads(vocab_size, train_args.adim)
                cleaned = {
                    (k[len("model."):] if k.startswith("model.") else k): v
                    for k, v in state_dict.items()
                }
                self.model.load_state_dict(cleaned, strict=False)
                print(f"Loaded Korean VSR checkpoint from {chaplin_checkpoint}")
            else:
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded English LRS3 pretrained weights from {chaplin_checkpoint}")
                self._replace_heads(vocab_size, train_args.adim)
        else:
            self._replace_heads(vocab_size, train_args.adim)

        self.vocab_size = vocab_size
        self.adim = train_args.adim
        self.sos_id = int(sos_id)
        self.eos_id = int(eos_id)
        self.ignore_id = int(ignore_id)

        self.model.odim = vocab_size
        self.model.sos = self.sos_id
        self.model.eos = self.eos_id

        # Label smoothing cross-entropy criterion for the Transformer Decoder
        length_norm = bool(getattr(train_args, "transformer_length_normalized_loss", False))
        self.criterion_att = LabelSmoothingLoss(
            self.vocab_size,
            self.ignore_id,
            lsm_weight,
            length_norm,
        )

    def _replace_heads(self, vocab_size, adim):
        """Replace CTC linear projection and Transformer Decoder embedding/output layers."""
        orig_vocab = self.model.odim
        if self.model.ctc is not None:
            self.model.ctc.ctc_lo = nn.Linear(adim, vocab_size)
            print(f"Replaced CTC head: {orig_vocab} -> {vocab_size}")

        if self.model.decoder is not None:
            if hasattr(self.model.decoder, "embed") and isinstance(self.model.decoder.embed, nn.Sequential):
                old_emb = self.model.decoder.embed[0]
                if isinstance(old_emb, nn.Embedding) and old_emb.num_embeddings != vocab_size:
                    self.model.decoder.embed[0] = nn.Embedding(vocab_size, adim)
                    print(f"Replaced Decoder embedding: {old_emb.num_embeddings} -> {vocab_size}")
            if hasattr(self.model.decoder, "output_layer") and self.model.decoder.output_layer is not None:
                self.model.decoder.output_layer = nn.Linear(adim, vocab_size)
                print(f"Replaced Decoder output layer: {orig_vocab} -> {vocab_size}")

    def forward(self, videos, video_lengths):
        """CTC-only forward pass (used for CTC Greedy decoding and CTC loss calculation).

        Args:
            videos (Tensor):        (B, 1, T_max, 88, 88) - normalized video
            video_lengths (Tensor): (B,) - actual frame counts

        Returns:
            logits (Tensor):         (B, T', vocab_size) - CTC logits (pre log_softmax)
            output_lengths (Tensor): (B,) - encoder output lengths
        """
        enc_output, _ = self.model.encoder(videos, None)
        logits = self.model.ctc.ctc_lo(self.model.ctc.dropout(enc_output))
        output_lengths = video_lengths.clone()
        return logits, output_lengths

    def forward_hybrid(self, videos, video_lengths, labels, label_lengths):
        """Hybrid CTC + Transformer Decoder forward pass (used in paper's training).

        Args:
            videos (Tensor):        (B, 1, T_max, 88, 88) - normalized video
            video_lengths (Tensor): (B,) - actual frame counts
            labels (Tensor):        (B, L_max) - padded token IDs (0-padded by collate_fn)
            label_lengths (Tensor): (B,) - actual label lengths

        Returns:
            ctc_logits (Tensor):     (B, T', vocab_size) - CTC logits
            output_lengths (Tensor): (B,) - encoder output lengths
            loss_att (Tensor):       scalar Transformer Decoder cross-entropy loss
        """
        padding_mask = make_non_pad_mask(video_lengths).to(videos.device).unsqueeze(-2)
        enc_output, enc_mask = self.model.encoder(videos, padding_mask)

        ctc_logits = self.model.ctc.ctc_lo(self.model.ctc.dropout(enc_output))
        output_lengths = video_lengths.clone()

        # Build ignore_id (-1) padded target tensor from (labels, label_lengths)
        ys_pad = labels.clone()
        for i, l in enumerate(label_lengths):
            if l < ys_pad.size(1):
                ys_pad[i, l:] = self.ignore_id

        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.sos_id, self.eos_id, self.ignore_id
        )
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.model.decoder(ys_in_pad, ys_mask, enc_output, enc_mask)
        loss_att = self.criterion_att(pred_pad, ys_out_pad)

        return ctc_logits, output_lengths, loss_att

    def encode(self, x):
        """Encode a single sample (inference)."""
        self.model.eval()
        return self.model.encode(x)


def load_model(chaplin_checkpoint, model_conf, vocab_size=8001, device="cpu"):
    """Helper to instantiate and move the model to device."""
    model = KoreanVSRModel(chaplin_checkpoint, model_conf, vocab_size, device=device)
    model = model.to(device)
    return model
