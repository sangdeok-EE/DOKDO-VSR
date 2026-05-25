"""
model.py - Korean VSR model wrapping the Chaplin E2E architecture.

Loads Chaplin pretrained weights, then replaces output layers with
Korean vocab size for full fine-tuning.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn

CHAPLIN_ROOT = "<path/to/chaplin>"
sys.path.insert(0, CHAPLIN_ROOT)

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class KoreanVSRModel(nn.Module):
    """Wrap Chaplin E2E model for Korean VSR fine-tuning.

    1. Load Chaplin pretrained model
    2. Replace CTC head + Decoder output layer with Korean vocab size
    3. Full fine-tuning of all parameters
    """

    def __init__(self, chaplin_checkpoint, model_conf, vocab_size, device="cpu"):
        """
        Args:
            chaplin_checkpoint: Path to Chaplin pretrained weight file.
            model_conf: Path to Chaplin model.json.
            vocab_size: tokenizer.vocab_size (e.g. 8001).
            device: torch device string.
        """
        super().__init__()

        with open(model_conf, "rb") as f:
            confs = json.load(f)
        args = confs if isinstance(confs, dict) else confs[2]
        train_args = argparse.Namespace(**args)

        orig_token_file = os.path.join(
            CHAPLIN_ROOT, "pipelines", "tokens", "unigram5000_units.txt"
        )
        if os.path.exists(orig_token_file):
            with open(orig_token_file, "r", encoding="utf-8-sig") as f:
                words = [line.split()[0] for line in f.read().splitlines() if line.strip()]
            orig_vocab_size = len(["<blank>"] + words + ["<eos>"])
        else:
            orig_vocab_size = 5002

        self.model = E2E(orig_vocab_size, train_args)
        state_dict = torch.load(chaplin_checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {chaplin_checkpoint}")

        adim = train_args.adim
        if self.model.ctc is not None:
            self.model.ctc.ctc_lo = nn.Linear(adim, vocab_size)
            print(f"Replaced CTC head: {orig_vocab_size} -> {vocab_size}")

        if self.model.decoder is not None and hasattr(self.model.decoder, 'output_layer'):
            self.model.decoder.output_layer = nn.Linear(adim, vocab_size)
            print(f"Replaced Decoder output layer: {orig_vocab_size} -> {vocab_size}")

        self.model.odim = vocab_size
        self.model.sos = vocab_size - 1
        self.model.eos = vocab_size - 1

        self.vocab_size = vocab_size
        self.adim = adim

    def forward(self, videos, video_lengths):
        """
        Args:
            videos (Tensor):        (B, 1, T_max, 88, 88) - normalized video
            video_lengths (Tensor): (B,) - actual frame counts

        Returns:
            logits (Tensor):         (B, T', vocab_size) - CTC logits (pre log_softmax)
            output_lengths (Tensor): (B,) - encoder output lengths
        """
        enc_output, enc_mask = self.model.encoder(videos, None)
        # enc_output: (B, T', adim)

        logits = self.model.ctc.ctc_lo(self.model.ctc.dropout(enc_output))
        # logits: (B, T', vocab_size)

        # Conv3dResNet does not subsample temporally (stride=1 in time)
        output_lengths = video_lengths.clone()

        return logits, output_lengths

    def encode(self, x):
        """Encode a single sample (inference)."""
        self.model.eval()
        return self.model.encode(x)


def load_model(chaplin_checkpoint, model_conf, vocab_size, device="cpu"):
    """Helper to instantiate and move the model to device."""
    model = KoreanVSRModel(chaplin_checkpoint, model_conf, vocab_size, device)
    model = model.to(device)
    return model
