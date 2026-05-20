#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Korean VSR inference model wrapper.

Loads a Chaplin E2E backbone whose CTC head AND decoder were retrained for
Korean (vocab=8001, blank=8000, sos=eos=8000).

Two decode modes:
  - "ctc"    : CTC-greedy + SentencePiece  (model 3, encoder/CTC-only training)
  - "hybrid" : joint CTC/attention beam search with weights
               {decoder: 1-ctc_weight, ctc: ctc_weight}  (model 4, ctc_weight=0.3)

NOTE on the Korean token layout (코드/tokenizer_utils.py):
    <unk>=0  <s>=1  </s>=2  ... sp pieces ...  CTC blank = 8000  (vocab=8001)
The training side set model.sos = model.eos = vocab-1 = 8000, so the decoder
was teacher-forced with sos=eos=8000.  Critically the CTC blank is 8000, NOT 0,
but espnet's stock CTCPrefixScorer hardcodes blank=0 — so the hybrid path uses
a blank-aware subclass below.
"""

import os
import json
import argparse
from typing import List

import torch
import torch.nn as nn
import sentencepiece as spm

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.ctc_prefix_score import CTCPrefixScore, CTCPrefixScoreTH


CTC_BLANK_ID = 8000
VOCAB_SIZE = 8001
SOS_EOS_ID = VOCAB_SIZE - 1  # = 8000  (training set model.sos=model.eos=vocab-1)


def _strip_state_dict(sd: dict) -> dict:
    """Unwrap {'model_state_dict': ...} and drop the 'model.' prefix."""
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    out = {}
    for k, v in sd.items():
        nk = k[len("model."):] if k.startswith("model.") else k
        out[nk] = v
    return out


class KoreanCTCPrefixScorer(CTCPrefixScorer):
    """CTCPrefixScorer that respects a non-zero blank id.

    espnet's stock scorer passes blank=0 to CTCPrefixScore(TH).  The Korean
    model places the CTC blank at id 8000, so we override the two init_state
    methods to forward the real blank id.
    """

    def __init__(self, ctc: torch.nn.Module, eos: int, blank: int):
        super().__init__(ctc, eos)
        self.blank = blank

    def init_state(self, x: torch.Tensor):
        import numpy as np
        logp = self.ctc.log_softmax(x.unsqueeze(0)).detach().squeeze(0).cpu().numpy()
        self.impl = CTCPrefixScore(logp, self.blank, self.eos, np)
        return 0, self.impl.initial_state()

    def batch_init_state(self, x: torch.Tensor):
        logp = self.ctc.log_softmax(x.unsqueeze(0))  # assuming batch_size = 1
        xlen = torch.tensor([logp.size(1)])
        self.impl = CTCPrefixScoreTH(logp, xlen, self.blank, self.eos)
        return None


class KoreanAVSR(nn.Module):
    def __init__(self, model_path: str, model_conf: str, spm_model: str,
                 device: str = "cuda:0", decode: str = "ctc",
                 ctc_weight: float = 0.3, beam_size: int = 10,
                 penalty: float = 0.0, sos_id: int = SOS_EOS_ID,
                 eos_id: int = SOS_EOS_ID):
        super().__init__()
        self.device = device
        self.decode = decode.lower().strip()
        assert self.decode in ("ctc", "hybrid"), f"unknown decode mode: {decode}"
        self.sos_id = int(sos_id)
        self.eos_id = int(eos_id)

        with open(model_conf, "rb") as f:
            confs = json.load(f)
        args = confs if isinstance(confs, dict) else confs[2]
        self.train_args = argparse.Namespace(**args)

        # Both Korean ckpts (3_encCTC, 4_hybrid) have decoder.embed/ctc_lo/
        # output_layer all at the Korean vocab (8001).  Build E2E directly at
        # VOCAB_SIZE so every one of the 767 tensors matches the ckpt — no
        # shape-mismatch, no silent drop of decoder.embed (the old 5049 build
        # only worked for the legacy CTC-only best_model.pt).
        self.model = E2E(VOCAB_SIZE, self.train_args)
        self.model.odim = VOCAB_SIZE
        self.model.sos = self.sos_id
        self.model.eos = self.eos_id

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        sd = _strip_state_dict(ckpt)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[KoreanAVSR] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        if missing:
            non_critical = [k for k in missing
                            if not (k.endswith("num_batches_tracked"))]
            if non_critical:
                print(f"[KoreanAVSR] missing keys ({len(non_critical)}): {non_critical[:5]}...")

        self.model.to(device=self.device).eval()

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_model)
        self.blank_id = CTC_BLANK_ID

        self.beam_search = None
        if self.decode == "hybrid":
            self.ctc_weight = float(ctc_weight)
            self.beam_size = int(beam_size)
            self._build_beam_search(penalty)
            print(f"[KoreanAVSR] decode=hybrid  ctc_weight={self.ctc_weight} "
                  f"decoder={1.0 - self.ctc_weight}  beam={self.beam_size}")
        else:
            print("[KoreanAVSR] decode=ctc (CTC-greedy)")

    def _build_beam_search(self, penalty: float):
        # token_list only needs the right length (and is used for debug logs).
        piece_size = self.sp.GetPieceSize()  # 8000
        token_list = [self.sp.IdToPiece(i) for i in range(piece_size)]
        token_list += ["<blank>"] * (VOCAB_SIZE - len(token_list))  # idx 8000

        scorers = {
            "decoder": self.model.decoder,
            "ctc": KoreanCTCPrefixScorer(self.model.ctc, self.eos_id, self.blank_id),
            "length_bonus": LengthBonus(VOCAB_SIZE),
        }
        weights = {
            "decoder": 1.0 - self.ctc_weight,
            "ctc": self.ctc_weight,
            "length_bonus": penalty,
        }
        self.beam_search = BatchBeamSearch(
            beam_size=self.beam_size,
            vocab_size=VOCAB_SIZE,
            weights=weights,
            scorers=scorers,
            sos=self.sos_id,
            eos=self.eos_id,
            token_list=token_list,
            pre_beam_score_key=None if self.ctc_weight == 1.0 else "decoder",
        )
        self.beam_search.to(device=self.device).eval()

    def _ctc_greedy(self, logits: torch.Tensor) -> List[int]:
        ids = logits.argmax(dim=-1).cpu().tolist()
        collapsed: List[int] = []
        prev = -1
        for tok in ids:
            if tok != prev:
                if tok != self.blank_id:
                    collapsed.append(tok)
                prev = tok
        return collapsed

    def _decode_pieces(self, ids: List[int]) -> str:
        piece_size = self.sp.GetPieceSize()
        # drop sos/eos/blank(8000) and the sp control ids <s>=1 / </s>=2
        ids = [i for i in ids if 0 <= i < piece_size and i not in (1, 2)]
        return self.sp.DecodeIds(ids)

    def infer(self, data) -> str:
        with torch.no_grad():
            if isinstance(data, tuple):
                x = data[0].to(self.device)
            else:
                x = data.to(self.device)

            # E2E.encode adds the batch dim (unsqueeze(0)) and returns (T, adim)
            enc_feats = self.model.encode(x)

            if self.decode == "hybrid":
                nbest = self.beam_search(enc_feats)
                if not nbest:
                    return ""
                yseq = nbest[0].yseq.tolist()
                # strip leading sos and trailing eos
                if yseq and yseq[0] == self.sos_id:
                    yseq = yseq[1:]
                if yseq and yseq[-1] == self.eos_id:
                    yseq = yseq[:-1]
                text = self._decode_pieces(yseq)
            else:
                ctc_logits = self.model.ctc.ctc_lo(enc_feats)
                ids = self._ctc_greedy(ctc_logits)
                text = self._decode_pieces(ids)
        return text.strip()
