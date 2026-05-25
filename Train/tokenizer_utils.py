"""
tokenizer_utils.py
==================
SentencePiece unigram8000 wrapper for Korean VSR fine-tuning.

Special token layout:
    <unk>  = 0
    <s>    = 1   (BOS)
    </s>   = 2   (EOS)
    CTC blank = 8000
    Total vocab size  = 8001
"""

import os
from pathlib import Path
from typing import List

try:
    import sentencepiece as spm
except ImportError as exc:
    raise ImportError(
        "sentencepiece is required.  Install via: pip install sentencepiece"
    ) from exc


UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
CTC_BLANK_ID = 8000
VOCAB_SIZE = 8001


class KoreanTokenizer:
    """SentencePiece-based tokenizer for Korean VSR."""

    def __init__(self, model_path: str) -> None:
        model_path = str(Path(model_path).resolve())
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"SentencePiece model not found: {model_path}"
            )

        self.sp = spm.SentencePieceProcessor()
        try:
            self.sp.Load(model_path)
        except OSError as exc:
            raise OSError(f"Failed to load SentencePiece model: {exc}") from exc

        self.vocab_size = VOCAB_SIZE
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
        self.blank_id = CTC_BLANK_ID

        print(f"[KoreanTokenizer] Loaded model from {model_path}")
        print(
            f"[KoreanTokenizer] vocab_size={self.vocab_size}, "
            f"bos={self.bos_id}, eos={self.eos_id}, blank={self.blank_id}"
        )

    def encode(self, text: str) -> List[int]:
        return self.sp.EncodeAsIds(text)

    def encode_with_special(self, text: str) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        return [self.bos_id] + ids + [self.eos_id]

    def encode_pieces(self, text: str) -> List[str]:
        return self.sp.EncodeAsPieces(text)

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if skip_special:
            ids = [
                i for i in ids
                if i not in (self.bos_id, self.eos_id, self.blank_id)
                and i != -1
            ]
        ids = [i for i in ids if 0 <= i < self.sp.GetPieceSize()]
        return self.sp.DecodeIds(ids)

    def decode_ctc(self, ids: List[int]) -> str:
        """Decode CTC output: remove consecutive duplicates and blank tokens."""
        collapsed: List[int] = []
        prev = -1
        for token_id in ids:
            if token_id != prev:
                if token_id != self.blank_id:
                    collapsed.append(token_id)
                prev = token_id
        return self.decode(collapsed, skip_special=True)

    def id_to_piece(self, token_id: int) -> str:
        if token_id == self.blank_id:
            return "<blank>"
        if 0 <= token_id < self.sp.GetPieceSize():
            return self.sp.IdToPiece(token_id)
        return "<oov>"

    def piece_to_id(self, piece: str) -> int:
        return self.sp.PieceToId(piece)

    def __len__(self) -> int:
        return self.vocab_size


def load_tokenizer(spm_model_path: str) -> KoreanTokenizer:
    return KoreanTokenizer(spm_model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the KoreanTokenizer.")
    parser.add_argument("--spm_model", type=str, required=True,
                        help="Path to unigram8000.model")
    parser.add_argument("--text", type=str, default="안녕하세요 반갑습니다",
                        help="Test sentence to encode/decode")
    args = parser.parse_args()

    print("=" * 60)
    print("KoreanTokenizer smoke test")
    print("=" * 60)

    tokenizer = load_tokenizer(args.spm_model)

    print(f"\nInput text : {args.text!r}")

    ids = tokenizer.encode(args.text)
    print(f"Encoded ids         : {ids}")

    ids_special = tokenizer.encode_with_special(args.text)
    print(f"Encoded (w/ special): {ids_special}")

    pieces = tokenizer.encode_pieces(args.text)
    print(f"Pieces              : {pieces}")

    decoded = tokenizer.decode(ids)
    print(f"Decoded             : {decoded!r}")

    ctc_ids = [ids[0], ids[0], CTC_BLANK_ID, ids[1]] if len(ids) >= 2 else ids
    ctc_text = tokenizer.decode_ctc(ctc_ids)
    print(f"CTC decoded ({ctc_ids}): {ctc_text!r}")

    print(f"\nVocab size  : {len(tokenizer)}")
    print(f"BOS piece   : {tokenizer.id_to_piece(BOS_ID)!r}")
    print(f"EOS piece   : {tokenizer.id_to_piece(EOS_ID)!r}")
    print(f"Blank piece : {tokenizer.id_to_piece(CTC_BLANK_ID)!r}")
    print("\nDone.")
