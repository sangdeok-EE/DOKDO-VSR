"""
train_tokenizer.py - Korean SentencePiece Unigram Tokenizer Training Script.

Trains the 8,000-subword Korean SentencePiece Unigram tokenizer (`unigram8000.model`)
used in DOKDO (Section II.2).

Note on Redistribution:
  Because the tokenizer vocabulary is induced directly from AI Hub Korean lip-reading
  corpora (OLKAVS / KMSAV), the binary `.model` file is subject to AI Hub's dataset
  redistribution policy. Users who obtain the dataset from AI Hub can reproduce the
  tokenizer layout (`<unk>=0, <s>=1, </s>=2, pieces=3..7999`, plus `CTC blank=8000`
  in `tokenizer_utils.py`, yielding `vocab_size=8,001`) using this script.

Usage:
  1. From AI Hub JSON annotation directory (OLKAVS / KMSAV):
     python Train/train_tokenizer.py --json_dir /path/to/json_labels \
         --output_prefix pipelines/tokens/unigram8000

  2. From preprocessed .npz directory:
     python Train/train_tokenizer.py --npz_dir /path/to/preprocessed/train \
         --output_prefix pipelines/tokens/unigram8000

  3. From a plain-text corpus (one sentence per line):
     python Train/train_tokenizer.py --corpus_txt /path/to/corpus.txt \
         --output_prefix pipelines/tokens/unigram8000
"""

import os
import glob
import json
import argparse
import tempfile
import numpy as np

try:
    import sentencepiece as spm
except ImportError as exc:
    raise ImportError(
        "sentencepiece is required. Install via: pip install sentencepiece"
    ) from exc


def collect_sentences_from_json(json_dir):
    sentences = []
    json_files = sorted(glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True))
    for jp in json_files:
        if os.path.basename(jp) == "split_info.json":
            continue
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                data = data[0]
            for sent in data.get("Sentence_info", []):
                txt = str(sent.get("sentence_text", "")).strip()
                if txt:
                    sentences.append(txt)
        except Exception as e:
            print(f"[warn] skipped {jp}: {e}")
    return sentences


def collect_sentences_from_npz(npz_dir):
    sentences = []
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))
    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
            if "text" in data:
                txt = str(data["text"]).strip()
                if txt:
                    sentences.append(txt)
        except Exception as e:
            print(f"[warn] skipped {npz_path}: {e}")
    return sentences


def main():
    parser = argparse.ArgumentParser(
        description="Train Korean SentencePiece Unigram 8000 tokenizer for DOKDO VSR"
    )
    parser.add_argument("--json_dir", type=str, default=None,
                        help="Directory containing AI Hub JSON label files")
    parser.add_argument("--npz_dir", type=str, default=None,
                        help="Directory containing preprocessed .npz files")
    parser.add_argument("--corpus_txt", type=str, default=None,
                        help="Plain text file (one sentence per line)")
    parser.add_argument("--output_prefix", type=str,
                        default="pipelines/tokens/unigram8000",
                        help="Output model prefix (creates <prefix>.model and <prefix>.vocab)")
    parser.add_argument("--vocab_size", type=int, default=8000,
                        help="SentencePiece vocabulary size (default: 8000; +1 CTC blank = 8001 total)")
    parser.add_argument("--character_coverage", type=float, default=0.9995,
                        help="Character coverage for Korean (default: 0.9995)")
    args = parser.parse_args()

    if not any([args.json_dir, args.npz_dir, args.corpus_txt]):
        parser.error("Provide at least one input source: --json_dir, --npz_dir, or --corpus_txt")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)

    temp_corpus_path = None
    if args.corpus_txt and not args.json_dir and not args.npz_dir:
        corpus_file = args.corpus_txt
    else:
        sentences = []
        if args.json_dir:
            s_json = collect_sentences_from_json(args.json_dir)
            print(f"[Tokenizer] Collected {len(s_json):,} sentences from JSON directory: {args.json_dir}")
            sentences.extend(s_json)
        if args.npz_dir:
            s_npz = collect_sentences_from_npz(args.npz_dir)
            print(f"[Tokenizer] Collected {len(s_npz):,} sentences from NPZ directory: {args.npz_dir}")
            sentences.extend(s_npz)
        if args.corpus_txt and os.path.isfile(args.corpus_txt):
            with open(args.corpus_txt, "r", encoding="utf-8") as f:
                s_txt = [line.strip() for line in f if line.strip()]
            print(f"[Tokenizer] Collected {len(s_txt):,} sentences from text file: {args.corpus_txt}")
            sentences.extend(s_txt)

        if not sentences:
            raise RuntimeError("No sentences found in the provided input sources.")

        fd, temp_corpus_path = tempfile.mkstemp(suffix=".txt", prefix="dokdo_spm_corpus_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")
        corpus_file = temp_corpus_path
        print(f"[Tokenizer] Wrote {len(sentences):,} total sentences to temporary corpus.")

    try:
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=args.output_prefix,
            vocab_size=args.vocab_size,
            model_type="unigram",
            character_coverage=args.character_coverage,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=-1,
        )
        print(f"[Tokenizer] Successfully trained: {args.output_prefix}.model and {args.output_prefix}.vocab")
        print(f"[Tokenizer] Token layout: <unk>=0, <s>=1, </s>=2, pieces=3..{args.vocab_size - 1}, CTC blank={args.vocab_size} (Total vocab_size={args.vocab_size + 1})")
    finally:
        if temp_corpus_path and os.path.exists(temp_corpus_path):
            os.remove(temp_corpus_path)


if __name__ == "__main__":
    main()
