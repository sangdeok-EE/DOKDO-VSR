"""
evaluate.py - Korean VSR Evaluation Script (CER & WER).

Evaluates a trained Korean VSR model on a preprocessed `.npz` test directory
(or compares a manifest TSV and prediction TSV) and reports:
  - CER (Character Error Rate, Levenshtein edit distance at syllable/character level)
  - WER (Word Error Rate, Levenshtein edit distance at whitespace-delimited eojeol level)

Supports both decoding modes implemented in `pipelines/model_korean.py`:
  - `--decode ctc`    : CTC Greedy decoding + SentencePiece detokenization (Paper Section II.2 / III.2)
  - `--decode hybrid` : Joint CTC + Attention Beam Search decoding

Usage:
  1. Evaluate on preprocessed .npz test directory (CTC Greedy, default):
     python Train/evaluate.py \
         --test_dir /path/to/preprocessed/test \
         --checkpoint korean_vsr_hybrid.pt \
         --spm_model pipelines/tokens/unigram8000.model \
         --decode ctc

  2. Evaluate with Joint CTC/Attention Beam Search:
     python Train/evaluate.py \
         --test_dir /path/to/preprocessed/test \
         --checkpoint korean_vsr_hybrid.pt \
         --spm_model pipelines/tokens/unigram8000.model \
         --decode hybrid --ctc_weight 0.3 --beam_size 10

  3. Evaluate pre-generated TSV predictions against a manifest TSV:
     python Train/evaluate.py --manifest manifest.tsv --pred_tsv predictions.tsv --strip_punct
"""

import os
import sys
import csv
import re
import unicodedata
import argparse
import torch
from torch.utils.data import DataLoader

TRAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRAIN_SCRIPT_DIR)
CHAPLIN_ROOT = os.environ.get("CHAPLIN_ROOT", REPO_ROOT)
if CHAPLIN_ROOT not in sys.path:
    sys.path.insert(0, CHAPLIN_ROOT)
if TRAIN_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, TRAIN_SCRIPT_DIR)

from dataset import KoreanVSRDataset, collate_fn
from tokenizer_utils import load_tokenizer
from pipelines.model_korean import KoreanAVSR


# =============================================================
# Text Normalization & Edit Distance Metrics (CER / WER)
# =============================================================

def normalize_text(text: str, strip_punct: bool = False) -> str:
    """Normalize Unicode NFC and optionally strip punctuation."""
    s = unicodedata.normalize("NFC", str(text))
    if strip_punct:
        s = re.sub(r"[^\w\s\u3131-\u318E\uAC00-\uD7A3]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def edit_distance(ref, hyp) -> int:
    """Standard Levenshtein edit distance (insertions + deletions + substitutions)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]


def compute_cer(predictions, references, strip_punct=False, ignore_space=False):
    """Compute micro-averaged Character Error Rate (CER)."""
    total_edits = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        r = normalize_text(ref, strip_punct=strip_punct)
        p = normalize_text(pred, strip_punct=strip_punct)
        if ignore_space:
            r = re.sub(r"\s+", "", r)
            p = re.sub(r"\s+", "", p)
        total_edits += edit_distance(list(r), list(p))
        total_chars += len(r)
    return total_edits / max(total_chars, 1)


def compute_wer(predictions, references, strip_punct=False):
    """Compute micro-averaged Word Error Rate (WER, whitespace-delimited eojeol units)."""
    total_edits = 0
    total_words = 0
    for pred, ref in zip(predictions, references):
        r_words = normalize_text(ref, strip_punct=strip_punct).split()
        p_words = normalize_text(pred, strip_punct=strip_punct).split()
        total_edits += edit_distance(r_words, p_words)
        total_words += len(r_words)
    return total_edits / max(total_words, 1)


def evaluate_tsv(manifest_path, pred_tsv_path, strip_punct=False, ignore_space=False, output_file=None):
    """Evaluate predictions from TSV files (manifest.tsv vs predictions.tsv)."""
    def _load(p):
        with open(p, "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    refs = {r["path"].replace("\\", "/"): r.get("text", "") for r in _load(manifest_path)}
    hyps = {r["path"].replace("\\", "/"): r.get("pred", "") for r in _load(pred_tsv_path)}
    common = sorted(set(refs) & set(hyps))
    if not common:
        raise RuntimeError("No matching paths between --manifest and --pred_tsv")

    ref_list = [refs[k] for k in common]
    hyp_list = [hyps[k] for k in common]
    cer = compute_cer(hyp_list, ref_list, strip_punct=strip_punct, ignore_space=ignore_space)
    wer = compute_wer(hyp_list, ref_list, strip_punct=strip_punct)

    print(f"\n{'=' * 60}")
    print(f"TSV Evaluation Results ({len(common)} / {len(refs)} samples)")
    print(f"{'=' * 60}")
    print(f"CER: {cer * 100:.2f}%")
    print(f"WER: {wer * 100:.2f}%")

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"CER: {cer * 100:.2f}%\nWER: {wer * 100:.2f}%\nSamples: {len(common)}\n\n")
            for idx, k in enumerate(common, 1):
                f.write(f"[{idx}] {k}\n  REF : {refs[k]}\n  PRED: {hyps[k]}\n\n")
        print(f"Saved detailed report to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Korean VSR model (CER / WER)")
    parser.add_argument("--test_dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "preprocessed", "test"),
                        help="Path to preprocessed test .npz directory")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(REPO_ROOT, "korean_vsr_hybrid.pt"),
                        help="Path to trained Korean VSR checkpoint (.pt)")
    parser.add_argument("--model_conf", type=str,
                        default=os.path.join(REPO_ROOT, "benchmarks", "LRS3", "models", "LRS3_V_WER19.1", "model.json"),
                        help="Path to model.json")
    parser.add_argument("--spm_model", type=str,
                        default=os.path.join(REPO_ROOT, "pipelines", "tokens", "unigram8000.model"),
                        help="Path to SentencePiece unigram8000.model")
    parser.add_argument("--decode", type=str, default="ctc", choices=["ctc", "hybrid"],
                        help="Decoding mode: 'ctc' (CTC Greedy, paper Section III.2) or 'hybrid' (Beam Search)")
    parser.add_argument("--ctc_weight", type=float, default=0.3,
                        help="CTC weight for hybrid beam search (default: 0.3)")
    parser.add_argument("--beam_size", type=int, default=10,
                        help="Beam size for hybrid beam search (default: 10)")
    parser.add_argument("--sos_id", type=int, default=1,
                        help="SOS token ID for hybrid decoding (default: 1)")
    parser.add_argument("--eos_id", type=int, default=2,
                        help="EOS token ID for hybrid decoding (default: 2)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for CTC greedy evaluation (default: 16)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Optional max frame truncation")
    parser.add_argument("--strip_punct", action="store_true",
                        help="Strip punctuation before computing CER/WER")
    parser.add_argument("--cer_ignore_space", action="store_true",
                        help="Remove whitespace when computing CER")
    parser.add_argument("--num_show", type=int, default=10,
                        help="Number of sample predictions to print (default: 10)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional path to save full evaluation results (.txt)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Optional manifest TSV for offline TSV evaluation")
    parser.add_argument("--pred_tsv", type=str, default=None,
                        help="Optional predictions TSV for offline TSV evaluation")
    args = parser.parse_args()

    if args.manifest and args.pred_tsv:
        evaluate_tsv(
            args.manifest, args.pred_tsv,
            strip_punct=args.strip_punct,
            ignore_space=args.cer_ignore_space,
            output_file=args.output_file,
        )
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Decode mode: {args.decode}")

    tokenizer = load_tokenizer(args.spm_model)
    test_dataset = KoreanVSRDataset(
        data_dir=args.test_dir,
        video_transform=None,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )
    print(f"Test dir: {args.test_dir} ({len(test_dataset)} samples)")

    avsr = KoreanAVSR(
        model_path=args.checkpoint,
        model_conf=args.model_conf,
        spm_model=args.spm_model,
        device=device,
        decode=args.decode,
        ctc_weight=args.ctc_weight,
        beam_size=args.beam_size,
        sos_id=args.sos_id,
        eos_id=args.eos_id,
    )

    all_predictions = []
    all_references = []

    if args.decode == "ctc":
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        n_total = len(test_loader)
        with torch.no_grad():
            for batch_idx, (videos, labels, video_lengths, label_lengths) in enumerate(test_loader):
                videos = videos.to(device)
                enc_out, _ = avsr.model.encoder(videos, None)
                logits = avsr.model.ctc.ctc_lo(enc_out)
                pred_ids = logits.argmax(dim=2)

                for i in range(pred_ids.shape[0]):
                    v_len = video_lengths[i].item()
                    ids = pred_ids[i, :v_len].cpu().tolist()
                    pred_text = tokenizer.decode_ctc(ids)
                    all_predictions.append(pred_text)

                    l_len = label_lengths[i].item()
                    ref_ids = labels[i, :l_len].tolist()
                    ref_text = tokenizer.decode(ref_ids)
                    all_references.append(ref_text)

                pct = (batch_idx + 1) / max(n_total, 1) * 100
                if (batch_idx + 1) == n_total or (batch_idx + 1) % max(1, n_total // 10) == 0:
                    print(f"  Eval: {batch_idx + 1}/{n_total} ({pct:.0f}%)")
    else:
        n_total = len(test_dataset)
        for idx in range(n_total):
            video, label, _, _ = test_dataset[idx]
            pred_text = avsr.infer(video)
            ref_text = tokenizer.decode(label.tolist())
            all_predictions.append(pred_text)
            all_references.append(ref_text)
            if (idx + 1) == n_total or (idx + 1) % max(1, n_total // 10) == 0:
                print(f"  Eval (hybrid beam search): {idx + 1}/{n_total} ({(idx + 1) / n_total * 100:.0f}%)")

    cer = compute_cer(
        all_predictions, all_references,
        strip_punct=args.strip_punct,
        ignore_space=args.cer_ignore_space,
    )
    wer = compute_wer(
        all_predictions, all_references,
        strip_punct=args.strip_punct,
    )

    print(f"\n{'=' * 60}")
    print(f"Results on {len(all_predictions)} test samples (decode={args.decode})")
    print(f"{'=' * 60}")
    print(f"CER: {cer * 100:.2f}%")
    print(f"WER: {wer * 100:.2f}%")

    print(f"\n{'=' * 60}")
    print("Sample predictions vs ground truth")
    print(f"{'=' * 60}")
    for i in range(min(args.num_show, len(all_predictions))):
        print(f"\n[{i + 1}]")
        print(f"  REF : {all_references[i]}")
        print(f"  PRED: {all_predictions[i]}")

    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f"Decode: {args.decode}\n")
            f.write(f"CER: {cer * 100:.2f}%\n")
            f.write(f"WER: {wer * 100:.2f}%\n")
            f.write(f"Test samples: {len(all_predictions)}\n\n")
            for i in range(len(all_predictions)):
                f.write(f"[{i + 1}]\n")
                f.write(f"  REF : {all_references[i]}\n")
                f.write(f"  PRED: {all_predictions[i]}\n\n")
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
