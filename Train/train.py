"""
train.py - Korean VSR Two-Stage Transfer Learning Script (DOKDO).

Implements the exact two-stage transfer learning protocol described in the DOKDO paper
(Section II.3 & Figure 1):
  - Base Architecture: Auto-AVSR (LRS3 visual-only pretrained, WER 19.1%)
  - Korean Vocab Head Replacement: 5,049 English units -> 8,001 Korean SentencePiece Unigram units
  - Hybrid Loss: L = ctc_weight * L_CTC + (1 - ctc_weight) * L_CE  (default: ctc_weight = 0.3)
  - Phase 1 (Epochs 1..freeze_epochs, default 5):
      Freeze Visual Frontend + Conformer Encoder, train newly initialized Head/Decoder (lr_head = 1e-3)
  - Phase 2 (Epochs freeze_epochs+1..epochs, default 6..109):
      Unfreeze entire network with differential learning rates (lr_encoder = 1e-5, lr_head = 1e-3)
  - Sequential Domain Adaptation (OLKAVS -> KMSAV):
      Pass `--pretrained_kr_ckpt <path/to/olkavs_best_model.pt>` to continue adaptation on KMSAV.
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

TRAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRAIN_SCRIPT_DIR)
CHAPLIN_ROOT = os.environ.get("CHAPLIN_ROOT", REPO_ROOT)
if CHAPLIN_ROOT not in sys.path:
    sys.path.insert(0, CHAPLIN_ROOT)
if TRAIN_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, TRAIN_SCRIPT_DIR)

from dataset import KoreanVSRDataset, collate_fn
from model import KoreanVSRModel
from tokenizer_utils import load_tokenizer

# =============================================================
# Default Configuration (Matches Paper Section II.3 & III.1)
# =============================================================

DEFAULT_TRAIN_DIR = os.path.join(REPO_ROOT, "data", "preprocessed", "train")
DEFAULT_VAL_DIR   = os.path.join(REPO_ROOT, "data", "preprocessed", "validation")
DEFAULT_TEST_DIR  = os.path.join(REPO_ROOT, "data", "preprocessed", "test")

DEFAULT_CHAPLIN_CHECKPOINT = os.path.join(
    CHAPLIN_ROOT, "benchmarks", "LRS3", "models", "LRS3_V_WER19.1", "model.pth"
)
DEFAULT_MODEL_CONF = os.path.join(
    CHAPLIN_ROOT, "benchmarks", "LRS3", "models", "LRS3_V_WER19.1", "model.json"
)
DEFAULT_SPM_MODEL = os.path.join(REPO_ROOT, "pipelines", "tokens", "unigram8000.model")
DEFAULT_OUTPUT_DIR = os.path.join(TRAIN_SCRIPT_DIR, "output")

# Paper default hyperparameters (Section II.3 / III.1)
EPOCHS        = 109
BATCH_SIZE    = 8
GRAD_ACCUM    = 2
WEIGHT_DECAY  = 0.01
NUM_WORKERS   = 4
MAX_FRAMES    = None

# Two-stage transfer learning (Phase 1: 5 epochs freeze -> Phase 2: full differential LR)
FREEZE_EPOCHS = 5
LR_ENCODER    = 1e-5
LR_HEAD       = 1e-3
WARMUP_EPOCHS = 3

# Hybrid CTC + Attention CE loss weight (L = 0.3 * L_CTC + 0.7 * L_CE)
LOSS_TYPE     = "hybrid"   # "hybrid" (paper Figure 1) or "ctc" (CTC-only ablation)
CTC_WEIGHT    = 0.3

# Mixed Precision
USE_AMP   = True
AMP_DTYPE = torch.bfloat16


# =============================================================
# Utilities
# =============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def ctc_loss_forward(ctc_loss_fn, logits, labels, output_lengths, label_lengths, device):
    log_probs = logits.float().log_softmax(2).permute(1, 0, 2)
    if device.type == "mps":
        loss = ctc_loss_fn(log_probs.cpu(), labels.cpu(), output_lengths.cpu(), label_lengths.cpu())
    else:
        loss = ctc_loss_fn(log_probs, labels, output_lengths, label_lengths)
    return loss


def compute_batch_loss(model, videos, labels, video_lengths, label_lengths,
                       ctc_loss_fn, device, loss_type="hybrid", ctc_weight=0.3):
    """Compute either Hybrid (CTC + Decoder CE) or CTC-only loss."""
    if loss_type == "hybrid" and ctc_weight < 1.0:
        logits, output_lengths, loss_att = model.forward_hybrid(
            videos, video_lengths, labels, label_lengths
        )
        loss_ctc = ctc_loss_forward(
            ctc_loss_fn, logits, labels, output_lengths, label_lengths, device
        )
        loss = ctc_weight * loss_ctc + (1.0 - ctc_weight) * loss_att
        return loss, logits, output_lengths
    else:
        logits, output_lengths = model(videos, video_lengths)
        loss = ctc_loss_forward(
            ctc_loss_fn, logits, labels, output_lengths, label_lengths, device
        )
        return loss, logits, output_lengths


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def get_phase(epoch, freeze_epochs):
    return 1 if epoch <= freeze_epochs else 2


def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s"


def build_scheduler_phase1(optimizer, freeze_epochs, warmup_epochs, lr_head, start_epoch=1):
    """Phase 1 (head only): warmup + cosine."""
    w_epochs = min(warmup_epochs, max(1, freeze_epochs - 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=w_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, freeze_epochs - w_epochs), eta_min=lr_head * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[w_epochs]
    )
    for _ in range(start_epoch - 1):
        scheduler.step()
    return scheduler


def build_scheduler_phase2(optimizer, epochs, freeze_epochs, lr_encoder, start_epoch=1):
    """Phase 2 (full fine-tuning): warmup + cosine."""
    remaining = max(1, epochs - freeze_epochs)
    warmup_steps = min(3, max(1, remaining - 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, remaining - warmup_steps), eta_min=lr_encoder * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
    phase2_elapsed = start_epoch - freeze_epochs - 1
    for _ in range(max(0, phase2_elapsed)):
        scheduler.step()
    return scheduler


# =============================================================
# Training / Validation
# =============================================================

def train_one_epoch(model, train_loader, ctc_loss_fn, optimizer, scaler, device,
                    epoch, epochs, freeze_epochs, grad_accum, loss_type, ctc_weight,
                    use_amp=True, amp_dtype=torch.bfloat16):
    model.train()
    total_loss = 0.0
    num_batches = 0
    n_total = len(train_loader)
    phase = get_phase(epoch, freeze_epochs)
    epoch_start = time.time()
    optimizer.zero_grad()

    for batch_idx, (videos, labels, video_lengths, label_lengths) in enumerate(train_loader):
        videos = videos.to(device)
        labels = labels.to(device)
        video_lengths = video_lengths.to(device)
        label_lengths = label_lengths.to(device)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")):
            loss, logits, output_lengths = compute_batch_loss(
                model, videos, labels, video_lengths, label_lengths,
                ctc_loss_fn, device, loss_type=loss_type, ctc_weight=ctc_weight
            )
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        total_loss += loss.item() * grad_accum
        num_batches += 1

        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == n_total:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        del videos, labels, logits, output_lengths, loss
        if device.type == "mps":
            torch.mps.empty_cache()

        pct = (batch_idx + 1) / n_total * 100
        avg = total_loss / num_batches
        elapsed = time.time() - epoch_start
        eta = elapsed / (batch_idx + 1) * (n_total - batch_idx - 1)
        stage = "Head" if phase == 1 else "Full"
        if (batch_idx + 1) == n_total or (batch_idx + 1) % max(1, n_total // 10) == 0:
            print(f"  Epoch {epoch}/{epochs} [{stage}] Train {batch_idx+1}/{n_total} ({pct:.0f}%) | Loss: {avg:.4f} | ETA: {fmt_time(eta)}")

    return total_loss / max(num_batches, 1)


def validate(model, data_loader, ctc_loss_fn, device, epoch, epochs, freeze_epochs,
             loss_type, ctc_weight, split_name="Val", use_amp=True, amp_dtype=torch.bfloat16):
    if data_loader is None or len(data_loader) == 0:
        return float("nan")
    model.eval()
    total_loss = 0.0
    num_batches = 0
    num_inf = 0
    n_total = len(data_loader)
    phase = get_phase(epoch, freeze_epochs)
    val_start = time.time()

    with torch.no_grad():
        for batch_idx, (videos, labels, video_lengths, label_lengths) in enumerate(data_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            video_lengths = video_lengths.to(device)
            label_lengths = label_lengths.to(device)

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")):
                loss, logits, output_lengths = compute_batch_loss(
                    model, videos, labels, video_lengths, label_lengths,
                    ctc_loss_fn, device, loss_type=loss_type, ctc_weight=ctc_weight
                )

            if torch.isinf(loss) or torch.isnan(loss):
                num_inf += 1
            else:
                total_loss += loss.item()
                num_batches += 1

            del videos, labels, logits, output_lengths, loss
            if device.type == "mps":
                torch.mps.empty_cache()

            pct = (batch_idx + 1) / n_total * 100
            avg = total_loss / max(num_batches, 1)
            elapsed = time.time() - val_start
            eta = elapsed / (batch_idx + 1) * (n_total - batch_idx - 1)
            stage = "Head" if phase == 1 else "Full"
            if (batch_idx + 1) == n_total or (batch_idx + 1) % max(1, n_total // 10) == 0:
                print(f"  Epoch {epoch}/{epochs} [{stage}] {split_name:5s} {batch_idx+1}/{n_total} ({pct:.0f}%) | Loss: {avg:.4f} | Inf: {num_inf} | ETA: {fmt_time(eta)}")

    if num_inf > 0:
        print(f"  [{split_name} WARNING] {num_inf}/{n_total} batches had inf/nan loss (skipped)")

    if num_batches == 0:
        return float("inf")
    return total_loss / num_batches


# =============================================================
# Argument Parser & Main
# =============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="DOKDO Korean VSR Two-Stage Transfer Learning (OLKAVS -> KMSAV)"
    )
    parser.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR,
                        help="Preprocessed train .npz directory")
    parser.add_argument("--val_dir", type=str, default=DEFAULT_VAL_DIR,
                        help="Preprocessed validation .npz directory")
    parser.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR,
                        help="Preprocessed test .npz directory (optional)")
    parser.add_argument("--chaplin_checkpoint", type=str, default=DEFAULT_CHAPLIN_CHECKPOINT,
                        help="LRS3 pretrained backbone checkpoint (model.pth)")
    parser.add_argument("--pretrained_kr_ckpt", type=str, default=None,
                        help="Stage-1 Korean checkpoint (e.g. OLKAVS best_model.pt) for KMSAV domain adaptation")
    parser.add_argument("--model_conf", type=str, default=DEFAULT_MODEL_CONF,
                        help="Path to model.json")
    parser.add_argument("--spm_model", type=str, default=DEFAULT_SPM_MODEL,
                        help="Path to SentencePiece unigram8000.model")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save checkpoints and train_metrics.pt")
    parser.add_argument("--loss_type", type=str, default=LOSS_TYPE, choices=["hybrid", "ctc"],
                        help="Loss function: 'hybrid' (0.3 CTC + 0.7 CE, paper Figure 1) or 'ctc' (CTC-only)")
    parser.add_argument("--ctc_weight", type=float, default=CTC_WEIGHT,
                        help="CTC loss weight lambda in hybrid loss (default: 0.3)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Total training epochs (default: 109)")
    parser.add_argument("--freeze_epochs", type=int, default=FREEZE_EPOCHS,
                        help="Phase 1 encoder freeze epochs (default: 5, matching paper Section II.3)")
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS,
                        help="Warmup epochs (default: 3)")
    parser.add_argument("--lr_encoder", type=float, default=LR_ENCODER,
                        help="Phase 2 Conformer encoder learning rate (default: 1e-5)")
    parser.add_argument("--lr_head", type=float, default=LR_HEAD,
                        help="Head / Decoder learning rate (default: 1e-3)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Per-step batch size (default: 8)")
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM,
                        help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES,
                        help="Optional max frame truncation")
    parser.add_argument("--sos_id", type=int, default=1,
                        help="Decoder SOS token ID (1 for <s> or 8000 for legacy)")
    parser.add_argument("--eos_id", type=int, default=2,
                        help="Decoder EOS token ID (2 for </s> or 8000 for legacy)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Mixed Precision: {USE_AMP} ({AMP_DTYPE})")
    print(f"Loss Type: {args.loss_type} (ctc_weight={args.ctc_weight})")
    print(f"Two-Stage Schedule: Phase 1 Freeze={args.freeze_epochs} epochs (lr_head={args.lr_head:.1e}) "
          f"-> Phase 2 Full up to {args.epochs} epochs (lr_encoder={args.lr_encoder:.1e}, lr_head={args.lr_head:.1e})")

    tokenizer = load_tokenizer(args.spm_model)
    print(f"Vocab size: {tokenizer.vocab_size}, Blank ID: {tokenizer.blank_id}")

    from pipelines.data.transforms import VideoTransform
    train_transform = VideoTransform(speed_rate=1)
    print("Train augmentation: VideoTransform enabled")

    train_dataset = KoreanVSRDataset(
        data_dir=args.train_dir, video_transform=train_transform,
        tokenizer=tokenizer, max_frames=args.max_frames
    )
    val_dataset = KoreanVSRDataset(
        data_dir=args.val_dir, video_transform=None,
        tokenizer=tokenizer, max_frames=args.max_frames
    )
    test_dataset = None
    if args.test_dir and os.path.isdir(args.test_dir):
        test_dataset = KoreanVSRDataset(
            data_dir=args.test_dir, video_transform=None,
            tokenizer=tokenizer, max_frames=args.max_frames
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = None
    if test_dataset is not None and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
        )

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples"
          + (f", Test: {len(test_dataset)} samples" if test_dataset else ""))
    print(f"Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} effective")

    init_ckpt = args.pretrained_kr_ckpt if args.pretrained_kr_ckpt else args.chaplin_checkpoint
    model = KoreanVSRModel(
        chaplin_checkpoint=init_ckpt,
        model_conf=args.model_conf,
        vocab_size=tokenizer.vocab_size,
        device=str(device),
        sos_id=args.sos_id,
        eos_id=args.eos_id,
    )
    model = model.to(device)

    # Separate Visual Frontend + Conformer Encoder from CTC Head / Decoder parameters
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("model.encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    # Phase 1 initial state: freeze encoder, train head/decoder only
    if args.freeze_epochs > 0:
        for p in encoder_params:
            p.requires_grad = False

    stats = count_parameters(model)
    print(f"Total: {stats['total']:,} | Trainable: {stats['trainable']:,} | Frozen: {stats['frozen']:,}")

    ctc_loss_train = nn.CTCLoss(blank=tokenizer.blank_id, reduction="mean", zero_infinity=True)
    ctc_loss_val   = nn.CTCLoss(blank=tokenizer.blank_id, reduction="mean", zero_infinity=False)

    scaler = GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))

    if args.freeze_epochs > 0:
        optimizer = torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)
        scheduler = build_scheduler_phase1(
            optimizer, args.freeze_epochs, args.warmup_epochs, args.lr_head
        )
    else:
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ], weight_decay=args.weight_decay)
        scheduler = build_scheduler_phase2(
            optimizer, args.epochs, 0, args.lr_encoder
        )

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    meta = {
        "mode": "staged_differential_lr",
        "loss_type": args.loss_type,
        "ctc_weight": float(args.ctc_weight),
        "vocab_size": tokenizer.vocab_size,
        "blank_id": tokenizer.blank_id,
        "sos_id": int(args.sos_id),
        "eos_id": int(args.eos_id),
        "lr_encoder": float(args.lr_encoder),
        "lr_head": float(args.lr_head),
        "freeze_epochs": int(args.freeze_epochs),
        "warmup_epochs": int(args.warmup_epochs),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "amp_dtype": str(AMP_DTYPE),
        "device": str(device),
    }

    hist = {
        "meta": meta,
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "lr": [],
        "epoch_time": [],
    }

    ckpt_last = os.path.join(ckpt_dir, "ckpt_last.pt")
    metrics_pt = os.path.join(args.output_dir, "train_metrics.pt")

    start_epoch = 1
    best_val_loss = float("inf")

    if os.path.exists(ckpt_last):
        ckpt = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        resumed_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model_state_dict"])

        next_phase = get_phase(resumed_epoch + 1, args.freeze_epochs)
        saved_phase = ckpt.get("phase", get_phase(resumed_epoch, args.freeze_epochs))

        if next_phase == 2:
            for p in encoder_params:
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": encoder_params, "lr": args.lr_encoder},
                {"params": head_params, "lr": args.lr_head},
            ], weight_decay=args.weight_decay)
            scheduler = build_scheduler_phase2(
                optimizer, args.epochs, args.freeze_epochs, args.lr_encoder,
                start_epoch=resumed_epoch + 1
            )
            if saved_phase == 2:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            optimizer = torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler = build_scheduler_phase1(
                optimizer, args.freeze_epochs, args.warmup_epochs, args.lr_head,
                start_epoch=resumed_epoch + 1
            )

        hist = ckpt.get("hist", hist)
        meta = ckpt.get("meta", meta)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = resumed_epoch + 1
        model = model.to(device)
        print(f"[RESUME] epoch {resumed_epoch} -> resuming from {start_epoch}")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()

            # Phase transition: Phase 1 (head-only) -> Phase 2 (full differential LR)
            if epoch == args.freeze_epochs + 1 and len(encoder_params) > 0 and not encoder_params[0].requires_grad:
                for p in encoder_params:
                    p.requires_grad = True
                optimizer = torch.optim.AdamW([
                    {"params": encoder_params, "lr": args.lr_encoder},
                    {"params": head_params, "lr": args.lr_head},
                ], weight_decay=args.weight_decay)
                scheduler = build_scheduler_phase2(
                    optimizer, args.epochs, args.freeze_epochs, args.lr_encoder
                )
                stats = count_parameters(model)
                print(f"  [Phase 2] Encoder unfrozen | Trainable: {stats['trainable']:,}")

            train_loss = train_one_epoch(
                model, train_loader, ctc_loss_train, optimizer, scaler, device,
                epoch, args.epochs, args.freeze_epochs, args.grad_accum,
                args.loss_type, args.ctc_weight, USE_AMP, AMP_DTYPE
            )
            val_loss = validate(
                model, val_loader, ctc_loss_val, device, epoch, args.epochs,
                args.freeze_epochs, args.loss_type, args.ctc_weight,
                split_name="Val", use_amp=USE_AMP, amp_dtype=AMP_DTYPE
            )
            test_loss = validate(
                model, test_loader, ctc_loss_val, device, epoch, args.epochs,
                args.freeze_epochs, args.loss_type, args.ctc_weight,
                split_name="Test", use_amp=USE_AMP, amp_dtype=AMP_DTYPE
            ) if test_loader is not None else float("nan")

            scheduler.step()

            elapsed = time.time() - epoch_start
            cur_lr = optimizer.param_groups[0]["lr"]
            phase = get_phase(epoch, args.freeze_epochs)

            hist["epochs"].append(epoch)
            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_loss)
            hist["test_loss"].append(test_loss)
            hist["lr"].append(cur_lr)
            hist["epoch_time"].append(elapsed)

            print(f"  => Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f} | LR: {cur_lr:.2e} | {fmt_time(elapsed)}")

            ckpt_tmp = ckpt_last + ".tmp"
            torch.save({
                "epoch": epoch,
                "phase": phase,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "hist": hist,
                "meta": meta,
                "best_val_loss": best_val_loss,
            }, ckpt_tmp)
            os.replace(ckpt_tmp, ckpt_last)

            if val_loss < best_val_loss and val_loss > 0:
                best_val_loss = val_loss
                best_tmp = os.path.join(ckpt_dir, "best_model.pt.tmp")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "test_loss": test_loss,
                    "meta": meta,
                }, best_tmp)
                os.replace(best_tmp, os.path.join(ckpt_dir, "best_model.pt"))
                print(f"  -> Best model saved (val_loss: {val_loss:.4f})")

            torch.save(hist, metrics_pt)

        print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    except KeyboardInterrupt:
        print("Training interrupted manually.")


if __name__ == "__main__":
    main()
