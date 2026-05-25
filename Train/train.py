"""
train.py - Korean VSR fine-tuning script.

Fine-tunes the Chaplin pretrained model for Korean VSR using CTC loss.
Two-stage training (head-only -> full) with differential LR.
- Mixed Precision (BF16/FP16) for memory savings, FP32 for checkpointing
- Data Augmentation (VideoTransform) for train split only
- LR Warmup to prevent CTC collapse
- Resume support (auto-resume from ckpt_last.pt)
- Metrics saving (train_metrics.pt) for plotting
- Train/Val CTC Loss separation: val uses zero_infinity=False for real loss
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHAPLIN_ROOT = "<path/to/chaplin>"
sys.path.insert(0, CHAPLIN_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from dataset import KoreanVSRDataset, collate_fn
from model import KoreanVSRModel
from tokenizer_utils import load_tokenizer

# =============================================================
# Configuration
# =============================================================

# Dataset paths
TRAIN_DIR = "<path/to/preprocessed>/train"
VAL_DIR   = "<path/to/preprocessed>/validation"
TEST_DIR  = "<path/to/preprocessed>/test"

# Chaplin pretrained model
CHAPLIN_CHECKPOINT = "<path/to/chaplin>/benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth"
MODEL_CONF         = "<path/to/chaplin>/benchmarks/LRS3/models/LRS3_V_WER19.1/model.json"

# SentencePiece tokenizer
SPM_MODEL = "./unigram/unigram8000.model"

# Output directory
OUTPUT_DIR = "./output"

# Training hyperparameters
EPOCHS       = 130
BATCH_SIZE   = 16
GRAD_ACCUM   = 2
WEIGHT_DECAY = 0.01
NUM_WORKERS  = 4
MAX_FRAMES   = None

# Two-stage training: head-only first, then full with differential LR
FREEZE_EPOCHS = 30
LR_ENCODER    = 5e-6
LR_HEAD       = 1e-4

# Warmup
WARMUP_EPOCHS = 3

# Mixed Precision
USE_AMP   = True
AMP_DTYPE = torch.bfloat16   # use BF16 on A100/4090, else switch to FP16

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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def get_phase(epoch):
    return 1 if epoch <= FREEZE_EPOCHS else 2


def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s"


def build_scheduler_phase1(optimizer, start_epoch=1):
    """Phase 1 (head only): warmup + cosine"""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FREEZE_EPOCHS - WARMUP_EPOCHS, eta_min=LR_HEAD * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )
    for _ in range(start_epoch - 1):
        scheduler.step()
    return scheduler


def build_scheduler_phase2(optimizer, start_epoch=1):
    """Phase 2 (full): warmup + cosine"""
    remaining = EPOCHS - FREEZE_EPOCHS
    warmup_steps = min(3, remaining)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining - warmup_steps, eta_min=LR_ENCODER * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
    phase2_elapsed = start_epoch - FREEZE_EPOCHS - 1
    for _ in range(max(0, phase2_elapsed)):
        scheduler.step()
    return scheduler

# =============================================================
# Training / Validation
# =============================================================

def train_one_epoch(model, train_loader, ctc_loss_fn, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    n_total = len(train_loader)
    phase = get_phase(epoch)
    epoch_start = time.time()
    optimizer.zero_grad()

    for batch_idx, (videos, labels, video_lengths, label_lengths) in enumerate(train_loader):
        videos = videos.to(device)
        labels = labels.to(device)
        video_lengths = video_lengths.to(device)
        label_lengths = label_lengths.to(device)

        with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(USE_AMP and device.type == "cuda")):
            logits, output_lengths = model(videos, video_lengths)
            loss = ctc_loss_forward(ctc_loss_fn, logits, labels, output_lengths, label_lengths, device)
            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()

        total_loss += loss.item() * GRAD_ACCUM
        num_batches += 1

        if (batch_idx + 1) % GRAD_ACCUM == 0 or (batch_idx + 1) == n_total:
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
            print(f"  Epoch {epoch}/{EPOCHS} [{stage}] Train {batch_idx+1}/{n_total} ({pct:.0f}%) | Loss: {avg:.4f} | ETA: {fmt_time(eta)}")

    return total_loss / max(num_batches, 1)


def validate(model, data_loader, ctc_loss_fn, device, epoch, split_name="Val"):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    num_inf = 0
    n_total = len(data_loader)
    phase = get_phase(epoch)
    val_start = time.time()

    with torch.no_grad():
        for batch_idx, (videos, labels, video_lengths, label_lengths) in enumerate(data_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            video_lengths = video_lengths.to(device)
            label_lengths = label_lengths.to(device)

            with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(USE_AMP and device.type == "cuda")):
                logits, output_lengths = model(videos, video_lengths)
                loss = ctc_loss_forward(ctc_loss_fn, logits, labels, output_lengths, label_lengths, device)

            # Skip inf/nan batches but count them
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
                print(f"  Epoch {epoch}/{EPOCHS} [{stage}] {split_name:5s} {batch_idx+1}/{n_total} ({pct:.0f}%) | Loss: {avg:.4f} | Inf: {num_inf} | ETA: {fmt_time(eta)}")

    if num_inf > 0:
        print(f"  [{split_name} WARNING] {num_inf}/{n_total} batches had inf/nan loss (skipped)")

    if num_batches == 0:
        return float("inf")
    return total_loss / num_batches

# =============================================================
# Main
# =============================================================

def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Mixed Precision: {USE_AMP} ({AMP_DTYPE})")

    tokenizer = load_tokenizer(SPM_MODEL)
    print(f"Vocab size: {tokenizer.vocab_size}, Blank ID: {tokenizer.blank_id}")

    from pipelines.data.transforms import VideoTransform
    train_transform = VideoTransform(speed_rate=1)
    print(f"Train augmentation: VideoTransform enabled")

    train_dataset = KoreanVSRDataset(data_dir=TRAIN_DIR, video_transform=train_transform, tokenizer=tokenizer, max_frames=MAX_FRAMES)
    val_dataset = KoreanVSRDataset(data_dir=VAL_DIR, video_transform=None, tokenizer=tokenizer, max_frames=MAX_FRAMES)
    test_dataset = KoreanVSRDataset(data_dir=TEST_DIR, video_transform=None, tokenizer=tokenizer, max_frames=MAX_FRAMES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples, Test: {len(test_dataset)} samples")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")

    model = KoreanVSRModel(
        chaplin_checkpoint=CHAPLIN_CHECKPOINT,
        model_conf=MODEL_CONF,
        vocab_size=tokenizer.vocab_size,
        device=str(device),
    )
    model = model.to(device)

    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "ctc_lo" in name or "output_layer" in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

    # Initial state: freeze encoder, train head only
    for p in encoder_params:
        p.requires_grad = False

    stats = count_parameters(model)
    print(f"Total: {stats['total']:,} | Trainable: {stats['trainable']:,} | Frozen: {stats['frozen']:,}")

    # Train: zero_infinity=True (stability); Val: zero_infinity=False (real loss)
    ctc_loss_train = nn.CTCLoss(blank=tokenizer.blank_id, reduction="mean", zero_infinity=True)
    ctc_loss_val   = nn.CTCLoss(blank=tokenizer.blank_id, reduction="mean", zero_infinity=False)

    # GradScaler is a no-op for BF16; enabled for FP16 only
    scaler = GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))

    optimizer = torch.optim.AdamW(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler_phase1(optimizer)

    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    meta = {
        "mode": "staged_differential_lr",
        "vocab_size": tokenizer.vocab_size,
        "blank_id": tokenizer.blank_id,
        "lr_encoder": float(LR_ENCODER),
        "lr_head": float(LR_HEAD),
        "freeze_epochs": int(FREEZE_EPOCHS),
        "warmup_epochs": int(WARMUP_EPOCHS),
        "epochs": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "grad_accum": int(GRAD_ACCUM),
        "amp_dtype": str(AMP_DTYPE),
        "device": str(device),
    }

    hist = {"meta": meta, "epochs": [], "train_loss": [], "val_loss": [], "test_loss": [], "lr": [], "epoch_time": []}

    CKPT_LAST = os.path.join(ckpt_dir, "ckpt_last.pt")
    METRICS_PT = os.path.join(OUTPUT_DIR, "train_metrics.pt")

    start_epoch = 1
    best_val_loss = float("inf")

    if os.path.exists(CKPT_LAST):
        ckpt = torch.load(CKPT_LAST, map_location="cpu", weights_only=False)
        resumed_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model_state_dict"])

        next_phase = get_phase(resumed_epoch + 1)
        saved_phase = ckpt.get("phase", get_phase(resumed_epoch))

        if next_phase == 2:
            for p in encoder_params:
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": encoder_params, "lr": LR_ENCODER},
                {"params": head_params, "lr": LR_HEAD},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = build_scheduler_phase2(optimizer, start_epoch=resumed_epoch + 1)
            if saved_phase == 2:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            optimizer = torch.optim.AdamW(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler = build_scheduler_phase1(optimizer, start_epoch=resumed_epoch + 1)

        hist = ckpt.get("hist", hist)
        meta = ckpt.get("meta", meta)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = resumed_epoch + 1
        model = model.to(device)
        print(f"[RESUME] epoch {resumed_epoch} -> resuming from {start_epoch}")

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            epoch_start = time.time()

            # Phase transition: head-only -> full (with warmup)
            if epoch == FREEZE_EPOCHS + 1 and not encoder_params[0].requires_grad:
                for p in encoder_params:
                    p.requires_grad = True
                optimizer = torch.optim.AdamW([
                    {"params": encoder_params, "lr": LR_ENCODER},
                    {"params": head_params, "lr": LR_HEAD},
                ], weight_decay=WEIGHT_DECAY)
                scheduler = build_scheduler_phase2(optimizer)
                stats = count_parameters(model)
                print(f"  [Phase 2] Encoder unfrozen | Trainable: {stats['trainable']:,}")

            train_loss = train_one_epoch(model, train_loader, ctc_loss_train, optimizer, scaler, device, epoch)
            val_loss = validate(model, val_loader, ctc_loss_val, device, epoch, split_name="Val")
            test_loss = validate(model, test_loader, ctc_loss_val, device, epoch, split_name="Test")
            scheduler.step()

            elapsed = time.time() - epoch_start
            cur_lr = optimizer.param_groups[0]["lr"]
            phase = get_phase(epoch)

            hist["epochs"].append(epoch)
            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_loss)
            hist["test_loss"].append(test_loss)
            hist["lr"].append(cur_lr)
            hist["epoch_time"].append(elapsed)

            print(f"  => Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f} | LR: {cur_lr:.2e} | {fmt_time(elapsed)}")

            # Atomic save (FP32)
            ckpt_tmp = CKPT_LAST + ".tmp"
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
            os.replace(ckpt_tmp, CKPT_LAST)

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

            torch.save(hist, METRICS_PT)

        print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    except KeyboardInterrupt:
        print("Training interrupted manually.")

    except Exception:
        raise


if __name__ == "__main__":
    main()
