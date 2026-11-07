#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
DOKDO: Korean Visual Speech Recognition — On-Device Webcam Demo

Supports both execution modes described in the DOKDO paper (Figure 1, Section II.4, Section III.3):
  1. Manual SPACE-toggle mode (`--mode manual`, default):
     - Press SPACE to start/stop recording an utterance clip (e.g., ~5.0s / 125 frames @ 25fps).
     - Runs the end-to-end preprocessing + VSR inference pipeline and prints the exact
       stage-by-stage latency breakdown (`t_preprocess`, `t_infer`, `t_total`, `RTF`,
       excluding recording duration, matching Table 2 / Section III.3: 5.0s -> 6.61s).
  2. Asynchronous MAR Producer-Consumer mode (`--mode async`):
     - 3-second closed-mouth MAR calibration -> automatic speech onset/offset detection
       with 15-frame pre-roll buffer -> MediaPipe 4-point ROI crop `(T, 96, 96)` uint8 ->
       `queue.Queue(maxsize=4)` with oldest-drop -> background Consumer VSR thread
       with real-time Korean subtitle overlay (Figure 1 & Figure 2).

Usage:
  python demo_webcam.py                         # Manual SPACE-toggle mode (Table 2 benchmark)
  python demo_webcam.py --mode async            # Async MAR Producer-Consumer mode (Figure 1)
  python demo_webcam.py --decode ctc            # Force CTC Greedy decoding (Paper Section II.2)
  python demo_webcam.py --decode hybrid         # Force Joint CTC/Attention Beam Search
"""

import os
import sys
import time
import signal
import queue as _queue
import threading
import traceback
import argparse

import cv2
import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from pipelines.pipeline_korean import KoreanInferencePipeline
from pipelines.data.transforms import VideoTransform
from util.lip_tracker_dokdo import LipTrackerAuto

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False

DEFAULT_CONFIG_FILE = os.path.join(REPO_DIR, "configs", "korean.ini")
RESULT_TXT = os.path.join(REPO_DIR, "result_korean.txt")
KOREAN_FONT_Candidates = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]


class KoreanSubtitle:
    """Thread-safe Korean subtitle renderer using PIL."""

    def __init__(self, duration_ms=8000):
        self.text = ""
        self.t0 = 0.0
        self.duration_ms = duration_ms
        self._font = None
        if _PIL_OK:
            for fp in KOREAN_FONT_Candidates:
                if os.path.exists(fp):
                    try:
                        self._font = ImageFont.truetype(fp, 28)
                        break
                    except Exception:
                        pass

    def set_text(self, text):
        if not text:
            return
        self.text = text.strip()
        self.t0 = time.time() * 1000.0

    def render(self, frame):
        if self._font is None or not self.text:
            return frame
        if time.time() * 1000.0 - self.t0 > self.duration_ms:
            return frame

        h, w = frame.shape[:2]
        margin = 12
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        max_w = w - margin * 2
        words = self.text.split()
        lines, cur = [], ""
        for word in words:
            trial = (cur + " " + word).strip()
            bbox = draw.textbbox((0, 0), trial, font=self._font)
            if (bbox[2] - bbox[0]) <= max_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        if len(lines) > 3:
            lines = lines[:3]
            lines[-1] = lines[-1] + " ..."

        line_h = self._font.size + 10
        box_h = line_h * len(lines) + margin
        box_y2 = h - margin
        box_y1 = box_y2 - box_h
        box_x1 = margin
        box_x2 = w - margin

        overlay = img.copy()
        odraw = ImageDraw.Draw(overlay)
        odraw.rectangle([box_x1, box_y1, box_x2, box_y2], fill=(0, 0, 0))
        img = Image.blend(img, overlay, 0.6)
        draw = ImageDraw.Draw(img)

        y = box_y1 + margin // 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=self._font)
            tw = bbox[2] - bbox[0]
            x = box_x1 + (max_w - tw) // 2
            draw.text((x, y), line, font=self._font, fill=(255, 255, 255))
            y += line_h
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_device(device_arg="auto"):
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu"  # CPU (ARM64) used for Apple Silicon on-device validation (Table 2)
    else:
        return "cpu"


def run_manual_mode(args, device):
    """Manual SPACE-toggle recording & E2E latency measurement mode (Table 2 / Section III.3)."""
    print("Loading model...")
    vsr_model = KoreanInferencePipeline(
        args.config,
        device=device,
        detector="mediapipe",
        face_track=True,
        decode_override=args.decode,
    )
    print("Model loaded!")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera (index={args.camera_index}). "
              "Use --camera_index or set CAMERA_INDEX env var.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    print(f"[Camera] {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
    print()
    print("=" * 60)
    print("  [Manual E2E Mode — Table 2 Latency Benchmark Protocol]")
    print("  SPACE : Start / Stop recording")
    print("  q     : Quit")
    print("=" * 60)
    print()

    subtitle = KoreanSubtitle()
    recording = False
    out = None
    output_path = ""
    frame_count = 0
    rec_start = 0.0
    latency_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        if recording:
            elapsed = int(time.time() - rec_start)
            label = f"REC {elapsed // 60:02d}:{elapsed % 60:02d}  ({frame_count} frames)"
            cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display, label, (50, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "READY (SPACE to record)", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        display = subtitle.render(display)
        cv2.imshow("DOKDO Korean VSR", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            if not recording:
                recording = True
                frame_count = 0
                rec_start = time.time()
                output_path = os.path.join(REPO_DIR, f"webcam_{int(time.time())}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, args.fps,
                                      (actual_w, actual_h), True)
                print("[Recording] Speak now...")
            else:
                recording = False
                if out is not None:
                    out.release()
                    out = None
                rec_duration = time.time() - rec_start
                print(f"[Stopped] Captured {frame_count} frames (recording duration: {rec_duration:.2f}s, excluded from processing latency)")

                if frame_count >= args.fps:  # at least 1 second
                    print("[Processing: Preprocessing + VSR Inference]")
                    try:
                        text, timings = vsr_model.forward_with_timing(output_path)
                        if not text:
                            text = "(empty)"
                        latency_history.append(timings["t_total"])
                        avg_lat = sum(latency_history) / len(latency_history)

                        print(f"\n>>> Result: {text}")
                        print(
                            f"    [Latency Breakdown] "
                            f"Utterance: {timings['utt_duration']:.2f}s ({timings['n_frames']} frames) | "
                            f"Preprocess: {timings['t_preprocess']:.2f}s | "
                            f"VSR Infer: {timings['t_infer']:.2f}s | "
                            f"Total: {timings['t_total']:.2f}s (RTF={timings['rtf']:.2f}, "
                            f"Session Avg={avg_lat:.2f}s over {len(latency_history)} runs)\n"
                        )
                        subtitle.set_text(text)

                        with open(RESULT_TXT, "a", encoding="utf-8") as f:
                            f.write(
                                f"{time.strftime('%H:%M:%S')} | manual | "
                                f"utt={timings['utt_duration']:.2f}s | "
                                f"prep={timings['t_preprocess']:.2f}s | "
                                f"infer={timings['t_infer']:.2f}s | "
                                f"total={timings['t_total']:.2f}s | {text}\n"
                            )
                    except Exception as e:
                        traceback.print_exc()
                        print(f"[Inference error] {e}")
                else:
                    print("[Too short (<1s), skipped]")

                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                except Exception:
                    pass

        if recording and out is not None:
            out.write(frame)
            frame_count += 1

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


def _async_consumer_loop(q, model, video_transform, subtitle, stop_event):
    """Background Consumer thread for Asynchronous Producer-Consumer mode (Figure 1)."""
    latency_history = []
    while not stop_event.is_set():
        try:
            item = q.get(timeout=0.5)
        except _queue.Empty:
            continue
        utt_id = item["id"]
        try:
            t_norm0 = time.perf_counter()
            video_t = torch.tensor(item["video"], dtype=torch.float32)
            video_t = video_transform(video_t)
            t_norm = time.perf_counter() - t_norm0

            t_inf0 = time.perf_counter()
            text = model.infer(video_t)
            t_infer = time.perf_counter() - t_inf0

            t_preprocess = item.get("t_preprocess", 0.0) + t_norm
            t_total = t_preprocess + t_infer
            utt_dur = item.get("utt_duration", item["n_frames"] / 25.0)
            rtf = t_total / max(utt_dur, 1e-6)
            latency_history.append(t_total)
            avg_lat = sum(latency_history) / len(latency_history)

            if not text:
                text = "(empty)"
            print(
                f"\n>>> [Async VSR utt #{utt_id:04d}] {text}\n"
                f"    [Latency Breakdown] "
                f"Utterance: {utt_dur:.2f}s ({item['n_frames']} frames) | "
                f"Preprocess: {t_preprocess:.2f}s | "
                f"VSR Infer: {t_infer:.2f}s | "
                f"Total: {t_total:.2f}s (RTF={rtf:.2f}, Session Avg={avg_lat:.2f}s)\n",
                flush=True,
            )
            try:
                with open(RESULT_TXT, "a", encoding="utf-8") as f:
                    f.write(
                        f"{time.strftime('%H:%M:%S')} | async#{utt_id:04d} | "
                        f"utt={utt_dur:.2f}s | prep={t_preprocess:.2f}s | "
                        f"infer={t_infer:.2f}s | total={t_total:.2f}s | {text}\n"
                    )
            except OSError as e:
                print(f"[warn] result file write failed: {e}")
            if text != "(empty)":
                subtitle.set_text(text)
        except Exception as e:
            print(f"[Consumer error] utt #{utt_id:04d}: {type(e).__name__}: {e}")
        finally:
            q.task_done()


def run_async_mode(args, device):
    """Asynchronous MAR-based Producer-Consumer webcam pipeline (Figure 1 & Section II.4)."""
    print("Loading model (Async Producer-Consumer mode)...")
    pipeline = KoreanInferencePipeline(
        args.config,
        device=device,
        detector="mediapipe",
        face_track=False,
        decode_override=args.decode,
    )
    print("Model loaded!")

    video_transform = VideoTransform(speed_rate=1.0)
    subtitle = KoreanSubtitle()

    q = _queue.Queue(maxsize=4)
    stop_event = threading.Event()
    consumer = threading.Thread(
        target=_async_consumer_loop,
        args=(q, pipeline.model, video_transform, subtitle, stop_event),
        name="dokdo-vsr-consumer",
        daemon=True,
    )
    consumer.start()

    tracker = LipTrackerAuto(
        output_queue=q,
        camera_index=args.camera_index,
        capture_width=args.width,
        capture_height=args.height,
        capture_fps=args.fps,
        display=True,
        subtitle_renderer=subtitle.render,
    )

    def _sigint(_s, _f):
        print("\n[Async Mode] SIGINT received, stopping...")
        tracker.stop()
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint)

    print()
    print("=" * 60)
    print("  [Async MAR Producer-Consumer Mode — Paper Figure 1]")
    print("  1. Keep mouth closed facing forward for 3s MAR calibration")
    print("  2. Speak naturally; utterance ends automatically after 1.0s silence")
    print("  3. Press 'q' or ESC to quit")
    print("=" * 60)
    print()

    try:
        tracker.run()
    finally:
        stop_event.set()
        consumer.join(timeout=3.0)
        print("[Async Mode] Shutdown complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DOKDO On-Device Korean Visual Speech Recognition Webcam Demo"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="manual",
        choices=["manual", "async"],
        help="'manual': SPACEbar start/stop recording with latency breakdown (Table 2); "
             "'async': MAR auto-triggered Producer-Consumer queue pipeline (Figure 1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="Path to INI config file (default: configs/korean.ini)",
    )
    parser.add_argument(
        "--decode",
        type=str,
        default=None,
        choices=["ctc", "hybrid"],
        help="Override decoding mode ('ctc' for CTC Greedy, 'hybrid' for Joint CTC/Attention Beam Search)",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=int(os.getenv("CAMERA_INDEX", "0")),
        help="OpenCV camera index (default: 0 or CAMERA_INDEX env var)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    parser.add_argument("--fps", type=int, default=25, help="Capture FPS (default: 25)")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cpu', 'cuda:0')")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"[Device] {device}")
    print(f"[Config] {args.config}")
    print(f"[Mode]   {args.mode}")

    if args.mode == "async":
        run_async_mode(args, device)
    else:
        run_manual_mode(args, device)


if __name__ == "__main__":
    main()
