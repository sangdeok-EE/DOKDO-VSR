#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Korean Visual Speech Recognition — Webcam Demo

Controls:
  SPACE  — Start / stop recording
  q      — Quit

Automatically runs VSR inference when recording stops.
"""

import os
import sys
import time
import traceback

import cv2
import torch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from pipelines.pipeline_korean import KoreanInferencePipeline

# ============================================================
# Settings
# ============================================================
CONFIG_FILE = os.path.join(REPO_DIR, "configs", "korean.ini")
RESULT_TXT = os.path.join(REPO_DIR, "result_korean.txt")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "1"))
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 25


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu"  # MPS has espnet compatibility issues
    else:
        return "cpu"


def main():
    device = get_device()
    print(f"[Device] {device}")

    print("Loading model...")
    vsr_model = KoreanInferencePipeline(
        CONFIG_FILE,
        device=device,
        detector="mediapipe",
        face_track=True,
    )
    print("Model loaded!")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera (index={CAMERA_INDEX}). "
              "Set CAMERA_INDEX env var to change.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Camera] {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
    print()
    print("=" * 50)
    print("  SPACE : Start / Stop recording")
    print("  q     : Quit")
    print("=" * 50)
    print()

    recording = False
    out = None
    output_path = ""
    frame_count = 0
    rec_start = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        if recording:
            elapsed = int(time.time() - rec_start)
            label = f"REC {elapsed//60:02d}:{elapsed%60:02d}  ({frame_count} frames)"
            cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display, label, (50, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "READY (SPACE to record)", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Korean VSR", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if not recording:
                recording = True
                frame_count = 0
                rec_start = time.time()
                output_path = os.path.join(REPO_DIR, f"webcam_{int(time.time())}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, CAPTURE_FPS,
                                      (actual_w, actual_h), True)
                print("[Recording] Speak now...")
            else:
                recording = False
                if out is not None:
                    out.release()
                    out = None
                print(f"[Stopped] {frame_count} frames")

                if frame_count >= CAPTURE_FPS:  # at least 1 second
                    print("[Inference]")
                    try:
                        text = vsr_model(output_path)
                        if not text:
                            text = "(empty)"
                        print(f"\n>>> Result: {text}\n")

                        with open(RESULT_TXT, "a", encoding="utf-8") as f:
                            f.write(f"{time.strftime('%H:%M:%S')} | {text}\n")
                    except Exception as e:
                        traceback.print_exc()
                        print(f"[Inference error] {e}")
                else:
                    print("[Too short, skipped]")

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


if __name__ == "__main__":
    main()
