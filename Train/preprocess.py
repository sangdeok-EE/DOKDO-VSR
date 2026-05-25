"""
preprocess.py - Korean VSR preprocessing script.

Converts raw video + JSON labels into per-sentence .npz segments.
Uses Chaplin VideoProcess to crop a 96x96 grayscale mouth region.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2

CHAPLIN_ROOT = "<path/to/chaplin>"
sys.path.insert(0, CHAPLIN_ROOT)

from pipelines.detectors.mediapipe.video_process import VideoProcess
from pipelines.detectors.mediapipe.detector import LandmarksDetector


def get_device():
    """CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def split_dataset(video_list, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split video list into (train, val, test) by video unit."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(video_list))
    train_end = int(len(video_list) * train_ratio)
    val_end = int(len(video_list) * (train_ratio + val_ratio))
    train_videos = [video_list[i] for i in indices[:train_end]]
    val_videos = [video_list[i] for i in indices[train_end:val_end]]
    test_videos = [video_list[i] for i in indices[val_end:]]
    return train_videos, val_videos, test_videos


def transform_bbox(raw_bboxes, orig_w, orig_h):
    """
    Convert JSON face bbox coordinates to the actual video coordinate system.

    JSON bbox is in portrait orientation (orig_h x orig_w, e.g. 1080x1920).
    Actual video is landscape (orig_w x orig_h, e.g. 1920x1080).

    Transform: rotate 90° CW -> horizontal flip.

    Args:
        raw_bboxes: list of [xtl, ytl, xbr, ybr] (JSON original, portrait).
        orig_w: actual video width (1920).
        orig_h: actual video height (1080).
    Returns:
        face_bboxes: list of [xtl, ytl, xbr, ybr] (actual video coords).
    """
    src_w = orig_h
    src_h = orig_w

    face_bboxes = []
    for bbox in raw_bboxes:
        xtl, ytl, xbr, ybr = bbox

        # Step 1: rotate 90° CW; (x, y) -> (src_h - y, x)
        r_xtl = src_h - ybr
        r_ytl = xtl
        r_xbr = src_h - ytl
        r_ybr = xbr

        # Step 2: horizontal flip
        f_xtl = orig_w - r_xbr
        f_ytl = r_ytl
        f_xbr = orig_w - r_xtl
        f_ybr = r_ybr

        face_bboxes.append([int(f_xtl), int(f_ytl), int(f_xbr), int(f_ybr)])

    return face_bboxes


def read_video_segment(video_path, start_frame, end_frame):
    """Read a specific frame range using cv2 (BGR -> RGB)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    return np.array(frames, dtype=np.uint8)


def detect_landmarks(video_frames, detector, face_bboxes=None):
    """Per-frame landmark extraction (mediapipe: 4-point).

    Args:
        video_frames: np.ndarray (N, H, W, 3) RGB uint8.
        detector: mediapipe LandmarksDetector instance.
        face_bboxes: unused (kept for compatibility).
    Returns:
        landmarks: list of length N, each element (4, 2) ndarray or None.
    """
    N = len(video_frames)
    landmarks = detector.detect(video_frames, detector.full_range_detector)
    if all(lm is None for lm in landmarks):
        landmarks = detector.detect(video_frames, detector.short_range_detector)
    if N >= 20:
        print(f"    Landmark: {N}/{N} (100%)")
    return landmarks


def preprocess_single_video(video_path, json_path, output_dir, video_process, detector):
    """Preprocess one video and save per-sentence .npz files.

    Returns:
        saved_count: number of segments saved.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"  Processing: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ERROR: Cannot open {video_path}")
        return 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"    Frames: {N}, Resolution: {W}x{H}, FPS: {fps}")

    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    if isinstance(json_data, list):
        json_data = json_data[0]

    sentences = json_data["Sentence_info"]
    raw_face_bboxes = json_data["Bounding_box_info"]["Face_bounding_box"]["xtl_ytl_xbr_ybr"]

    face_bboxes = transform_bbox(raw_face_bboxes, W, H)

    saved_count = 0
    total_sents = len(sentences)
    for sent_idx, sent_info in enumerate(sentences):
        sentence_id = sent_info["ID"]
        sentence_text = sent_info["sentence_text"]
        start_time = sent_info["start_time"]
        end_time = sent_info["end_time"]

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        start_frame = max(0, min(start_frame, N))
        end_frame = max(0, min(end_frame, N))

        pct = (sent_idx + 1) / total_sents * 100
        print(f"\r    Sentence {sent_idx+1}/{total_sents} ({pct:.0f}%) - sent{sentence_id:02d}", end="", flush=True)

        if end_frame <= start_frame:
            continue

        video_segment = read_video_segment(video_path, start_frame, end_frame)
        if video_segment is None or len(video_segment) == 0:
            print(f"    Skipped sent{sentence_id:02d}: failed to read segment")
            continue

        landmarks_segment = detect_landmarks(video_segment, detector)

        try:
            processed_video = video_process(video_segment, landmarks_segment)
        except Exception as e:
            print(f"    Skipped sent{sentence_id:02d}: VideoProcess error - {e}")
            continue

        if processed_video is None:
            print(f"    Skipped sent{sentence_id:02d}: VideoProcess returned None")
            continue

        out_path = os.path.join(output_dir, f"{video_name}_sent{sentence_id:02d}.npz")
        np.savez(out_path, video=processed_video, text=sentence_text)
        saved_count += 1

    print(f"\n    Saved: {saved_count}/{total_sents} segments")
    return saved_count


def save_split_info(output_dir, train_videos, val_videos, test_videos, seed,
                    train_count, val_count, test_count):
    """Save split_info.json."""
    total_videos = len(train_videos) + len(val_videos) + len(test_videos)
    split_info = {
        "train_videos": [os.path.splitext(os.path.basename(v))[0] for v in train_videos],
        "val_videos": [os.path.splitext(os.path.basename(v))[0] for v in val_videos],
        "test_videos": [os.path.splitext(os.path.basename(v))[0] for v in test_videos],
        "split_ratio": {
            "train": len(train_videos) / max(total_videos, 1),
            "val": len(val_videos) / max(total_videos, 1),
            "test": len(test_videos) / max(total_videos, 1),
        },
        "random_seed": seed,
        "total_train_segments": train_count,
        "total_val_segments": val_count,
        "total_test_segments": test_count,
    }
    info_path = os.path.join(output_dir, "split_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    print(f"\nSplit info saved: {info_path}")


def find_video_json_pairs(data_root, json_root=None):
    """Collect (video, json) pairs from given directories."""
    if json_root is None:
        json_root = data_root

    pairs = []
    video_exts = {'.mp4', '.MP4'}

    for fname in sorted(os.listdir(data_root)):
        if os.path.splitext(fname)[1] in video_exts:
            video_path = os.path.join(data_root, fname)
            base_name = os.path.splitext(fname)[0]
            json_path = os.path.join(json_root, base_name + ".json")
            if os.path.exists(json_path):
                pairs.append((video_path, json_path))
            else:
                print(f"Warning: JSON not found for {fname}, skipping")

    return pairs


def main(data_root, output_dir, json_root=None, detector_type="retinaface",
         train_ratio=0.8, val_ratio=0.1, seed=42):
    """Main preprocessing pipeline."""
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    video_process = VideoProcess(convert_gray=True)

    device = get_device()
    print(f"Device: {device}")
    detector = LandmarksDetector()

    pairs = find_video_json_pairs(data_root, json_root)
    if not pairs:
        print(f"Error: No video-JSON pairs found in {data_root} (json_root={json_root})")
        return

    video_list = [p[0] for p in pairs]
    print(f"Found {len(video_list)} video-JSON pairs")

    train_videos, val_videos, test_videos = split_dataset(video_list, train_ratio, val_ratio, seed)
    train_set = set(train_videos)
    val_set = set(val_videos)
    print(f"Split: {len(train_videos)} train / {len(val_videos)} val / {len(test_videos)} test videos")

    train_count = 0
    val_count = 0
    test_count = 0

    for video_path, json_path in pairs:
        if video_path in train_set:
            dest_dir = train_dir
        elif video_path in val_set:
            dest_dir = val_dir
        else:
            dest_dir = test_dir

        count = preprocess_single_video(video_path, json_path, dest_dir, video_process, detector)

        if video_path in train_set:
            train_count += count
        elif video_path in val_set:
            val_count += count
        else:
            test_count += count

    save_split_info(output_dir, train_videos, val_videos, test_videos, seed,
                    train_count, val_count, test_count)

    print(f"\nDone! Train: {train_count}, Val: {val_count}, Test: {test_count} segments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Korean VSR Preprocessing")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Raw video directory (MP4)")
    parser.add_argument("--json_root", type=str, default=None,
                        help="JSON label directory (defaults to data_root)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output root directory for preprocessed data")
    parser.add_argument("--detector", type=str, default="retinaface",
                        choices=["retinaface", "mediapipe"],
                        help="Detector type (default: retinaface)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train ratio (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()
    main(args.data_root, args.output_dir, args.json_root, args.detector,
         args.train_ratio, args.val_ratio, args.seed)
