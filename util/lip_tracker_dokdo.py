"""
lip_tracker_dokdo.py — DOKDO Real-Time Asynchronous Producer (MAR-based Utterance Detector).

Implements the Producer side of the asynchronous Producer–Consumer webcam pipeline
shown in Figure 1 and Section II.4 / III.3 of the DOKDO paper:
  1. 3-second frontal closed-mouth calibration -> personalized `OPEN_THRESHOLD = mean(MAR) + 0.04`
  2. Real-time MediaPipe FaceMesh 3D Mouth Aspect Ratio (MAR) + frontal face check
  3. 15-frame pre-roll ring buffer (deque) merged on speech onset to preserve initial consonants
  4. 1.0-second silence detection + 2-stage frame count verification
     (`min_total_frames > 20` and `min_valid_mar_frames >= 5`)
  5. Per-utterance ROI extraction via MediaPipe 4-point `LandmarksDetector` + `VideoProcess`
     (12-frame smoothing, similarity affine warp, 96x96 grayscale mouth ROI)
  6. Non-blocking enqueue of `(T, 96, 96)` uint8 numpy array + timing metadata into
     `queue.Queue(maxsize=4)` with oldest-drop overflow policy.
"""

import os
import platform
import time
import queue as _queue
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from pipelines.detectors.mediapipe.detector import LandmarksDetector
from pipelines.detectors.mediapipe.video_process import VideoProcess


class LipTrackerAuto:
    # MediaPipe FaceMesh 478-point indices for 3D Mouth Aspect Ratio (MAR)
    LEFT_LIP = 78
    RIGHT_LIP = 308
    TOP_LIP = 13
    BOTTOM_LIP = 14

    def __init__(
        self,
        output_queue,
        *,
        camera_index=0,
        capture_width=1280,
        capture_height=720,
        capture_fps=25,
        silence_threshold=1.0,
        min_total_frames=20,
        min_valid_mar_frames=5,
        mar_margin=0.04,
        pre_roll_frames=15,
        calibration_seconds=3.0,
        default_open_threshold=0.15,
        save_dir=None,
        save_npz=False,
        display=True,
        window_title="DOKDO Korean VSR (Async MAR)",
        subtitle_renderer=None,
    ):
        self.output_queue = output_queue
        self.camera_index = camera_index
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.capture_fps = capture_fps
        self.silence_threshold = silence_threshold
        self.min_total_frames = min_total_frames
        self.min_valid_mar_frames = min_valid_mar_frames
        self.mar_margin = mar_margin
        self.pre_roll_frames = pre_roll_frames
        self.calibration_seconds = calibration_seconds
        self.default_open_threshold = default_open_threshold
        self.save_dir = save_dir
        self.save_npz = save_npz
        self.display = display
        self.window_title = window_title
        self.subtitle_renderer = subtitle_renderer

        self._stop = False
        self._utt_counter = 0

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True
        )
        self._lm_detector = LandmarksDetector()
        self._video_process = VideoProcess(convert_gray=True)

        if self.save_npz and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    # ---------- MAR & Frontal Face Helpers ----------

    def _get_mar_3d(self, landmarks):
        l = np.array([landmarks[self.LEFT_LIP].x,
                      landmarks[self.LEFT_LIP].y,
                      landmarks[self.LEFT_LIP].z])
        r = np.array([landmarks[self.RIGHT_LIP].x,
                      landmarks[self.RIGHT_LIP].y,
                      landmarks[self.RIGHT_LIP].z])
        t = np.array([landmarks[self.TOP_LIP].x,
                      landmarks[self.TOP_LIP].y,
                      landmarks[self.TOP_LIP].z])
        b = np.array([landmarks[self.BOTTOM_LIP].x,
                      landmarks[self.BOTTOM_LIP].y,
                      landmarks[self.BOTTOM_LIP].z])
        w = np.linalg.norm(l - r)
        h = np.linalg.norm(t - b)
        if w == 0:
            return 0.0
        return float(h / w)

    @staticmethod
    def _is_face_frontal(landmarks):
        nose_x = landmarks[1].x
        left_eye_x = landmarks[33].x
        right_eye_x = landmarks[263].x
        d_l = abs(nose_x - left_eye_x)
        d_r = abs(nose_x - right_eye_x)
        if d_r == 0:
            return False
        ratio = d_l / d_r
        return 0.6 < ratio < 1.4

    def _calibrate(self, cap):
        print("\n" + "=" * 55)
        print("[Calibration] Face forward with mouth closed for 3 seconds...")
        values = []
        t0 = time.time()
        while time.time() - t0 < self.calibration_seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                if self._is_face_frontal(lms):
                    values.append(self._get_mar_3d(lms))
            if self.display:
                cv2.putText(
                    frame, "CALIBRATING (Keep mouth closed)...",
                    (24, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA
                )
                cv2.imshow(self.window_title, frame)
                cv2.waitKey(1)
        if values:
            thr = sum(values) / len(values) + self.mar_margin
            print(f"[Calibration] OPEN_THRESHOLD = {thr:.3f} (baseline={thr - self.mar_margin:.3f} + margin={self.mar_margin:.2f})")
            return thr
        print(f"[Calibration] Fallback to default threshold = {self.default_open_threshold}")
        return self.default_open_threshold

    def _detect_face_landmarks(self, rgb_frames):
        landmarks = self._lm_detector.detect(
            rgb_frames, self._lm_detector.full_range_detector
        )
        if all(lm is None for lm in landmarks):
            landmarks = self._lm_detector.detect(
                rgb_frames, self._lm_detector.short_range_detector
            )
        return landmarks

    def _process_utterance(self, frames_bgr):
        """Run MediaPipe 4-point detection + 12-frame smoothing + affine warp + 96x96 ROI crop."""
        if not frames_bgr:
            return None, 0.0
        t0 = time.perf_counter()
        rgb_frames = np.stack(
            [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr], axis=0
        )
        landmarks = self._detect_face_landmarks(rgb_frames)
        if all(lm is None for lm in landmarks):
            print("[Async Producer] No face landmarks detected in utterance, skipping.")
            return None, 0.0
        try:
            processed = self._video_process(rgb_frames, landmarks)
        except Exception as e:
            print(f"[Async Producer] VideoProcess failed: {e}")
            return None, 0.0
        t_preprocess = time.perf_counter() - t0
        if processed is None:
            print("[Async Producer] VideoProcess returned None, skipping.")
            return None, t_preprocess
        return np.asarray(processed, dtype=np.uint8), t_preprocess

    def _enqueue(self, video_arr, t_preprocess, actual_fps):
        self._utt_counter += 1
        n_frames = int(video_arr.shape[0])
        item = {
            "id": self._utt_counter,
            "video": video_arr,  # (T, 96, 96) uint8 grayscale
            "n_frames": n_frames,
            "utt_duration": n_frames / max(float(actual_fps), 1.0),
            "t_preprocess": float(t_preprocess),
            "timestamp": time.time(),
        }
        try:
            self.output_queue.put_nowait(item)
        except _queue.Full:
            try:
                dropped = self.output_queue.get_nowait()
                print(f"[Async Producer] Queue full (maxsize=4), dropped oldest utt {dropped['id']:04d}")
            except _queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(item)
            except _queue.Full:
                pass
        if self.save_npz and self.save_dir:
            self._save_npz(item)
        return item

    def _save_npz(self, item):
        path = os.path.join(self.save_dir, f"utt_{item['id']:04d}.npz")
        try:
            np.savez_compressed(
                path,
                video=item["video"],
                text="",
                timestamp=np.float64(item["timestamp"]),
                n_frames=np.int64(item["n_frames"]),
            )
        except Exception as e:
            print(f"[Async Producer] npz save failed: {e}")

    # ---------- Main Capture Loop ----------

    def run(self):
        cap = None
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        elif platform.system() == "Linux":
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera (index={self.camera_index})")

        if self.capture_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.capture_width))
        if self.capture_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.capture_height))
        if self.capture_fps:
            cap.set(cv2.CAP_PROP_FPS, float(self.capture_fps))

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or self.capture_fps
        print(f"[Camera] index={self.camera_index}: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

        open_threshold = self._calibrate(cap)

        ring_buffer = deque(maxlen=self.pre_roll_frames)
        frames_buffer = []
        is_speaking = False
        silence_t0 = None
        valid_speech_count = 0

        try:
            while not self._stop and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    continue

                display_frame = frame.copy() if self.display else None
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._face_mesh.process(rgb)

                cur_mar = 0.0
                is_valid = False
                if results.multi_face_landmarks:
                    lms = results.multi_face_landmarks[0].landmark
                    if self._is_face_frontal(lms):
                        cur_mar = self._get_mar_3d(lms)
                        if cur_mar > open_threshold:
                            is_valid = True

                if is_valid:
                    valid_speech_count += 1
                    if not is_speaking:
                        is_speaking = True
                        frames_buffer.extend(ring_buffer)
                        ring_buffer.clear()
                        print("\n[Async Producer] Speech start detected (15-frame pre-roll merged)")
                    silence_t0 = None
                else:
                    if is_speaking and silence_t0 is None:
                        silence_t0 = time.time()

                if is_speaking:
                    frames_buffer.append(frame)
                else:
                    ring_buffer.append(frame)

                if (is_speaking and silence_t0 is not None
                        and (time.time() - silence_t0) > self.silence_threshold):
                    enough_total = len(frames_buffer) > self.min_total_frames
                    enough_valid = valid_speech_count >= self.min_valid_mar_frames
                    if enough_total and enough_valid:
                        print(f"[Async Producer] Speech end: {len(frames_buffer)} raw frames "
                              f"(valid MAR frames={valid_speech_count})")
                        video_arr, t_prep = self._process_utterance(frames_buffer)
                        if video_arr is not None:
                            item = self._enqueue(video_arr, t_prep, actual_fps)
                            print(f"[Async Producer] Enqueued utt #{item['id']:04d} "
                                  f"shape={item['video'].shape} (preprocess={t_prep:.2f}s)")
                    else:
                        print(f"[Async Producer] Discarded noise clip: total={len(frames_buffer)}, "
                              f"valid_mar={valid_speech_count}")
                    is_speaking = False
                    frames_buffer = []
                    valid_speech_count = 0
                    silence_t0 = None

                if self.display and display_frame is not None:
                    self._draw_overlay(display_frame, is_speaking, cur_mar, open_threshold)
                    if self.subtitle_renderer is not None:
                        try:
                            display_frame = self.subtitle_renderer(display_frame)
                        except Exception as e:
                            print(f"[Async Producer] Subtitle render error: {e}")
                    cv2.imshow(self.window_title, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        print("[Async Producer] Quit key pressed.")
                        break
        finally:
            cap.release()
            if self.display:
                cv2.destroyAllWindows()

    def _draw_overlay(self, frame, speaking, mar, threshold):
        h, w = frame.shape[:2]
        if speaking:
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1, cv2.LINE_AA)
            label = f"SPEAKING  MAR={mar:.3f}/{threshold:.3f}"
            color = (0, 0, 255)
        else:
            label = f"READY     MAR={mar:.3f}/{threshold:.3f}"
            color = (0, 200, 0)
        cv2.putText(frame, label, (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' or ESC to quit", (24, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def stop(self):
        self._stop = True
