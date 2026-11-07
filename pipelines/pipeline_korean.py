#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Korean inference pipeline: video -> cropped lip ROI -> KoreanAVSR -> text.

Supports stage-by-stage latency instrumentation (`forward_with_timing`) to
reproduce the preprocessing + inference latency breakdown reported in Section III.3
and Table 2 of the DOKDO paper (e.g., 5.0s utterance -> 6.61s processing time on
MacBook Pro M5 Pro CPU, excluding video capture duration).
"""

import os
import time
import pickle
from configparser import ConfigParser

import torch

from pipelines.model_korean import KoreanAVSR
from pipelines.data.data_module import AVSRDataLoader


class KoreanInferencePipeline(torch.nn.Module):
    def __init__(self, config_filename, detector="mediapipe",
                 face_track=True, device="cuda:0", decode_override=None):
        super().__init__()
        assert os.path.isfile(config_filename), f"config not found: {config_filename}"

        config = ConfigParser()
        config.read(config_filename)

        self.modality = config.get("input", "modality")
        self.input_v_fps = config.getfloat("input", "v_fps")
        self.model_v_fps = config.getfloat("model", "v_fps")

        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def _resolve(p):
            return p if os.path.isabs(p) else os.path.join(repo_dir, p)

        model_path = _resolve(config.get("model", "model_path"))
        model_conf = _resolve(config.get("model", "model_conf"))
        spm_model = _resolve(config.get("model", "spm_model"))

        # Decode strategy (can be overridden from CLI via decode_override)
        decode = decode_override or config.get("model", "decode", fallback="ctc")
        ctc_weight = config.getfloat("model", "ctc_weight", fallback=0.3)
        beam_size = config.getint("model", "beam_size", fallback=10)
        sos_id = config.getint("model", "sos_id", fallback=1)
        eos_id = config.getint("model", "eos_id", fallback=2)

        self.dataloader = AVSRDataLoader(
            self.modality,
            speed_rate=self.input_v_fps / self.model_v_fps,
            detector=detector,
        )
        self.model = KoreanAVSR(
            model_path, model_conf, spm_model, device=device,
            decode=decode, ctc_weight=ctc_weight, beam_size=beam_size,
            sos_id=sos_id, eos_id=eos_id,
        )

        if face_track and self.modality in ("video", "audiovisual"):
            if detector == "mediapipe":
                from pipelines.detectors.mediapipe.detector import LandmarksDetector
                self.landmarks_detector = LandmarksDetector()
            elif detector == "retinaface":
                from pipelines.detectors.retinaface.detector import LandmarksDetector
                self.landmarks_detector = LandmarksDetector(device="cuda:0")
            else:
                self.landmarks_detector = None
        else:
            self.landmarks_detector = None

    def process_landmarks(self, data_filename, landmarks_filename):
        if self.modality == "audio":
            return None
        if isinstance(landmarks_filename, str):
            return pickle.load(open(landmarks_filename, "rb"))
        return self.landmarks_detector(data_filename)

    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(data_filename), f"video not found: {data_filename}"
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        return self.model.infer(data)

    def forward_with_timing(self, data_filename, landmarks_filename=None):
        """Run E2E preprocessing + VSR inference and return detailed latency metrics.

        Measurement definition (Paper Section III.3):
          - Recording / capture duration is excluded.
          - `t_preprocess`: MediaPipe 4-keypoint landmark extraction + 12-frame
            temporal smoothing + similarity affine warp + 96x96 grayscale ROI crop
            + 88x88 center crop & normalization.
          - `t_infer`: 3D Conv + ResNet-18 + Conformer Encoder forward pass +
            CTC Greedy / Hybrid Beam decoding + SentencePiece detokenization.
          - `t_total`: `t_preprocess + t_infer`.

        Returns:
            text (str): recognized Korean sentence
            timings (dict): {
                "n_frames": int,
                "utt_duration": float (sec),
                "t_preprocess": float (sec),
                "t_infer": float (sec),
                "t_total": float (sec),
                "rtf": float
            }
        """
        assert os.path.isfile(data_filename), f"video not found: {data_filename}"
        t0 = time.perf_counter()
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        t1 = time.perf_counter()
        text = self.model.infer(data)
        t2 = time.perf_counter()

        n_frames = int(data.shape[1]) if isinstance(data, torch.Tensor) and data.dim() >= 2 else 0
        utt_duration = n_frames / max(self.model_v_fps, 1.0)
        t_preprocess = t1 - t0
        t_infer = t2 - t1
        t_total = t2 - t0
        rtf = t_total / max(utt_duration, 1e-6)

        timings = {
            "n_frames": n_frames,
            "utt_duration": utt_duration,
            "t_preprocess": t_preprocess,
            "t_infer": t_infer,
            "t_total": t_total,
            "rtf": rtf,
        }
        return text, timings
