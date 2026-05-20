#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Korean inference pipeline: video → cropped lip ROI → KoreanAVSR → text.
"""

import os
import pickle
from configparser import ConfigParser

import torch

from pipelines.model_korean import KoreanAVSR
from pipelines.data.data_module import AVSRDataLoader


class KoreanInferencePipeline(torch.nn.Module):
    def __init__(self, config_filename, detector="mediapipe",
                 face_track=True, device="cuda:0"):
        super().__init__()
        assert os.path.isfile(config_filename), f"config not found: {config_filename}"

        config = ConfigParser()
        config.read(config_filename)

        self.modality = config.get("input", "modality")
        input_v_fps = config.getfloat("input", "v_fps")
        model_v_fps = config.getfloat("model", "v_fps")

        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def _resolve(p):
            return p if os.path.isabs(p) else os.path.join(repo_dir, p)

        model_path = _resolve(config.get("model", "model_path"))
        model_conf = _resolve(config.get("model", "model_conf"))
        spm_model = _resolve(config.get("model", "spm_model"))

        # decode strategy (backward compatible: defaults to legacy CTC-greedy)
        decode = config.get("model", "decode", fallback="ctc")
        ctc_weight = config.getfloat("model", "ctc_weight", fallback=0.3)
        beam_size = config.getint("model", "beam_size", fallback=10)
        sos_id = config.getint("model", "sos_id", fallback=8000)
        eos_id = config.getint("model", "eos_id", fallback=8000)

        self.dataloader = AVSRDataLoader(
            self.modality,
            speed_rate=input_v_fps / model_v_fps,
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
