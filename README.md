# DOKDO: On-Device Korean Lip-Reading System via Two-Stage Transfer Learning

**Paper Title**: *DOKDO: 전이학습 기반 온디바이스 한국어 립리딩 시스템의 실환경 검증*  
(*DOKDO: Real-World Validation of an On-Device Korean Lip-Reading System Based on Transfer Learning*)

Visual Speech Recognition (VSR) — reading lips from silent video to produce Korean text.  
DOKDO transfers an English visual speech representation ([Auto-AVSR](https://github.com/mpc001/auto_avsr) / [Chaplin](https://github.com/amanvirparhar/chaplin)) to Korean via **8,001-token SentencePiece Unigram head replacement** and **two-stage differential-LR fine-tuning**, and validates real-time on-device inference on a single webcam without external server communication.

<p align="center">
  <img src="assets/pipeline.jpeg" width="95%" />
</p>

---

## 1. Inherited Backbone vs. DOKDO Korean Adaptation

To clearly separate the English pretrained components from DOKDO's Korean-specific contributions:

| Module / Stage | Source | Role & Specification |
| :--- | :--- | :--- |
| **Visual Frontend** | Inherited ([Auto-AVSR](https://github.com/mpc001/auto_avsr) LRS3 pretrained) | 3D Convolution (`5x7x7`) + 2D ResNet-18 extracting 512-D spatial-temporal features per frame |
| **Conformer Encoder** | Inherited (Fine-tuned in Phase 2) | 12-layer Conformer (`adim=768`, `aheads=12`, `eunits=3072`, Macaron FFN + MHSA + Conv module `kernel=31`) |
| **Korean Subword Tokenizer** | **DOKDO (Ours)** | Korean SentencePiece Unigram (`vocab_size=8,001`: `<unk>=0, <s>=1, </s>=2`, subwords `3..7999`, `CTC blank=8000`) |
| **CTC & Decoder Output Heads** | **DOKDO (Ours)** | Replaced English 5,049-unit linear/embedding layers with Korean 8,001-unit `ctc.ctc_lo` (`768 -> 8001`) and Transformer Decoder layers (`Train/model.py`) |
| **2-Stage Transfer Learning** | **DOKDO (Ours)** | Phase 1 Head/Decoder warm-up (Encoder frozen, Epochs 1–5) $\rightarrow$ Phase 2 Full fine-tuning with differential LR (`1e-5` vs `1e-3`) + Hybrid CTC/CE loss (`Train/train.py`) |
| **On-Device Webcam Pipeline** | **DOKDO (Ours)** | MediaPipe 4-keypoint affine ROI crop (`96x96` $\rightarrow$ `88x88`) + Manual E2E & Async MAR Producer–Consumer Queue (`demo_webcam.py`, `util/lip_tracker_dokdo.py`) |

---

## 2. Two-Stage Transfer Learning & Training Protocol

### 2.1 Training Objective (Hybrid CTC + Attention CE Loss)
During training (`Train/model.py`, `Train/train.py`), the Conformer Encoder and 6-layer Transformer Decoder are jointly optimized using a hybrid objective (Figure 1):

$$L = \lambda L_{\text{CTC}} + (1 - \lambda) L_{\text{CE}}, \quad \lambda = 0.3$$

- **$L_{\text{CTC}}$ ($\lambda = 0.3$)**: Frame-level Connectionist Temporal Classification loss (`blank_id=8000`) enforcing monotonic alignment between lip frame features and Korean subword tokens.
- **$L_{\text{CE}}$ ($1 - \lambda = 0.7$)**: Transformer Decoder teacher-forcing Cross-Entropy loss with label smoothing (`lsm_weight=0.1`, `sos_id=1`, `eos_id=2`) modeling inter-token Korean linguistic context.
- **Inference Decoding (`pipelines/model_korean.py`)**:
  - **`decode=ctc` (Paper Default)**: Uses only the Visual Frontend + Conformer Encoder + CTC linear projection (`ctc.ctc_lo`) with CTC Greedy decoding ($\arg\max$ $\rightarrow$ collapse consecutive duplicates $\rightarrow$ remove `blank=8000`) and SentencePiece detokenization, omitting the autoregressive Transformer Decoder for fast on-device CPU execution.
  - **`decode=hybrid` (Optional)**: Joint CTC + Attention Beam Search (`beam_size=10`, `ctc_weight=0.3`, `decoder_weight=0.7`) via `KoreanCTCPrefixScorer`.

### 2.2 Two-Stage Differential-LR Schedule & Dataset Curriculum

| Stage / Phase | Epochs | Visual Frontend + Conformer Encoder | Korean CTC Head & Decoder | Learning Rate (`AdamW`, `wd=0.01`) | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 1 (Head Warm-up)** | Epoch 1 – 5 | **Frozen** (`242.6M` params) | **Trainable** (`12.3M` params) | `lr_head = 1e-3` (Linear warmup 3 ep + Cosine) | Stabilize randomly initialized 8,001-class Korean head without corrupting pretrained visual features |
| **Phase 2 (Full Fine-Tuning)** | Epoch 6 – 109 | **Unfrozen / Trainable** | **Trainable** | `lr_encoder = 1e-5`, `lr_head = 1e-3` (CosineAnnealingLR) | Adapt visual-phonetic representations to Korean articulatory patterns while preventing catastrophic forgetting |

- **Sequential Domain Adaptation**:
  1. **Stage A — OLKAVS (AI Hub 립리딩 데이터셋)**: Trained on ~1.5 TB subset (~14% of the full 14 TB studio dataset; `train:val:test = 8:1:1` video-level split, `seed=42`) for 109 epochs (`batch_size=8`, `grad_accum=2`, `BF16`, gradient clipping `max_norm=5.0`). Best validation loss achieved at Epoch 95 (`val_loss=1.1027`).
  2. **Stage B — KMSAV (AI Hub 시청각 발화 데이터셋)**: Initialized from the OLKAVS checkpoint (`--pretrained_kr_ckpt`) and fine-tuned on ~100 GB of KMSAV YouTube wild videos to evaluate domain adaptation under diverse illumination, pose, and resolution variations.

---

## 3. Quantitative Results & Comparison with Prior Korean VSR Works

### 3.1 Benchmark Evaluation (Table 1)

| Model / Study | Input Modality | Architecture | Dataset | Split Condition | WER (%) | CER (%) | Open-Source / E2E Demo |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| Cho & Yoon (2024) | Visual-only | 3D-CNN + ResNet-18 + Transformer (CTC) | AI Hub (OLKAVS) | Seen-speaker | — | 37.27 | No |
| KM-AVSR (2025) | Audio + Visual | Backbone + LLM | AI Hub (OLKAVS) | Seen-speaker | — | 15.66 | No |
| **DOKDO (Ours)** | **Visual-only** | **Auto-AVSR Conformer + SentencePiece (8,001)** | **OLKAVS (~1.5 TB)** | **Seen-speaker** | **41.36** | **22.90** | **Yes (Weights + Code + Webcam E2E)** |
| **DOKDO (Ours)** | **Visual-only** | **Auto-AVSR Conformer + SentencePiece (8,001)** | **OLKAVS (~1.5 TB)** | **Unseen-speaker** | 89.55 | 72.70 | **Yes** |
| **DOKDO (Ours)** | **Visual-only** | **Auto-AVSR Conformer + SentencePiece (8,001)** | **KMSAV (~100 GB)** | **In-the-wild (YouTube)** | 99.47 | 95.83 | **Yes** |

> **Definition of "Natural Multi-Word Sentence Utterances (자연발화)"**: In this paper, *natural utterance* refers to continuous sentence-level Korean speech containing multiple words/eojeols (as opposed to isolated word/digit classification in earlier Korean lip-reading studies), rather than unscripted spontaneous conversation.

---

## 4. On-Device Webcam Pipeline & Latency Measurement Protocol

<p align="center">
  <img src="assets/result.jpeg" width="85%" />
</p>

### 4.1 Preprocessing & ROI Extraction Order (`pipelines/detectors/mediapipe/`)
Both offline dataset preprocessing (`Train/preprocess.py`) and live webcam inference (`demo_webcam.py`) execute the exact same visual preprocessing sequence:
1. **Face & 4-Keypoint Detection**: MediaPipe Face Detection extracts 4 facial keypoints per frame (left eye, right eye, nose tip, mouth center), falling back from `full_range_detector` to `short_range_detector` if needed.
2. **Temporal Landmark Smoothing**: Linear interpolation fills missing frames, and a **12-frame sliding window** (`window_margin=12`) averages landmarks to suppress frame-to-frame jitter.
3. **Similarity Affine Warp**: `warp_img` estimates a 2D similarity transformation matrix aligning the smoothed landmarks to the canonical LRS3 mean face (`256x256`).
4. **Mouth ROI Crop (`96x96`)**: Crops a `96x96` bounding box centered on the transformed mouth landmark (`start_height=116, stop_height=212, start_width=79, stop_width=175`) and converts to single-channel grayscale `(T, 96, 96)` `uint8`.
5. **Spatial Crop & Normalization (`88x88`)**: `VideoTransform` scales pixels to `[0, 1]`, applies `CenterCrop(88)` (random crop + horizontal flip during training), and normalizes with `(x - 0.421) / 0.165` to produce `(1, T, 88, 88)`.

### 4.2 Two Webcam Execution Modes (`demo_webcam.py`)
1. **Manual E2E Benchmark Mode (`python demo_webcam.py --mode manual`, default)**:
   - User presses `SPACE` to start and stop recording an utterance clip (`1280x720 @ 25fps`).
   - Immediately upon stopping, `KoreanInferencePipeline.forward_with_timing()` executes preprocessing + VSR inference and logs the exact stage-by-stage latency breakdown (`t_preprocess`, `t_infer`, `t_total`, `RTF`).
2. **Asynchronous MAR Producer–Consumer Mode (`python demo_webcam.py --mode async`, Figure 1)**:
   - **Producer (`util/lip_tracker_dokdo.py`)**: Calibrates the user's closed-mouth 3D Mouth Aspect Ratio (MAR) for 3 seconds (`OPEN_THRESHOLD = mean(MAR) + 0.04`). When `MAR > OPEN_THRESHOLD` under frontal face conditions, merges a **15-frame pre-roll ring buffer** (`deque(maxlen=15)`) to preserve word-initial consonants. After **1.0s of silence** and 2-stage frame count verification (`total > 20`, `valid_mar >= 5`), runs `VideoProcess` to create a `(T, 96, 96)` `uint8` array and enqueues it into `queue.Queue(maxsize=4)` (dropping the oldest item if full).
   - **Consumer (`demo_webcam.py`)**: Dequeues `(T, 96, 96)` in a background daemon thread, applies `88x88` normalization, runs `KoreanAVSR.infer()`, and renders Korean subtitles on the live preview without blocking the camera loop.

### 4.3 On-Device Latency Specification (Table 2 & Section III.3)

| Item | Specification |
| :--- | :--- |
| **Device** | MacBook Pro (Apple M5 Pro, 48 GB Unified Memory) |
| **Execution Backend** | CPU (`15-core ARM64`, `device="cpu"`, FP32) |
| **Camera Input** | `1280x720 @ 25fps` |
| **Measured Utterance Length** | Average **5.0 s** (`~125 frames @ 25fps`, e.g., *"요즘 연애를 하고 싶은데 일상이 바빠서 할 수가 없어."*) |
| **Latency Measurement Boundary** | **Recording time excluded.** Measured from utterance completion ($t_0$) through MediaPipe preprocessing ($t_1$) to final Korean text output ($t_2$): $t_{\text{total}} = (t_1 - t_0) + (t_2 - t_1)$ |
| **Average Processing Time** | **6.61 s** per 5.0 s utterance ($\text{RTF} \approx 1.32$) |
| **Figure 2 Demo Example** | **GT**: `"요즘 연애를 하고 싶은데 일상이 바빠서 할 수가 없어."` (10 words)<br>**PRED**: `"요즘 연애 하고 싶은데 일상 바빠서 할수"` (Core content words preserved; visually ambiguous short grammatical particles `를/이` and sentence-final `가 없어` omitted due to subtle lip aperture) |

---

## 5. Repository Structure & Paper-to-Code Mapping

| File Path | Paper Reference | Role & Description |
| :--- | :---: | :--- |
| `demo_webcam.py` | Sec. II.4, III.3, Fig. 1, Fig. 2, Table 2 | On-device webcam demo supporting `--mode manual` (SPACEbar E2E with latency breakdown) and `--mode async` (MAR Producer–Consumer queue) |
| `util/lip_tracker_dokdo.py` | Sec. II.4, Fig. 1 | Real-time MAR utterance detector (3s calibration, 15-frame pre-roll, 1.0s silence trigger, `(T, 96, 96)` ROI enqueue with oldest-drop) |
| `pipelines/pipeline_korean.py` | Sec. II.1, III.3 | End-to-end inference pipeline (`forward` & `forward_with_timing`) |
| `pipelines/model_korean.py` | Sec. II.2, II.3 | `KoreanAVSR` inference wrapper supporting `decode=ctc` (CTC Greedy) and `decode=hybrid` (Joint CTC/Attention Beam Search) |
| `pipelines/detectors/mediapipe/` | Sec. II.1, Fig. 1 | MediaPipe 4-keypoint `LandmarksDetector` + 12-frame smoothing, similarity affine warp, and `96x96` grayscale mouth crop (`VideoProcess`) |
| `pipelines/data/transforms.py` | Sec. II.1 | `VideoTransform`: `88x88` crop, frame-rate alignment, and `Normalize(0.421, 0.165)` |
| `configs/korean.ini` | Sec. II.2, III.2 | Runtime configuration (`model_path`, `spm_model`, `decode=ctc` or `hybrid`) |
| `Train/preprocess.py` | Sec. II.1, III.1 | Raw MP4 + AI Hub JSON $\rightarrow$ per-sentence `(T, 96, 96)` `.npz` segment generator with `8:1:1` video-level split (`split_info.json`) |
| `Train/train_tokenizer.py` | Sec. II.2 | Trains the 8,000-piece Korean SentencePiece Unigram model (`unigram8000.model`) from AI Hub JSON / `.npz` / text transcripts |
| `Train/tokenizer_utils.py` | Sec. II.2 | `KoreanTokenizer` wrapper (`vocab_size=8,001`: pieces `0..7999`, `CTC blank=8000`) |
| `Train/dataset.py` | Sec. II.1, III.1 | `KoreanVSRDataset` & `collate_fn` loading `.npz` clips and applying `VideoTransform` |
| `Train/model.py` | Sec. II.2, II.3, Fig. 1 | `KoreanVSRModel`: replaces LRS3 5,049 heads with Korean 8,001 heads and computes Hybrid CTC + Decoder CE loss |
| `Train/train.py` | Sec. II.3, III.1, Fig. 1 | Two-stage differential-LR training script (`freeze_epochs=5`, `lr_encoder=1e-5`, `lr_head=1e-3`, `ctc_weight=0.3`, `--pretrained_kr_ckpt`) |
| `Train/evaluate.py` | Sec. III.2, Table 1 | Computes micro-averaged CER (syllable edit distance) and WER (eojeol edit distance) on `.npz` test sets or TSV manifests |

---

## 6. Setup & Reproducible Commands

### 6.1 Environment Installation
```bash
git clone https://github.com/sangdeok-EE/DOKDO-VSR.git
cd DOKDO-VSR

conda create -n dokdo-vsr python=3.10 -y
conda activate dokdo-vsr
pip install -r requirements.txt
```

### 6.2 Pretrained Weights & Tokenizer Setup
1. **Korean VSR Model Weights (`korean_vsr_hybrid.pt`)**:  
   Download from [HuggingFace (`dlvusgks/dokdo-vsr`)](https://huggingface.co/dlvusgks/dokdo-vsr) and place `korean_vsr_hybrid.pt` in the repository root (`DOKDO-VSR/korean_vsr_hybrid.pt`).
2. **Korean SentencePiece Tokenizer (`pipelines/tokens/unigram8000.model`)**:  
   > **License Note**: Because the SentencePiece vocabulary is induced from AI Hub Korean Lip-Reading corpora (OLKAVS / KMSAV), the tokenizer binary is not committed directly to this public git repository under AI Hub's data redistribution policy.
   - If you already have `unigram8000.model`, place it at `pipelines/tokens/unigram8000.model`.
   - To generate `unigram8000.model` directly from your AI Hub JSON annotations or preprocessed `.npz` files:
     ```bash
     python Train/train_tokenizer.py \
         --json_dir /path/to/OLKAVS/json_labels \
         --output_prefix pipelines/tokens/unigram8000 \
         --vocab_size 8000
     ```

### 6.3 Step-by-Step Pipeline Execution

#### (1) Data Preprocessing (`MP4 + JSON` $\rightarrow$ `.npz`)
```bash
python Train/preprocess.py \
    --data_root /path/to/raw_videos \
    --json_root /path/to/json_labels \
    --output_dir data/preprocessed \
    --detector mediapipe \
    --train_ratio 0.8 --val_ratio 0.1 --seed 42
```

#### (2) Two-Stage Transfer Learning (Stage 1: OLKAVS $\rightarrow$ Stage 2: KMSAV)
```bash
# Stage 1: OLKAVS Transfer Learning (Phase 1: 5 epochs head warm-up -> Phase 2: full fine-tuning, Hybrid 0.3 CTC + 0.7 CE)
python Train/train.py \
    --train_dir data/preprocessed/train \
    --val_dir data/preprocessed/validation \
    --test_dir data/preprocessed/test \
    --chaplin_checkpoint benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth \
    --model_conf benchmarks/LRS3/models/LRS3_V_WER19.1/model.json \
    --spm_model pipelines/tokens/unigram8000.model \
    --output_dir Train/output_olkavs \
    --loss_type hybrid --ctc_weight 0.3 \
    --freeze_epochs 5 --lr_encoder 1e-5 --lr_head 1e-3 \
    --epochs 109 --batch_size 8 --grad_accum 2

# Stage 2: KMSAV Wild Domain Adaptation (initialized from OLKAVS best checkpoint)
python Train/train.py \
    --train_dir data/kmsav_preprocessed/train \
    --val_dir data/kmsav_preprocessed/validation \
    --pretrained_kr_ckpt Train/output_olkavs/checkpoints/best_model.pt \
    --model_conf benchmarks/LRS3/models/LRS3_V_WER19.1/model.json \
    --spm_model pipelines/tokens/unigram8000.model \
    --output_dir Train/output_kmsav \
    --loss_type hybrid --ctc_weight 0.3 \
    --freeze_epochs 5 --lr_encoder 1e-5 --lr_head 1e-3
```

#### (3) Quantitative Evaluation (CER & WER — Table 1)
```bash
# Evaluate with CTC Greedy Decoding (Paper Section II.2 / III.2)
python Train/evaluate.py \
    --test_dir data/preprocessed/test \
    --checkpoint korean_vsr_hybrid.pt \
    --spm_model pipelines/tokens/unigram8000.model \
    --decode ctc \
    --output_file eval_result_ctc.txt

# Evaluate with Joint CTC/Attention Beam Search
python Train/evaluate.py \
    --test_dir data/preprocessed/test \
    --checkpoint korean_vsr_hybrid.pt \
    --spm_model pipelines/tokens/unigram8000.model \
    --decode hybrid --ctc_weight 0.3 --beam_size 10 \
    --output_file eval_result_hybrid.txt
```

#### (4) On-Device Webcam Demo & Latency Validation (Figure 1, Figure 2, Table 2)
```bash
# Mode A: Manual SPACEbar Recording + Stage-by-Stage Latency Breakdown (Table 2: 5.0s -> 6.61s)
python demo_webcam.py --mode manual --decode ctc

# Mode B: Asynchronous MAR Auto-Triggered Producer-Consumer Queue Pipeline (Figure 1)
python demo_webcam.py --mode async --decode ctc
```

---
---

# DOKDO: 전이학습 기반 온디바이스 한국어 립리딩 시스템의 실환경 검증 (한국어 안내)

본 저장소는 논문 **「DOKDO : 전이학습 기반 온디바이스 한국어 립리딩 시스템의 실환경 검증」** 의 공식 구현 코드 및 재현 가이드를 제공합니다.  
영어 대규모 시각 음성 인식 모델([Auto-AVSR](https://github.com/mpc001/auto_avsr) / [Chaplin](https://github.com/amanvirparhar/chaplin))의 시각적 특징 추출 능력을 한국어로 전이하기 위해 **8,001개 규모의 한국어 SentencePiece Unigram 출력층 교체**와 **2단계 차등 학습률(Differential LR) 전이학습**을 수행하고, 외부 서버 통신 없이 **단일 노트북(MacBook Pro M5 Pro CPU) 웹캠 환경에서 End-to-End 실시간/온디바이스 구동**을 검증하였습니다.

---

## 1. 베이스 모델(Auto-AVSR) 상속 부분과 DOKDO의 한국어 확장 구분

| 구성 모듈 | 출처 구분 | 상세 구조 및 역할 |
| :--- | :---: | :--- |
| **Visual Frontend** | 베이스 모델 상속 (LRS3 사전학습) | 3D Convolution (`5x7x7`) + 2D ResNet-18 구조로 입술 ROI 프레임당 512차원 시공간 특징 추출 |
| **Conformer Encoder** | 베이스 모델 상속 (Phase 2 미세조정) | 12-Layer Conformer (`adim=768`, `aheads=12`, `eunits=3072`, Macaron FFN + MHSA + Conv `kernel=31`) |
| **한국어 토크나이저** | **DOKDO 자체 구축** | 한국어 SentencePiece Unigram (`vocab_size=8,001`: `<unk>=0, <s>=1, </s>=2`, 서브워드 `3..7999`, `CTC blank=8000`) |
| **출력 헤드 및 디코더 교체** | **DOKDO 자체 구현** | 기존 영어 5,049 클래스 출력층을 제거하고 8,001 차원 한국어 `ctc.ctc_lo` 및 Transformer Decoder 임베딩/출력층으로 교체 (`Train/model.py`) |
| **2단계 차등 학습률 전이학습** | **DOKDO 자체 구현** | Phase 1(Epoch 1–5) 백본 동결 및 헤드 워밍업 $\rightarrow$ Phase 2(Epoch 6+) 전체 해동 및 차등 학습률(`1e-5` / `1e-3`) + 혼합 손실($0.3 L_{\text{CTC}} + 0.7 L_{\text{CE}}$) 적용 (`Train/train.py`) |
| **온디바이스 웹캠 E2E 파이프라인** | **DOKDO 자체 구현** | MediaPipe 4점 검출·12프레임 스무딩·Affine 정렬·`96x96` $\rightarrow$ `88x88` 정규화 및 MAR 기반 비동기 생산자-소비자 큐(`Queue(maxsize=4)`) 구현 (`demo_webcam.py`, `util/lip_tracker_dokdo.py`) |

---

## 2. 2단계 전이학습 및 손실 함수 명세 (논문 II.3절 및 그림 1)

- **혼합 손실 함수 (Hybrid Loss)**:
  $$L = 0.3 \cdot L_{\text{CTC}} + 0.7 \cdot L_{\text{CE}}$$
  - 학습 시에는 Conformer 인코더 위의 CTC 분기($\lambda=0.3$)와 6-Layer Transformer 디코더의 교차 엔트로피 손실($1-\lambda=0.7$, Label Smoothing `0.1`)을 결합하여 프레임 단위 정렬과 한국어 문맥 학습을 동시에 수행합니다.
  - **학습 경로와 추론 경로의 구분**: 온디바이스 실환경 추론(`decode=ctc`, 논문 기본값) 시에는 연산량이 큰 자기회귀 Transformer 디코더를 생략하고, **Conformer 인코더 + CTC 선형 계층(`ctc.ctc_lo`)의 CTC Greedy 디코딩 + SentencePiece 역토큰화**만 수행하여 CPU 지연을 최소화합니다. (필요 시 `--decode hybrid`로 CTC+Attention 빔 서치도 선택 가능)

| 학습 단계 | 에폭 구간 | 학습 대상 모듈 | 동결(Freeze) 모듈 | 학습률 (AdamW, `wd=0.01`) | 단계별 목적 |
| :--- | :---: | :--- | :--- | :---: | :--- |
| **Phase 1 (Head Warm-up)** | Epoch 1 – 5 | 한국어 CTC 헤드 + 디코더 (`12.3M` 파라미터) | Visual Frontend + Conformer 인코더 (`242.6M` 파라미터) | `LR_HEAD = 1e-3` (3 에폭 선형 워밍업) | 무작위 초기화된 한국어 출력 헤드의 초기 큰 그래디언트로부터 사전학습 백본 보호 |
| **Phase 2 (Full Fine-Tuning)** | Epoch 6 – 109 | **전체 네트워크** (`254.9M` 파라미터) | 없음 (전체 해동) | `LR_ENCODER = 1e-5`, `LR_HEAD = 1e-3` (CosineAnnealingLR) | 파괴적 망각(Catastrophic Forgetting)을 억제하며 한국어 입모양 조음 패턴에 맞게 백본 미세조정 |

---

## 3. 전처리·추론 실행 순서 및 지연 시간(6.61초) 측정 기준

### 3.1 전처리 단계별 순서 (`Train/preprocess.py`, `pipelines/detectors/mediapipe/`)
1. **얼굴 4점 랜드마크 검출**: MediaPipe Face Detection으로 프레임별 양안·코끝·입 중심 4개 키포인트 추출
2. **12프레임 윈도우 스무딩**: 결측 프레임 선형 보간 후 12프레임 이동 평균으로 랜드마크 떨림 제거
3. **유사 변환(Similarity Affine Warp)**: LRS3 평균 얼굴 좌표계(`256x256`)로 회전·스케일 정렬
4. **`96x96` 입술 ROI 크롭 및 그레이스케일 변환**: 입 중심 기준 `(T, 96, 96)` `uint8` 텐서 생성
5. **`88x88` 크롭 및 정규화**: `[0, 1]` 스케일링 $\rightarrow$ `CenterCrop(88)` (학습 시 RandomCrop + HorizontalFlip) $\rightarrow$ `Normalize(mean=0.421, std=0.165)` $\rightarrow$ `(1, T, 88, 88)` 모델 입력

### 3.2 평균 5.0초 발화 $\rightarrow$ 6.61초 처리 지연 측정 프로토콜 (논문 III.3절 및 표 2)
- **측정 기기**: MacBook Pro (Apple M5 Pro, 48GB RAM, 15-core ARM64 CPU, `1280x720 @ 25fps`)
- **측정 구간 정의**: **사용자가 말을 하는 녹화 시간(예: 5.0초, 약 125프레임)은 처리 시간에서 제외**합니다. 발화 녹화가 완료된 시점($t_0$)부터 MediaPipe 랜드마크 검출·Affine 정렬·ROI 정규화 전처리 완료 시점($t_1$, `t_preprocess`), 그리고 Conformer 인코더 및 CTC Greedy 디코딩을 거쳐 최종 한국어 텍스트가 출력되는 시점($t_2$, `t_infer`)까지의 합산 시간($t_{\text{total}} = t_{\text{preprocess}} + t_{\text{infer}}$)을 측정합니다.
- `python demo_webcam.py --mode manual` 실행 시 매 발화마다 콘솔과 `result_korean.txt`에 `Utterance`, `Preprocess`, `VSR Infer`, `Total`, `RTF`, `Session Avg`가 자동 출력 및 기록됩니다.

---

## Acknowledgments

This project builds upon [Chaplin](https://github.com/amanvirparhar/chaplin) and [Auto-AVSR](https://github.com/mpc001/auto_avsr) (Imperial College London), using [ESPnet](https://github.com/espnet/espnet) for the Conformer/Transformer backbone. Korean lip-reading datasets (OLKAVS, KMSAV) were provided by [AI Hub](https://www.aihub.or.kr/).

## License

This project is licensed under the Apache-2.0 License. Model weights are subject to the licenses of the underlying pretrained models and datasets.
