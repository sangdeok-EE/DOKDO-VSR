# DOKDO: Korean Visual Speech Recognition

Webcam-based Korean lip reading system. Recognizes Korean speech from lip movements only (no audio).

Built by transfer-learning [Auto-AVSR](https://github.com/mpc001/auto_avsr) (LRS3, WER 19.1%) with Korean datasets (OLKAVS, KMSAV). See our paper for details.

## Architecture

![DOKDO Pipeline](assets/pipeline.jpeg)

- **Frontend:** Conv3D + ResNet-18 spatiotemporal feature extractor
- **Encoder:** 12-layer Conformer (768-dim, 254.95M params), pretrained on LRS3 (English, WER 19.1%)
- **Decoder:** Hybrid CTC + CE Loss (L = 0.3 · L_CTC + 0.7 · L_CE)
- **Tokenizer:** SentencePiece Unigram (vocab=8,001, blank=8000)
- **Preprocessing:** MediaPipe 4-keypoint face detection → affine warp to mean face → 96x96 grayscale lip crop → center crop(88x88) + normalization
- **Training:** 2-phase transfer learning on single RTX 4090
  - Phase 1 (epoch 1-5): Encoder frozen, CTC head only — stabilize Korean token prediction
  - Phase 2 (epoch 6+): Full network unfreeze, differential LR (lower for encoder, higher for head) — preserve pretrained visual features while adapting to Korean
- **Inference latency:** ~6.61s per 5.0s utterance (on MacBook Pro M5, CPU only)

## Training Data

| Dataset | Source | Size Used | Description |
|---------|--------|-----------|-------------|
| [OLKAVS](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=538) | AI Hub | ~1.5 TB (of 14 TB) | Studio-recorded Korean speech, multi-view, 1,107 speakers |
| [KMSAV](https://github.com/etri/kmsav) | ETRI | ~100 GB | Korean multi-speaker spontaneous audiovisual speech (YouTube) |

OLKAVS was used for primary training (~109 epochs), then KMSAV was added for domain diversity.

## Performance

![DOKDO Performance](assets/result.jpeg)

| Dataset | WER (%) | CER (%) |
|---------|---------|---------|
| OLKAVS (Seen-speaker) | 41.36 | 22.9 |
| OLKAVS (Unseen-speaker) | 89.55 | 72.7 |
| KMSAV (YouTube) | 99.47 | 95.83 |

## Setup

### 1. Clone & install

```bash
git clone https://github.com/sangdeok-EE/DOKDO-VSR.git
cd DOKDO-VSR
pip install -r requirements.txt
```

> Python 3.9+, PyTorch 2.0+

### 2. Download files

| File | Where to place | Link |
|------|---------------|------|
| `korean_vsr_hybrid.pt` (~1 GB) | Project root | [Download](https://huggingface.co/dlvusgks/dokdo-vsr) |
| `model.json` (8 KB) | `benchmarks/LRS3/models/LRS3_V_WER19.1/` | [Auto-AVSR checkpoint](https://drive.google.com/file/d/1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6/view) |

### 3. Prepare tokenizer

**The tokenizer is not included in this repository.** You need to train your own SentencePiece Unigram tokenizer using your Korean speech dataset.

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='your_korean_transcripts.txt',   # one sentence per line
    model_prefix='unigram8000',
    model_type='unigram',
    vocab_size=8000,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=-1,
)
```

Place the trained `unigram8000.model` file in `pipelines/tokens/` and update the path in `configs/korean.ini`:

```ini
spm_model=pipelines/tokens/unigram8000.model
```

> **Note:** We used `vocab_size=8000` due to hardware constraints, but you can adjust this value to fit your setup. For best results, the tokenizer should match the vocabulary used during model training. If you use a different tokenizer, the model output may be degraded.

## Usage

```bash
python demo_webcam.py
```

| Key | Action |
|-----|--------|
| SPACE | Start / stop recording |
| q | Quit |

Recording stops → inference runs automatically → Korean text printed to console.

### Camera selection

```bash
CAMERA_INDEX=1 python demo_webcam.py
```

## Project Structure

```
├── demo_webcam.py                          # Webcam demo (25fps capture → inference → Korean text)
├── configs/
│   └── korean.ini                          # Model & decode config (hybrid, ctc_weight, beam_size)
├── pipelines/
│   ├── pipeline_korean.py                  # End-to-end inference pipeline (video → text)
│   ├── model_korean.py                     # KoreanAVSR model wrapper + hybrid beam search
│   ├── data/
│   │   ├── data_module.py                  # AVSRDataLoader (video loading + speed adjustment)
│   │   └── transforms.py                   # center crop(88×88), normalization (mean=0.421, std=0.165)
│   ├── detectors/mediapipe/
│   │   ├── detector.py                     # MediaPipe 4-keypoint face/lip landmark detection
│   │   ├── video_process.py                # Affine warp → 96×96 grayscale lip ROI crop
│   │   └── 20words_mean_face.npy           # Reference face landmarks for alignment
│   └── tokens/                             # Place your SentencePiece tokenizer here
├── espnet/                                 # E2E model architecture (Conformer encoder + Transformer decoder)
│   ├── nets/pytorch_backend/
│   │   ├── e2e_asr_transformer.py          # Main E2E model (encoder-decoder with CTC)
│   │   ├── backbones/                      # Conv3D + ResNet-18 visual frontend
│   │   └── transformer/                    # Attention, encoder/decoder layers, embedding
│   └── ...
├── Train/                                  # Training code (2-phase transfer learning)
│   ├── train.py                            # Training script (Phase1: freeze → Phase2: unfreeze, differential LR)
│   ├── model.py                            # KoreanVSRModel (E2E wrapper + CTC head replacement)
│   ├── dataset.py                          # PyTorch Dataset (.npz lip sequences)
│   ├── preprocess.py                       # Video → .npz preprocessing (face detect + lip crop)
│   ├── tokenizer_utils.py                  # SentencePiece tokenizer wrapper
│   └── requirements.txt                    # Training dependencies
├── benchmarks/LRS3/models/LRS3_V_WER19.1/
│   └── model.json                          # Auto-AVSR architecture config (required to build E2E model)
├── assets/                                 # Architecture diagram & performance charts
├── requirements.txt                        # Inference dependencies
└── LICENSE                                 # Apache 2.0
```

## Acknowledgements

- [Chaplin](https://github.com/amanvirparhar/chaplin) by Amanvir Parhar
- [Auto-AVSR](https://github.com/mpc001/auto_avsr) by Pingchuan Ma, Imperial College London
- [ESPnet](https://github.com/espnet/espnet)

## License

Apache 2.0

---

# DOKDO: 한국어 시각 발화 인식

웹캠으로 실시간 한국어 입모양 인식. 음성 없이 입술 움직임만으로 한국어를 인식합니다.

[Auto-AVSR](https://github.com/mpc001/auto_avsr) (LRS3, WER 19.1%) 사전학습 모델을 한국어 데이터셋(OLKAVS, KMSAV)으로 전이학습하였습니다. 자세한 내용은 논문을 참고하세요.

## 아키텍처

![DOKDO 파이프라인](assets/pipeline.jpeg)

- **프론트엔드:** Conv3D + ResNet-18 시공간 특징 추출기
- **인코더:** 12층 Conformer (768차원, 254.95M 파라미터), LRS3 (영어, WER 19.1%) 사전학습
- **디코더:** Hybrid CTC + CE Loss (L = 0.3 · L_CTC + 0.7 · L_CE)
- **토크나이저:** SentencePiece Unigram (vocab=8,001, blank=8000)
- **전처리:** MediaPipe 4-keypoint 얼굴 검출 → mean face 기준 어파인 변환 → 96x96 흑백 입술 크롭 → center crop(88x88) + 정규화
- **학습:** 단일 RTX 4090에서 2단계 전이학습
  - 1단계 (epoch 1-5): 인코더 동결, CTC 헤드만 학습 — 한국어 토큰 예측 안정화
  - 2단계 (epoch 6+): 전체 네트워크 unfreeze, 차등 학습률 (인코더 낮게, 헤드 높게) — 사전학습된 시각 특징 유지하며 한국어 적응
- **추론 지연:** 5.0초 발화 기준 ~6.61초 (MacBook Pro M5, CPU 전용)

## 학습 데이터

| 데이터셋 | 출처 | 사용량 | 설명 |
|---------|------|--------|------|
| [OLKAVS](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=538) | AI Hub | ~1.5 TB (전체 14 TB 중) | 스튜디오 녹화 한국어 음성, 다시점, 1,107명 화자 |
| [KMSAV](https://github.com/etri/kmsav) | ETRI | ~100 GB | 한국어 다화자 자발적 시청각 음성 (YouTube) |

OLKAVS로 주요 학습(~109 epochs) 후, KMSAV를 추가하여 도메인 다양성을 확보하였습니다.

## 성능

![DOKDO 성능](assets/result.jpeg)

| 데이터셋 | WER (%) | CER (%) |
|---------|---------|---------|
| OLKAVS (Seen-speaker) | 41.36 | 22.9 |
| OLKAVS (Unseen-speaker) | 89.55 | 72.7 |
| KMSAV (YouTube) | 99.47 | 95.83 |

## 설치

### 1. 클론 & 설치

```bash
git clone https://github.com/sangdeok-EE/DOKDO-VSR.git
cd DOKDO-VSR
pip install -r requirements.txt
```

> Python 3.9 이상, PyTorch 2.0 이상

### 2. 파일 다운로드

| 파일 | 위치 | 링크 |
|------|------|------|
| `korean_vsr_hybrid.pt` (~1 GB) | 프로젝트 루트 | [다운로드](https://huggingface.co/dlvusgks/dokdo-vsr) |
| `model.json` (8 KB) | `benchmarks/LRS3/models/LRS3_V_WER19.1/` | [Auto-AVSR 체크포인트](https://drive.google.com/file/d/1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6/view) |

### 3. 토크나이저 준비

**토크나이저는 이 저장소에 포함되어 있지 않습니다.** 한국어 음성 데이터셋을 사용하여 직접 SentencePiece Unigram 토크나이저를 학습해야 합니다.

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='your_korean_transcripts.txt',   # 한 줄에 문장 하나
    model_prefix='unigram8000',
    model_type='unigram',
    vocab_size=8000,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=-1,
)
```

학습된 `unigram8000.model` 파일을 `pipelines/tokens/`에 넣고, `configs/korean.ini`에서 경로를 수정하세요:

```ini
spm_model=pipelines/tokens/unigram8000.model
```

> **참고:** 저희는 하드웨어 제약으로 `vocab_size=8000`을 사용했지만, 환경에 맞게 조정할 수 있습니다. 모델 학습 시 사용된 어휘와 동일한 토크나이저를 사용해야 최적의 성능을 얻을 수 있습니다. 다른 토크나이저를 사용하면 모델 출력이 저하될 수 있습니다.

## 사용법

```bash
python demo_webcam.py
```

| 키 | 동작 |
|----|------|
| 스페이스바 | 녹화 시작 / 종료 |
| q | 종료 |

녹화 종료 시 자동으로 추론 → 인식된 한국어 텍스트가 콘솔에 출력됩니다.

### 카메라 선택

```bash
CAMERA_INDEX=1 python demo_webcam.py
```

## 프로젝트 구조

```
├── demo_webcam.py                          # 웹캠 데모 (25fps 캡처 → 추론 → 한국어 텍스트)
├── configs/
│   └── korean.ini                          # 모델 & 디코딩 설정 (hybrid, ctc_weight, beam_size)
├── pipelines/
│   ├── pipeline_korean.py                  # E2E 추론 파이프라인 (영상 → 텍스트)
│   ├── model_korean.py                     # KoreanAVSR 모델 래퍼 + 하이브리드 빔서치
│   ├── data/
│   │   ├── data_module.py                  # AVSRDataLoader (영상 로딩 + 속도 조정)
│   │   └── transforms.py                   # center crop(88×88), 정규화 (mean=0.421, std=0.165)
│   ├── detectors/mediapipe/
│   │   ├── detector.py                     # MediaPipe 4-keypoint 얼굴/입술 랜드마크 검출
│   │   ├── video_process.py                # 어파인 변환 → 96×96 흑백 입술 ROI 크롭
│   │   └── 20words_mean_face.npy           # 정렬 기준 얼굴 랜드마크
│   └── tokens/                             # SentencePiece 토크나이저를 여기에 넣으세요
├── espnet/                                 # E2E 모델 아키텍처 (Conformer 인코더 + Transformer 디코더)
│   ├── nets/pytorch_backend/
│   │   ├── e2e_asr_transformer.py          # 메인 E2E 모델 (인코더-디코더 + CTC)
│   │   ├── backbones/                      # Conv3D + ResNet-18 시각 프론트엔드
│   │   └── transformer/                    # Attention, 인코더/디코더 레이어, 임베딩
│   └── ...
├── Train/                                  # 학습 코드 (2단계 전이학습)
│   ├── train.py                            # 학습 스크립트 (1단계: 동결 → 2단계: 해제, 차등 학습률)
│   ├── model.py                            # KoreanVSRModel (E2E 래퍼 + CTC 헤드 교체)
│   ├── dataset.py                          # PyTorch Dataset (.npz 입술 시퀀스)
│   ├── preprocess.py                       # 영상 → .npz 전처리 (얼굴 검출 + 입술 크롭)
│   ├── tokenizer_utils.py                  # SentencePiece 토크나이저 래퍼
│   └── requirements.txt                    # 학습 의존성
├── benchmarks/LRS3/models/LRS3_V_WER19.1/
│   └── model.json                          # Auto-AVSR 아키텍처 설정 (E2E 모델 구축에 필요)
├── assets/                                 # 아키텍처 다이어그램 & 성능 차트
├── requirements.txt                        # 추론 의존성
└── LICENSE                                 # Apache 2.0
```

## 감사의 글

- [Chaplin](https://github.com/amanvirparhar/chaplin) -- Amanvir Parhar
- [Auto-AVSR](https://github.com/mpc001/auto_avsr) -- Pingchuan Ma, Imperial College London
- [ESPnet](https://github.com/espnet/espnet)

## 라이선스

Apache 2.0
