# DOKDO: Korean Visual Speech Recognition

Webcam-based Korean lip reading system. Recognizes Korean speech from lip movements only (no audio).

Built by transfer-learning [Auto-AVSR](https://github.com/mpc001/auto_avsr) (LRS3, WER 19.1%) with Korean datasets (OLKAVS, KMSAV). See our paper for details.

## Setup

### 1. Clone & install

```bash
git clone https://github.com/sangdeok-EE/DOKDO-VSR.git
cd dokdo-vsr
pip install -r requirements.txt
```

> Python 3.9+, PyTorch 2.0+

### 2. Download files

| File | Where to place | Link |
|------|---------------|------|
| `korean_vsr_hybrid.pt` (~1 GB) | Project root | [Download](https://huggingface.co/dlvusgks/dokdo-vsr) |
| `model.json` (8 KB) | `benchmarks/LRS3/models/LRS3_V_WER19.1/` | [Auto-AVSR checkpoint](https://drive.google.com/file/d/1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6/view) |

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
├── demo_webcam.py              # Webcam demo
├── configs/korean.ini          # Model config
├── pipelines/
│   ├── model_korean.py         # Model loader
│   ├── pipeline_korean.py      # Inference pipeline
│   ├── data/                   # Preprocessing
│   ├── detectors/mediapipe/    # Face detection + lip ROI crop
│   └── tokens/                 # SentencePiece tokenizer
├── espnet/                     # E2E model architecture
└── requirements.txt
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

## 설치

### 1. 클론 & 설치

```bash
git clone https://github.com/sangdeok-EE/DOKDO-VSR.git
cd dokdo-vsr
pip install -r requirements.txt
```

> Python 3.9 이상, PyTorch 2.0 이상

### 2. 파일 다운로드

| 파일 | 위치 | 링크 |
|------|------|------|
| `korean_vsr_hybrid.pt` (~1 GB) | 프로젝트 루트 | [다운로드](https://huggingface.co/dlvusgks/dokdo-vsr) |
| `model.json` (8 KB) | `benchmarks/LRS3/models/LRS3_V_WER19.1/` | [Auto-AVSR 체크포인트](https://drive.google.com/file/d/1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6/view) |

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
├── demo_webcam.py              # 웹캠 데모
├── configs/korean.ini          # 모델 설정
├── pipelines/
│   ├── model_korean.py         # 모델 로더
│   ├── pipeline_korean.py      # 추론 파이프라인
│   ├── data/                   # 전처리
│   ├── detectors/mediapipe/    # 얼굴 검출 + 입술 ROI 크롭
│   └── tokens/                 # SentencePiece 토크나이저
├── espnet/                     # E2E 모델 아키텍처
└── requirements.txt
```

## 감사의 글

- [Chaplin](https://github.com/amanvirparhar/chaplin) -- Amanvir Parhar
- [Auto-AVSR](https://github.com/mpc001/auto_avsr) -- Pingchuan Ma, Imperial College London
- [ESPnet](https://github.com/espnet/espnet)

## 라이선스

Apache 2.0
