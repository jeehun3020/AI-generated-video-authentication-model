# AI-generated Video Authentication Model

I SEE YOU 팀의 동영상 진위 판별 실험 코드입니다.

현재 프로젝트 목표는 업로드된 동영상이 들어왔을 때 다음을 수행하는 것입니다.
- 프레임 추출
- 얼굴 또는 전체 프레임 기반 특징 추출
- `real / generated / deepfake` 혹은 `real / generated` 분류
- 시계열(temporal), RGB, FFT 신호를 결합한 video-level 판별

이 레포는 캡스톤에서 바로 실험을 이어갈 수 있도록 구성한 baseline/experiment 코드베이스입니다.

## 핵심 특징
- PyTorch 기반 학습/평가/추론 파이프라인
- config 기반 실험 관리
- frame 모델 + temporal 모델 + ensemble 추론
- YouTube Shorts 수집/다운로드 스크립트 포함
- leakage 방지를 위한 group-based split
- binary / multiclass 실험 모두 지원
- FFT, face crop, background-masked view 실험 축 지원

## 프로젝트 구조

```text
AI-generated-video-authentication-model/
├── configs/                 # 실험 설정 파일
├── iseeyou/                 # 핵심 라이브러리 코드
│   ├── data/
│   ├── engine/
│   ├── models/
│   └── utils/
├── scripts/                 # 수집/다운로드/실험 실행 스크립트
├── prepare_data.py          # raw -> processed/manifests
├── train.py                 # frame 모델 학습
├── eval.py                  # frame 모델 평가
├── inference.py             # frame 모델 단일 추론
├── train_temporal.py        # temporal 모델 학습
├── eval_temporal.py         # temporal 모델 평가
├── inference_temporal.py    # temporal 단일 추론
├── inference_ensemble.py    # frame + temporal + FFT 앙상블 추론
├── requirements.txt
└── pyproject.toml
```

## 권장 환경
- macOS / Linux
- Python `3.10+`
- `ffmpeg` 설치 권장
- 가상환경 사용 권장

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

참고:
- Python 3.7에서는 최신 `torch`가 설치되지 않습니다.
- macOS에서 OpenMP 이슈가 있을 경우, 현재 코드가 일부 `num_workers=0`으로 자동 보정합니다.

## 데이터셋
현재 코드베이스는 아래 데이터셋들을 사용할 수 있게 구성되어 있습니다.

### 실사용/YouTube 중심
- YouTube Shorts `real`
- YouTube Shorts `generated`
- real hard-negative Shorts
- query 기반 real hard-negative Shorts
- GenImage

### 추가 동영상 데이터셋
- UCF101
- FaceForensics++
- Celeb-DF
- VoxCeleb2 (현재 데이터 준비가 끝난 경우에만 사용)
- StyleGAN 계열 이미지셋 (선택)

## 데이터 경로 예시
대용량 데이터는 레포에 포함하지 않습니다. 아래처럼 로컬에 준비한 뒤 config의 `datasets.*.root`와 맞춰 사용하면 됩니다.

```text
data/raw/
├── youtube_shorts_binary/
│   ├── real/
│   └── generated/
├── youtube_shorts_real_hardnegative/
│   └── real/
└── youtube_shorts_real_hardnegative_query/
    └── real/

UCF101-Action Recognition/
FaceForensics++_C23/
Celeb_V2/
gen_img/
```

## 전처리

```bash
python prepare_data.py --config configs/baseline_binary_allvideo.yaml
```

생성 결과:
- `data/processed/...`
- `data/manifests.../*.csv`
- `data/manifests.../preprocess_report.json`

전처리 시 다음을 수행합니다.
- raw sample 수집
- leakage 방지 split
- 영상 -> frame extraction
- detector crop / full-frame / background-masked view 생성
- manifest 작성

손상된 영상은 경고 후 skip하도록 처리되어 있습니다.

## 학습

### 1. Frame 모델
```bash
python train.py --config configs/baseline_binary_allvideo.yaml
```

### 2. Temporal 모델
```bash
python train_temporal.py --config configs/temporal_binary_allvideo.yaml
```

### 3. FFT 모델
```bash
python train.py --config configs/baseline_binary_allvideo_fft.yaml
```

## 평가

### Frame 평가
```bash
python eval.py \
  --config configs/baseline_binary_allvideo.yaml \
  --split test \
  --checkpoint outputs/checkpoints_allvideo_binary/best.pt
```

### Temporal 평가
```bash
python eval_temporal.py \
  --config configs/temporal_binary_allvideo.yaml \
  --split test \
  --checkpoint outputs/checkpoints_temporal_allvideo_binary/best.pt
```

## 추론

### 단일 영상 파일
```bash
python inference.py \
  --config configs/baseline_binary_allvideo.yaml \
  --checkpoint outputs/checkpoints_allvideo_binary/best.pt \
  --video-path /absolute/path/to/input_video.mp4 \
  --aggregation confidence_mean \
  --min-confidence 0.55 \
  --save-frame-csv
```

### YouTube Shorts URL
```bash
python inference.py \
  --config configs/baseline_binary_allvideo.yaml \
  --checkpoint outputs/checkpoints_allvideo_binary/best.pt \
  --youtube-url "https://www.youtube.com/shorts/VIDEO_ID" \
  --aggregation confidence_mean \
  --min-confidence 0.55 \
  --save-frame-csv
```

### 앙상블 추론
```bash
python inference_ensemble.py \
  --config configs/temporal_binary_allvideo.yaml \
  --youtube-url "https://www.youtube.com/shorts/VIDEO_ID" \
  --frame-checkpoint outputs/checkpoints_allvideo_binary/best.pt \
  --temporal-checkpoint outputs/checkpoints_temporal_allvideo_binary/best.pt \
  --freq-checkpoint outputs/checkpoints_allvideo_binary_fft/best.pt
```

## 주요 실험 트랙

### 1. YouTube Shorts binary
- `configs/baseline_binary_shorts*.yaml`
- `configs/temporal_binary_shorts*.yaml`

### 2. Hold-out binary
- `configs/baseline_binary_shorts_holdout.yaml`
- `configs/temporal_binary_shorts_holdout.yaml`

### 3. All-video binary
- `configs/baseline_binary_allvideo.yaml`
- `configs/temporal_binary_allvideo.yaml`
- `configs/baseline_binary_allvideo_fft.yaml`

### 4. Multiclass
- `configs/baseline_multiclass.yaml`
- `configs/baseline_multiclass_allvideo.yaml`

## 실행 스크립트
- `scripts/run_allvideo_binary_experiment.sh`
- `scripts/run_allvideo_multicue_tuning.sh`
- `scripts/run_shorts_binary_experiment.sh`
- `scripts/run_shorts_binary_inference.sh`
- `scripts/run_ultraquick_baseline.sh`

## YouTube 수집 관련
`scripts/`에는 아래 기능이 포함되어 있습니다.
- Shorts URL 수집
- 채널 seed 기반 수집
- query 기반 수집
- yt-dlp 다운로드
- hard-negative 후보 구성
- batch inference 및 정책 비교

## 현재 권장 방향
현재 실험 기준으로는 다음 조합이 가장 중요합니다.
- `full-frame RGB`
- `temporal`
- `FFT`

즉, 최종 목표는 얼굴에만 국한되지 않고 동영상 전체의
- 시각적 질감
- 시계열 일관성
- 주파수 artifact
를 함께 보는 video authenticity classifier입니다.

## 주의
- 이 레포에는 대용량 데이터셋과 실험 출력물이 포함되어 있지 않습니다.
- `data/raw`, `data/processed`, `outputs/`는 로컬에서 생성/관리해야 합니다.
- pretrained 모델 다운로드나 YouTube 다운로드에는 네트워크가 필요합니다.

## 빠른 시작

```bash
python prepare_data.py --config configs/baseline_binary_allvideo.yaml
python train.py --config configs/baseline_binary_allvideo.yaml
python eval.py --config configs/baseline_binary_allvideo.yaml --split test --checkpoint outputs/checkpoints_allvideo_binary/best.pt
```

그 다음:
```bash
python train_temporal.py --config configs/temporal_binary_allvideo.yaml
python train.py --config configs/baseline_binary_allvideo_fft.yaml
bash scripts/run_allvideo_multicue_tuning.sh
```

## Text-Mask 실험

Shorts 영상의 자막, 제목, 워터마크 같은 텍스트 bias를 줄이기 위해 `text-mask` 실험 트랙을 추가했습니다.

핵심 아이디어:
- 전처리에서 상단/하단 밴드를 마스킹해 텍스트 정보 약화
- 학습 augmentation에서 랜덤 밴드 마스킹 추가
- 모델이 텍스트보다 배경, 인물, 물체, 질감, 움직임을 더 보도록 유도

관련 설정:
- `configs/baseline_binary_allvideo_textmask.yaml`
- `configs/temporal_binary_allvideo_textmask.yaml`
- `configs/baseline_binary_allvideo_fft_textmask.yaml`

관련 코드:
- `iseeyou/utils/masking.py`
- `iseeyou/data/preprocess.py`
- `iseeyou/data/transforms.py`
- `inference.py`
- `inference_ensemble.py`
