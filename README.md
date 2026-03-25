# I SEE YOU - Face Authenticity Baseline

심화캡스톤디자인 "I SEE YOU" 팀용 baseline 코드입니다.

목표:
- 입력 영상에서 얼굴 프레임 추출
- 프레임별 분류 (`real` / `generated` / `deepfake`)
- 영상 단위 최종 예측 (mean / vote aggregation)
- 멀티큐 결합 (RGB + 주파수 FFT + 시계열 frame-diff)

우선순위:
- SOTA보다 **안정적으로 돌아가는 구조**
- 유지보수 가능한 모듈화
- config 기반 실험 관리

## 1) 프로젝트 구조

```text
Capstone/
├── configs/
│   ├── baseline_multiclass.yaml
│   └── baseline_binary.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── outputs/
│   ├── checkpoints/
│   ├── eval/
│   └── inference/
├── iseeyou/
│   ├── config.py
│   ├── constants.py
│   ├── data/
│   │   ├── adapters.py
│   │   ├── dataset.py
│   │   ├── manifest.py
│   │   ├── preprocess.py
│   │   ├── split.py
│   │   ├── transforms.py
│   │   └── detectors/
│   │       ├── base.py
│   │       ├── factory.py
│   │       └── mtcnn_detector.py
│   ├── engine/
│   │   ├── evaluator.py
│   │   └── trainer.py
│   ├── models/
│   │   └── builder.py
│   └── utils/
│       ├── metrics.py
│       ├── seed.py
│       └── video.py
├── prepare_data.py
├── train.py
├── eval.py
├── inference.py
├── requirements.txt
└── pyproject.toml
```

## 2) 데이터 파이프라인

`prepare_data.py`에서 수행:
1. raw dataset 스캔 (`adapters.py`)
2. leakage 방지 group split (`split.py`)
3. video -> frame extraction (`utils/video.py`)
4. face detection/crop (`detectors/*`)
5. face crop 저장 + manifest CSV 생성 (`preprocess.py`, `manifest.py`)

생성 결과:
- `data/processed/faces/{train|val|test}/.../*.jpg`
- `data/manifests/{train|val|test|all}.csv`
- `data/manifests/preprocess_report.json`

## 3) Leakage 방지 기준

group key 우선순위(기본):
- `original_id` -> `identity_id` -> `source_id` -> `video_id`

즉, 같은 원본/같은 identity/같은 source가 train/val/test에 중복되지 않게 split합니다.

## 4) 모델/학습

- Backbone: `timm` 기반 (`efficientnet_b0`, `convnext_tiny` 등)
- Loss: CrossEntropy
- Metric: Accuracy, F1(macro), AUC
- 멀티클래스/바이너리 전환:
  - `configs/baseline_multiclass.yaml`
  - `configs/baseline_binary.yaml`

## 5) 영상 단위 추론

`inference.py` 흐름:
1. 영상 프레임 샘플링
2. 프레임 얼굴 검출 + crop
3. 프레임별 예측
4. `mean` 또는 `vote`로 video-level prediction 생성

출력:
- JSON: `outputs/inference/*.json`
- (옵션) frame CSV: `--save-frame-csv`

## 6) 빠른 시작

### 6-1. 환경 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6-2. 데이터 위치 정리

압축 해제 후 아래 경로를 맞추거나, config의 `datasets.*.root`를 수정:
- `data/raw/UCF101`
- `data/raw/VoxCeleb2`
- `data/raw/StyleGAN`
- `data/raw/FaceForensics++`
- `data/raw/CelebDF-v2`

### 6-3. 전처리

```bash
python prepare_data.py --config configs/baseline_multiclass.yaml
```

### 6-4. 학습

```bash
python train.py --config configs/baseline_multiclass.yaml
```

### 6-5. 평가

```bash
python eval.py --config configs/baseline_multiclass.yaml --split test
```

video-level 집계를 명시하고 싶을 때:

```bash
python eval.py \
  --config configs/baseline_multiclass.yaml \
  --split test \
  --video-aggregation confidence_mean
```

### 6-6. 단일 영상 추론

```bash
python inference.py \
  --config configs/baseline_multiclass.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --video-path /absolute/path/to/input_video.mp4 \
  --aggregation confidence_mean \
  --min-confidence 0.55 \
  --save-frame-csv
```

YouTube URL 직접 입력:

```bash
python inference.py \
  --config configs/baseline_multiclass.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --aggregation confidence_mean \
  --min-confidence 0.55 \
  --save-frame-csv
```

### 6-7. 데이터 품질 점검 (권장)

```bash
python analyze_data_health.py --manifests-dir data/manifests
python analyze_data_health.py --manifests-dir data/manifests_ultraquick
```

## 9) 시계열(Temporal) 모델

프레임 독립 분류가 아니라 영상 시퀀스 자체를 학습하려면 temporal 파이프라인 사용:

```bash
python train_temporal.py --config configs/temporal_ultraquick_rgb.yaml
python eval_temporal.py --config configs/temporal_ultraquick_rgb.yaml --split test
python inference_temporal.py \
  --config configs/temporal_ultraquick_rgb.yaml \
  --checkpoint outputs/checkpoints_temporal_ultraquick_rgb/best.pt \
  --video-path /absolute/path/to/input_video.mp4
```

Temporal 추론도 YouTube URL 직접 지원:

```bash
python inference_temporal.py \
  --config configs/temporal_ultraquick_rgb.yaml \
  --checkpoint outputs/checkpoints_temporal_ultraquick_rgb/best.pt \
  --youtube-url "https://www.youtube.com/shorts/VIDEO_ID"
```

Frame + Temporal 앙상블 추론:

```bash
python inference_ensemble.py \
  --config configs/temporal_ultraquick_rgb.yaml \
  --video-path /absolute/path/to/input_video.mp4 \
  --frame-checkpoint outputs/checkpoints_ultraquick/best.pt \
  --temporal-checkpoint outputs/checkpoints_temporal_ultraquick_rgb/best.pt \
  --save-frame-csv
```

보수적 가짜 판정(가짜 놓침 최소화):

```bash
python inference_ensemble.py \
  --config configs/temporal_ultraquick_rgb.yaml \
  --youtube-url "https://www.youtube.com/shorts/VIDEO_ID" \
  --frame-checkpoint outputs/checkpoints_ultraquick/best.pt \
  --temporal-checkpoint outputs/checkpoints_temporal_ultraquick_rgb/best.pt \
  --decision-policy conservative_fake \
  --fake-threshold 0.45
```

Temporal 구조:
- Frame encoder: `EfficientNet/ConvNeXt` (timm backbone)
- Sequence encoder: `BiLSTM`
- Output: video-level class probability (`real/generated/deepfake`)
- `frame_mode: frame_diff`를 켜면 프레임 차분 기반 시계열 학습 가능
- 안정 기본값은 `configs/temporal_ultraquick_rgb.yaml` (frame_mode=rgb)

## 8) 즉시 실행용 Ultraquick 베이스라인

`VoxCeleb2`가 아직 준비되지 않았거나, 빠르게 end-to-end 데모를 돌릴 때 사용:

```bash
bash scripts/run_ultraquick_baseline.sh
```

직접 실행:

```bash
python prepare_data.py --config configs/baseline_ultraquick.yaml
python train.py --config configs/baseline_ultraquick.yaml
python eval.py --config configs/baseline_ultraquick.yaml --split test --checkpoint outputs/checkpoints_ultraquick/best.pt
```

샘플 추론:

```bash
python inference.py \
  --config configs/baseline_ultraquick.yaml \
  --checkpoint outputs/checkpoints_ultraquick/best.pt \
  --video-path "UCF101-Action Recognition/test/BalanceBeam/v_BalanceBeam_g24_c03.avi" \
  --aggregation mean \
  --save-frame-csv
```

## 10) 멀티큐 강건 모드 (RGB + FFT + Temporal)

1) RGB 프레임 모델 학습

```bash
python train.py --config configs/baseline_ultraquick.yaml
```

2) FFT 프레임 모델 학습

```bash
python train.py --config configs/baseline_ultraquick_fft.yaml
```

3) Temporal(rgb 기본) 모델 학습

```bash
python train_temporal.py --config configs/temporal_ultraquick_rgb.yaml
```

4) 3-컴포넌트 앙상블 추론

```bash
python inference_ensemble.py \
  --config configs/temporal_ultraquick_rgb.yaml \
  --youtube-url "https://www.youtube.com/shorts/VIDEO_ID" \
  --frame-checkpoint outputs/checkpoints_ultraquick/best.pt \
  --freq-checkpoint outputs/checkpoints_ultraquick_fft/best.pt \
  --temporal-checkpoint outputs/checkpoints_temporal_ultraquick_rgb/best.pt \
  --frame-input-representation rgb \
  --freq-input-representation fft \
  --temporal-input-representation rgb \
  --temporal-frame-mode rgb \
  --frame-weight 0.4 \
  --freq-weight 0.2 \
  --temporal-weight 0.4 \
  --decision-policy argmax
```

입력 표현 옵션:
- `rgb`: 일반 RGB
- `fft`: log-FFT magnitude
- `rgb_fft`: RGB 2채널 + FFT 1채널 혼합

가짜 놓침 최소화 모드(정밀도보다 재현율 우선):

```bash
python inference_ensemble.py \
  --config configs/temporal_ultraquick_rgb.yaml \
  --youtube-url "https://www.youtube.com/shorts/VIDEO_ID" \
  --frame-checkpoint outputs/checkpoints_ultraquick/best.pt \
  --freq-checkpoint outputs/checkpoints_ultraquick_fft/best.pt \
  --temporal-checkpoint outputs/checkpoints_temporal_ultraquick_rgb/best.pt \
  --decision-policy conservative_fake \
  --fake-threshold 0.45
```

## 11) YouTube Shorts URL 자동 수집

YouTube Shorts 데이터 수집용 URL CSV를 자동 생성:

```bash
python scripts/collect_shorts_urls.py \
  --query "ai cooking dog" \
  --query "home cooking recipe" \
  --max-per-query 200 \
  --suggested-label generated \
  --output-csv data/youtube/shorts_urls.csv
```

seed query 파일 사용(추천):

```bash
# generated 후보 수집
python scripts/collect_shorts_urls.py \
  --queries-file scripts/seeds/shorts_queries_generated.txt \
  --max-per-query 200 \
  --suggested-label generated \
  --output-csv data/youtube/shorts_urls.csv

# real 후보 수집 (같은 CSV에 append)
python scripts/collect_shorts_urls.py \
  --queries-file scripts/seeds/shorts_queries_real.txt \
  --max-per-query 200 \
  --suggested-label real \
  --output-csv data/youtube/shorts_urls.csv
```

채널 기반 수집:

```bash
python scripts/collect_shorts_urls.py \
  --channels-file scripts/seeds/shorts_channels_template.txt \
  --max-per-channel 300 \
  --suggested-label real \
  --output-csv data/youtube/shorts_urls.csv
```

AI 생성 채널 seed 바로 사용:

```bash
python scripts/collect_shorts_urls.py \
  --channels-file scripts/seeds/shorts_channels_generated.txt \
  --max-per-channel 300 \
  --suggested-label generated \
  --output-csv data/youtube/shorts_urls.csv
```

출력 CSV 컬럼:
- `video_id`, `shorts_url`, `title`, `uploader`, `duration_sec`, `source_type`, `source_value`, `suggested_label`

## 12) YouTube Shorts 실험용 다운로드/학습

수집한 URL CSV를 실제 학습용 raw video 폴더로 다운로드:

```bash
python scripts/download_shorts_dataset.py \
  --csv data/youtube/shorts_urls_channels_real.csv \
  --csv data/youtube/shorts_urls_channels_generated.csv \
  --output-root data/raw/youtube_shorts \
  --limit-per-label 200 \
  --report-path outputs/inference/shorts_download_report.json
```

Shorts 실험용 config:
- `configs/baseline_ultraquick_shortsmix.yaml`
- `configs/baseline_ultraquick_shortsmix_fft.yaml`
- `configs/temporal_ultraquick_shortsmix.yaml`

이 config는:
- `real`: YouTube Shorts real 채널
- `generated`: YouTube Shorts generated 채널
- `deepfake`: FaceForensics++ / CelebDF

그리고 split 기준을 `source_id` 우선으로 둬서 같은 YouTube 채널이 train/val/test에 동시에 들어가지 않게 설정함.

전체 pilot 실행:

```bash
bash scripts/run_shortsmix_experiment.sh
```

Shorts가 `real/generated`만 있는 경우에는 deepfake를 섞지 말고 전용 binary 트랙으로 돌리는 편이 더 정확함:

- `configs/baseline_binary_shorts_pilot.yaml`
- `configs/baseline_binary_shorts.yaml`
- `scripts/run_shorts_binary_experiment.sh`

Pilot 전처리/학습/평가:

```bash
python prepare_data.py --config configs/baseline_binary_shorts_pilot.yaml
python train.py --config configs/baseline_binary_shorts_pilot.yaml
python eval.py \
  --config configs/baseline_binary_shorts_pilot.yaml \
  --split val \
  --checkpoint outputs/checkpoints_shorts_binary_pilot/best.pt
```

CSV 다운로드부터 binary 실험까지 한 번에:

```bash
bash scripts/run_shorts_binary_experiment.sh
```

현재 best pilot(`RGB + Temporal`)을 바로 YouTube/로컬 영상 판별에 연결:

```bash
bash scripts/run_shorts_binary_inference.sh \
  --youtube-url "https://www.youtube.com/shorts/VIDEO_ID" \
  --save-frame-csv
```

로컬 영상:

```bash
bash scripts/run_shorts_binary_inference.sh \
  --video-path /absolute/path/to/input_video.mp4 \
  --save-frame-csv
```

## 13) 현재 추천 운영 모델 (All-Video)

현재 보유한 전체 동영상/이미지 데이터셋을 통합한 `all-video` 실험에서 가장 강한 운영 기준은:

- `text-mask all-video frame`
- config: `configs/baseline_binary_allvideo_textmask.yaml`
- checkpoint: `outputs/checkpoints_allvideo_binary_textmask/best.pt`

바로 추론:

```bash
bash scripts/run_allvideo_inference.sh \
  --youtube-url "https://www.youtube.com/shorts/VIDEO_ID" \
  --aggregation confidence_mean \
  --save-frame-csv
```

```bash
bash scripts/run_allvideo_inference.sh \
  --video-path /absolute/path/to/input_video.mp4 \
  --aggregation confidence_mean \
  --save-frame-csv
```

현재 `video-level test` 기준 성능:

| 모델 | 조건 | Accuracy | F1 | AUC |
|---|---|---:|---:|---:|
| Frame | original all-video | 0.9792 | 0.9791 | 0.9984 |
| Frame | text-mask all-video | **0.9834** | **0.9833** | 0.9978 |
| Temporal | original all-video | 0.9345 | 0.9344 | 0.9870 |
| Temporal | text-mask all-video | 0.9210 | 0.9209 | 0.9810 |
| FFT | original all-video | 0.9387 | 0.9382 | 0.9897 |
| FFT | text-mask all-video | 0.9480 | 0.9476 | 0.9874 |
| Multicue | original RGB + Temporal + FFT | 0.9678 | 0.9676 | 0.9960 |
| Multicue | text-mask Frame + original Temporal + text-mask FFT | 0.9678 | 0.9676 | 0.9960 |

해석:

- `text-mask`는 `Frame`, `FFT`에는 이득
- `text-mask`는 `Temporal`에는 손해
- 현재 기준 최종 운영 1순위는 `text-mask all-video frame`

## 7) TODO (의도적으로 남긴 확장 포인트)

- RetinaFace detector 실제 구현 (`mtcnn_detector.py` placeholder)
- temporal aggregation (LSTM/Transformer/TCN) 확장
- hard example mining / class imbalance 대응
- dataset-specific 메타데이터 정밀 파싱 강화
- video-level calibration 및 threshold tuning
