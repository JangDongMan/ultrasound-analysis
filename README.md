# 5MHz 초음파 피부층 두께 측정 — 프로젝트 개요

5MHz A-mode 초음파 신호에서 진피·근막 경계를 자동 검출하고,  
NXP MCXNx4x N1-16 NPU에서 실시간 추론하는 시스템 개발 프로젝트.

---

## 목차

1. [데이터](#1-데이터)
2. [신호 구조](#2-신호-구조)
3. [알고리즘 — 통계 기반 경계 검출](#3-알고리즘--통계-기반-경계-검출)
4. [알고리즘 — 표피 자동 검출](#4-알고리즘--표피-자동-검출)
5. [유틸리티 디렉토리](#5-유틸리티-디렉토리)
6. [분석 스크립트](#6-분석-스크립트)
7. [ML 모델 학습](#7-ml-모델-학습)
8. [추론 시스템 — system/](#8-추론-시스템--system-float32-c)
9. [추론 시스템 — system2/](#9-추론-시스템--system2-int8-tflite--npu)
10. [NPU 포팅 로드맵](#10-npu-포팅-로드맵)
11. [서버 환경](#11-서버-환경)

---

## 1. 데이터

### 구버전 — `data/` (234개)
- 19명 환자 + 캘리브레이션 세트
- 형식: `time(s), voltage(V)` CSV (2줄 헤더), 100MHz, 2000샘플
- 수동 레이블: `manual_boundaries/*_positions.json` (시작점·진피·근막)

```json
{
  "start_point_us": 17.37,
  "positions": [
    {"position_name": "Dermis", "time_us": 19.49, "thickness_mm": 1.63},
    {"position_name": "Fascia", "time_us": 22.11, "thickness_mm": 3.65}
  ]
}
```

### 신버전 — `usdata/` (669개)
- 20명 환자 / 3일 (2026-02-25~27)
- 경로: `usdata/data/날짜/환자명/파일명.csv`
- 파일명: `김선일_20260225_161114_(01)이마 중앙_M.csv`
- 형식: ADC 정수값 (0~255), 헤더 없음, 1250샘플 (저장 전 트림 완료)
- 자동 레이블: 각 CSV 옆 `*_positions.json` (utils4/auto_mark_usdata.py 생성)

---

## 2. 신호 구조

```
ADC 샘플레이트: 100 MHz (10 ns/sample)
유효 구간    : DROP_SAMPLES=1200 제거 후 1250샘플 사용 (12.0 ~ 24.49 μs)

구간별 특성 (envelope = |hilbert(ADC - 128)|):

  [노이즈 ≤50]  [표피 클러스터 ≥100] [갭] [진피/근막 클러스터]
                T₀                  진피경계
                (envelope 100↑)     (envelope 100↓)
```

| 신호 특징 | 설명 |
|----------|------|
| `epi_mm` | 표피 두께 (T₀ ~ 진피경계) |
| `gap_us` | 표피 끝 ~ 두 번째 클러스터 시작 |
| `second_span_us` | 두 번째 클러스터 폭 (뼈 여부 판별) |
| `n_clusters` | 전체 클러스터 수 |

---

## 3. 알고리즘 — 통계 기반 경계 검출

234개 수동 레이블 통계 기반 파라미터:

```
진피(Dermis): T₀ + 2.33 ± 0.33 μs → 1.80 ± 0.25 mm
근막(Fascia): T₀ + 5.16 ± 0.79 μs → 3.97 ± 0.61 mm
T₀ 자동 검출 정확도: ±0.04 μs
```

| 경계 | 평균 오차 | 0.5mm 이내 |
|------|---------|-----------|
| 진피 | 0.20 mm | 91.7% |
| 근막 | 0.61 mm | 54.2% |

구현: `utils4/boundary_detector.c` + Python 래퍼 `utils4/boundary_detector_wrapper.py`

---

## 4. 알고리즘 — 표피 자동 검출

**`analyze_epidermis.py`** — 669/669 파일 100% 성공

```python
env = abs(hilbert(ADC - 128))
T0           = envelope이 100 이상으로 첫 상승하는 샘플
dermis_bound = envelope이 100 아래로 첫 하강하는 샘플
epi_mm       = (dermis_bound - T0) × 10ns × 1540 / 2
```

**부위별 표피 두께 통계 (669건):**

| 부위 | 평균 (mm) |
|------|----------|
| 이마 | 1.01 |
| 볼·코 | 1.03~1.04 |
| 눈 주변·턱 | 1.07 |
| **전체** | **1.03 ± 0.12** (범위 0.52~1.92) |

---

## 5. 유틸리티 디렉토리

### `utils/` — Python 공용 유틸리티
| 파일 | 기능 |
|------|------|
| `data_loader.py` | CSV 로딩·메타데이터 파싱 |
| `normalizer.py` | StandardScaler / MinMax / Robust |
| `preprocessor.py` | 밴드패스 필터·FFT·특징 추출 |
| `dataset.py` | PyTorch Dataset·train/val/test 분할 |

### `utils3/` — ADC 캡처 GUI (Windows)
- `adc_capture_gui.py`: VB5K 장치 시리얼 통신 캡처 (CustomTkinter)
- `serial_comm.py`: 시리얼 통신 모듈

```bash
# Windows에서 실행
python adc_capture_gui.py
```

### `utils4/` — C 경계 검출 + 자동 마킹
- `boundary_detector.c / .h`: 통계 기반 C 구현
- `boundary_detector_wrapper.py`: Python ctypes 래퍼
- `auto_mark_usdata.py`: usdata/ 전체 자동 레이블 생성

```bash
cd utils4
make
./boundary_detector ../usdata/data/260225/장동만/파일.csv

# usdata 전체 자동 레이블
python3 auto_mark_usdata.py
python3 auto_mark_usdata.py --overwrite    # 기존 JSON 덮어씀
python3 auto_mark_usdata.py --dry-run      # 파일 목록만 확인
```

---

## 6. 분석 스크립트

| 스크립트 | 입력 | 출력 |
|---------|------|------|
| `analyze_epidermis.py` | usdata/ | `results/epidermis_analysis.csv` |
| `stats_by_region.py` | usdata/ | `results/stats_by_region.csv/.png` |
| `analyze_by_posnum.py` | usdata/ | `results/posnum_analysis.png` |
| `analyze_features_by_posnum.py` | usdata/ | `results/feature_by_posnum.png` |
| `analyze_t0_epidermis.py` | usdata/ | `results/t0_epidermis_stats.csv` |
| `detect_layers.py` | usdata/ | `results/layer_detection.csv` |
| `export_to_excel.py` | results/ | `results/skin_layer_analysis_summary.xlsx` |
| `evaluate_detection_accuracy.py` | data/ + manual_boundaries/ | 정확도 리포트 |

---

## 7. ML 모델 학습

### 7-1. 1D-CNN (메인 모델) — `train_1dcnn.py` / `train_1dcnn.ipynb`

```
입력 : 1250샘플 정규화 ADC 신호 → [-1, +1]  ((ADC - 128) / 128)
출력 : [dermis_mm, fascia_mm]
분할 : Leave-One-Patient-Out Cross-Validation (LOPO CV, 24명)
프레임워크: PyTorch
```

**모델 구조 (SkinCNN, n_filters=16, 파라미터 37,218개):**

```
Conv1d(1→16,  k=15, s=2) + BN + ReLU → (16, 625)
Conv1d(16→32, k=7,  s=2) + BN + ReLU → (32, 313)
Conv1d(32→64, k=5,  s=2) + BN + ReLU → (64, 157)
Conv1d(64→64, k=5,  s=2) + BN + ReLU → (64, 79)
AdaptiveAvgPool1d(1)                  → (64,)
Linear(64→32) + ReLU
Linear(32→2)                          → [dermis_mm, fascia_mm]
```

**LOPO CV 결과:**

| 경계 | MAE (mm) | 목표 |
|------|---------|------|
| 진피 | **0.135** | < 0.10 |
| 근막 | **0.067** | < 0.10 |

**하이퍼파라미터 그리드 서치 최적값:**

```python
HP = dict(lr=5e-4, batch=16, n_filters=16, dropout=0.2)
```

저장 모델: `results/best_1dcnn.pth`

```bash
python3 train_1dcnn.py                   # 기본 학습
python3 train_1dcnn.py --search          # 그리드 서치
python3 train_1dcnn.py --search --quick  # 빠른 탐색
```

### 7-2. 특징 기반 모델 — `train_feature_model.py`

- Random Forest 기반
- 입력: epi_mm, gap_us, second_span_us, n_clusters 등 신호 특징
- 저장: `results/feature_model.pkl`

### 7-3. 부위 분류기 — `train_region_classifier.py`

- 신호 특징으로 측정 부위(이마·볼·코 등) 예측
- 저장: `results/region_classifier.pkl`

---

## 8. 추론 시스템 — `system/` (float32 C)

순수 C99, 외부 라이브러리 의존 없음 (서버/PC 검증용).

```
system/
├── export_weights.py   ← PyTorch .pth → C float 배열 헤더 (BN fused)
├── model_weights.h     ← 자동 생성 (144 KB, 36,866 floats)
├── skin_cnn.h / .c     ← Conv1D·Pool·FC 추론 엔진
├── main.c              ← CSV 읽기 → 추론 → 결과 출력
└── Makefile
```

**메모리 사용량:**

| | Flash | SRAM |
|-|-------|------|
| 가중치 (`model_weights.h`) | **144 KB** | — |
| 스크래치 버퍼 (2 × 40 KB) | — | **80 KB** |

**빌드 및 실행:**

```bash
cd system
make                         # 가중치 내보내기 + 빌드
./skin_infer 파일.csv         # 단일 파일 추론

# 배치 (파일명에 공백 있을 경우)
find ../usdata/data -name "*.csv" ! -name "*_positions*" | xargs -d '\n' ./skin_infer
```

**출력 예시:**

```
파일                                              진피(mm)  근막(mm)
-----------------------------------------------------------------------
김희락_20260226_110618_(04)이마 끝 관자놀이_M.csv     1.886     4.129
```

> **메모 :** float32 C 버전은 MCU 메모리 부족으로 직접 사용 불가.  
> MCU 배포는 `system2/` 사용.

---

## 9. 추론 시스템 — `system2/` (INT8 TFLite + NPU)

NXP MCXNx4x N1-16 NPU 타겟. TFLite Micro 기반.

```
system2/
├── export_tflite.py    ← PyTorch → ONNX → TFLite INT8 변환
├── model_int8.tflite   ← 47 KB (full INT8 양자화)
├── model_int8.h        ← C 바이트 배열 (MCU Flash 포함용)
├── skin_npu.h          ← C API 헤더
├── skin_npu.cc         ← TFLite Micro + N1-16 NPU C++ 구현
└── CMakeLists.txt      ← NXP MCUXpresso 빌드 설정
```

**메모리 절감 (vs system/):**

| | system/ float32 | system2/ INT8 | 절감 |
|-|----------------|---------------|------|
| Flash | 144 KB | **47 KB** | **−97 KB** |
| SRAM (Tensor Arena) | 80 KB | **28 KB** | **−52 KB** |
| 정확도 손실 | 기준 | Δ 0.000 mm | — |

**변환 파이프라인:**

```
PyTorch best_1dcnn.pth
  → ONNX (torch.onnx.export, opset 18)
  → TFLite Full INT8 (onnx2tf, per-channel 양자화)
  → model_int8.h (C 바이트 배열)
```

**변환 실행 (최초 1회):**

```bash
cd system2
python3 export_tflite.py
```

**MCU C API:**

```c
#include "skin_npu.h"

// 초기화 (시작 시 1회)
skin_npu_init();

// 추론 (매 측정마다)
float dermis_mm, fascia_mm;
skin_npu_infer(adc_buf, 1250, &dermis_mm, &fascia_mm);
// adc_buf: uint8[1250] — VB5K ADC 데이터 (TRIM 완료된 것)
// adc_buf: uint8[2500+] — 원시 2500샘플 (TRIM_START=1200 자동 적용)
```

**MCUXpresso 빌드 (NXP SDK 필요):**

```bash
# SDK_PATH 환경변수 설정 후
cmake -DSDK_PATH=/path/to/nxp-sdk -DENABLE_NPU=ON -B build
cmake --build build
```

---

## 10. NPU 포팅 로드맵

| Phase | 내용 | 상태 |
|-------|------|------|
| 1. 데이터 수집 | 669개 usdata/ + 234개 구버전 | ✅ 완료 |
| 2. 전처리 | ADC → [-1,+1] 정규화, LOPO 분할 | ✅ 완료 |
| 3-a. 모델 학습 | PyTorch 1D-CNN, LOPO CV | ✅ 완료 |
| 3-b. INT8 양자화 | ONNX → TFLite Full INT8 | ✅ 완료 |
| 3-c. 진피 MAE 개선 | 현재 0.135mm → 목표 0.10mm | 🔲 진행 중 |
| 4-a. MCUXpresso 셋업 | FRDM-MCXN947 + eIQ Toolkit | 🔲 미시작 |
| 4-b. TFLite Micro 통합 | skin_npu.cc MCU 빌드 | 🔲 미시작 |
| 4-c. NPU 가속 | N1-16 NPU 위임, 실시간 검증 | 🔲 미시작 |

**현재 위치: Phase 3-c — 진피 MAE 0.10mm 달성 → MCU 포팅**

---

## 11. 서버 환경

| 항목 | 내용 |
|------|------|
| 서버 | spark-a70d (DGX Spark, aarch64) |
| OS | Ubuntu, Linux 6.14.0-1015-nvidia |
| Python | 시스템 python3 (3.12) — `.venv`는 x86이라 사용 불가 |
| 작업 경로 | `/home/dmjang/work/ultrasound_analysis/` |
| Git | master 브랜치, push는 터미널에서 직접 |

**주요 디렉토리:**

```
ultrasound_analysis/
├── data/                    # 234개 구버전 CSV (time, voltage)
├── manual_boundaries/       # 234개 수동 레이블 JSON
├── usdata/                  # 669개 신버전 CSV (ADC 정수값)
│   └── data/날짜/환자명/    # *.csv + *_positions.json
├── utils/                   # Python 유틸리티
├── utils3/                  # ADC 캡처 GUI (Windows, VB5K)
├── utils4/                  # C 경계 검출 + 자동 마킹
├── results/                 # 분석 결과, 학습 모델
│   ├── best_1dcnn.pth       # 학습된 PyTorch 모델
│   ├── feature_model.pkl    # Random Forest 모델
│   ├── region_classifier.pkl
│   └── improved/            # 환자별 레이어 분석 PNG (24개)
├── system/                  # float32 C 추론 (PC 검증용)
├── system2/                 # INT8 TFLite + NPU (MCU 배포용)
├── train_1dcnn.ipynb        # Jupyter 학습 노트북
├── PROJECT_ROADMAP.md       # 상세 로드맵
└── README.md                # 이 파일
```
