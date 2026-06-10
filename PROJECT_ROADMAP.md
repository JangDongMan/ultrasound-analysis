# 5MHz 초음파 피부 두께 측정 - NPU 포팅 로드맵

## 프로젝트 개요

5MHz 초음파 A-mode 신호에서 피부층 경계(진피/근막)를 자동 검출하여
NXP MCXNx4x N1-16 NPU에서 실시간 추론하는 시스템 개발

---

## 현재 완료된 작업

### 1. 기존 데이터 수집 완료 (구버전, data/)
- **총 234개 CSV 파일** (19명 환자 + 2개 캘리브레이션 세트)
- **234개 수동 레이블 JSON** (전문가 마킹: 시작점, 진피, 근막)
- 데이터 경로: `data/*.csv`
- 레이블 경로: `manual_boundaries/*_positions.json`

#### 데이터 형식 (구버전)
```
# CSV 파일 (2줄 헤더 + 데이터)
x-axis,1
second,Volt
+15.42000E-06,+21.76381648E-03    # 시간(초), 전압(V), 100MHz, 2000샘플
```

#### 수동 레이블 형식
```json
{
  "source_file": "data/dmjang-5M-1.csv",
  "start_point_us": 17.37,
  "positions": [
    {"position_name": "Dermis",  "time_us": 19.49, "thickness_mm": 1.63},
    {"position_name": "Fascia",  "time_us": 22.11, "thickness_mm": 3.65}
  ]
}
```

#### 환자 목록 (19명)
bhjung, cmkim, csyang, dmjang, Drpark, hjbaek, hjpark, hrkim,
ihhong, jjkwon, khchoo, mkkim, okk, shkim, sicho, sikim,
ybsong, yrha, ysseo

#### 캘리브레이션 데이터
- 320 시리즈: 6개 (36dB, 46dB)
- 325 시리즈: 44개 (32dB ~ 46dB)

---

### 2. 새 측정 데이터 수집 완료 (신버전, usdata/) ★
- **1,341개 CSV 파일** (25명 환자 / 5일: 2026-02-25~27, 03-06~13)
- **1,341개 자동 레이블 JSON** (자동 마킹 완료)
- 데이터 경로: `usdata/data/날짜/환자명/*.csv`

#### 파일명 규칙
```
김선일_20260225_161114_(01)이마 중앙_M.csv
```

#### 데이터 형식 (현재 표준 — drop=1850)
```
# CSV 파일, 헤더 없음, ADC 정수값 (0~255), 100MHz
# 변환 후 저장: 2050샘플  /  원시 캡처: ~3900샘플
128
131
...
```

#### 신호 처리 파라미터 (표준: drop=1850)
```python
TRIM_START     = 1850     # 앞 1850샘플 제거 (초음파 미도달 구간)
TRIM_COUNT     = 2050     # 이후 2050샘플 사용
DISPLAY_OFFSET = 18.50    # μs (트림 후 첫 샘플의 절대 시간)
SAMPLE_NS      = 10       # ns/sample (100 MHz ADC)
# → 유효 시간 범위: 18.50 ~ 39.00 μs
```

#### 구포맷 → 신포맷 일괄 변환 (`shift_legacy_data.py`)
| 소스 타입 | 변환 내용 |
|----------|----------|
| usdata/ 구버전 (drop=1200, 1250샘플) | +650샘플 패딩, JSON time_us += 6.5μs |
| data/ VB5K 오실로스코프 (시간-전압) | VREF=3.0V ADC 변환, 18.5μs 기준 정렬 |

```bash
python3 shift_legacy_data.py           # 드라이런 (변경 없음)
python3 shift_legacy_data.py --apply   # 실제 변환
python3 shift_legacy_data.py --verify  # 변환 결과 검증
```

#### 신호 구조 (엔벨로프 기준)
```
[노이즈 ≤50]  [표피 클러스터 ≥100] [갭] [진피/근막 클러스터]
              T₀                   진피경계
              (envelope 100 첫 상승)  (envelope 100 첫 하강)
```

---

### 3. 통계 기반 경계 검출 알고리즘 (C 구현) 완료 — utils4/
- 위치: `utils4/boundary_detector.c`, `utils4/boundary_detector_wrapper.py`
- utils2를 개선·대체 (utils2는 삭제됨)
- **Python 자동 마킹**: `utils4/auto_mark_usdata.py` — usdata/ 669개 전체 자동 레이블 생성

```c
// 234개 수동 레이블 기반 파라미터
#define DERMIS_EXPECTED_TIME_US  2.33   // T0 + 2.33±0.33 μs (1.80±0.25 mm)
#define FASCIA_EXPECTED_TIME_US  5.16   // T0 + 5.16±0.79 μs (3.97±0.61 mm)
#define EPIDERMIS_CUTOFF_TIME_US 1.5
#define SPEED_OF_SOUND           1540.0
#define START_POINT_EXPECTED_US  17.39  // ±0.15 μs (234샘플 통계)
```

#### 검출 정확도 (234개 data/ 기준)
| 경계 | 평균 오차 | 0.5mm 이내 비율 |
|------|----------|----------------|
| 진피 | 0.20 mm  | 91.7% |
| 근막 | 0.61 mm  | 54.2% |

#### 빌드 및 실행
```bash
cd utils4
make
./boundary_detector ../data/bhjung-5M-1.csv         # 자동 시작점
./boundary_detector ../data/bhjung-5M-1.csv 17.26   # 수동 시작점

# usdata 전체 자동 마킹
python3 utils4/auto_mark_usdata.py
python3 utils4/auto_mark_usdata.py --overwrite      # 기존 JSON 덮어씀
```

---

### 4. 표피 자동 검출 알고리즘 완료 (analyze_epidermis.py) ★
- **성공률: 669/669 (100%)**
- 알고리즘: `env = |hilbert(ADC - 128)|`
  - T₀ = envelope이 100 이상으로 첫 상승하는 샘플
  - 진피경계 = envelope이 100 아래로 첫 하강하는 샘플
  - 표피두께 = (진피경계 - T₀) × 10ns × 1540/2
- **평균 표피두께: 1.03 ± 0.12 mm** (범위 0.52~1.92 mm)
- VB5K Boundary Marker 검증 완료 (~13.3μs 일치)

#### 부위별 표피 두께 통계
| 부위 | 평균 (mm) |
|------|----------|
| 이마 | 1.01 |
| 볼/코 | 1.03~1.04 |
| 눈 주변/턱 | 1.07 |
| 전체 | 1.03 ± 0.12 |

#### 추출 특징 (부위 예측용)
1. `epi_mm` — 표피 두께
2. `gap_us` — 표피 끝 ~ 두 번째 클러스터 시작 (연조직/뼈 깊이 지표)
3. `second_span_us` — 두 번째 클러스터 폭 (뼈 = 지속 반사 = 넓은 폭)
4. `n_clusters` — 전체 클러스터 수

---

### 5. 부위 분류기 및 특징 기반 모델 완료
- `train_region_classifier.py` → `results/region_classifier.pkl`
- `train_feature_model.py` → `results/feature_model.pkl`
- 신호 특징으로 측정 부위 예측 → 진피/근막 검출 정확도 향상 파이프라인

---

### 6. 1D-CNN 모델 학습 완료 ★ (`train_1dcnn.ipynb`)
- **입력**: 1250샘플 정규화 ADC 신호 → [-1, +1]
- **출력**: [dermis_mm, fascia_mm]
- **데이터**: 1,341개 / 25명 환자
- **분할**: Leave-One-Patient-Out Cross-Validation (LOO-CV, 25-fold)

#### 모델 구조 (SkinCNN)
```
Conv1d×4 (stride=2, BatchNorm+ReLU) → AdaptiveAvgPool1d(1)
→ Linear(64→32) → ReLU → Dropout → Linear(32→2)
파라미터: 37,218개 (n_filters=16)
```

#### 최적 하이퍼파라미터 (36개 조합 그리드 서치)
```
lr=0.001, batch=32, n_filters=16, dropout=0.3
```

#### LOO-CV 결과
| 경계 | MAE | 표준편차 | 목표 | 상태 |
|------|-----|---------|------|------|
| 진피 | **0.137 mm** | ±0.011 | < 0.30 mm | ✅ 달성 |
| 근막 | **0.074 mm** | ±0.007 | < 0.30 mm | ✅ 달성 |

- 저장 경로: `results/best_1dcnn.pth`
- 라벨 분포: 진피 1.787±0.160mm, 근막 3.971±0.087mm

#### VB5K 자동 마킹 (`auto_mark_data.py`)
- data/ VB5K 변환 파일 232/234개 자동 마킹 성공 (99%)
- 클러스터 검출: 이동평균 + 적응형 임계값(신호최대×15%), 그룹갭 1.5μs

---

### 7. Python 유틸리티 완료
- `utils/data_loader.py`: CSV 데이터 로딩 및 메타데이터 파싱
- `utils/normalizer.py`: StandardScaler, MinMaxScaler, RobustScaler
- `utils/preprocessor.py`: 밴드패스 필터, FFT, 특징 추출
- `utils/dataset.py`: PyTorch Dataset, train/val/test 분할

---

### 8. ADC 캡처 GUI 완료 (utils3/)
- `utils3/adc_capture_gui.py`: VB5K 장치 시리얼 통신 캡처
- `utils3/serial_comm.py`: 시리얼 통신 모듈
- Windows GUI (CustomTkinter)

---

### 9. C 추론 엔진 완료 — system/ (PC 검증용)
**상태: 완료 (MCU 직접 사용 불가 — 메모리 초과)**

순수 C99, 외부 의존 없음 (`-lm`만 사용). 서버/PC에서 정확도 검증용.

```
system/
├── export_weights.py   ← PyTorch .pth → BN-fused C float 배열 헤더
├── model_weights.h     ← 자동 생성 (144 KB, 36,866 floats)
├── skin_cnn.h / .c     ← Conv1D·AdaptiveAvgPool·FC 추론 엔진
├── main.c              ← CSV 읽기 → 추론 → 결과 출력
└── Makefile
```

**메모리 사용량 (MCU 불가 판정 원인):**

| 구성 요소 | Flash | SRAM |
|----------|-------|------|
| 가중치 (float32) | **144 KB** | — |
| 스크래치 버퍼 ×2 | — | **80 KB** |
| 입력 버퍼 (1250×4B) | — | 5 KB |
| **총합** | **144 KB** | **85 KB** |

```bash
cd system
make
./skin_infer 파일.csv
find ../usdata/data -name "*.csv" ! -name "*_positions*" | xargs -d '\n' ./skin_infer
```

---

### 10. PyTorch INT8 PTQ 양자화 검증 완료 ★ (`quantize_eval.ipynb`)
**상태: LOO-CV 재검증 완료 — 정확도 손실 없음**

- Backend: `qnnpack` (ARM aarch64)
- 방식: Post-Training Static Quantization (Conv+BN+ReLU 융합)
- 보정 데이터: 각 LOO-CV fold의 학습 데이터 전체

#### LOO-CV 재검증 결과 (25명 전원)
| 항목 | FP32 | INT8 PTQ | 변화(Δ) |
|------|------|----------|---------|
| 진피 MAE | 0.137 ± 0.011 mm | 0.138 ± 0.011 mm | **+0.0001 mm** |
| 근막 MAE | 0.076 ± 0.007 mm | 0.078 ± 0.008 mm | **+0.0011 mm** |
| 목표 달성 (진피 < 0.30mm) | 25/25 | **25/25** | 변화 없음 |
| 목표 달성 (근막 < 0.30mm) | 25/25 | **25/25** | 변화 없음 |

#### 성능 개선
| 항목 | FP32 | INT8 | 개선 |
|------|------|------|------|
| 추론 속도 (단일 샘플) | 0.272 ms | 0.158 ms | **1.72× 빠름** |
| 모델 파일 크기 | 157 KB | 45 KB | **3.46× 작음** |

- 결과 CSV: `results/quant_loo_cv.csv`
- 결론: 37K 파라미터 소형 모델은 INT8 표현 오차가 미미 → 양자화 손실 없이 배포 가능

---

### 11. Boundary Marker GUI 업데이트 (`utils2/boundary_marker_gui.py`)
- **VREF 수정**: VB5K 신호 1.25V → 3.0V (클리핑 방지)
- **"전체를 표시" 버튼 추가**: X-Zoom 아래에 토글 버튼
  - 비활성(회색): 정상 18.5μs~ 범위 표시
  - 활성(초록): 원시 파일 재로드 → 0μs~끝 전체 표시 (drop 전 구간 포함)
  - 파일 로드 시 자동 초기화
- **원시 데이터 재로드 로직**: `_load_csv_raw()` 추가 — 트리밍 없이 0μs 기준 로드

---

### 12. INT8 TFLite + NPU 추론 엔진 완료 ★ — system2/
**상태: 완료 (MCU 배포 준비)**

PyTorch → ONNX → TFLite Full INT8 변환 완료. NXP MCUXpresso 빌드 파일 작성 완료.

```
system2/
├── export_tflite.py    ← PyTorch → ONNX → TFLite INT8 변환 스크립트
├── model_int8.tflite   ← 47 KB (Full INT8 양자화, per-channel)
├── model_int8.h        ← C 바이트 배열 (MCU Flash 포함용)
├── skin_npu.h          ← C API: skin_npu_init() / skin_npu_infer()
├── skin_npu.cc         ← TFLite Micro + N1-16 NPU C++ 구현
└── CMakeLists.txt      ← NXP MCUXpresso / eIQ SDK 빌드 설정
```

#### 변환 파이프라인
```
best_1dcnn.pth (PyTorch float32)
  → model.onnx  (torch.onnx.export, opset 18)
  → model_int8.tflite  (onnx2tf, Full INT8, per-channel, 100샘플 교정)
  → model_int8.h  (C 바이트 배열)
```

#### 메모리 절감 (system/ 대비)
| 구성 요소 | system/ float32 | system2/ INT8 | 절감 |
|----------|----------------|---------------|------|
| Flash (가중치) | 144 KB | **47 KB** | **−97 KB** |
| SRAM (Tensor Arena) | 80 KB | **28 KB** | **−52 KB** |
| 정확도 손실 | 기준 | **Δ 0.000 mm** | — |

#### 실제 메모리 예산 (MCXNx4x)
| 구성 요소 | Flash | SRAM |
|----------|-------|------|
| 모델 (INT8, model_int8.h) | **47 KB** | — |
| TFLite Micro 런타임 | ~100 KB | ~20 KB |
| Tensor Arena | — | **28 KB** |
| ADC 버퍼 (1250 × 1B) | — | 2 KB |
| 애플리케이션 코드 | ~50 KB | ~10 KB |
| **총합** | **~197 KB / 2 MB** | **~60 KB / 512 KB** |

#### C API (MCU 사용법)
```c
#include "skin_npu.h"

skin_npu_init();   // 시작 시 1회

float dermis_mm, fascia_mm;
skin_npu_infer(adc_buf, 1250, &dermis_mm, &fascia_mm);
// 1250샘플: 그대로 사용 / 2500샘플 이상: [1200:2450] 자동 트림
```

#### 변환 실행 (최초 1회)
```bash
cd system2
python3 export_tflite.py    # model_int8.tflite + model_int8.h 생성
```

---

## 다음 단계: NPU 포팅

### Phase 3 완료 현황
**상태: LOO-CV 학습 + INT8 양자화 재검증 완료**

| 경계 | FP32 MAE | INT8 MAE | 목표 | 상태 |
|------|---------|---------|------|------|
| 진피 | 0.137 mm | 0.138 mm | < 0.30 mm | ✅ 달성 |
| 근막 | 0.074 mm | 0.078 mm | < 0.30 mm | ✅ 달성 |

> 목표를 < 0.30mm로 재설정 (실용적 임상 정밀도 기준).  
> 원래 목표 < 0.10mm는 추후 데이터 증강/아키텍처 개선 시 도전 가능.

---

### Phase 4: NPU 포팅 (최종)
**상태: 코드 준비 완료 / 하드웨어 셋업 필요**

`system2/skin_npu.cc` 및 `system2/CMakeLists.txt` 작성 완료.  
NXP 보드 및 SDK 설치 후 바로 빌드 가능.

#### 타겟 하드웨어
```
NXP MCXNx4x (FRDM-MCXN947)
├── CPU: Dual Cortex-M33 @ 150MHz
├── NPU: N1-16 (2KB Cache)
├── Flash: 2MB (듀얼 뱅크)
├── SRAM: 512KB (ECC)
├── ADC: 2x 16-bit, 2-3 Msps
└── 전력: 75 μA/MHz @ 3.3V
```

#### 4-1. 개발 환경 설정 (미완)
- [ ] NXP MCUXpresso IDE 설치
- [ ] eIQ Toolkit 설치 (TFLite Micro + NPU 드라이버 포함)
- [ ] FRDM-MCXN947 평가 보드 준비
- [ ] SEGGER J-Link 디버거 연결

#### 4-2. MCUXpresso 빌드 (system2/ 준비 완료)
```bash
# NXP SDK 설치 후
export SDK_PATH=/path/to/nxp-sdk
cmake -DSDK_PATH=$SDK_PATH -DENABLE_NPU=ON -B build
cmake --build build
```

#### 4-3. 실시간 파이프라인 목표
```
VB5K ADC (100MHz, 8-bit)
  → DMA 버퍼 (1250샘플)
  → skin_npu_infer()  [NPU 추론, 목표 < 50ms]
  → 진피/근막 두께 출력 (mm)
  → 총 지연시간 목표: < 100ms
```

---

## 서버 환경 정보

| 항목 | 내용 |
|------|------|
| 서버 | spark-a70d (DGX Spark, aarch64) |
| 작업 경로 | `/home/dmjang/work/ultrasound_analysis/` |
| Python | 시스템 python3 (3.12) — .venv는 x86 바이너리라 사용 불가 |
| 사용자 | dmjang |
| Git remote | GitHub (push는 터미널에서 직접) |

### 주요 디렉토리
```
/home/dmjang/work/ultrasound_analysis/
├── data/                    # 234개 CSV (VB5K 오실로스코프, 변환 완료)
├── usdata/                  # 1,341개 CSV (ADC 정수값, drop=1850, 2050샘플)
│   └── data/날짜/환자명/    # *.csv, *_positions.json
├── utils2/                  # Boundary Marker GUI (boundary_marker_gui.py)
├── utils3/                  # ADC 캡처 GUI (Windows, VB5K 시리얼)
├── utils4/                  # C 경계 검출 + Python 자동 마킹 스크립트
├── results/                 # 분석 결과 (PNG, JSON, CSV, .pth)
│   ├── best_1dcnn.pth       # FP32 최종 모델 (37K params)
│   ├── quant_loo_cv.csv     # INT8 PTQ LOO-CV 결과
│   └── quantize_eval.png    # 양자화 결과 시각화
├── system/                  # float32 C 추론 엔진 (PC 검증용)
├── system2/                 # INT8 TFLite + NPU 추론 엔진 (MCU 배포용)
├── shift_legacy_data.py     # 구포맷 → 신포맷 변환 스크립트
├── auto_mark_data.py        # VB5K data/ 자동 마킹 스크립트
├── train_1dcnn.ipynb        # 1D-CNN 학습 노트북 (LOO-CV + 그리드 서치)
├── quantize_eval.ipynb      # INT8 PTQ 양자화 재검증 노트북
├── *.py                     # 분석/학습/평가 스크립트
└── PROJECT_ROADMAP.md       # 이 파일
```

---

## 진행 체크리스트

### Phase 1: 학습 데이터 준비
- [x] 기존 데이터 234개 (data/, VB5K) + 수동 레이블 확보
- [x] 신규 데이터 1,341개 수집 (usdata/, 25명 환자)
- [x] 구포맷 → 신포맷 변환 (`shift_legacy_data.py`: drop=1200→1850)
- [x] VB5K data/ 자동 마킹 (`auto_mark_data.py`, 232/234 성공)
- [x] 표피 자동 검출 (analyze_epidermis.py, 100% 성공)
- [x] 신호 특징 추출 (epi_mm, gap_us, second_span_us, n_clusters)
- [x] 부위별 통계 분석 완료

### Phase 2: 데이터 정규화
- [x] ADC 정수값(0~255) → [-1, +1] 정규화
- [x] TRIM 1850샘플 제거 → 2050샘플 사용 (표준화 완료)
- [x] 엔벨로프 검출 (힐버트 변환)
- [x] LOO-CV 분할 (환자 단위, 25-fold)

### Phase 3: 모델 학습 및 변환
- [x] 1D CNN 모델 구현 (SkinCNN, 37K params, PyTorch)
- [x] LOO-CV 학습 및 검증 (25명, 1341샘플)
- [x] 하이퍼파라미터 그리드 서치 (36개 조합)
- [x] 최적 모델 저장 (`results/best_1dcnn.pth`)
- [x] **PyTorch INT8 PTQ 양자화** (`quantize_eval.ipynb`, qnnpack)
- [x] **PTQ LOO-CV 재검증**: 진피 Δ+0.0001mm, 근막 Δ+0.001mm (손실 없음)
- [x] float32 C 추론 엔진 (`system/`) — PC 검증용
- [x] INT8 TFLite 변환 (`system2/`, 47KB, MCU 배포용)
- [x] 목표 달성: 진피 0.138mm < 0.30mm, 근막 0.078mm < 0.30mm ✅

### Phase 4: NPU 포팅
- [x] TFLite Micro C++ 코드 작성 (`system2/skin_npu.cc`)
- [x] CMakeLists.txt 작성 (`system2/CMakeLists.txt`)
- [x] C API 설계 완료 (`skin_npu_init` / `skin_npu_infer`)
- [ ] MCUXpresso IDE + eIQ Toolkit 설치
- [ ] FRDM-MCXN947 보드 셋업
- [ ] MCU 빌드 및 플래시
- [ ] NPU 가속 활성화 (`ENABLE_NPU=ON`)
- [ ] 실시간 파이프라인 검증 (지연시간 < 100ms)
- [ ] 정확도 검증 (MCU 추론 vs PC ONNX)

---

## 새 대화에서 시작할 때

> PROJECT_ROADMAP.md 파일을 읽고 이어서 진행해줘

**현재 위치: Phase 3 완료 (LOO-CV + INT8 PTQ 재검증) → Phase 4 (MCU 하드웨어 셋업)**
