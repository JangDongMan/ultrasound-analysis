# 5MHz 초음파 피부 두께 측정 - NPU 포팅 로드맵

## 프로젝트 개요

5MHz 초음파 A-mode 신호에서 피부층 경계(진피/근막)를 자동 검출하여
NXP MCXNx4x N1-16 NPU에서 실시간 추론하는 시스템 개발

---

## 현재 완료된 작업

### 1. 데이터 수집 완료
- **총 234개 CSV 파일** (19명 환자 + 2개 캘리브레이션 세트)
- **234개 수동 레이블 JSON** (전문가 마킹: 시작점, 진피, 근막)
- 데이터 경로: `data/*.csv`
- 레이블 경로: `manual_boundaries/*_positions.json`

#### 데이터 형식
```
# CSV 파일 (2줄 헤더 + 데이터)
x-axis,1
second,Volt
+15.42000E-06,+21.76381648E-03    # 시간(초), 전압(V)
+15.43000E-06,+6.68843975E-03
...
```

#### 수동 레이블 형식
```json
{
  "source_file": "data/dmjang-5M-1.csv",
  "start_point_us": 17.37,          // 피부 표면 시작점 (μs)
  "speed_of_sound": 1540,            // 음속 (m/s)
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

### 2. 통계 기반 경계 검출 알고리즘 (C 구현) 완료
- 위치: `utils2/boundary_detector.c`
- **234개 수동 레이블 분석 결과 기반 파라미터:**

```c
// 진피 (Dermis): T0 + 2.33±0.33 μs (1.80±0.25 mm)
#define DERMIS_EXPECTED_TIME_US  2.33
#define DERMIS_TIME_STD_US       0.33

// 근막 (Fascia): T0 + 5.16±0.79 μs (3.97±0.61 mm)
#define FASCIA_EXPECTED_TIME_US  5.16
#define FASCIA_TIME_STD_US       0.79

// 기타 파라미터
#define THICKNESS_GAP_MM         2.18   // 진피-근막 거리
#define EPIDERMIS_CUTOFF_TIME_US 1.5    // 표피 제외 시간
#define SPEED_OF_SOUND           1540.0 // 음속 (m/s)
#define MAX_DISTANCE_MM          6.0    // 분석 범위

// 시작점 자동 검출 (234개 샘플 통계)
#define START_POINT_EXPECTED_US      17.39
#define START_POINT_STD_US           0.15
#define START_POINT_SEARCH_WINDOW_US 1.0
```

#### 검출 알고리즘 흐름
1. **시작점 자동 검출**: 16.39~18.39 μs 범위에서 최대 기울기의 30% 임계값
2. **표피 제외**: 시작점 이후 1.5 μs 이전 영역 제외
3. **진피 검출**: 2차 미분 변곡점 기반 (2.0~2.7 μs)
4. **근막 검출**: 피크 검출 + 통계 기반 예상값 (4.4~5.9 μs)

#### 현재 정확도
- 시작점 자동 검출: ±0.04 μs (수동 레이블 대비)
- 진피 평균 오차: 0.20 mm (91.7%가 0.5mm 이내)
- 근막 평균 오차: 0.61 mm (54.2%가 0.5mm 이내)

#### 빌드 및 실행
```bash
cd utils2
make clean && make
./boundary_detector ../data/bhjung-5M-1.csv          # 시작점 자동 검출
./boundary_detector ../data/bhjung-5M-1.csv 17.26    # 시작점 수동 지정
```

### 3. Python 유틸리티 완료
- `utils/data_loader.py`: CSV 데이터 로딩 및 메타데이터 파싱
- `utils/normalizer.py`: StandardScaler, MinMaxScaler, RobustScaler
- `utils/preprocessor.py`: 밴드패스 필터, FFT, 특징 추출
- `utils/dataset.py`: PyTorch Dataset, train/val/test 분할

### 4. ADC 캡처 GUI 완료
- `utils3/adc_capture_gui.py`: VB5K 장치에서 시리얼 통신으로 데이터 캡처
- Windows GUI (CustomTkinter)

---

## 다음 단계: NPU 포팅 로드맵

### Phase 1: 학습 데이터 준비
**상태: 미시작**

#### 1-1. 데이터 정리 및 검증
```python
# 이미 완료된 항목:
# - 234개 CSV 파일 (data/)
# - 234개 수동 레이블 JSON (manual_boundaries/)

# 해야 할 작업:
# - 이상치 데이터 필터링
# - 캘리브레이션 데이터(320, 325)와 환자 데이터 분리
# - 데이터 품질 검증 (SNR, 신호 무결성)
```

#### 1-2. 학습용 입출력 데이터 생성
```python
# 입력: A-scan 신호 (시작점 이후 512 또는 1024 샘플)
# 출력: 진피 깊이(mm), 근막 깊이(mm)

# 데이터 구조:
# X: (N, 512, 1) - 정규화된 신호
# y: (N, 2) - [dermis_mm, fascia_mm]
```

#### 1-3. 데이터 증강
- 노이즈 추가 (가우시안)
- 시간축 미세 이동
- 진폭 스케일링
- 목표: 최소 1000개 이상의 학습 샘플

### Phase 2: 데이터 정규화
**상태: 미시작**

#### 2-1. 전처리 파이프라인
```python
def preprocess_for_npu(raw_signal, start_us):
    # 1. 시작점 이후 데이터 추출
    start_idx = find_start_index(time_us, start_us)
    signal = voltage[start_idx:]

    # 2. DC offset 제거
    signal = signal - np.mean(signal)

    # 3. 밴드패스 필터 (2-8 MHz)
    signal = bandpass_filter(signal, fs=sample_rate, low=2e6, high=8e6)

    # 4. 엔벨로프 검출 (힐버트 변환)
    envelope = np.abs(hilbert(signal))

    # 5. 고정 길이 (512 샘플)
    envelope = envelope[:512]

    # 6. 정규화 (0~1)
    envelope = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope))

    return envelope
```

#### 2-2. 정규화 방법 선택
| 방법 | 설명 | NPU 적합성 |
|------|------|------------|
| Min-Max (0~1) | 범위 고정 | INT8 양자화에 최적 |
| Z-score | 평균0, 분산1 | 음수값 처리 필요 |
| Abs-Max | 최대값 기준 | 간단하지만 이상치 민감 |

**권장: Min-Max (0~1)** → INT8 양자화 시 0~255 매핑 용이

### Phase 3: 모델 학습
**상태: 미시작**

#### 3-1. 1D CNN 모델 (1순위 추천)
```python
import tensorflow as tf

def create_1d_cnn_model(input_shape=(512, 1)):
    model = tf.keras.Sequential([
        # Conv Block 1
        tf.keras.layers.Conv1D(16, 7, strides=2, activation='relu',
                               input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(2),

        # Conv Block 2
        tf.keras.layers.Conv1D(32, 5, strides=2, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),

        # Conv Block 3
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),

        # Regression Head
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2)  # [dermis_mm, fascia_mm]
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 예상 사양:
# - 파라미터: ~15K-30K
# - 모델 크기: ~50KB (INT8)
# - 추론 시간: < 50ms @ 150MHz
```

#### 3-2. 학습 설정
```python
model = create_1d_cnn_model()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)
```

#### 3-3. INT8 양자화
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
# 저장: ultrasound_thickness_int8.tflite
```

#### 3-4. 목표 성능
- MAE < 0.1 mm (진피, 근막 각각)
- 양자화 정확도 손실 < 1%
- 모델 크기 < 60KB

### Phase 4: NPU 포팅 (최종, 최고 난이도)
**상태: 미시작**

#### 타겟 하드웨어
```
NXP MCXNx4x
├── CPU: Dual Cortex-M33 @ 150MHz
├── NPU: N1-16 (2KB Cache)
├── Flash: 2MB (듀얼 뱅크)
├── SRAM: 512KB (ECC)
├── ADC: 2x 16-bit, 2-3 Msps
└── 전력: 75 μA/MHz @ 3.3V
```

#### 4-1. 개발 환경 설정
- NXP MCUXpresso IDE 설치
- eIQ Toolkit 설치 (TFLite Micro 포함)
- FRDM-MCXN947 평가 보드 준비
- SEGGER J-Link 디버거

#### 4-2. TFLite Micro 통합
```c
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// 모델 데이터 (Flash에 저장)
extern const unsigned char ultrasound_model[];

// Tensor Arena (SRAM에 할당)
constexpr int kTensorArenaSize = 30 * 1024;  // 30KB
uint8_t tensor_arena[kTensorArenaSize];

void setup_inference() {
    const tflite::Model* model = tflite::GetModel(ultrasound_model);

    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddQuantize();

    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter.AllocateTensors();
}

float run_inference(int16_t* signal, int length) {
    TfLiteTensor* input = interpreter->input(0);

    // 전처리 + 입력 설정
    for (int i = 0; i < 512; i++) {
        input->data.int8[i] = (int8_t)((signal[i] - 128) / 2);
    }

    interpreter->Invoke();

    TfLiteTensor* output = interpreter->output(0);
    float thickness_mm = output->data.f[0];
    return thickness_mm;
}
```

#### 4-3. NPU 가속 활성화
- eIQ Toolkit에서 N1-16 NPU 타겟으로 모델 변환
- Conv1D, Dense 레이어 NPU 오프로드
- CPU/NPU 하이브리드 실행 프로파일링

#### 4-4. 실시간 파이프라인
```
ADC (2 Msps, 16-bit)
  → DMA 버퍼 (512 샘플)
  → 전처리 (CPU: DC제거, 필터링, 정규화)
  → NPU 추론 (< 50ms)
  → 결과 출력 (진피/근막 깊이 mm)
  → 총 지연시간 목표: < 100ms
```

#### 4-5. 메모리 예산
| 구성 요소 | Flash | SRAM |
|----------|-------|------|
| 모델 파라미터 | 30KB | - |
| TFLite Micro 런타임 | 100KB | 20KB |
| Tensor Arena | - | 30KB |
| 입출력 버퍼 | - | 2KB |
| 애플리케이션 코드 | 50KB | 10KB |
| **총합** | **~180KB / 2MB** | **~62KB / 512KB** |

---

## 서버 환경 정보

| 항목 | 내용 |
|------|------|
| 서버 | spark-a70d (DGX Spark) |
| 작업 경로 | `/home/dmjang/work/ultrasound_analysis/` |
| Python | 3.10 (.venv 가상환경) |
| 사용자 | dmjang |
| Git remote | https://github.com/JangDongMan/ultrasound-analysis.git |

### 주요 디렉토리
```
/home/dmjang/work/ultrasound_analysis/
├── data/                    # 234개 CSV 데이터 파일
├── manual_boundaries/       # 234개 수동 레이블 JSON
├── utils/                   # Python 유틸리티 (로더, 정규화, 전처리, 데이터셋)
├── utils2/                  # C 경계 검출 알고리즘 (boundary_detector)
├── utils3/                  # ADC 캡처 GUI (Windows용)
├── results/                 # 분석 결과 (PNG, JSON, Excel)
├── saved_models/            # 학습된 모델 (.keras)
├── label_org.xlsx           # 원본 레이블 (Excel)
├── training_dataset.npz     # 전처리된 학습 데이터 (7.3MB)
└── PROJECT_ROADMAP.md       # 이 파일
```

---

## 진행 체크리스트

### Phase 1: 학습 데이터 준비
- [ ] 이상치 데이터 필터링
- [ ] 캘리브레이션/환자 데이터 분리
- [ ] 시작점 이후 512 샘플 추출
- [ ] 레이블 매칭 (진피mm, 근막mm)
- [ ] 데이터 증강 (노이즈, 시프트, 스케일링)
- [ ] train/val/test 분할 (환자 단위)

### Phase 2: 데이터 정규화
- [ ] DC offset 제거
- [ ] 밴드패스 필터 적용
- [ ] 엔벨로프 검출 (힐버트 변환)
- [ ] Min-Max 정규화 (0~1)
- [ ] 정규화 파라미터 저장

### Phase 3: 모델 학습
- [ ] 1D CNN 모델 구현
- [ ] 학습 및 검증
- [ ] 하이퍼파라미터 튜닝
- [ ] INT8 양자화
- [ ] 양자화 전후 정확도 비교
- [ ] TFLite 모델 저장

### Phase 4: NPU 포팅
- [ ] MCUXpresso IDE + eIQ Toolkit 설정
- [ ] FRDM-MCXN947 보드 셋업
- [ ] TFLite Micro 통합
- [ ] NPU 가속 활성화
- [ ] 실시간 파이프라인 구현
- [ ] 성능 프로파일링 및 최적화
- [ ] 정확도 검증 (MCU vs PC)

---

## 새 대화에서 시작할 때

새 대화를 시작할 때 다음과 같이 말하세요:

> PROJECT_ROADMAP.md 파일을 읽고 Phase N부터 이어서 진행해줘

Claude가 이 파일을 읽으면 전체 프로젝트 맥락을 파악하고 바로 작업을 시작할 수 있습니다.
