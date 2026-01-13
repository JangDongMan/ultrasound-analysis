# Utils2 - C++ 기반 피부층 경계 검출 라이브러리

## 개요

통계 기반 피부층 경계 자동 검출 알고리즘을 C++로 구현한 고성능 라이브러리입니다.

### 특징

- **고성능**: C++로 구현되어 Python 대비 빠른 처리 속도
- **통계 기반**: 234개 수동 레이블 분석 결과 기반 알고리즘
- **Python 연동**: ctypes를 통한 간편한 Python 바인딩
- **검증된 정확도**: 진피 검출 평균 오차 0.20mm

## 알고리즘 파라미터

통계 분석 결과를 기반으로 한 하드코딩 파라미터:

```cpp
DERMIS_EXPECTED_TIME_US = 2.33    // 진피 예상 시간 (μs)
DERMIS_TIME_STD_US = 0.33         // 진피 표준편차
FASCIA_EXPECTED_TIME_US = 5.16    // 근막 예상 시간 (μs)
FASCIA_TIME_STD_US = 0.79         // 근막 표준편차
THICKNESS_GAP_MM = 2.18           // 진피-근막 두께 차이 (mm)
EPIDERMIS_CUTOFF_TIME_US = 1.5    // 표피 제외 시간 (μs)
SPEED_OF_SOUND = 1540.0           // 음속 (m/s)
MAX_DISTANCE_MM = 6.0             // 분석 범위 (mm)
```

## 빌드

### 요구사항

- g++ (C++11 이상)
- Linux 또는 macOS

### 컴파일

```bash
cd utils2
./build.sh
```

컴파일이 성공하면 `libboundary_detector.so` 파일이 생성됩니다.

## 사용법

### Python에서 사용

```python
from utils2.boundary_detector_wrapper import BoundaryDetector
import numpy as np

# 검출기 초기화
detector = BoundaryDetector()

# 데이터 준비
time_us = np.array([...])        # 시간 데이터 (μs)
voltage = np.array([...])         # 전압 데이터
reference_start_us = 17.26        # 시작점 시간 (μs)

# 경계 검출
dermis_idx, fascia_idx, success = detector.detect(
    time_us, voltage, reference_start_us
)

if success:
    print(f"Dermis index: {dermis_idx}")
    print(f"Fascia index: {fascia_idx}")
```

### 편의 함수 사용

```python
from utils2.boundary_detector_wrapper import detect_skin_boundaries

dermis_idx, fascia_idx, success = detect_skin_boundaries(
    time_us, voltage, reference_start_us
)
```

## 테스트

### 기본 테스트

```bash
cd utils2
python3 boundary_detector_wrapper.py
```

### 실제 데이터 테스트

```bash
cd utils2
python3 test_with_real_data.py
```

출력 예시:
```
bhjung Position 1
────────────────────────────────────────────
Manual Labels:
  Dermis: 1.80 mm
  Fascia: 3.52 mm

C++ Detection:
  Dermis: 1.80 mm (error: 0.00 mm)
  Fascia: 3.90 mm (error: 0.38 mm)

Result: ✓
```

## 파일 구조

```
utils2/
├── boundary_detector.cpp          # C++ 구현
├── boundary_detector_wrapper.py   # Python 래퍼
├── build.sh                       # 빌드 스크립트
├── test_with_real_data.py        # 실제 데이터 테스트
├── libboundary_detector.so       # 컴파일된 라이브러리 (빌드 후)
└── README.md                     # 이 파일
```

## 알고리즘 상세

### Step 1: 표피 영역 제외

1.5μs 이전 영역을 표피로 간주하고 완전히 제외합니다.

```cpp
epidermis_cutoff_time_us = 1.5
```

### Step 2: 진피 검출

1. **탐색 범위**: 2.0 ~ 2.7 μs (DERMIS_EXPECTED ± STD)
2. **검출 방법**:
   - 절댓값의 2차 미분을 통한 변곡점 검출
   - 상승 변곡점(음→양) 중 예상 시간에 가장 가까운 지점 선택
   - 변곡점이 없으면 피크 중 예상 시간에 가장 가까운 것 선택

### Step 3: 근막 검출

1. **탐색 범위**:
   - 시간 기반: 4.4 ~ 5.9 μs (FASCIA_EXPECTED ± STD)
   - 진피 기반: dermis_time + 2.18mm
   - 두 값의 평균 사용

2. **검출 방법**:
   - 피크 검출 (prominence = std × 0.3)
   - 예상 시간에 가장 가까운 피크 선택

## 성능

### 정확도 (24개 테스트 샘플)

| 지표 | 진피 | 근막 |
|------|------|------|
| 평균 오차 | 0.20 mm | 0.61 mm |
| 중앙값 오차 | 0.15 mm | 0.43 mm |
| < 0.5mm 정확도 | 91.7% | 54.2% |
| < 1.0mm 정확도 | 100% | 70.8% |

### 속도

C++ 구현으로 Python 대비 약 10-50배 빠른 처리 속도를 보입니다.

## Python 구현과 비교

Python 구현(`visualize_signal_improved.py`)과 동일한 알고리즘이지만:

- **장점**:
  - 빠른 처리 속도
  - 메모리 효율성
  - 독립 실행 가능

- **단점**:
  - 컴파일 필요
  - 디버깅이 상대적으로 어려움

## 주의사항

1. **플랫폼 의존성**: Linux/macOS용으로 빌드됨. Windows는 MinGW 또는 WSL 필요
2. **라이브러리 경로**: `boundary_detector_wrapper.py`는 같은 디렉토리에 `.so` 파일이 있어야 함
3. **데이터 형식**: numpy array는 `float64` (double) 타입이어야 함

## 문제 해결

### 라이브러리를 찾을 수 없음

```
FileNotFoundError: Shared library not found
```

**해결**: `build.sh`를 실행하여 라이브러리 컴파일

### 컴파일 오류

```
g++: command not found
```

**해결**: g++ 설치
```bash
# Ubuntu/Debian
sudo apt-get install g++

# macOS
xcode-select --install
```

## 향후 개선

- [ ] Windows 지원 (.dll 빌드)
- [ ] CMake 기반 빌드 시스템
- [ ] 적응형 파라미터 지원
- [ ] GPU 가속 (CUDA)
- [ ] 멀티스레드 배치 처리

## 라이선스

이 코드는 초음파 피부층 분석 프로젝트의 일부입니다.

## 참고

- 원본 Python 구현: `visualize_signal_improved.py`
- 알고리즘 리포트: `DETECTION_ALGORITHM_REPORT.md`
- 통계 분석: `analyze_manual_labels.py`
