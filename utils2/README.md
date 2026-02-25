# Utils2 - C 기반 피부층 경계 검출 라이브러리

### 특징

- **고성능**: 순수 C로 구현되어 Python 대비 빠른 처리 속도
- **통계 기반**: 234개 수동 레이블 분석 결과 기반 알고리즘
- **Python 연동**: ctypes를 통한 간편한 Python 바인딩
- **검증된 정확도**: 진피 검출 평균 오차 0.20mm
- **의존성 최소화**: 표준 C 라이브러리만 사용

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

- gcc (C 컴파일러)
- Linux 또는 macOS
- Python 3.x (Python 바인딩 사용 시)

### 컴파일

**방법 1: Makefile 사용 (권장)**

```bash
cd utils2
make           # 라이브러리 빌드
make test      # 빌드 및 기본 테스트
make test-real # 실제 데이터 테스트
```

**방법 2: 빌드 스크립트 사용**

```bash
cd utils2
./build.sh
```

컴파일이 성공하면 `libboundary_detector.so` 파일이 생성됩니다.

**Makefile 주요 명령어:**
```bash
make           # 빌드
make clean     # 빌드 산출물 제거
make rebuild   # 클린 후 재빌드
make test      # 기본 테스트
make test-real # 실제 데이터 테스트
make debug     # 디버그 빌드
make info      # 빌드 설정 정보
make help      # 도움말
```

## 사용법

### 1. 독립 실행파일 사용 (권장)


```bash
# 빌드
make

# 실행 (CSV 또는 TXT 파일 모두 지원, 시작점 자동 검출)
./boundary_detector <data_file> [start_time_us]

# 예시
./boundary_detector data/bhjung-5M-1.csv          # 시작점 자동 검출 (권장)
./boundary_detector data/bhjung-5M-1.csv 17.26    # 시작점 수동 지정
./boundary_detector data/bhjung-5M-1.txt          # TXT 파일도 지원
```

**출력 예시:**
```
========================================
Detection Results
========================================

Start Point: 17.25 μs (auto-detected)

Dermis (진피):
  Index:    235
  Time:     2.35 μs (from start)
  Depth:    1.81 mm

Fascia (근막):
  Index:    507
  Time:     5.07 μs (from start)
  Depth:    3.90 mm

Thickness (두께):
  Dermis-Fascia: 2.09 mm
```

### 2. Python에서 사용

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

### 3. 편의 함수 사용

```python
from utils2.boundary_detector_wrapper import detect_skin_boundaries

dermis_idx, fascia_idx, success = detect_skin_boundaries(
    time_us, voltage, reference_start_us
)
```

## 테스트

### Makefile 사용 (권장)

```bash
cd utils2
make test       # 기본 테스트
make test-real  # 실제 데이터 테스트
```

### 수동 실행

**기본 테스트:**
```bash
cd utils2
python3 boundary_detector_wrapper.py
```

**실제 데이터 테스트:**
```bash
python3 test_c_detector.py
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
├── boundary_detector.c            # C 구현 (라이브러리)
├── boundary_detector_cli.c        # CLI 프로그램
├── boundary_detector_wrapper.py   # Python 래퍼
├── Makefile                       # Make 빌드 파일
├── build.sh                       # 빌드 스크립트 (대체)
├── boundary_detector             # 실행파일 (빌드 후)
├── libboundary_detector.so       # 공유 라이브러리 (빌드 후)
└── README.md                     # 이 파일
```

## 알고리즘 상세

### Step 0: 시작점 자동 검출 (선택적)

시작점을 지정하지 않으면 자동으로 검출합니다 (정확도: ±0.04 μs):

1. **탐색 범위**: 16.39 ~ 18.39 μs (통계 기반: 평균 17.39±0.15 μs)
2. **검출 방법**:
   - 탐색 범위 내에서 최대 전압 변화율(기울기) 계산
   - 처음으로 최대 기울기의 30% 이상 상승하는 지점 검출
   - 이것이 초음파 펄스의 실제 시작점 (피부 표면)

```cpp
START_POINT_EXPECTED_US = 17.39    // 통계적 평균
START_POINT_SEARCH_WINDOW_US = 1.0 // 탐색 범위: ±1.0 μs
```

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
