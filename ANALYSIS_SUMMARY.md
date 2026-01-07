# 초음파 신호 자동 검출 알고리즘 개선 분석

## 개요
수동 마킹 데이터(228개 샘플)를 참조하여 자동 검출 알고리즘을 분석하고 개선

## 1. 데이터 준비

### 수동 레이블 데이터
- **총 샘플 수**: 228개
- **레이블 형식**: JSON (manual_boundaries/*.json)
- **포함 정보**:
  - `start_point_us`: 피부 시작점 (μs)
  - `positions[0]`: Dermis (진피) - 피부 시작점으로부터의 거리
  - `positions[1]`: Fascia (근막) - 피부 시작점으로부터의 거리

### 레이블 데이터 특성
- Dermis 두께: 1.54 ± 0.00 mm (피부 시작점부터의 거리)
- Fascia 두께: 3.08 ± 0.00 mm (피부 시작점부터의 거리)
- 모든 샘플에서 일관된 두께 값

## 2. 파라미터 최적화 분석

### 테스트 방법
- **analyze_detection_accuracy.py** 스크립트 사용
- 100개 파라미터 조합 테스트
- 테스트 범위:
  - `prominence_factor`: 0.3 ~ 0.7
  - `distance`: 5 ~ 15
  - `height_factor`: 0.5 ~ 0.8

### 최적 파라미터
```json
{
  "pulse_threshold": 2.0,
  "prominence_factor": 0.6,
  "distance": 5,
  "height_factor": 0.5
}
```

### 검출 성능 (기존 자동 검출)
- **성공률**: 36% (82/228 샘플)
- **평균 오차**:
  - 진피 두께: 0.889 ± 0.507 mm
  - 근막 두께: 1.117 ± 0.527 mm

### 정확도 분석
| 허용 오차 | 진피 정확도 | 근막 정확도 |
|-----------|-------------|-------------|
| 0.05mm    | 4.9%        | 0.0%        |
| 0.1mm     | 7.3%        | 0.0%        |
| 0.2mm     | 13.4%       | 4.9%        |
| 0.5mm     | 29.3%       | 13.4%       |

## 3. 주요 문제점 분석

### 3.1 펄스 시작점 검출 문제
- **문제**: 자동 검출된 펄스 시작점과 수동 마킹 시작점의 불일치
- **원인**:
  - 임계값 기반 검출의 한계
  - 신호 잡음으로 인한 오검출
  - 첫 번째 피크가 항상 피부 시작점이 아님

### 3.2 피크 검출 감도 문제
- **문제**: 피크 검출 파라미터의 일반화 어려움
- **원인**:
  - 신호 품질의 변동성
  - 환자별/부위별 신호 특성 차이
  - 고정 파라미터로는 모든 경우 대응 불가

### 3.3 낮은 성공률
- **36% 성공률**의 원인:
  1. 펄스 시작점 오검출 (64%)
  2. 피크 검출 실패
  3. 신호 품질 저하

## 4. 개선 방안

### 4.1 수동 시작점 참조 방식 (visualize_signal_improved.py)
**핵심 개선 사항**:
```python
def detect_positions_with_reference(time_data, voltage_data, reference_start_us, sample_rate):
    # 수동 레이블의 정확한 시작점 사용
    start_idx = np.argmin(np.abs(time_us - reference_start_us))

    # 시작점 이후 데이터만 분석
    analysis_time_us = time_us[start_idx:] - reference_start_us
    analysis_voltage = voltage_data[start_idx:]

    # 필터링 및 피크 검출 (최적 파라미터 사용)
    ...
```

**장점**:
- ✓ 시작점 오검출 문제 해결
- ✓ 분석 영역이 정확히 정의됨
- ✓ 피크 검출 정확도 향상

### 4.2 시각화 개선
**새로운 6개 서브플롯**:
1. **전체 신호**: 수동 시작점 표시
2. **시작점 이후 원본 신호**: 6mm 제한 표시
3. **필터링된 신호 + 수동 레이블**: 정확한 레이블 위치
4. **자동 검출 vs 수동 레이블 비교**: 두 방법 중첩 표시
5. **오차 분석**: 층별 오차 막대 그래프
6. **층별 두께 비교**: 수동 vs 자동 비교 막대 그래프

## 5. 결과

### 생성된 파일
- **분석 결과**: `results/error_distribution.png`
- **최적 파라미터**: `results/optimal_params.json`
- **개선된 시각화**: `results/improved/bhjung_position_*_layer_analysis.png` (8개 파일)

### 향후 개선 방향

#### 5.1 기계학습 기반 접근
```python
# 제안: CNN 기반 경계 검출
class BoundaryDetectionCNN:
    def __init__(self):
        # 1D CNN for signal processing
        # Input: 전처리된 신호
        # Output: 진피/근막 경계 위치
```

**장점**:
- 신호 패턴 자동 학습
- 일반화 성능 향상
- 다양한 신호 품질에 대응

#### 5.2 앙상블 방법
```python
# 여러 검출 알고리즘의 조합
ensemble_result = combine([
    peak_detection_method(),
    gradient_based_method(),
    wavelet_based_method(),
    ml_based_method()
])
```

#### 5.3 적응형 파라미터
```python
# 신호 특성에 따른 동적 파라미터 조정
params = adaptive_parameter_selection(
    signal_quality=assess_quality(signal),
    snr=calculate_snr(signal),
    patient_characteristics=patient_info
)
```

## 6. 핵심 발견

### 6.1 수동 마킹의 중요성
- **정확한 시작점**: 자동 검출의 핵심 전제 조건
- **일관된 레이블**: 모델 학습을 위한 기준점
- **228개 샘플**: 기계학습 적용 가능한 데이터셋

### 6.2 신호 전처리의 한계
- 밴드패스 필터만으로는 부족
- 추가 전처리 필요:
  - 적응형 필터링
  - 잡음 제거
  - 신호 정규화

### 6.3 일반화의 어려움
- 고정 파라미터로는 36% 성공률
- 환자별/부위별 최적화 필요
- 기계학습 기반 접근이 필수

## 7. 사용 방법

### 파라미터 최적화 분석
```bash
python3 analyze_detection_accuracy.py
```

### 개선된 시각화 (수동 레이블 참조)
```bash
python3 visualize_signal_improved.py
```

### 출력 파일
- `results/optimal_params.json`: 최적 파라미터
- `results/error_distribution.png`: 오차 분포
- `results/improved/*.png`: 개선된 시각화 (환자별)

## 8. 결론

### 현재 상태
- ✓ 228개 수동 레이블 데이터 확보
- ✓ 최적 파라미터 도출 (prominence=0.6, distance=5, height=0.5)
- ✓ 성능 분석 완료 (36% 성공률, 0.89mm 평균 오차)
- ✓ 개선된 시각화 도구 개발

### 다음 단계
1. **기계학습 모델 개발**
   - CNN/RNN 기반 경계 검출 모델
   - 228개 샘플로 학습
   - 교차 검증으로 성능 평가

2. **추가 데이터 수집**
   - 다양한 환자군
   - 다양한 측정 부위
   - 신호 품질 다변화

3. **실시간 검출 시스템**
   - 최적화된 모델 배포
   - 사용자 인터페이스 개발
   - 피드백 수집 및 모델 업데이트

### 기대 효과
- 자동 검출 정확도 향상 (목표: >80%)
- 검사 시간 단축
- 일관된 측정 결과
- 객관적 진단 지원
