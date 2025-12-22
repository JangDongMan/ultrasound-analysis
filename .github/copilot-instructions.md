# GitHub Copilot Instructions for Ultrasound Analysis ML Project

## 프로젝트 개요
이 프로젝트는 딥러닝 모델(CNN, U-Net, 앙상블 아키텍처)을 사용한 초음파 영상 분석 시스템입니다. 의료 초음파 이미지를 처리하여 질병 진단, 종양 검출, 조직 분할 등의 자동화된 분석을 수행합니다.

**개발 환경**: Ubuntu 20.04 Docker 컨테이너에서 VS Code를 사용하여 초음파 분석 모델 개발 및 실험 진행 예정

**참고**: 이 프로젝트는 혈당 예측 프로젝트와 함께 운영되는 의료 AI 프로젝트 중 하나입니다.

## 아키텍처 및 데이터 흐름

### 핵심 구성 요소
- **데이터 처리**: 원본 DICOM/초음파 이미지 → 전처리 및 증강
- **모델 학습**: GPU 최적화된 CNN 기반 아키텍처(U-Net, ResNet, EfficientNet)
- **실시간 분석**: 저장된 모델로부터 실시간 초음파 영상 분석
- **평가**: 의료 영상 특화 메트릭(Dice, IoU, Sensitivity, Specificity)

### 주요 디렉토리 (계획)
- `data/`: 날짜/환자별로 정리된 초음파 이미지 데이터
- `saved_models/`: 학습된 모델 및 평가 메트릭
- `processed_images/`: 전처리된 이미지 및 증강 데이터
- `models/`: 실험 버전별 학습 스크립트
- `inference/`: 실시간 분석 유틸리티

## 필수 개발자 워크플로우

### 환경 설정
```bash
# venv_docker에서 Python 3.10+ 사용
source venv_docker/bin/activate
pip install -r requirements.txt
# 추가 의존성: pydicom, opencv-python, albumentations, segmentation-models-pytorch
```

### 데이터 전처리
```python
# 새로운 초음파 데이터에 대한 전처리 실행
python preprocess_ultrasound.py
# 이미지 정규화, 크기 조정, 마스크 생성
```

### 모델 학습
```python
# GPU 최적화로 모델 학습
python train_ultrasound_model.py
# 혼합 정밀도, 조기 종료, 교차 검증 사용
```

### 실시간 분석
```python
# 저장된 모델로 분석
from inference.ultrasound_analyzer import UltrasoundAnalyzer
analyzer = UltrasoundAnalyzer(model_path, config_path)
```

## 프로젝트별 패턴 및 관행

### 모델 아키텍처 규칙
- **입력 형태**: `(height, width, channels)` - 일반적으로 `(224, 224, 3)` 또는 `(512, 512, 1)`
- **CNN 백본**: ResNet, EfficientNet, DenseNet 등 사전 학습된 네트워크
- **세그먼테이션**: U-Net, DeepLab, FPN 아키텍처
- **분류 모델**: CNN 기반 다중 클래스 분류
- **앙상블 모델**: 여러 모델의 예측 결합

### 데이터 처리 패턴
```python
# 이미지 전처리 및 증강
transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3),
])
# 항상 NaN/inf 값 처리 및 데이터 검증
```

### 커스텀 손실 함수 및 메트릭
```python
# Dice Loss for segmentation
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# IoU 및 Dice 계수 계산
def calculate_metrics(y_true, y_pred):
    dice = dice_coefficient(y_true, y_pred)
    iou = jaccard_index(y_true, y_pred)
    return {'dice': dice, 'iou': iou}
```

### 파일 명명 규칙
- **모델**: `unet_model_*.h5` 또는 `*.keras`
- **데이터**: `ultrasound_images_YYYYMMDD_HHMMSS.npy`
- **결과**: `predictions_YYYYMMDD_HHMMSS.csv`
- **타임스탬프**: 항상 `datetime.now().strftime('%Y%m%d_%H%M%S')` 사용

## 통합 포인트

### GPU 최적화
```python
# 항상 혼합 정밀도 및 메모리 증가 활성화
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.experimental.set_memory_growth(gpu, True)
```

### 데이터 증강 파이프라인
```python
# Albumentations를 사용한 고급 증강
augmentation_pipeline = A.Compose([
    A.RandomCrop(width=224, height=224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(blur_limit=3, p=0.1),
])
```

### 교차 검증 전략
```python
# 환자별 계층화된 K-fold (데이터 누출 방지)
patient_ids = df['patient_id'].unique()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

## 일반적인 패턴 및 모범 사례

### 오류 처리
```python
# 항상 모델 연산을 래핑
try:
    predictions = model.predict(images, verbose=0)
except Exception as e:
    print(f"분석 실패: {e}")
    return None
```

### 메모리 관리
```python
# 대용량 이미지 데이터 처리 후 메모리 정리
del images, masks
import gc
gc.collect()
```

### 로깅 및 진행 상황
```python
# 추적을 위한 타임스탬프 출력 사용
print(f"[{datetime.now()}] {len(dataset)}개 이미지 처리 중")
print(f"Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
```

## 참고할 핵심 파일 (계획)

### 주요 학습 스크립트
- `models/train_unet.py`: U-Net 기반 세그먼테이션 학습
- `models/train_classifier.py`: CNN 기반 분류 모델 학습

### 실시간 분석
- `inference/ultrasound_analyzer.py`: 프로덕션 분석 인터페이스
- `inference/batch_processor.py`: 배치 분석 도구

### 모델 아키텍처
- `models/architectures.py`: 다양한 모델 아키텍처 구현

## 성능 목표
- **Dice 계수**: > 0.85 (세그먼테이션)
- **IoU**: > 0.80 (세그먼테이션)
- **정확도**: > 95% (분류)
- **학습 시간**: RTX 3060에서 에폭당 10분 미만

## 디버깅 팁
- 이미지 차원이 모델 input_shape와 일치하는지 검증
- 마스크 레이블이 올바르게 인코딩되었는지 확인
- GPU 메모리 사용량 모니터링
- 학습 전 `model.summary()`로 아키텍처 검증