# Ultrasound Position Marker - Windows Executable

초음파 데이터에서 피부층 위치를 수동으로 마킹하는 도구입니다.

## 빌드 방법 (Windows에서만)

### 1. Python 3.12 설치
- https://www.python.org/downloads/ 에서 Python 3.12 이상 다운로드
- 설치 시 **"Add Python to PATH"** 체크 필수

### 2. 빌드 실행

**방법 1: 배치 파일 실행 (가장 쉬움)**
```
BUILD.bat 더블클릭
```

**방법 2: Python 스크립트 실행**
```cmd
python BUILD.py
```

### 3. 결과
- 빌드 완료 후 `utils/dist/UltrasoundMarker.exe` 생성됨
- 파일 크기: 약 80-100 MB

## 사용 방법

### 1. 프로그램 실행
- `dist/UltrasoundMarker.exe` 더블클릭

### 2. 라벨 파일 준비 (선택사항)
- `D:/util/masker/data/label.csv` 파일을 준비
- CSV 형식:
  ```csv
  filename,thickness1,thickness2
  bhjung-5M-1.csv,0.0015,0.0035
  ```
- thickness1: P1까지 거리 (미터 단위)
- thickness2: P2까지 거리 (미터 단위)

### 3. CSV 파일 열기
- "Open CSV" 버튼 클릭
- 초음파 데이터 CSV 파일 선택
- label.csv가 있으면 자동으로 로드되어 수동 측정값 표시

### 4. 위치 마킹
1. 그래프에서 **시작점 (Epidermis)** 클릭
2. **P1 (Dermis Start)** 클릭
3. **P2 (Fascia Start)** 클릭
4. 마우스를 움직이면 실시간으로 시작점부터의 거리 표시

### 5. 저장
- "Save Positions" 버튼 클릭
- JSON 파일로 저장됨

## 파일 구조

```
utils/
├── BUILD.bat              # 빌드 배치 파일
├── BUILD.py               # 빌드 Python 스크립트
├── manual_position_marker.py  # 메인 프로그램
├── dist/                  # 빌드된 실행 파일 위치
│   └── UltrasoundMarker.exe
└── README.md              # 이 파일
```

## 문제 해결

### Python을 찾을 수 없음
- BUILD.py 파일을 열어서 PYTHON 경로를 수정하세요:
  ```python
  PYTHON = r"C:\Users\사용자명\AppData\Local\Programs\Python\Python312\python.exe"
  ```

### tkinter 모듈 오류
```cmd
pip install tk
```

### matplotlib 오류
```cmd
pip install matplotlib --upgrade
```

## 기술 정보

### 포함된 라이브러리
- tkinter (GUI)
- matplotlib (그래프)
- numpy (수치 계산)
- pandas (데이터 처리)
- openpyxl (Excel 파일)
- scipy (신호 처리)

### 제외된 라이브러리 (파일 크기 감소)
- torch, tensorflow (딥러닝)
- sklearn, opencv (머신러닝)
- IPython, jupyter (노트북)
