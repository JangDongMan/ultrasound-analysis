#!/bin/bash

# 피부층 경계 검출 알고리즘 컴파일 스크립트 (C)

echo "=========================================="
echo "Building Boundary Detector Library"
echo "=========================================="

# 컴파일러 확인
if ! command -v gcc &> /dev/null; then
    echo "Error: gcc not found. Please install gcc."
    exit 1
fi

# 소스 파일
SOURCE="boundary_detector.c"
OUTPUT="libboundary_detector.so"

# 컴파일 옵션
# -O3: 최적화 레벨 3
# -shared: 공유 라이브러리 생성
# -fPIC: Position Independent Code (공유 라이브러리에 필요)
# -lm: 수학 라이브러리 링크
# -Wall: 모든 경고 표시
CFLAGS="-O3 -shared -fPIC -Wall -lm"

echo ""
echo "Compiling $SOURCE..."
echo "Command: gcc $CFLAGS -o $OUTPUT $SOURCE"
echo ""

# 컴파일 실행
gcc $CFLAGS -o $OUTPUT $SOURCE

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Build successful!"
    echo "=========================================="
    echo "Output: $OUTPUT"
    echo "Size: $(ls -lh $OUTPUT | awk '{print $5}')"
    echo ""
    echo "You can now use the detector:"
    echo "  python3 boundary_detector_wrapper.py"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Build failed!"
    echo "=========================================="
    exit 1
fi
