# Makefile for Boundary Detector Library
# 피부층 경계 검출 알고리즘 C 라이브러리 빌드

# 컴파일러 및 플래그
CC = gcc
CFLAGS = -O3 -fPIC -Wall -Wextra
LDFLAGS = -shared -lm

# 소스 및 출력 파일
SRC = boundary_detector.c
TARGET = libboundary_detector.so
OBJ = $(SRC:.c=.o)

# Python 테스트 스크립트
PYTHON = python3
TEST_SCRIPT = boundary_detector_wrapper.py
REAL_TEST = ../test_c_detector.py

# 기본 타겟
.PHONY: all
all: $(TARGET)

# 공유 라이브러리 빌드
$(TARGET): $(SRC)
	@echo "=========================================="
	@echo "Building Boundary Detector Library"
	@echo "=========================================="
	@echo ""
	@echo "Compiling $(SRC)..."
	@echo "Command: $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<"
	@echo ""
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	@echo ""
	@echo "=========================================="
	@echo "✓ Build successful!"
	@echo "=========================================="
	@echo "Output: $(TARGET)"
	@ls -lh $(TARGET) | awk '{print "Size: " $$5}'
	@echo ""

# 기본 테스트
.PHONY: test
test: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo "Running Basic Test"
	@echo "=========================================="
	@echo ""
	$(PYTHON) $(TEST_SCRIPT)
	@echo ""

# 실제 데이터 테스트
.PHONY: test-real
test-real: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo "Running Real Data Test"
	@echo "=========================================="
	@echo ""
	cd .. && $(PYTHON) test_c_detector.py
	@echo ""

# 클린
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET) $(OBJ)
	rm -rf __pycache__
	@echo "✓ Clean complete"

# 재빌드
.PHONY: rebuild
rebuild: clean all

# 설치 (시스템 경로로 복사, root 권한 필요)
.PHONY: install
install: $(TARGET)
	@echo "Installing $(TARGET) to /usr/local/lib..."
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Error: Installation requires root privileges."; \
		echo "Please run: sudo make install"; \
		exit 1; \
	fi
	install -m 755 $(TARGET) /usr/local/lib/
	ldconfig
	@echo "✓ Installation complete"

# 제거
.PHONY: uninstall
uninstall:
	@echo "Uninstalling $(TARGET) from /usr/local/lib..."
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Error: Uninstallation requires root privileges."; \
		echo "Please run: sudo make uninstall"; \
		exit 1; \
	fi
	rm -f /usr/local/lib/$(TARGET)
	ldconfig
	@echo "✓ Uninstallation complete"

# 도움말
.PHONY: help
help:
	@echo "Boundary Detector Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Build the shared library (default)"
	@echo "  test        - Run basic test"
	@echo "  test-real   - Run test with real ultrasound data"
	@echo "  clean       - Remove build artifacts"
	@echo "  rebuild     - Clean and rebuild"
	@echo "  install     - Install library to /usr/local/lib (requires sudo)"
	@echo "  uninstall   - Remove library from /usr/local/lib (requires sudo)"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build library"
	@echo "  make test         # Build and test"
	@echo "  make clean        # Clean build"
	@echo "  sudo make install # Install system-wide"

# 디버그 빌드
.PHONY: debug
debug: CFLAGS = -g -fPIC -Wall -Wextra -DDEBUG
debug: $(TARGET)
	@echo "✓ Debug build complete"

# 정보 출력
.PHONY: info
info:
	@echo "Build Configuration:"
	@echo "  Compiler: $(CC)"
	@echo "  CFLAGS: $(CFLAGS)"
	@echo "  LDFLAGS: $(LDFLAGS)"
	@echo "  Source: $(SRC)"
	@echo "  Target: $(TARGET)"
	@echo "  Python: $(PYTHON)"
