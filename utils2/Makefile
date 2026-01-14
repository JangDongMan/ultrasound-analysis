# Makefile for Boundary Detector Library
# 피부층 경계 검출 알고리즘 C 라이브러리 빌드

# 컴파일러 및 플래그
CC = gcc
CFLAGS = -O3 -fPIC -Wall -Wextra
LDFLAGS = -shared -lm
LDFLAGS_EXE = -lm

# 소스 및 출력 파일
SRC = boundary_detector.c
CLI_SRC = boundary_detector_cli.c
TARGET = libboundary_detector.so
CLI_TARGET = boundary_detector
OBJ = $(SRC:.c=.o)

# Python 테스트 스크립트
PYTHON = python3
TEST_SCRIPT = boundary_detector_wrapper.py
REAL_TEST = ../test_c_detector.py

# 기본 타겟
.PHONY: all
all: $(TARGET) $(CLI_TARGET)

# 공유 라이브러리 빌드
$(TARGET): $(SRC)
	@echo "=========================================="
	@echo "Building Shared Library"
	@echo "=========================================="
	@echo ""
	@echo "Compiling $(SRC)..."
	@echo "Command: $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<"
	@echo ""
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	@echo ""
	@echo "✓ Shared library built: $(TARGET)"
	@ls -lh $(TARGET) | awk '{print "  Size: " $$5}'
	@echo ""

# 실행파일 빌드
$(CLI_TARGET): $(SRC) $(CLI_SRC)
	@echo "=========================================="
	@echo "Building Standalone Executable"
	@echo "=========================================="
	@echo ""
	@echo "Compiling $(CLI_TARGET)..."
	@echo "Command: $(CC) $(CFLAGS) -o $@ $(SRC) $(CLI_SRC) $(LDFLAGS_EXE)"
	@echo ""
	$(CC) $(CFLAGS) -o $@ $(SRC) $(CLI_SRC) $(LDFLAGS_EXE)
	@echo ""
	@echo "✓ Executable built: $(CLI_TARGET)"
	@ls -lh $(CLI_TARGET) | awk '{print "  Size: " $$5}'
	@echo ""
	@echo "=========================================="
	@echo "Build Complete!"
	@echo "=========================================="
	@echo "  Library: $(TARGET)"
	@echo "  Executable: $(CLI_TARGET)"
	@echo ""
	@echo "Usage:"
	@echo "  ./$(CLI_TARGET) <csv_file> <start_time_us>"
	@echo ""
	@echo "Example:"
	@echo "  ./$(CLI_TARGET) ../data/bhjung-5M-1.csv 17.26"
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

# CLI 실행 테스트
.PHONY: run
run: $(CLI_TARGET)
	@echo ""
	@echo "=========================================="
	@echo "Running CLI Test"
	@echo "=========================================="
	@echo ""
	./$(CLI_TARGET) ../data/bhjung-5M-1.csv 17.26
	@echo ""

# 클린
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET) $(CLI_TARGET) $(OBJ)
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
	@echo "  all         - Build library and executable (default)"
	@echo "  test        - Run basic Python test"
	@echo "  test-real   - Run test with real ultrasound data"
	@echo "  run         - Run standalone executable test"
	@echo "  clean       - Remove build artifacts"
	@echo "  rebuild     - Clean and rebuild"
	@echo "  debug       - Build with debug symbols"
	@echo "  install     - Install library to /usr/local/lib (requires sudo)"
	@echo "  uninstall   - Remove library from /usr/local/lib (requires sudo)"
	@echo "  info        - Show build configuration"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build library and executable"
	@echo "  make run          # Run standalone executable"
	@echo "  make test         # Build and test with Python"
	@echo "  make clean        # Clean build"
	@echo "  sudo make install # Install system-wide"
	@echo ""
	@echo "Standalone executable usage:"
	@echo "  ./boundary_detector <csv_file> <start_time_us>"
	@echo "  ./boundary_detector ../data/bhjung-5M-1.csv 17.26"

# 디버그 빌드
.PHONY: debug
debug: CFLAGS = -g -fPIC -Wall -Wextra -DDEBUG
debug: clean all
	@echo "✓ Debug build complete"

# 정보 출력
.PHONY: info
info:
	@echo "Build Configuration:"
	@echo "  Compiler: $(CC)"
	@echo "  CFLAGS: $(CFLAGS)"
	@echo "  LDFLAGS: $(LDFLAGS)"
	@echo "  LDFLAGS_EXE: $(LDFLAGS_EXE)"
	@echo ""
	@echo "Source Files:"
	@echo "  Library: $(SRC)"
	@echo "  CLI: $(CLI_SRC)"
	@echo ""
	@echo "Output Files:"
	@echo "  Library: $(TARGET)"
	@echo "  Executable: $(CLI_TARGET)"
	@echo ""
	@echo "Python: $(PYTHON)"
