CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c99
LDFLAGS = -lm

TARGET  = skin_infer
SRCS    = main.c skin_cnn.c
OBJS    = $(SRCS:.c=.o)

.PHONY: all clean weights run

all: weights $(TARGET)

## 1단계: PyTorch 가중치 → C 헤더 생성
weights: model_weights.h

model_weights.h: ../results/best_1dcnn.pth export_weights.py
	python3 export_weights.py

## 2단계: C 빌드
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c skin_cnn.h model_weights.h
	$(CC) $(CFLAGS) -c -o $@ $<

## 테스트: usdata 첫 파일로 추론
run: $(TARGET)
	@CSV=$$(find ../usdata/data -name "*.csv" ! -name "*_positions*" | head -1); \
	if [ -z "$$CSV" ]; then echo "usdata CSV 없음"; exit 1; fi; \
	echo "테스트 파일: $$CSV"; \
	./$(TARGET) "$$CSV"

clean:
	rm -f $(OBJS) $(TARGET)

cleanall: clean
	rm -f model_weights.h
