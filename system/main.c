#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "skin_cnn.h"
#include "model_weights.h"

#define MAX_SAMPLES 4096

static int load_adc_csv(const char *path, unsigned char *adc, int max_len)
{
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "파일 열기 실패: %s\n", path); return -1; }

    int n = 0;
    char line[64];
    while (fgets(line, sizeof(line), fp) && n < max_len) {
        /* 헤더(숫자가 아닌 줄) 자동 건너뜀 */
        char *end;
        long val = strtol(line, &end, 10);
        if (end == line) continue;          /* 변환 실패 → 헤더 */
        if (val < 0 || val > 255) continue; /* 범위 초과 → 건너뜀 */
        adc[n++] = (unsigned char)val;
    }
    fclose(fp);
    return n;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("사용법: %s <adc_csv_file> [adc_csv_file ...]\n", argv[0]);
        printf("        ADC CSV 파일 (정수 0~255, 헤더 없음)\n");
        printf("        TRIM_START=%d, TRIM_COUNT=%d 자동 적용\n",
               TRIM_START, TRIM_COUNT);
        return 1;
    }

    printf("%-50s  %8s  %8s\n", "파일", "진피(mm)", "근막(mm)");
    printf("%s\n", "-----------------------------------------------------------------------");

    unsigned char adc[MAX_SAMPLES];
    int all_ok = 1;

    for (int fi = 1; fi < argc; fi++) {
        const char *path = argv[fi];
        int n = load_adc_csv(path, adc, MAX_SAMPLES);

        if (n < 0) { all_ok = 0; continue; }

        if (n < TRIM_COUNT) {
            fprintf(stderr, "샘플 부족: %s  (%d개, 최소 %d 필요)\n",
                    path, n, TRIM_COUNT);
            all_ok = 0;
            continue;
        }

        float dermis, fascia;
        int ret = skin_cnn_infer(adc, n, &dermis, &fascia);
        if (ret != 0) {
            fprintf(stderr, "추론 오류: %s\n", path);
            all_ok = 0;
            continue;
        }

        /* 파일명만 출력 (경로 제거) */
        const char *fname = strrchr(path, '/');
        fname = fname ? fname + 1 : path;

        printf("%-50s  %8.3f  %8.3f\n", fname, dermis, fascia);
    }

    return all_ok ? 0 : 1;
}
