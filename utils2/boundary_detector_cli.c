/**
 * 피부층 경계 검출 CLI 프로그램
 *
 * 사용법:
 *   ./boundary_detector <csv_file> <start_time_us>
 *
 * 예시:
 *   ./boundary_detector ../data/bhjung-5M-1.csv 17.26
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* boundary_detector.c의 함수들을 사용하기 위한 외부 선언 */
typedef struct {
    int dermis_idx;
    int fascia_idx;
    int success;
} DetectionResult;

extern DetectionResult detect_skin_boundaries(
    const double *time_us,
    const double *voltage,
    int n,
    double reference_start_us
);

/* CSV 파일 파싱 */
int parse_csv_file(const char *filename, double **time_us, double **voltage) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return -1;
    }

    /* 헤더 2줄 건너뛰기 */
    char line[1024];
    if (!fgets(line, sizeof(line), fp)) return -1;
    if (!fgets(line, sizeof(line), fp)) return -1;

    /* 데이터 읽기 (동적 배열) */
    int capacity = 100000;
    int count = 0;
    *time_us = (double*)malloc(capacity * sizeof(double));
    *voltage = (double*)malloc(capacity * sizeof(double));

    while (fgets(line, sizeof(line), fp)) {
        double t, v;
        if (sscanf(line, "%lf,%lf", &t, &v) == 2) {
            if (count >= capacity) {
                capacity *= 2;
                *time_us = (double*)realloc(*time_us, capacity * sizeof(double));
                *voltage = (double*)realloc(*voltage, capacity * sizeof(double));
            }
            (*time_us)[count] = t * 1e6;  /* 초 → μs 변환 */
            (*voltage)[count] = v;
            count++;
        }
    }

    fclose(fp);
    return count;
}

/* 결과 출력 */
void print_results(DetectionResult result, const double *time_us, int n, double start_us) {
    if (!result.success) {
        printf("Detection failed!\n");
        return;
    }

    /* 시작점 찾기 */
    int start_idx = 0;
    double min_diff = fabs(time_us[0] - start_us);
    for (int i = 1; i < n; i++) {
        double diff = fabs(time_us[i] - start_us);
        if (diff < min_diff) {
            min_diff = diff;
            start_idx = i;
        }
    }

    /* 상대 시간 계산 */
    double dermis_time_us = time_us[start_idx + result.dermis_idx] - start_us;
    double fascia_time_us = time_us[start_idx + result.fascia_idx] - start_us;

    /* 깊이 계산 (음속: 1540 m/s) */
    double speed_of_sound = 1540.0;  /* m/s */
    double dermis_depth_mm = (dermis_time_us * 1e-6 * speed_of_sound / 2.0) * 1000.0;
    double fascia_depth_mm = (fascia_time_us * 1e-6 * speed_of_sound / 2.0) * 1000.0;

    /* 결과 출력 */
    printf("\n");
    printf("========================================\n");
    printf("Detection Results\n");
    printf("========================================\n");
    printf("\n");
    printf("Start Point: %.2f μs\n", start_us);
    printf("\n");
    printf("Dermis (진피):\n");
    printf("  Index:    %d\n", result.dermis_idx);
    printf("  Time:     %.2f μs (from start)\n", dermis_time_us);
    printf("  Depth:    %.2f mm\n", dermis_depth_mm);
    printf("\n");
    printf("Fascia (근막):\n");
    printf("  Index:    %d\n", result.fascia_idx);
    printf("  Time:     %.2f μs (from start)\n", fascia_time_us);
    printf("  Depth:    %.2f mm\n", fascia_depth_mm);
    printf("\n");
    printf("Thickness (두께):\n");
    printf("  Dermis-Fascia: %.2f mm\n", fascia_depth_mm - dermis_depth_mm);
    printf("\n");
    printf("========================================\n");
}

int main(int argc, char *argv[]) {
    /* 인자 확인 */
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <csv_file> <start_time_us>\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "  %s ../data/bhjung-5M-1.csv 17.26\n", argv[0]);
        fprintf(stderr, "\n");
        return 1;
    }

    const char *csv_file = argv[1];
    double start_time_us = atof(argv[2]);

    printf("\n");
    printf("========================================\n");
    printf("Boundary Detector CLI\n");
    printf("========================================\n");
    printf("\n");
    printf("Input file:  %s\n", csv_file);
    printf("Start time:  %.2f μs\n", start_time_us);
    printf("\n");

    /* CSV 파일 읽기 */
    double *time_us = NULL;
    double *voltage = NULL;
    int n = parse_csv_file(csv_file, &time_us, &voltage);

    if (n < 0) {
        return 1;
    }

    printf("Data loaded: %d samples\n", n);

    /* 경계 검출 */
    printf("\nDetecting boundaries...\n");
    DetectionResult result = detect_skin_boundaries(time_us, voltage, n, start_time_us);

    /* 결과 출력 */
    print_results(result, time_us, n, start_time_us);

    /* 메모리 해제 */
    free(time_us);
    free(voltage);

    return result.success ? 0 : 1;
}
