/**
 * 통계 기반 피부층 경계 자동 검출 알고리즘 (C 구현)
 *
 * 234개 수동 레이블 분석 결과 기반:
 * - 진피 (Dermis): T0 + 2.33±0.33 μs (1.80±0.25 mm)
 * - 근막 (Fascia): T0 + 5.16±0.79 μs (3.97±0.61 mm)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* 통계 기반 파라미터 */
#define DERMIS_EXPECTED_TIME_US 2.33
#define DERMIS_TIME_STD_US 0.33
#define FASCIA_EXPECTED_TIME_US 5.16
#define FASCIA_TIME_STD_US 0.79
#define THICKNESS_GAP_MM 2.18
#define EPIDERMIS_CUTOFF_TIME_US 1.5
#define SPEED_OF_SOUND 1540.0
#define MAX_DISTANCE_MM 6.0

/* 시작점 자동 검출 파라미터 (새 포맷 drop=1850: 데이터 시작 18.50μs) */
#define START_POINT_EXPECTED_US 18.80
#define START_POINT_STD_US 0.30
#define START_POINT_SEARCH_WINDOW_US 1.50

/* 검출 결과 구조체 */
typedef struct {
    int dermis_idx;
    int fascia_idx;
    int success;
    double actual_start_us;  /* 실제 사용된 시작점 (자동 검출된 경우 포함) */
} DetectionResult;

/* 동적 배열 구조체 */
typedef struct {
    int *data;
    int size;
    int capacity;
} IntArray;

/* IntArray 초기화 */
IntArray* create_int_array(int initial_capacity) {
    IntArray *arr = (IntArray*)malloc(sizeof(IntArray));
    arr->data = (int*)malloc(initial_capacity * sizeof(int));
    arr->size = 0;
    arr->capacity = initial_capacity;
    return arr;
}

/* IntArray에 요소 추가 */
void append_int_array(IntArray *arr, int value) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = (int*)realloc(arr->data, arr->capacity * sizeof(int));
    }
    arr->data[arr->size++] = value;
}

/* IntArray 해제 */
void free_int_array(IntArray *arr) {
    free(arr->data);
    free(arr);
}

/* 절댓값 계산 */
static inline double fabs_d(double x) {
    return x < 0.0 ? -x : x;
}

/* 최소값 */
static inline int min_i(int a, int b) {
    return a < b ? a : b;
}

/* 최대값 */
static inline int max_i(int a, int b) {
    return a > b ? a : b;
}

/* 최소값 (double) */
static inline double min_d(double a, double b) {
    return a < b ? a : b;
}

/* 최대값 (double) */
static inline double max_d(double a, double b) {
    return a > b ? a : b;
}

/**
 * 피크 검출
 */
IntArray* find_peaks(const double *signal, int n, double prominence_threshold, int min_distance) {
    IntArray *peaks = create_int_array(100);

    for (int i = 1; i < n - 1; i++) {
        /* 로컬 최대값 확인 */
        if (signal[i] > signal[i-1] && signal[i] > signal[i+1]) {
            /* Prominence 계산 */
            double left_min = signal[i];
            double right_min = signal[i];

            int left_start = max_i(0, i - 50);
            int right_end = min_i(n, i + 50);

            for (int j = i - 1; j >= left_start; j--) {
                left_min = min_d(left_min, signal[j]);
            }
            for (int j = i + 1; j < right_end; j++) {
                right_min = min_d(right_min, signal[j]);
            }

            double prominence = signal[i] - max_d(left_min, right_min);

            if (prominence >= prominence_threshold) {
                /* 거리 조건 확인 */
                int too_close = 0;
                for (int k = 0; k < peaks->size; k++) {
                    if (abs(i - peaks->data[k]) < min_distance) {
                        too_close = 1;
                        break;
                    }
                }

                if (!too_close) {
                    append_int_array(peaks, i);
                }
            }
        }
    }

    return peaks;
}

/**
 * 시작점 자동 검출
 *
 * 예상 범위(18.80±1.5 μs)에서
 * 신호의 급격한 상승이 시작되는 지점을 검출
 *
 * 새 포맷 (drop=1850): 데이터 시작 18.50μs, T0 ≈ 18.5~19.5μs
 * 검색 범위: 17.30~20.30 μs
 */
double auto_detect_start_point(const double *time_us, const double *voltage, int n) {
    if (n < 10) {
        return START_POINT_EXPECTED_US;
    }

    /* 예상 범위 설정: 평균 ± 1.0 μs */
    double search_start_us = START_POINT_EXPECTED_US - START_POINT_SEARCH_WINDOW_US;
    double search_end_us = START_POINT_EXPECTED_US + START_POINT_SEARCH_WINDOW_US;

    /* 탐색 범위 인덱스 찾기 */
    int start_idx = 0;
    int end_idx = n - 1;

    for (int i = 0; i < n; i++) {
        if (time_us[i] >= search_start_us && start_idx == 0) {
            start_idx = i;
        }
        if (time_us[i] >= search_end_us) {
            end_idx = i;
            break;
        }
    }

    if (start_idx >= end_idx - 10 || start_idx < 1) {
        /* 범위가 좁거나 시작점이 너무 앞쪽이면 기본값 사용 */
        return START_POINT_EXPECTED_US;
    }

    /* 전체 탐색 범위에서 최대 기울기 계산 */
    double max_gradient = 0.0;
    for (int i = start_idx; i < end_idx - 1; i++) {
        double abs_voltage_curr = fabs_d(voltage[i]);
        double abs_voltage_next = fabs_d(voltage[i + 1]);
        double gradient = abs_voltage_next - abs_voltage_curr;
        if (gradient > max_gradient) {
            max_gradient = gradient;
        }
    }

    /* 처음으로 큰 상승이 시작되는 지점 찾기 (최대 기울기의 30% 이상) */
    double threshold_gradient = max_gradient * 0.3;
    int best_idx = start_idx;

    for (int i = start_idx; i < end_idx - 1; i++) {
        double abs_voltage_curr = fabs_d(voltage[i]);
        double abs_voltage_next = fabs_d(voltage[i + 1]);
        double gradient = abs_voltage_next - abs_voltage_curr;

        /* 처음으로 임계값을 넘는 상승 시작점 */
        if (gradient > threshold_gradient) {
            best_idx = i;
            break;
        }
    }

    return time_us[best_idx];
}

/**
 * 변곡점 검출
 */
IntArray* find_inflection_points(const double *signal, int n) {
    IntArray *inflections = create_int_array(50);

    if (n < 3) return inflections;

    /* 1차 미분 (gradient) */
    double *gradient = (double*)malloc(n * sizeof(double));
    for (int i = 1; i < n - 1; i++) {
        gradient[i] = (signal[i+1] - signal[i-1]) / 2.0;
    }
    gradient[0] = signal[1] - signal[0];
    gradient[n-1] = signal[n-1] - signal[n-2];

    /* 2차 미분 */
    double *gradient2 = (double*)malloc(n * sizeof(double));
    for (int i = 1; i < n - 1; i++) {
        gradient2[i] = (gradient[i+1] - gradient[i-1]) / 2.0;
    }

    /* 상승 변곡점 찾기 (음수→양수, gradient > 0) */
    for (int i = 1; i < n - 1; i++) {
        if (gradient2[i-1] < 0 && gradient2[i] > 0 && gradient[i] > 0) {
            append_int_array(inflections, i);
        }
    }

    free(gradient);
    free(gradient2);

    return inflections;
}

/**
 * 통계 기반 피부층 경계 검출
 *
 * reference_start_us가 0.0이면 자동 검출
 */
DetectionResult detect_skin_boundaries(
    const double *time_us,
    const double *voltage,
    int n,
    double reference_start_us
) {
    DetectionResult result = {-1, -1, 0, reference_start_us};

    if (n < 100) return result;

    /* 시작점 자동 검출 (reference_start_us == 0.0인 경우) */
    if (reference_start_us == 0.0) {
        reference_start_us = auto_detect_start_point(time_us, voltage, n);
        result.actual_start_us = reference_start_us;  /* 자동 검출된 값 저장 */
    }

    /* 레퍼런스 시작점 찾기 */
    int start_idx = 0;
    double min_diff = fabs_d(time_us[0] - reference_start_us);
    for (int i = 1; i < n; i++) {
        double diff = fabs_d(time_us[i] - reference_start_us);
        if (diff < min_diff) {
            min_diff = diff;
            start_idx = i;
        }
    }

    /* 시작점 이후 데이터 */
    int analysis_n = n - start_idx;
    double *analysis_time_us = (double*)malloc(analysis_n * sizeof(double));
    double *analysis_voltage = (double*)malloc(analysis_n * sizeof(double));

    for (int i = 0; i < analysis_n; i++) {
        analysis_time_us[i] = time_us[start_idx + i] - reference_start_us;
        analysis_voltage[i] = voltage[start_idx + i];
    }

    /* 6mm 범위까지만 분석 */
    double max_time_us = (MAX_DISTANCE_MM / 1000.0) * 2.0 / SPEED_OF_SOUND * 1e6;
    int skin_end_idx = 0;
    for (int i = 0; i < analysis_n; i++) {
        if (analysis_time_us[i] >= max_time_us) {
            skin_end_idx = i;
            break;
        }
    }
    if (skin_end_idx == 0) skin_end_idx = analysis_n;

    if (skin_end_idx < 100) {
        free(analysis_time_us);
        free(analysis_voltage);
        return result;
    }

    /* Step 1: 표피 영역 제외 (1.5μs 이전) */
    int epidermis_end = 0;
    for (int i = 0; i < skin_end_idx; i++) {
        if (analysis_time_us[i] >= EPIDERMIS_CUTOFF_TIME_US) {
            epidermis_end = i;
            break;
        }
    }
    epidermis_end = min_i(epidermis_end, skin_end_idx - 100);

    if (epidermis_end < 10) {
        free(analysis_time_us);
        free(analysis_voltage);
        return result;
    }

    /* Step 2: 진피 검출 (2.0 ~ 2.7 μs 범위) */
    double dermis_min_time_us = DERMIS_EXPECTED_TIME_US - DERMIS_TIME_STD_US;
    double dermis_max_time_us = DERMIS_EXPECTED_TIME_US + DERMIS_TIME_STD_US;

    /* 탐색 범위 설정 */
    int dermis_min_idx = 0;
    int dermis_max_idx = skin_end_idx - 1;
    for (int i = 0; i < skin_end_idx; i++) {
        if (analysis_time_us[i] >= dermis_min_time_us && dermis_min_idx == 0) {
            dermis_min_idx = i;
        }
        if (analysis_time_us[i] >= dermis_max_time_us) {
            dermis_max_idx = i;
            break;
        }
    }

    int search_start = max_i(epidermis_end, dermis_min_idx - 50);
    int search_end = min_i(skin_end_idx, dermis_max_idx + 100);

    if (search_start >= search_end - 20) {
        free(analysis_time_us);
        free(analysis_voltage);
        return result;
    }

    /* 예상 범위 내 조정 */
    search_start = max_i(search_start, dermis_min_idx - 20);
    search_end = min_i(search_end, dermis_max_idx + 20);

    if (search_start >= search_end - 20) {
        free(analysis_time_us);
        free(analysis_voltage);
        return result;
    }

    /* 탐색 영역 추출 및 절댓값 변환 */
    int search_n = search_end - search_start;
    double *abs_search = (double*)malloc(search_n * sizeof(double));
    for (int i = 0; i < search_n; i++) {
        abs_search[i] = fabs_d(analysis_voltage[search_start + i]);
    }

    /* 변곡점 검출 */
    IntArray *inflection_points = find_inflection_points(abs_search, search_n);

    int dermis_idx = -1;
    if (inflection_points->size > 0) {
        /* 예상 시간에 가장 가까운 변곡점 선택 */
        int best_inflection = inflection_points->data[0];
        double best_diff = fabs_d(analysis_time_us[search_start + best_inflection] - DERMIS_EXPECTED_TIME_US);

        for (int k = 0; k < inflection_points->size; k++) {
            int ip = inflection_points->data[k];
            double ip_time = analysis_time_us[search_start + ip];
            if (ip_time >= dermis_min_time_us && ip_time <= dermis_max_time_us) {
                double time_diff = fabs_d(ip_time - DERMIS_EXPECTED_TIME_US);
                if (time_diff < best_diff) {
                    best_diff = time_diff;
                    best_inflection = ip;
                }
            }
        }
        dermis_idx = search_start + best_inflection;
    } else {
        /* 변곡점이 없으면 피크 사용 */
        IntArray *peaks = find_peaks(abs_search, search_n, 0.0, 10);
        if (peaks->size > 0) {
            int best_peak = peaks->data[0];
            double best_diff = fabs_d(analysis_time_us[search_start + best_peak] - DERMIS_EXPECTED_TIME_US);

            for (int k = 0; k < peaks->size; k++) {
                int peak = peaks->data[k];
                double peak_time = analysis_time_us[search_start + peak];
                double time_diff = fabs_d(peak_time - DERMIS_EXPECTED_TIME_US);
                if (time_diff < best_diff) {
                    best_diff = time_diff;
                    best_peak = peak;
                }
            }
            dermis_idx = search_start + best_peak;
        } else {
            /* 예상 시간 위치 사용 */
            for (int i = 0; i < skin_end_idx; i++) {
                if (analysis_time_us[i] >= DERMIS_EXPECTED_TIME_US) {
                    dermis_idx = i;
                    break;
                }
            }
        }
        free_int_array(peaks);
    }

    free_int_array(inflection_points);
    free(abs_search);

    if (dermis_idx < 0) {
        free(analysis_time_us);
        free(analysis_voltage);
        return result;
    }

    /* Step 3: 근막 검출 (4.4 ~ 5.9 μs 또는 진피 + 2.18mm) */
    double fascia_min_time_us = FASCIA_EXPECTED_TIME_US - FASCIA_TIME_STD_US;
    double fascia_max_time_us = FASCIA_EXPECTED_TIME_US + FASCIA_TIME_STD_US;

    /* 진피로부터의 예상 거리 */
    double expected_gap_time_us = (THICKNESS_GAP_MM / 1000.0) * 2.0 / SPEED_OF_SOUND * 1e6;
    double dermis_time_us = analysis_time_us[dermis_idx];
    double fascia_from_dermis_us = dermis_time_us + expected_gap_time_us;

    /* 두 예상값의 평균 */
    double fascia_expected_us = (FASCIA_EXPECTED_TIME_US + fascia_from_dermis_us) / 2.0;

    /* 탐색 범위 */
    int fascia_search_start = dermis_idx + 20;
    int fascia_min_idx = 0;
    int fascia_max_idx = skin_end_idx - 1;

    for (int i = 0; i < skin_end_idx; i++) {
        if (analysis_time_us[i] >= fascia_min_time_us && fascia_min_idx == 0) {
            fascia_min_idx = i;
        }
        if (analysis_time_us[i] >= fascia_max_time_us) {
            fascia_max_idx = i;
            break;
        }
    }

    fascia_search_start = max_i(fascia_search_start, fascia_min_idx - 30);
    int fascia_search_end = min_i(skin_end_idx, fascia_max_idx + 50);

    int fascia_idx = -1;
    if (fascia_search_start >= fascia_search_end - 10) {
        /* 범위가 좁으면 진피 기반 추정 */
        double time_step = analysis_time_us[1] - analysis_time_us[0];
        fascia_idx = dermis_idx + (int)(expected_gap_time_us / time_step);
        if (fascia_idx >= skin_end_idx) {
            free(analysis_time_us);
            free(analysis_voltage);
            return result;
        }
    } else {
        /* 피크 검출 */
        int fascia_search_n = fascia_search_end - fascia_search_start;
        double *abs_fascia_search = (double*)malloc(fascia_search_n * sizeof(double));
        for (int i = 0; i < fascia_search_n; i++) {
            abs_fascia_search[i] = fabs_d(analysis_voltage[fascia_search_start + i]);
        }

        /* 표준편차 계산 */
        double mean = 0.0;
        for (int i = 0; i < fascia_search_n; i++) {
            mean += abs_fascia_search[i];
        }
        mean /= fascia_search_n;

        double sq_sum = 0.0;
        for (int i = 0; i < fascia_search_n; i++) {
            double diff = abs_fascia_search[i] - mean;
            sq_sum += diff * diff;
        }
        double std_val = sqrt(sq_sum / fascia_search_n);

        IntArray *peaks = find_peaks(abs_fascia_search, fascia_search_n, std_val * 0.3, 10);

        if (peaks->size > 0) {
            /* 예상 시간에 가장 가까운 피크 */
            int best_peak = peaks->data[0];
            double best_diff = fabs_d(analysis_time_us[fascia_search_start + best_peak] - fascia_expected_us);

            for (int k = 0; k < peaks->size; k++) {
                int peak = peaks->data[k];
                double peak_time = analysis_time_us[fascia_search_start + peak];
                double time_diff = fabs_d(peak_time - fascia_expected_us);
                if (time_diff < best_diff) {
                    best_diff = time_diff;
                    best_peak = peak;
                }
            }
            fascia_idx = fascia_search_start + best_peak;
        } else {
            /* 피크가 없으면 예상 위치 사용 */
            for (int i = 0; i < skin_end_idx; i++) {
                if (analysis_time_us[i] >= fascia_expected_us) {
                    fascia_idx = i;
                    break;
                }
            }
        }

        free_int_array(peaks);
        free(abs_fascia_search);
    }

    free(analysis_time_us);
    free(analysis_voltage);

    if (fascia_idx < 0) return result;

    result.dermis_idx = dermis_idx;
    result.fascia_idx = fascia_idx;
    result.success = 1;

    return result;
}

/* Python 바인딩을 위한 함수 */
DetectionResult detect_boundaries_c(
    const double *time_us,
    const double *voltage,
    int n,
    double reference_start_us
) {
    return detect_skin_boundaries(time_us, voltage, n, reference_start_us);
}
