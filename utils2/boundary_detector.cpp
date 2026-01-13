/**
 * 통계 기반 피부층 경계 자동 검출 알고리즘
 *
 * 234개 수동 레이블 분석 결과 기반:
 * - 진피 (Dermis): T0 + 2.33±0.33 μs (1.80±0.25 mm)
 * - 근막 (Fascia): T0 + 5.16±0.79 μs (3.97±0.61 mm)
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

// 통계 기반 파라미터
const double DERMIS_EXPECTED_TIME_US = 2.33;
const double DERMIS_TIME_STD_US = 0.33;
const double FASCIA_EXPECTED_TIME_US = 5.16;
const double FASCIA_TIME_STD_US = 0.79;
const double THICKNESS_GAP_MM = 2.18;
const double EPIDERMIS_CUTOFF_TIME_US = 1.5;
const double SPEED_OF_SOUND = 1540.0;  // m/s
const double MAX_DISTANCE_MM = 6.0;

struct DetectionResult {
    int dermis_idx;
    int fascia_idx;
    bool success;
};

/**
 * 피크 검출
 */
std::vector<int> find_peaks(const std::vector<double>& signal,
                            double prominence_threshold,
                            int min_distance) {
    std::vector<int> peaks;
    int n = signal.size();

    for (int i = 1; i < n - 1; i++) {
        // 로컬 최대값 확인
        if (signal[i] > signal[i-1] && signal[i] > signal[i+1]) {
            // Prominence 계산 (간단한 버전)
            double left_min = signal[i];
            double right_min = signal[i];

            for (int j = i - 1; j >= std::max(0, i - 50); j--) {
                left_min = std::min(left_min, signal[j]);
            }
            for (int j = i + 1; j < std::min(n, i + 50); j++) {
                right_min = std::min(right_min, signal[j]);
            }

            double prominence = signal[i] - std::max(left_min, right_min);

            if (prominence >= prominence_threshold) {
                // 거리 조건 확인
                bool too_close = false;
                for (int peak : peaks) {
                    if (std::abs(i - peak) < min_distance) {
                        too_close = true;
                        break;
                    }
                }

                if (!too_close) {
                    peaks.push_back(i);
                }
            }
        }
    }

    return peaks;
}

/**
 * 변곡점 검출
 */
std::vector<int> find_inflection_points(const std::vector<double>& signal) {
    std::vector<int> inflections;
    int n = signal.size();

    if (n < 3) return inflections;

    // 1차 미분 (gradient)
    std::vector<double> gradient(n);
    for (int i = 1; i < n - 1; i++) {
        gradient[i] = (signal[i+1] - signal[i-1]) / 2.0;
    }
    gradient[0] = signal[1] - signal[0];
    gradient[n-1] = signal[n-1] - signal[n-2];

    // 2차 미분
    std::vector<double> gradient2(n);
    for (int i = 1; i < n - 1; i++) {
        gradient2[i] = (gradient[i+1] - gradient[i-1]) / 2.0;
    }

    // 상승 변곡점 찾기 (음수→양수, gradient > 0)
    for (int i = 1; i < n - 1; i++) {
        if (gradient2[i-1] < 0 && gradient2[i] > 0 && gradient[i] > 0) {
            inflections.push_back(i);
        }
    }

    return inflections;
}

/**
 * 통계 기반 피부층 경계 검출
 */
DetectionResult detect_skin_boundaries(
    const std::vector<double>& time_us,
    const std::vector<double>& voltage,
    double reference_start_us
) {
    DetectionResult result = {-1, -1, false};

    int n = time_us.size();
    if (n < 100) return result;

    // 레퍼런스 시작점 찾기
    int start_idx = 0;
    double min_diff = std::abs(time_us[0] - reference_start_us);
    for (int i = 1; i < n; i++) {
        double diff = std::abs(time_us[i] - reference_start_us);
        if (diff < min_diff) {
            min_diff = diff;
            start_idx = i;
        }
    }

    // 시작점 이후 데이터
    std::vector<double> analysis_time_us;
    std::vector<double> analysis_voltage;
    for (int i = start_idx; i < n; i++) {
        analysis_time_us.push_back(time_us[i] - reference_start_us);
        analysis_voltage.push_back(voltage[i]);
    }

    // 6mm 범위까지만 분석
    double max_time_us = (MAX_DISTANCE_MM / 1000.0) * 2.0 / SPEED_OF_SOUND * 1e6;
    int skin_end_idx = 0;
    for (int i = 0; i < analysis_time_us.size(); i++) {
        if (analysis_time_us[i] >= max_time_us) {
            skin_end_idx = i;
            break;
        }
    }
    if (skin_end_idx == 0) skin_end_idx = analysis_time_us.size();

    std::vector<double> skin_voltage(analysis_voltage.begin(),
                                     analysis_voltage.begin() + skin_end_idx);

    if (skin_voltage.size() < 100) return result;

    // Step 1: 표피 영역 제외 (1.5μs 이전)
    int epidermis_end = 0;
    for (int i = 0; i < analysis_time_us.size() && i < skin_voltage.size(); i++) {
        if (analysis_time_us[i] >= EPIDERMIS_CUTOFF_TIME_US) {
            epidermis_end = i;
            break;
        }
    }
    epidermis_end = std::min(epidermis_end, (int)skin_voltage.size() - 100);

    if (epidermis_end < 10) return result;

    // Step 2: 진피 검출 (2.0 ~ 2.7 μs 범위)
    double dermis_min_time_us = DERMIS_EXPECTED_TIME_US - DERMIS_TIME_STD_US;
    double dermis_max_time_us = DERMIS_EXPECTED_TIME_US + DERMIS_TIME_STD_US;

    // 탐색 범위 설정
    int dermis_min_idx = 0;
    int dermis_max_idx = skin_voltage.size() - 1;
    for (int i = 0; i < analysis_time_us.size() && i < skin_voltage.size(); i++) {
        if (analysis_time_us[i] >= dermis_min_time_us && dermis_min_idx == 0) {
            dermis_min_idx = i;
        }
        if (analysis_time_us[i] >= dermis_max_time_us) {
            dermis_max_idx = i;
            break;
        }
    }

    int search_start = std::max(epidermis_end, dermis_min_idx - 50);
    int search_end = std::min((int)skin_voltage.size(), dermis_max_idx + 100);

    if (search_start >= search_end - 20) return result;

    // 예상 범위 내 조정
    int expected_start_idx = dermis_min_idx;
    int expected_end_idx = dermis_max_idx;

    search_start = std::max(search_start, expected_start_idx - 20);
    search_end = std::min(search_end, expected_end_idx + 20);

    if (search_start >= search_end - 20) return result;

    // 탐색 영역 추출
    std::vector<double> search_region(skin_voltage.begin() + search_start,
                                     skin_voltage.begin() + search_end);

    // 절댓값 변환
    std::vector<double> abs_search(search_region.size());
    for (int i = 0; i < search_region.size(); i++) {
        abs_search[i] = std::abs(search_region[i]);
    }

    // 변곡점 검출
    std::vector<int> inflection_points = find_inflection_points(abs_search);

    int dermis_idx = -1;
    if (!inflection_points.empty()) {
        // 예상 시간에 가장 가까운 변곡점 선택
        int best_inflection = inflection_points[0];
        double best_diff = std::abs(analysis_time_us[search_start + best_inflection]
                                   - DERMIS_EXPECTED_TIME_US);

        for (int ip : inflection_points) {
            double ip_time = analysis_time_us[search_start + ip];
            if (ip_time >= dermis_min_time_us && ip_time <= dermis_max_time_us) {
                double time_diff = std::abs(ip_time - DERMIS_EXPECTED_TIME_US);
                if (time_diff < best_diff) {
                    best_diff = time_diff;
                    best_inflection = ip;
                }
            }
        }
        dermis_idx = search_start + best_inflection;
    } else {
        // 변곡점이 없으면 피크 사용
        std::vector<int> peaks = find_peaks(abs_search, 0.0, 10);
        if (!peaks.empty()) {
            // 예상 시간에 가장 가까운 피크
            int best_peak = peaks[0];
            double best_diff = std::abs(analysis_time_us[search_start + best_peak]
                                       - DERMIS_EXPECTED_TIME_US);

            for (int peak : peaks) {
                double peak_time = analysis_time_us[search_start + peak];
                double time_diff = std::abs(peak_time - DERMIS_EXPECTED_TIME_US);
                if (time_diff < best_diff) {
                    best_diff = time_diff;
                    best_peak = peak;
                }
            }
            dermis_idx = search_start + best_peak;
        } else {
            // 예상 시간 위치 사용
            for (int i = 0; i < analysis_time_us.size() && i < skin_voltage.size(); i++) {
                if (analysis_time_us[i] >= DERMIS_EXPECTED_TIME_US) {
                    dermis_idx = i;
                    break;
                }
            }
        }
    }

    if (dermis_idx < 0) return result;

    // Step 3: 근막 검출 (4.4 ~ 5.9 μs 또는 진피 + 2.18mm)
    double fascia_min_time_us = FASCIA_EXPECTED_TIME_US - FASCIA_TIME_STD_US;
    double fascia_max_time_us = FASCIA_EXPECTED_TIME_US + FASCIA_TIME_STD_US;

    // 진피로부터의 예상 거리
    double expected_gap_time_us = (THICKNESS_GAP_MM / 1000.0) * 2.0 / SPEED_OF_SOUND * 1e6;
    double dermis_time_us = analysis_time_us[dermis_idx];
    double fascia_from_dermis_us = dermis_time_us + expected_gap_time_us;

    // 두 예상값의 평균
    double fascia_expected_us = (FASCIA_EXPECTED_TIME_US + fascia_from_dermis_us) / 2.0;

    // 탐색 범위
    int fascia_search_start = dermis_idx + 20;
    int fascia_min_idx = 0;
    int fascia_max_idx = skin_voltage.size() - 1;

    for (int i = 0; i < analysis_time_us.size() && i < skin_voltage.size(); i++) {
        if (analysis_time_us[i] >= fascia_min_time_us && fascia_min_idx == 0) {
            fascia_min_idx = i;
        }
        if (analysis_time_us[i] >= fascia_max_time_us) {
            fascia_max_idx = i;
            break;
        }
    }

    fascia_search_start = std::max(fascia_search_start, fascia_min_idx - 30);
    int fascia_search_end = std::min((int)skin_voltage.size(), fascia_max_idx + 50);

    int fascia_idx = -1;
    if (fascia_search_start >= fascia_search_end - 10) {
        // 범위가 좁으면 진피 기반 추정
        double time_step = analysis_time_us[1] - analysis_time_us[0];
        fascia_idx = dermis_idx + (int)(expected_gap_time_us / time_step);
        if (fascia_idx >= skin_voltage.size()) {
            return result;
        }
    } else {
        // 피크 검출
        std::vector<double> fascia_search_region(skin_voltage.begin() + fascia_search_start,
                                                 skin_voltage.begin() + fascia_search_end);
        std::vector<double> abs_fascia_search(fascia_search_region.size());
        for (int i = 0; i < fascia_search_region.size(); i++) {
            abs_fascia_search[i] = std::abs(fascia_search_region[i]);
        }

        // 표준편차 계산
        double mean = std::accumulate(abs_fascia_search.begin(), abs_fascia_search.end(), 0.0)
                     / abs_fascia_search.size();
        double sq_sum = 0.0;
        for (double val : abs_fascia_search) {
            sq_sum += (val - mean) * (val - mean);
        }
        double std_val = std::sqrt(sq_sum / abs_fascia_search.size());

        std::vector<int> peaks = find_peaks(abs_fascia_search, std_val * 0.3, 10);

        if (!peaks.empty()) {
            // 예상 시간에 가장 가까운 피크
            int best_peak = peaks[0];
            double best_diff = std::abs(analysis_time_us[fascia_search_start + best_peak]
                                       - fascia_expected_us);

            for (int peak : peaks) {
                double peak_time = analysis_time_us[fascia_search_start + peak];
                double time_diff = std::abs(peak_time - fascia_expected_us);
                if (time_diff < best_diff) {
                    best_diff = time_diff;
                    best_peak = peak;
                }
            }
            fascia_idx = fascia_search_start + best_peak;
        } else {
            // 피크가 없으면 예상 위치 사용
            for (int i = 0; i < analysis_time_us.size() && i < skin_voltage.size(); i++) {
                if (analysis_time_us[i] >= fascia_expected_us) {
                    fascia_idx = i;
                    break;
                }
            }
        }
    }

    if (fascia_idx < 0) return result;

    result.dermis_idx = dermis_idx;
    result.fascia_idx = fascia_idx;
    result.success = true;

    return result;
}

// Python 바인딩을 위한 extern "C" 함수
extern "C" {
    DetectionResult detect_boundaries_c(
        const double* time_us,
        const double* voltage,
        int n,
        double reference_start_us
    ) {
        std::vector<double> time_vec(time_us, time_us + n);
        std::vector<double> voltage_vec(voltage, voltage + n);
        return detect_skin_boundaries(time_vec, voltage_vec, reference_start_us);
    }
}
