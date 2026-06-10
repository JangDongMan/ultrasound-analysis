#!/usr/bin/env python3
"""
auto_mark_usdata.py  v4
신호 기반 뼈 감지: Hilbert 엔벨로프 + 진피/근막 구간 대비 비율 판단

변경 사항 (v3 → v4):
  - 뼈 감지 모드 선택 가능: --bone-mode signal(기본)|filename|none
    signal  : 파일명 무관, 순수 신호 분석으로 뼈 판별
              (진피/근막 구간 최대값 대비 BONE_PRE_RATIO 이상 + BONE_MIN_AMP 이상)
    filename: 파일명 위치 키워드(NO_BONE_KEYWORDS)로 탐색 제외
    none    : 필터 없이 모든 파일 탐색

사용법:
  python auto_mark_usdata.py                             # signal 모드, 미마킹만
  python auto_mark_usdata.py --overwrite                 # signal 모드, 덮어쓰기
  python auto_mark_usdata.py --bone-mode filename        # 파일명 기반 필터
  python auto_mark_usdata.py --bone-mode none            # 필터 없음
  python auto_mark_usdata.py --dry-run                   # 목록만 출력
"""

import os, sys, json, argparse, re
import numpy as np
from scipy.signal import hilbert, find_peaks

# ─── 경로 설정 ────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# usdata/data 자동 탐지 (서버: project/usdata/data, 로컬: project/data)
_c1 = os.path.join(PROJECT_DIR, "usdata", "data")
_c2 = os.path.join(PROJECT_DIR, "data")
USDATA = _c1 if os.path.isdir(_c1) else _c2

# ─── 신호 파라미터 ────────────────────────────────────────────
TRIM_START     = 1850
TRIM_COUNT     = 2050
DISPLAY_OFFSET = 18.50   # μs (트림 후 첫 샘플의 절대 시간)
SAMPLE_NS      = 10      # ns/sample
SPEED_OF_SOUND = 1540.0  # m/s

# ─── 검출 파라미터 (boundary_detector.c 동일) ─────────────────
T0_THRESHOLD          = 100.0   # 힐버트 엔벨로프 T0 임계값
EPIDERMIS_CUTOFF_US   = 1.5     # 표피 제외 구간 (μs)
DERMIS_EXPECTED_US    = 2.33    # 진피 예상 시간 (T0 기준 μs)
DERMIS_STD_US         = 0.33
FASCIA_EXPECTED_US    = 5.16    # 근막 예상 시간 (T0 기준 μs)
FASCIA_STD_US         = 0.79
MAX_DISTANCE_MM       = 6.0     # 분석 최대 깊이 (mm) — 뼈 미검출 시 폴백용

# ─── 뼈 검출 파라미터 ─────────────────────────────────────────
BONE_SEARCH_MIN_US    = 6.0     # T0 기준 뼈 탐색 시작 μs (≈4.6mm)
BONE_SEARCH_MAX_US    = 11.5    # T0 기준 뼈 탐색 종료 μs (≈8.9mm) — 12μs 이후 시스템 artifact 제거
BONE_MARGIN_US        = 1.0     # 뼈-근막 최소 이격 μs (≈0.77mm)

# ADC 포화 기반 얕은 뼈 검출 (이마에서 뼈가 4.5~6μs에 있는 케이스)
BONE_SAT_SEARCH_MIN   = 4.0     # 포화 탐색 시작 μs
BONE_SAT_THRESHOLD    = 125     # |raw| >= 이 값 → ADC 포화로 판단
BONE_SAT_MIN_COUNT    = 8       # 포화 샘플 최소 개수 (잡음성 포화 제거용)

# signal 모드: 신호 기반 뼈 판별 파라미터
BONE_MIN_AMP          = 25.0    # 뼈 후보 최소 절대 진폭 (Hilbert 엔벨로프)
BONE_PRE_RATIO        = 0.40    # 뼈 진폭 / 진피-근막 구간 최대값 비율 하한
# signal 모드: 뼈를 근막 탐색 기준으로 사용할 최소 max_prom
# 이마 95.5% 유지, 볼/잎술 24.5% 오참조 제거 (max_prom 분리도 기반)
BONE_USE_REF_MIN_PROM = 30.0
# 4.0~6.0μs 구간에서 ADC 포화 없이도 뼈로 인정할 최소 prominence
# 연조직 에코(20~60) 제외, 뼈 강반사(80~150)만 허용
BONE_HIGH_PROM        = 80.0
# 조기 구간(4.0~6.0μs) 뼈: 정상 구간(6.0~11.5μs) 최강 prom 대비 이 배수 이상이어야 뼈로 인정
# (정상 구간에 강한 뼈가 있으면 조기 에코는 연조직으로 처리)
BONE_EARLY_DOMINANCE  = 2.0
# 정상 구간(6.0~11.5μs) 뼈 SNR: 구간 median 대비 피크 진폭의 최소 비율
# 구간 전체가 균질한 잡음(median=10~15, max=25)인 경우 오검출 방지
BONE_ZONE_SNR_MIN     = 2.5

# filename 모드: 탐색 제외 키워드
NO_BONE_KEYWORDS      = ('볼', '잎술', '입술', '인중', '코옆')

DERMIS_MAX_BONE_RATIO = 0.40    # 진피 탐색 상한: 뼈 깊이의 40% 이내
FASCIA_WIDER_STD_US   = 1.5     # 이마형: 뼈 기준 근막 허용 범위 확장 (±μs)

# ─── v3 추가: 뼈 미검출 시 확장 탐색 ─────────────────────────
FASCIA_FAR_MAX_US     = 9.0     # 뼈 미검출 시 근막 탐색 최대 μs (볼·입술 대응)


# ─── CSV 로드 ─────────────────────────────────────────────────
def load_adc(filepath):
    adc = []
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                v = int(line)
                if 0 <= v <= 255:
                    adc.append(v)
            except ValueError:
                pass

    adc = np.array(adc, dtype=np.float64)

    if len(adc) > TRIM_COUNT:
        end = TRIM_START + TRIM_COUNT
        adc = adc[TRIM_START:end] if len(adc) >= end else adc[TRIM_START:]

    n       = len(adc)
    time_us = np.arange(n) * SAMPLE_NS / 1000.0 + DISPLAY_OFFSET
    return adc, time_us


# ─── T0 검출 (힐버트 엔벨로프) ───────────────────────────────
def detect_t0(adc, time_us):
    env = np.abs(hilbert(adc - 128.0))
    for i, v in enumerate(env):
        if v >= T0_THRESHOLD:
            return time_us[i]
    return None


# ─── 뼈 에코 검출 ────────────────────────────────────────────
def detect_bone(signal, rel_t, use_ratio=True):
    """
    뼈 에코 검출 — Hilbert 엔벨로프 기반

    우선순위 1 (ADC 포화 탐지):
        4.0~6.0μs 구간에 |raw| >= BONE_SAT_THRESHOLD 인 샘플이
        BONE_SAT_MIN_COUNT개 이상이면 → 해당 구간 Hilbert 최대 지점을 뼈로 반환.
        (이마 얕은 뼈: ADC 포화 ≈ 뼈 강반사의 물리적 증거)

    우선순위 2 (일반 prominence 탐지):
        BONE_SEARCH_MIN_US~BONE_SEARCH_MAX_US 구간에서 prominence 기반 탐색.

    Returns: (bone_rel_us, bone_amp, max_prom, first_prom)  또는  (None, None, 0, 0)
    """
    if len(signal) < 10:
        return None, None, 0.0, 0.0

    env = np.abs(hilbert(signal))  # Hilbert 엔벨로프: 5MHz 반주기 진동 제거

    # ── 우선순위 1: ADC 포화 기반 얕은 뼈 탐지 ─────────────────
    si0 = int(np.searchsorted(rel_t, BONE_SAT_SEARCH_MIN))
    si1 = int(np.searchsorted(rel_t, BONE_SEARCH_MIN_US))
    if si1 > si0:
        sat_seg = signal[si0:si1]
        sat_count = int(np.sum(np.abs(sat_seg) >= BONE_SAT_THRESHOLD))
        if sat_count >= BONE_SAT_MIN_COUNT:
            sat_env = env[si0:si1]
            peak_in_sat = int(np.argmax(sat_env))
            bone_rel_us = float(rel_t[si0 + peak_in_sat])
            bone_amp    = float(sat_env[peak_in_sat])
            # first_prom=0 → bone_strong=False → 확장 근막 탐색 경로 사용
            # (얕은 뼈: BONE_MARGIN_US 제약으로 근막 창이 좁아지는 것 방지)
            return bone_rel_us, bone_amp, float(sat_count), 0.0

    # ── 우선순위 2: 전체 구간 prominence 기반 탐지 ──────────────
    # 4.0~11.5μs 전체를 한 번에 탐색해 prominence를 정확히 계산.
    # 경계 6.0μs에서 잘라내면 그 직전 강에코의 꼬리가 기준점이 되어
    # 6.03μs 피크의 within-segment prominence가 허위로 낮아지는 문제 방지.
    if use_ratio:
        pre_start = int(np.searchsorted(rel_t, EPIDERMIS_CUTOFF_US))
        pre_end   = int(np.searchsorted(rel_t, BONE_SEARCH_MIN_US))
        pre_max   = float(np.max(env[pre_start:pre_end])) if pre_end > pre_start else 0.0
    else:
        pre_max = 0.0

    bi0 = int(np.searchsorted(rel_t, BONE_SAT_SEARCH_MIN))   # 4.0μs
    bi1 = int(np.searchsorted(rel_t, BONE_SEARCH_MAX_US))    # 11.5μs
    bi1 = min(bi1, len(rel_t) - 1)
    if bi0 >= bi1:
        return None, None, 0.0, 0.0

    seg   = env[bi0:bi1 + 1]
    t_seg = rel_t[bi0:bi1 + 1]

    noise_level = max(float(np.percentile(env, 20)), 5.0)
    threshold   = noise_level * 3.0

    # 낮은 임계값으로 후보 피크를 모두 추출
    peaks, props = find_peaks(seg, prominence=threshold / 2, distance=10)
    max_prom = float(props['prominences'].max()) if len(peaks) > 0 else 0.0

    # 정상 구간(6.0~11.5μs) 통계 계산
    # - bone_zone_thr: 20th percentile × 3 (prominence 임계값)
    # - zone_median: median (SNR 기준선 — 구간이 균질 잡음인지 판별)
    norm_seg_start = int(np.searchsorted(t_seg, BONE_SEARCH_MIN_US))
    if norm_seg_start < len(seg):
        norm_zone = seg[norm_seg_start:]
        bone_zone_noise = max(float(np.percentile(norm_zone, 20)), 5.0)
        zone_median     = max(float(np.median(norm_zone)), 5.0)
    else:
        bone_zone_noise = noise_level
        zone_median     = noise_level
    bone_zone_thr = bone_zone_noise * 3.0

    # 정상 구간(6.0~11.5μs) 최대 prominence 계산 — 조기 구간 우선 탐지 판단용
    norm_max_prom = 0.0
    for i, p in enumerate(peaks):
        if float(t_seg[p]) >= BONE_SEARCH_MIN_US:
            prom = float(props['prominences'][i])
            if prom > norm_max_prom:
                norm_max_prom = prom

    for i, p in enumerate(peaks):
        peak_t     = float(t_seg[p])
        amp        = float(seg[p])
        first_prom = float(props['prominences'][i])

        if peak_t < BONE_SEARCH_MIN_US:
            # 조기 구간(4.0~6.0μs): 두 조건 모두 충족 시 뼈로 허용
            # 조건1: 고 prominence — 연조직 에코(prom 20~60) 제외
            if first_prom < BONE_HIGH_PROM:
                continue
            # 조건2: 정상 구간 최강 prom 대비 BONE_EARLY_DOMINANCE배 이상
            #        정상 구간에 강한 뼈가 있으면 조기 에코는 연조직으로 처리
            if norm_max_prom > 0 and first_prom < norm_max_prom * BONE_EARLY_DOMINANCE:
                continue
        else:
            # 정상 구간(6.0~11.5μs): 로컬 노이즈 임계값 + SNR 이중 검사
            # 구간 전체가 균질한 잡음(SNR < BONE_ZONE_SNR_MIN)이면 오검출 방지
            if first_prom < bone_zone_thr:
                continue
            if amp < zone_median * BONE_ZONE_SNR_MIN:
                continue
            if use_ratio:
                if amp < BONE_MIN_AMP:
                    continue
                if pre_max > 0 and amp < pre_max * BONE_PRE_RATIO:
                    continue

        return peak_t, amp, max_prom, first_prom

    return None, None, 0.0, 0.0


# ─── 피부층 경계 검출 (Python 구현) ──────────────────────────
def detect_boundaries(adc, time_us, t0_us, bone_mode="signal"):
    """
    T0 이후 신호에서 진피(Dermis)·근막(Fascia) 경계 검출
    boundary_detector.c 알고리즘과 동일한 파라미터 사용

    Returns:
        dict with dermis_rel_us, dermis_mm, fascia_rel_us, fascia_mm,
                   bone_rel_us, bone_mm (뼈 미검출 시 None)
        or None if detection failed
    """
    # T0 이후 상대 시간으로 변환
    mask = time_us >= t0_us
    if mask.sum() < 100:
        return None

    rel_t  = time_us[mask] - t0_us   # 0부터 시작하는 상대 시간
    signal = adc[mask] - 128.0       # 중심 이동
    abs_sig = np.abs(signal)

    # ── Step 1: 뼈 에코 검출 ──────────────────────────────────
    if bone_mode == "skip":
        bone_rel_us, bone_amp, bone_max_prom, bone_first_prom = None, None, 0.0, 0.0
    else:  # "signal" / "none" — Hilbert prominence 임계값만 사용, 비율 필터 없음
        bone_rel_us, bone_amp, bone_max_prom, bone_first_prom = detect_bone(signal, rel_t, use_ratio=False)

    # signal 모드: 선택된 첫 피크 자체의 prominence로 강도 판단
    # max_prom(창 전체 최대)이 아닌 first_prom을 사용해야
    # "노이즈 스파이크가 선택되었지만 다른 강한 피크 때문에 max_prom 통과" 오류를 방지
    if bone_mode == "signal" and bone_rel_us is not None:
        bone_strong = bone_first_prom >= BONE_USE_REF_MIN_PROM
    else:
        bone_strong = True  # filename/none 모드는 기존대로

    bone_mm = bone_rel_us * SPEED_OF_SOUND / 2 / 1000 if bone_rel_us is not None else None

    # 분석 범위:
    #   - 강한 뼈 OR 포화 탐지 얕은 뼈 → 뼈 이전까지 제한
    #   - 약한 뼈(연조직 오검출 가능성) → 폴백 거리
    bone_is_shallow = (bone_rel_us is not None) and (bone_rel_us < BONE_SEARCH_MIN_US)
    if bone_rel_us is not None and (bone_strong or bone_is_shallow):
        max_t_us = bone_rel_us         # 뼈 에코 이전까지만 분석
    else:
        max_t_us = MAX_DISTANCE_MM / SPEED_OF_SOUND * 2 * 1e6

    end_idx = np.searchsorted(rel_t, max_t_us)
    if end_idx < 100:
        end_idx = len(rel_t)
    rel_t_analysis  = rel_t[:end_idx]
    abs_sig_analysis = abs_sig[:end_idx]

    # ── Step 2: 진피 검출 ─────────────────────────────────────
    d_min = DERMIS_EXPECTED_US - DERMIS_STD_US
    if bone_rel_us is not None and bone_rel_us >= BONE_SEARCH_MIN_US:
        # 일반 뼈 깊이: 뼈 깊이의 40% 이내로 진피 상한 제한
        d_max = min(DERMIS_EXPECTED_US + DERMIS_STD_US,
                    bone_rel_us * DERMIS_MAX_BONE_RATIO)
    else:
        # 얕은 뼈(ADC 포화 탐지) 또는 뼈 없음: 비율 제약 없이 표준 범위 사용
        d_max = DERMIS_EXPECTED_US + DERMIS_STD_US

    d_i0 = int(np.searchsorted(rel_t_analysis, d_min))
    d_i1 = int(np.searchsorted(rel_t_analysis, d_max))
    d_i1 = min(d_i1, len(rel_t_analysis) - 1)

    if d_i0 >= d_i1:
        return None

    seg = abs_sig_analysis[d_i0:d_i1]

    # 변곡점(2차 미분 부호 전환) 우선, 없으면 최댓값 지점
    grad1   = np.gradient(seg)
    grad2   = np.gradient(grad1)
    inflect = np.where((grad2[:-1] < 0) & (grad2[1:] > 0) & (grad1[1:] > 0))[0]

    if len(inflect) > 0:
        best_local = inflect[np.argmin(np.abs(rel_t_analysis[d_i0 + inflect] - DERMIS_EXPECTED_US))]
        dermis_idx = d_i0 + best_local
    else:
        dermis_idx = d_i0 + int(np.argmax(seg))

    dermis_rel_us = float(rel_t_analysis[dermis_idx])
    dermis_mm     = dermis_rel_us * SPEED_OF_SOUND / 2 / 1000

    # ── Step 3: 근막 탐색 창 결정 ────────────────────────────────
    # 2-way 분기:
    #   A) 뼈 검출: AND 제약 (뼈 거리 무관) — v2와 동일
    #   B) 뼈 미검출: 최대 FASCIA_FAR_MAX_US까지 확장 (볼·입술 대응)
    if bone_rel_us is not None and bone_strong and (bone_rel_us - dermis_rel_us) > 2.5:
        # A) 강한 뼈 에코 검출: 뼈 기준 근막 탐색 창 제한
        f_min = max(dermis_rel_us + 1.5,
                    FASCIA_EXPECTED_US - FASCIA_WIDER_STD_US)
        f_max = min(bone_rel_us - BONE_MARGIN_US,
                    FASCIA_EXPECTED_US + FASCIA_WIDER_STD_US)
        if f_max - f_min < 0.5:   # 창이 너무 좁으면 마진 축소
            f_min = dermis_rel_us + 0.5
            f_max = bone_rel_us   - 0.3
        use_bone_ref = True
    else:
        # B) 뼈 미검출 또는 약한 뼈(signal 모드, max_prom 낮음): 확장 탐색
        # 볼/잎술 연조직 에코 오검출이 근막 창을 제한하지 않도록 함
        t_fascia_dermis = dermis_rel_us + (FASCIA_EXPECTED_US - DERMIS_EXPECTED_US)
        t_fascia_center = (FASCIA_EXPECTED_US + t_fascia_dermis) / 2
        f_min = max(dermis_rel_us + 1.0,
                    t_fascia_center - FASCIA_STD_US * 2)
        f_max = FASCIA_FAR_MAX_US
        use_bone_ref = False

    # ── Step 4: 근막 검출 ─────────────────────────────────────
    f_i0 = int(np.searchsorted(rel_t_analysis, f_min))
    f_i1 = int(np.searchsorted(rel_t_analysis, f_max))
    f_i1 = min(f_i1, len(rel_t_analysis) - 1)

    if f_i0 >= f_i1:
        if use_bone_ref:
            # 폴백: 전체 진피-뼈 구간에서 탐색
            f_i0 = dermis_idx + 1
            f_i1 = min(end_idx - 1, len(rel_t_analysis) - 1)
        else:
            f_i0 = dermis_idx + 1
            f_i1 = len(rel_t_analysis) - 1

    if f_i0 >= f_i1:
        # 근막 탐색 창이 없음 — 뼈가 검출된 경우 근막 없음으로 처리 (이마에서 정상)
        if bone_rel_us is not None:
            return {
                "dermis_rel_us": dermis_rel_us, "dermis_mm": dermis_mm,
                "fascia_rel_us": None,          "fascia_mm": None,
                "bone_rel_us":      bone_rel_us,  "bone_mm":         bone_mm,
                "bone_amp":         bone_amp,     "bone_max_prom":   bone_max_prom,
                "bone_first_prom":  bone_first_prom,
                "bone_strong":      bone_strong,  "bone_used":       False,
            }
        return None  # 뼈도 근막도 없음 → 검출 실패

    seg_f = abs_sig_analysis[f_i0:f_i1]
    std_f = float(np.std(seg_f))
    # 피크 선택 기준점: 해부학적 예상 위치 (v1/v2 방식 유지)
    t_fascia_dermis = dermis_rel_us + (FASCIA_EXPECTED_US - DERMIS_EXPECTED_US)
    t_fascia_center = (FASCIA_EXPECTED_US + t_fascia_dermis) / 2

    peaks, _ = find_peaks(seg_f, prominence=std_f * 0.3)
    if len(peaks) > 0:
        best_peak  = peaks[np.argmin(np.abs(rel_t_analysis[f_i0 + peaks] - t_fascia_center))]
        fascia_idx = f_i0 + best_peak
    else:
        fascia_idx = f_i0 + int(np.argmax(seg_f))

    fascia_rel_us = float(rel_t_analysis[fascia_idx])
    fascia_mm     = fascia_rel_us * SPEED_OF_SOUND / 2 / 1000

    # 근막-뼈 간격 검사: 근막 뒤에 최소 1mm 근육층이 있어야 정상
    # 간격 < 1mm이면 근막 오검출 → 근막 없음으로 처리
    if bone_mm is not None and (bone_mm - fascia_mm) < 1.0:
        fascia_rel_us = None
        fascia_mm     = None

    return {
        "dermis_rel_us": dermis_rel_us,
        "dermis_mm":     dermis_mm,
        "fascia_rel_us": fascia_rel_us,
        "fascia_mm":     fascia_mm,
        "bone_rel_us":      bone_rel_us,
        "bone_mm":          bone_mm,
        "bone_amp":         bone_amp,
        "bone_max_prom":    bone_max_prom,
        "bone_first_prom":  bone_first_prom,
        "bone_strong":      bone_strong,
        "bone_used":        use_bone_ref,
    }


# ─── JSON 저장 ────────────────────────────────────────────────
def save_json(csv_path, n_samples, t0_us, det):
    dermis_abs   = t0_us + det["dermis_rel_us"]
    fascia_found = det["fascia_rel_us"] is not None

    # 뼈 정보 (검출된 경우만)
    bone_info = None
    if det["bone_rel_us"] is not None:
        bone_info = {
            "time_us":           round(t0_us + det["bone_rel_us"], 4),
            "rel_us":            round(det["bone_rel_us"],          4),
            "depth_mm":          round(det["bone_mm"],              4),
            "amplitude":         round(det["bone_amp"],             2),
            "first_prominence":  round(det["bone_first_prom"],      2),
            "max_prominence":    round(det["bone_max_prom"],        2),
            "strong":            det["bone_strong"],
            "used_as_reference": det["bone_used"],
        }

    # positions: 진피 + 근막(검출된 경우만)
    # 뼈 검출 시 근막 없음 가능 (이마 해부학적 정상)
    # 뼈 미검출 시 근막 항상 존재 (볼/입술)
    positions = [
        {
            "position_number": 1,
            "position_name":   "피하지방시작",
            "time_us":         round(dermis_abs,       4),
            "thickness_mm":    round(det["dermis_mm"], 4),
            "depth_start_mm":  0.0,
            "depth_end_mm":    round(det["dermis_mm"], 4),
        }
    ]
    if fascia_found:
        fascia_abs = t0_us + det["fascia_rel_us"]
        positions.append({
            "position_number": 2,
            "position_name":   "Fascia",
            "time_us":         round(fascia_abs,       4),
            "thickness_mm":    round(det["fascia_mm"], 4),
            "depth_start_mm":  round(det["dermis_mm"], 4),
            "depth_end_mm":    round(det["fascia_mm"], 4),
        })

    data = {
        "source_file":        os.path.basename(csv_path),
        "start_point_us":     round(t0_us,           4),
        "num_positions":      len(positions),
        "speed_of_sound":     SPEED_OF_SOUND,
        "sample_interval_ns": SAMPLE_NS,
        "num_samples":        n_samples,
        "auto_marked":        True,
        "bone_echo":          bone_info,
        "positions":          positions,
    }

    out = os.path.splitext(csv_path)[0] + "_positions.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out


# ─── 파일 수집 ────────────────────────────────────────────────
def collect_csv(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # (펄서) 디렉토리 제외
        dirnames[:] = [d for d in dirnames if "(펄서)" not in d]
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                files.append(os.path.join(dirpath, fname))
    return sorted(files)


# ─── 메인 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite",  action="store_true", help="기존 JSON 덮어쓰기")
    parser.add_argument("--dry-run",    action="store_true", help="파일 목록만 출력")
    parser.add_argument("--bone-mode",  default="signal",
                        choices=["signal", "filename", "none"],
                        help="뼈 감지 모드: signal(기본,신호기반)|filename(파일명기반)|none(동일)")
    args = parser.parse_args()

    if not os.path.isdir(USDATA):
        print(f"[ERROR] usdata 디렉토리 없음: {USDATA}")
        sys.exit(1)

    all_csv = collect_csv(USDATA)
    total   = len(all_csv)

    if total == 0:
        print(f"[WARN] CSV 파일이 없습니다: {USDATA}")
        sys.exit(0)

    print("=" * 60)
    print("  auto_mark_usdata.py  (pure Python, 외부 바이너리 불필요)")
    print(f"  대상: {USDATA}")
    print(f"  총 CSV: {total} 개")
    if args.overwrite: print("  모드: 전체 덮어쓰기 (--overwrite)")
    if args.dry_run:   print("  모드: 시험 실행 (--dry-run)")
    print(f"  뼈 감지: --bone-mode {args.bone_mode}")
    print("=" * 60)

    ok = skip = fail = 0

    for idx, csv_path in enumerate(all_csv, 1):
        fname     = os.path.basename(csv_path)
        json_path = os.path.splitext(csv_path)[0] + "_positions.json"

        # (0) 또는 (00) 번호 = 허공 측정 → 항상 스킵
        if re.search(r'\(0+\)', fname):
            skip += 1
            print(f"[{idx:4d}/{total}] SKIP  {fname}  (허공 측정)")
            continue

        if os.path.isfile(json_path) and not args.overwrite:
            skip += 1
            print(f"[{idx:4d}/{total}] SKIP  {fname}")
            continue

        if args.dry_run:
            print(f"[{idx:4d}/{total}] DRY   {fname}")
            continue

        try:
            adc, time_us = load_adc(csv_path)
        except Exception as e:
            fail += 1
            print(f"[{idx:4d}/{total}] FAIL  {fname}  (로드 오류: {e})")
            continue

        if len(adc) == 0:
            fail += 1
            print(f"[{idx:4d}/{total}] FAIL  {fname}  (빈 파일)")
            continue

        t0_us = detect_t0(adc, time_us)
        if t0_us is None:
            fail += 1
            print(f"[{idx:4d}/{total}] FAIL  {fname}  (T0 검출 실패)")
            continue

        if args.bone_mode == "filename":
            m_pos = re.search(r'\(\d+\)(.+?)_[MF]\.csv$', fname, re.IGNORECASE)
            pos_name = m_pos.group(1).strip() if m_pos else ""
            bmode = "skip" if any(kw in pos_name for kw in NO_BONE_KEYWORDS) else "none"
        else:
            bmode = args.bone_mode

        det = detect_boundaries(adc, time_us, t0_us, bone_mode=bmode)
        if det is None:
            fail += 1
            print(f"[{idx:4d}/{total}] FAIL  {fname}  (경계 검출 실패)")
            continue

        save_json(csv_path, len(adc), t0_us, det)

        if det["bone_rel_us"] is not None:
            bone_str = (f"  ▶뼈={det['bone_mm']:.2f}mm"
                        f"({'기준' if det['bone_used'] else '감지'})"
                        f"  amp={det['bone_amp']:.0f}")
        else:
            bone_str = "  뼈=없음"
        fascia_str = f"F={det['fascia_mm']:.2f}mm" if det["fascia_rel_us"] is not None else "F=없음"
        print(
            f"[{idx:4d}/{total}] OK    {fname:<52s}"
            f"  T0={t0_us:.2f}μs"
            f"  D={det['dermis_mm']:.2f}mm"
            f"  {fascia_str}"
            f"{bone_str}"
        )
        ok += 1

    print()
    print("=" * 60)
    print(f"  완료: OK={ok}  SKIP={skip}  FAIL={fail}  / 전체={total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
