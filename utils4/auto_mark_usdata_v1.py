#!/usr/bin/env python3
"""
auto_mark_usdata.py
boundary_detector 알고리즘을 Python으로 구현 — 외부 바이너리 불필요
Windows / Linux / macOS 공용

사용법:
  python auto_mark_usdata.py                  # 미마킹 파일만 처리
  python auto_mark_usdata.py --overwrite      # 기존 JSON도 덮어씀
  python auto_mark_usdata.py --dry-run        # 파일 목록만 출력 (실행 안함)
"""

import os, sys, json, argparse
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
MAX_DISTANCE_MM       = 6.0     # 분석 최대 깊이 (mm)


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


# ─── 피부층 경계 검출 (Python 구현) ──────────────────────────
def detect_boundaries(adc, time_us, t0_us):
    """
    T0 이후 신호에서 진피(Dermis)·근막(Fascia) 경계 검출
    boundary_detector.c 알고리즘과 동일한 파라미터 사용

    Returns:
        dict with dermis_rel_us, dermis_mm, fascia_rel_us, fascia_mm
        or None if detection failed
    """
    # T0 이후 상대 시간으로 변환
    mask = time_us >= t0_us
    if mask.sum() < 100:
        return None

    rel_t  = time_us[mask] - t0_us         # 0부터 시작하는 상대 시간
    signal = adc[mask] - 128.0             # 중심 이동

    # 분석 범위: 6mm 이내
    max_t_us = MAX_DISTANCE_MM / SPEED_OF_SOUND * 2 * 1e6
    end_idx  = np.searchsorted(rel_t, max_t_us)
    if end_idx < 100:
        end_idx = len(rel_t)
    rel_t  = rel_t[:end_idx]
    signal = signal[:end_idx]

    abs_sig = np.abs(signal)

    # ── Step 1: 진피 검출 (DERMIS_EXPECTED ± STD μs) ──────────
    d_min = DERMIS_EXPECTED_US - DERMIS_STD_US
    d_max = DERMIS_EXPECTED_US + DERMIS_STD_US

    d_i0 = int(np.searchsorted(rel_t, d_min))
    d_i1 = int(np.searchsorted(rel_t, d_max))
    d_i1 = min(d_i1, len(rel_t) - 1)

    if d_i0 >= d_i1:
        return None

    seg = abs_sig[d_i0:d_i1]

    # 변곡점(2차 미분 부호 전환) 우선, 없으면 최댓값 지점
    grad1   = np.gradient(seg)
    grad2   = np.gradient(grad1)
    inflect = np.where((grad2[:-1] < 0) & (grad2[1:] > 0) & (grad1[1:] > 0))[0]

    if len(inflect) > 0:
        # 예상 시간에 가장 가까운 변곡점
        best_local = inflect[np.argmin(np.abs(rel_t[d_i0 + inflect] - DERMIS_EXPECTED_US))]
        dermis_idx = d_i0 + best_local
    else:
        dermis_idx = d_i0 + int(np.argmax(seg))

    dermis_rel_us = float(rel_t[dermis_idx])
    dermis_mm     = dermis_rel_us * SPEED_OF_SOUND / 2 / 1000

    # ── Step 2: 근막 검출 ─────────────────────────────────────
    # 시간 기반 + 진피 기반 두 예상값의 평균
    t_fascia_time  = FASCIA_EXPECTED_US
    t_fascia_dermis = dermis_rel_us + (FASCIA_EXPECTED_US - DERMIS_EXPECTED_US)
    t_fascia_center = (t_fascia_time + t_fascia_dermis) / 2

    f_min = t_fascia_center - FASCIA_STD_US
    f_max = t_fascia_center + FASCIA_STD_US

    f_i0 = int(np.searchsorted(rel_t, f_min))
    f_i1 = int(np.searchsorted(rel_t, f_max))
    f_i1 = min(f_i1, len(rel_t) - 1)

    if f_i0 >= f_i1:
        # 범위 밖이면 전체 나머지에서 피크 탐색
        f_i0 = dermis_idx + 1
        f_i1 = len(rel_t) - 1

    if f_i0 >= f_i1:
        return None

    seg_f = abs_sig[f_i0:f_i1]
    std_f = float(np.std(seg_f))

    peaks, _ = find_peaks(seg_f, prominence=std_f * 0.3)
    if len(peaks) > 0:
        # 예상 시간에 가장 가까운 피크
        best_peak = peaks[np.argmin(np.abs(rel_t[f_i0 + peaks] - t_fascia_center))]
        fascia_idx = f_i0 + best_peak
    else:
        fascia_idx = f_i0 + int(np.argmax(seg_f))

    fascia_rel_us = float(rel_t[fascia_idx])
    fascia_mm     = fascia_rel_us * SPEED_OF_SOUND / 2 / 1000

    return {
        "dermis_rel_us": dermis_rel_us,
        "dermis_mm":     dermis_mm,
        "fascia_rel_us": fascia_rel_us,
        "fascia_mm":     fascia_mm,
    }


# ─── JSON 저장 ────────────────────────────────────────────────
def save_json(csv_path, n_samples, t0_us, det):
    dermis_abs = t0_us + det["dermis_rel_us"]
    fascia_abs = t0_us + det["fascia_rel_us"]

    data = {
        "source_file":        os.path.basename(csv_path),
        "start_point_us":     round(t0_us,           4),
        "num_positions":      2,
        "speed_of_sound":     SPEED_OF_SOUND,
        "sample_interval_ns": SAMPLE_NS,
        "num_samples":        n_samples,
        "auto_marked":        True,
        "positions": [
            {
                "position_number": 1,
                "position_name":   "피하지방시작",
                "time_us":         round(dermis_abs,       4),
                "thickness_mm":    round(det["dermis_mm"], 4),
                "depth_start_mm":  0.0,
                "depth_end_mm":    round(det["dermis_mm"], 4),
            },
            {
                "position_number": 2,
                "position_name":   "Fascia",
                "time_us":         round(fascia_abs,       4),
                "thickness_mm":    round(det["fascia_mm"], 4),
                "depth_start_mm":  round(det["dermis_mm"], 4),
                "depth_end_mm":    round(det["fascia_mm"], 4),
            },
        ],
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
    parser.add_argument("--overwrite", action="store_true", help="기존 JSON 덮어쓰기")
    parser.add_argument("--dry-run",   action="store_true", help="파일 목록만 출력")
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
    print("=" * 60)

    ok = skip = fail = 0

    for idx, csv_path in enumerate(all_csv, 1):
        fname     = os.path.basename(csv_path)
        json_path = os.path.splitext(csv_path)[0] + "_positions.json"

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

        det = detect_boundaries(adc, time_us, t0_us)
        if det is None:
            fail += 1
            print(f"[{idx:4d}/{total}] FAIL  {fname}  (경계 검출 실패)")
            continue

        save_json(csv_path, len(adc), t0_us, det)

        print(
            f"[{idx:4d}/{total}] OK    {fname:<52s}"
            f"  T0={t0_us:.2f}μs"
            f"  D={det['dermis_mm']:.2f}mm"
            f"  F={det['fascia_mm']:.2f}mm"
        )
        ok += 1

    print()
    print("=" * 60)
    print(f"  완료: OK={ok}  SKIP={skip}  FAIL={fail}  / 전체={total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
