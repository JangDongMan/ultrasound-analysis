#!/bin/bash
# auto_mark_usdata.sh
# utils4/boundary_detector 를 usdata/ 전체 CSV에 순회 실행하여
# _positions.json 자동 생성
#
# 사용법:
#   ./auto_mark_usdata.sh                  # 미마킹 파일만 처리
#   ./auto_mark_usdata.sh --overwrite      # 기존 JSON도 덮어씀
#   ./auto_mark_usdata.sh --dry-run        # 실행하지 않고 대상 파일만 출력

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTOR="$SCRIPT_DIR/utils4/boundary_detector"
USDATA="$SCRIPT_DIR/usdata/data"
TMPDIR_PREFIX="/tmp/auto_mark_$$"

OVERWRITE=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --overwrite) OVERWRITE=1 ;;
        --dry-run)   DRY_RUN=1   ;;
    esac
done

# ─── 사전 점검 ────────────────────────────────────────────────
if [ ! -f "$DETECTOR" ]; then
    echo "[ERROR] boundary_detector 없음: $DETECTOR"
    echo "        cd utils4 && make 으로 먼저 빌드하세요."
    exit 1
fi
if [ ! -d "$USDATA" ]; then
    echo "[ERROR] usdata 디렉토리 없음: $USDATA"
    exit 1
fi
if ! python3 -c "import numpy, scipy" 2>/dev/null; then
    echo "[ERROR] python3에 numpy/scipy 가 없습니다."
    exit 1
fi

mkdir -p "$TMPDIR_PREFIX"
trap 'rm -rf "$TMPDIR_PREFIX"' EXIT

# ─── 파이썬 헬퍼: T0 검출 + 임시 CSV 생성 + JSON 저장 ─────────
PYHELPER="$TMPDIR_PREFIX/helper.py"
cat > "$PYHELPER" << 'PYEOF'
#!/usr/bin/env python3
"""
auto_mark helper
사용법(내부):
  python3 helper.py detect_t0  <csv>           → "OK <t0_us>" or "FAIL"
  python3 helper.py make_tmp   <csv> <tmp_out>  → old-format temp CSV 생성
  python3 helper.py save_json  <csv> <start_us> <dermis_rel_us> <dermis_mm> <fascia_rel_us> <fascia_mm>
"""
import sys, os, json
import numpy as np
from scipy.signal import hilbert

TRIM_START       = 1200
TRIM_COUNT       = 1250
DISPLAY_OFFSET   = 12.00   # μs
SAMPLE_NS        = 10      # ns/sample
SPEED_OF_SOUND   = 1540.0
THRESHOLD        = 100.0   # envelope threshold for T0


def load_adc(filepath):
    adc = []
    with open(filepath) as f:
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
        if len(adc) >= end:
            adc = adc[TRIM_START:end]
        elif len(adc) > TRIM_START:
            adc = adc[TRIM_START:]
    n = len(adc)
    time_us = np.arange(n) * SAMPLE_NS / 1000.0 + DISPLAY_OFFSET
    return adc, time_us


def detect_t0(adc, time_us):
    centered = adc - 128.0
    env = np.abs(hilbert(centered))
    above = env >= THRESHOLD
    for i, flag in enumerate(above):
        if flag:
            return time_us[i]
    return None


def make_tmp_csv(adc, time_us, out_path):
    """old-format CSV: 2 header lines + time_sec,voltage"""
    voltage = (adc - 128.0) / 128.0   # normalize -1..+1
    with open(out_path, 'w') as f:
        f.write("x-axis,1\n")
        f.write("second,Volt\n")
        for t, v in zip(time_us, voltage):
            f.write(f"{t*1e-6:.10f},{v:.6f}\n")


def save_json(csv_path, num_samples, start_us,
              dermis_rel_us, dermis_mm, fascia_rel_us, fascia_mm):
    dermis_abs = start_us + dermis_rel_us
    fascia_abs = start_us + fascia_rel_us
    data = {
        "source_file":       os.path.basename(csv_path),
        "start_point_us":    round(start_us, 4),
        "num_positions":     2,
        "speed_of_sound":    SPEED_OF_SOUND,
        "sample_interval_ns": SAMPLE_NS,
        "num_samples":       num_samples,
        "auto_marked":       True,
        "positions": [
            {
                "position_number": 1,
                "position_name":   "피하지방시작",
                "time_us":         round(dermis_abs, 4),
                "thickness_mm":    round(dermis_mm,  4),
                "depth_start_mm":  0.0,
                "depth_end_mm":    round(dermis_mm,  4),
            },
            {
                "position_number": 2,
                "position_name":   "Fascia",
                "time_us":         round(fascia_abs, 4),
                "thickness_mm":    round(fascia_mm,  4),
                "depth_start_mm":  round(dermis_mm,  4),
                "depth_end_mm":    round(fascia_mm,  4),
            },
        ],
    }
    base     = os.path.splitext(csv_path)[0]
    out_path = base + "_positions.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(out_path)   # stdout으로 저장 경로 반환


# ─── entry ────────────────────────────────────────────────────
cmd = sys.argv[1]

if cmd == "detect_t0":
    adc, time_us = load_adc(sys.argv[2])
    t0 = detect_t0(adc, time_us)
    if t0 is None:
        print("FAIL")
    else:
        print(f"OK {t0:.6f} {len(adc)}")

elif cmd == "make_tmp":
    adc, time_us = load_adc(sys.argv[2])
    make_tmp_csv(adc, time_us, sys.argv[3])

elif cmd == "save_json":
    csv_path      = sys.argv[2]
    num_samples   = int(sys.argv[3])
    start_us      = float(sys.argv[4])
    dermis_rel_us = float(sys.argv[5])
    dermis_mm     = float(sys.argv[6])
    fascia_rel_us = float(sys.argv[7])
    fascia_mm     = float(sys.argv[8])
    save_json(csv_path, num_samples, start_us,
              dermis_rel_us, dermis_mm, fascia_rel_us, fascia_mm)
PYEOF

# ─── CSV 파일 수집 ─────────────────────────────────────────────
mapfile -d '' ALL_CSV < <(find "$USDATA" -name "*.csv" -print0 | sort -z)
TOTAL=${#ALL_CSV[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "[WARN] CSV 파일이 없습니다: $USDATA"
    exit 0
fi

echo "========================================"
echo "  auto_mark_usdata.sh"
echo "  대상: $USDATA"
echo "  총 CSV: $TOTAL 개"
[ "$OVERWRITE" -eq 1 ] && echo "  모드: 전체 덮어쓰기 (--overwrite)"
[ "$DRY_RUN"   -eq 1 ] && echo "  모드: 시험 실행 (--dry-run)"
echo "========================================"

OK=0; SKIP=0; FAIL=0; IDX=0

for CSV in "${ALL_CSV[@]}"; do
    IDX=$(( IDX + 1 ))
    BASE="${CSV%.csv}"
    JSON="${BASE}_positions.json"
    FNAME="$(basename "$CSV")"

    # 이미 마킹된 파일 건너뜀
    if [ -f "$JSON" ] && [ "$OVERWRITE" -eq 0 ]; then
        SKIP=$(( SKIP + 1 ))
        printf "[%d/%d] SKIP  %s\n" "$IDX" "$TOTAL" "$FNAME"
        continue
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        printf "[%d/%d] DRY   %s\n" "$IDX" "$TOTAL" "$FNAME"
        continue
    fi

    # T0 자동 검출
    T0_RESULT=$(python3 "$PYHELPER" detect_t0 "$CSV" 2>/dev/null)
    if [[ "$T0_RESULT" != OK* ]]; then
        printf "[%d/%d] FAIL  %s  (T0 검출 실패)\n" "$IDX" "$TOTAL" "$FNAME"
        FAIL=$(( FAIL + 1 ))
        continue
    fi
    T0_US=$(echo "$T0_RESULT" | awk '{print $2}')
    N_SAMPLES=$(echo "$T0_RESULT" | awk '{print $3}')

    # 임시 CSV 생성 (old-format)
    TMP_CSV="$TMPDIR_PREFIX/tmp_$$.csv"
    python3 "$PYHELPER" make_tmp "$CSV" "$TMP_CSV" 2>/dev/null

    # boundary_detector 실행
    DETECTOR_OUT=$("$DETECTOR" "$TMP_CSV" "$T0_US" 2>/dev/null)

    rm -f "$TMP_CSV"

    # 출력 파싱
    START_US=$(echo "$DETECTOR_OUT"   | grep -oP 'Start Point:\s*\K[\d.]+')
    DERMIS_T=$(echo "$DETECTOR_OUT"   | grep -A3 'Dermis' | grep -oP 'Time:\s*\K[\d.]+' | head -1)
    DERMIS_MM=$(echo "$DETECTOR_OUT"  | grep -A3 'Dermis' | grep -oP 'Depth:\s*\K[\d.]+' | head -1)
    FASCIA_T=$(echo "$DETECTOR_OUT"   | grep -A3 'Fascia' | grep -oP 'Time:\s*\K[\d.]+' | head -1)
    FASCIA_MM=$(echo "$DETECTOR_OUT"  | grep -A3 'Fascia' | grep -oP 'Depth:\s*\K[\d.]+' | head -1)

    if [ -z "$START_US" ] || [ -z "$DERMIS_T" ] || [ -z "$FASCIA_T" ]; then
        printf "[%d/%d] FAIL  %s  (detector 출력 파싱 실패)\n" "$IDX" "$TOTAL" "$FNAME"
        FAIL=$(( FAIL + 1 ))
        continue
    fi

    # JSON 저장
    OUT_PATH=$(python3 "$PYHELPER" save_json \
        "$CSV" "$N_SAMPLES" "$START_US" \
        "$DERMIS_T" "$DERMIS_MM" \
        "$FASCIA_T" "$FASCIA_MM" 2>/dev/null)

    printf "[%d/%d] OK    %-55s  T0=%.2fμs  D=%.2fmm  F=%.2fmm\n" \
        "$IDX" "$TOTAL" "$FNAME" "$T0_US" "$DERMIS_MM" "$FASCIA_MM"
    OK=$(( OK + 1 ))
done

echo ""
echo "========================================"
printf "  완료: OK=%d  SKIP=%d  FAIL=%d  / 전체=%d\n" "$OK" "$SKIP" "$FAIL" "$TOTAL"
echo "========================================"
