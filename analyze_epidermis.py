"""
표피 두께 자동 분석 스크립트
- usdata/ 하위 모든 CSV를 일괄 처리
- 파일명에서 환자/부위 정보 파싱
- Hilbert envelope + 클러스터 검출로 표피/진피/근막 경계 자동 추출
- 결과: results/epidermis_analysis.csv
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert

# ── 상수 ──────────────────────────────────────────────────────────────
SAMPLE_INTERVAL_NS = 10          # 10ns/sample (100MHz ADC)
DROP_SAMPLES       = 1200        # 드롭한 샘플 수 → 저장 시작 절대시간
SOUND_SPEED_MM_US  = 1.54        # mm/μs (= 1540 m/s)
THRESHOLD_ABS      = 100         # 절대 임계값: 기준선(~50) 위로 올라오는 기준
MIN_CLUSTER_WIDTH  = 10          # 유효 클러스터 최소 폭 (샘플)

DATA_ROOT   = Path(__file__).parent / "usdata"
RESULT_DIR  = Path(__file__).parent / "results"
RESULT_FILE = RESULT_DIR / "epidermis_analysis.csv"

# ── 파일명 파싱 ───────────────────────────────────────────────────────
FILENAME_RE = re.compile(
    r'^(?P<name>.+)_(?P<date>\d{8})_(?P<time>\d{6})_'
    r'\((?P<pos_num>\d+)\)(?P<position>.+?)_(?P<gender>[MF])\.csv$'
)

def parse_filename(path: Path) -> dict | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return {
        "name":     m.group("name"),
        "date":     m.group("date"),
        "time":     m.group("time"),
        "pos_num":  int(m.group("pos_num")),
        "position": m.group("position").strip(),
        "gender":   m.group("gender"),
    }

# ── 신호 분석 ─────────────────────────────────────────────────────────
def detect_clusters(env: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """envelope에서 임계값 이상인 구간(클러스터)을 반환."""
    above = env >= threshold
    clusters = []
    in_c, start = False, 0
    for i, v in enumerate(above):
        if v and not in_c:
            in_c, start = True, i
        elif not v and in_c:
            in_c = False
            if i - 1 - start >= MIN_CLUSTER_WIDTH:
                clusters.append((start, i - 1))
    if in_c and len(env) - 1 - start >= MIN_CLUSTER_WIDTH:
        clusters.append((start, len(env) - 1))
    return clusters


def analyze_signal(adc: np.ndarray) -> dict:
    """
    ADC 배열(0-255, 1250샘플)을 받아 표피/진피/근막 경계를 검출.

    원리:
      - 첫 번째 클러스터 시작(T0) = 표면 반사 진입점
      - 첫 번째 클러스터 끝(dermis) = envelope이 고값 유지하다 떨어지는 지점
        → 표피/진피 경계 (VB5K Boundary Marker의 Dermis 라인과 동일)
      - 표피 두께 = (dermis - T0) × dt × v_sound / 2

    Returns dict with keys: t0_us, dermis_us, fascia_us, epi_mm, dermis_mm, ok, note
    """
    result = dict(t0_us=np.nan, dermis_us=np.nan, fascia_us=np.nan,
                  epi_mm=np.nan, dermis_mm=np.nan,
                  gap_us=np.nan,           # 표피 끝 ~ 두 번째 클러스터 시작 (연조직/뼈 구분)
                  second_span_us=np.nan,   # 두 번째 클러스터 폭 (뼈 = 넓음)
                  n_clusters=0,            # 전체 클러스터 수
                  ok=False, note="")

    if len(adc) == 0:
        result["note"] = "empty"
        return result

    s   = adc.astype(float) - 128.0
    env = np.abs(hilbert(s))

    clusters = detect_clusters(env, THRESHOLD_ABS)
    result["n_clusters"] = len(clusters)
    if len(clusters) < 1:
        result["note"] = f"clusters<1 ({len(clusters)})"
        return result

    # 절대시간 변환 (μs)
    def sample_to_us(samp):
        return (DROP_SAMPLES + samp) * SAMPLE_INTERVAL_NS / 1000.0

    # 거리 → 깊이 (mm): 왕복시간 × 음속 / 2
    def delta_to_mm(s1, s2):
        dt_us = (s2 - s1) * SAMPLE_INTERVAL_NS / 1000.0
        return dt_us * SOUND_SPEED_MM_US / 2.0

    # T0 = 첫 번째 클러스터 시작, dermis = 첫 번째 클러스터 끝
    t0_sample     = clusters[0][0]
    dermis_sample = clusters[0][1]

    result["t0_us"]     = sample_to_us(t0_sample)
    result["dermis_us"] = sample_to_us(dermis_sample)
    result["epi_mm"]    = delta_to_mm(t0_sample, dermis_sample)

    # 두 번째 클러스터: gap, span, 피크 위치 추출
    if len(clusters) >= 2:
        c1_start, c1_end = clusters[1]
        def cluster_peak(c):
            s_i, e_i = c
            return s_i + int(np.argmax(env[s_i:e_i + 1]))
        fascia_sample = cluster_peak(clusters[1])
        result["fascia_us"]     = sample_to_us(fascia_sample)
        result["dermis_mm"]     = delta_to_mm(dermis_sample, fascia_sample)
        result["gap_us"]        = (c1_start - dermis_sample) * SAMPLE_INTERVAL_NS / 1000.0
        result["second_span_us"] = (c1_end - c1_start) * SAMPLE_INTERVAL_NS / 1000.0

    # 유효성 검사 (얼굴 표피: 0.2 ~ 2.5 mm)
    if not (0.2 <= result["epi_mm"] <= 2.5):
        result["note"] = f"epi_mm out of range ({result['epi_mm']:.2f})"
    else:
        result["ok"] = True

    return result

# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    csv_files = sorted(DATA_ROOT.rglob("*.csv"))
    if not csv_files:
        print(f"CSV 파일 없음: {DATA_ROOT}")
        sys.exit(1)

    print(f"총 {len(csv_files)}개 파일 처리 중...\n")
    RESULT_DIR.mkdir(exist_ok=True)

    rows = []
    errors = []

    for i, path in enumerate(csv_files, 1):
        meta = parse_filename(path)
        if meta is None:
            errors.append(str(path))
            continue

        try:
            adc = np.loadtxt(path, dtype=float)
        except Exception as e:
            errors.append(f"{path}: {e}")
            continue

        res = analyze_signal(adc)

        row = {**meta, **res, "file": path.name}
        rows.append(row)

        status = "OK" if res["ok"] else f"WARN({res['note']})"
        epi    = f"{res['epi_mm']:.2f}mm" if not np.isnan(res['epi_mm']) else "N/A"
        print(f"[{i:3d}/{len(csv_files)}] {status:30s}  epi={epi:8s}  {meta['name']} ({meta['position']})")

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

    # ── 요약 출력 ───────────────────────────────────────────────
    ok_df = df[df["ok"] == True]
    print(f"\n{'='*60}")
    print(f"처리 완료:  총 {len(rows)}개  /  성공 {len(ok_df)}개  /  경고 {len(rows)-len(ok_df)}개")
    if errors:
        print(f"파싱 실패:  {len(errors)}개")
    if len(ok_df):
        print(f"\n[표피 두께 전체 통계]")
        print(f"  평균: {ok_df['epi_mm'].mean():.2f} mm")
        print(f"  std : {ok_df['epi_mm'].std():.2f} mm")
        print(f"  min : {ok_df['epi_mm'].min():.2f} mm")
        print(f"  max : {ok_df['epi_mm'].max():.2f} mm")

        print(f"\n[환자별 평균 표피 두께]")
        for name, grp in ok_df.groupby("name"):
            print(f"  {name}: {grp['epi_mm'].mean():.2f} ± {grp['epi_mm'].std():.2f} mm  (n={len(grp)})")

    print(f"\n결과 저장: {RESULT_FILE}")


if __name__ == "__main__":
    main()
