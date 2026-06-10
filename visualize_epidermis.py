"""
표피 검출 알고리즘 시각화 검증
- OK 케이스와 WARN 케이스를 각각 샘플링해서 플롯
- 각 신호에서 T0 / 진피 / 근막 경계 표시
- 결과: results/epidermis_validation.png
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=_font_path).get_name()
import numpy as np
import pandas as pd
from scipy.signal import hilbert

# analyze_epidermis.py 와 동일한 상수/함수
SAMPLE_INTERVAL_NS = 10
DROP_SAMPLES       = 1200
SOUND_SPEED_MM_US  = 1.54
THRESHOLD_ABS      = 100
MIN_CLUSTER_WIDTH  = 10

DATA_ROOT   = Path(__file__).parent / "usdata"
RESULT_DIR  = Path(__file__).parent / "results"
RESULT_FILE = RESULT_DIR / "epidermis_analysis.csv"

FILENAME_RE = re.compile(
    r'^(?P<name>.+)_(?P<date>\d{8})_(?P<time>\d{6})_'
    r'\((?P<pos_num>\d+)\)(?P<position>.+?)_(?P<gender>[MF])\.csv$'
)


def find_csv(file_name: str) -> Path | None:
    for p in DATA_ROOT.rglob(file_name):
        return p
    return None


def detect_clusters(env_smooth, threshold):
    above = env_smooth > threshold
    clusters = []
    in_c, start = False, 0
    for i, v in enumerate(above):
        if v and not in_c:
            in_c, start = True, i
        elif not v and in_c:
            in_c = False
            if i - 1 - start >= MIN_CLUSTER_WIDTH:
                clusters.append((start, i - 1))
    if in_c and len(env_smooth) - 1 - start >= MIN_CLUSTER_WIDTH:
        clusters.append((start, len(env_smooth) - 1))
    return clusters


def plot_signal(ax, adc, title, result_row):
    s        = adc.astype(float) - 128.0
    env      = np.abs(hilbert(s))
    clusters = detect_clusters(env, THRESHOLD_ABS)

    t_us = (DROP_SAMPLES + np.arange(len(adc))) * SAMPLE_INTERVAL_NS / 1000.0

    ax.plot(t_us, s, color="#4a9eff", lw=0.7, alpha=0.7, label="Signal")
    ax.plot(t_us, env, color="orange", lw=1.0, alpha=0.8, label="Envelope")
    ax.axhline(THRESHOLD_ABS, color="gray", lw=0.8, ls="--", label=f"Threshold ({THRESHOLD_ABS})")

    # T0 = 첫 클러스터 시작, dermis = 첫 클러스터 끝
    if len(clusters) >= 1:
        t0_us     = (DROP_SAMPLES + clusters[0][0]) * SAMPLE_INTERVAL_NS / 1000.0
        dermis_us = (DROP_SAMPLES + clusters[0][1]) * SAMPLE_INTERVAL_NS / 1000.0
        ax.axvline(t0_us,     color="#ffaa00", lw=2.0, ls="-",  label=f"T0 (표면) {t0_us:.2f}μs")
        ax.axvline(dermis_us, color="#ff3333", lw=2.0, ls="--", label=f"진피경계 {dermis_us:.2f}μs")
        ax.fill_betweenx([-128, 128], t0_us, dermis_us, color="#ffaa00", alpha=0.10)

    # 두 번째 클러스터 = 진피/근막 영역
    if len(clusters) >= 2:
        def cluster_peak(c):
            return c[0] + int(np.argmax(env[c[0]:c[1]+1]))
        fascia_peak_us = (DROP_SAMPLES + cluster_peak(clusters[1])) * SAMPLE_INTERVAL_NS / 1000.0
        ax.axvline(fascia_peak_us, color="#cc00ff", lw=1.5, ls=":",
                   label=f"심층반사 {fascia_peak_us:.2f}μs")

    ok_str  = "OK" if result_row["ok"] else "WARN"
    epi_str = f"{result_row['epi_mm']:.2f}mm" if not np.isnan(result_row["epi_mm"]) else "N/A"
    ax.set_title(f"{title}\n[{ok_str}] 표피={epi_str}", fontsize=9)
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("ADC - 128")
    ax.set_ylim(-140, 140)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)


def main():
    df = pd.read_csv(RESULT_FILE)

    ok_df   = df[df["ok"] == True].sample(min(8, df["ok"].sum()), random_state=42)
    warn_df = df[df["ok"] == False].sample(min(4, (~df["ok"]).sum()), random_state=42)
    samples = pd.concat([ok_df, warn_df]).reset_index(drop=True)

    n = len(samples)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = np.array(axes).flatten()

    for i, (_, row) in enumerate(samples.iterrows()):
        path = find_csv(row["file"])
        if path is None:
            axes[i].set_title(f"파일 없음: {row['file']}", fontsize=8)
            continue
        adc = np.loadtxt(path, dtype=float)
        title = f"{row['name']} | {row['position']}"
        plot_signal(axes[i], adc, title, row)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("표피 검출 알고리즘 검증 (OK × 8, WARN × 4)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = RESULT_DIR / "epidermis_validation.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"저장 완료: {out}")


if __name__ == "__main__":
    main()
