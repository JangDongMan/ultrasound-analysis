"""
표피 시작점(T0)과 두께(epi_mm) 부위별 상세 분석
- 15개 세부 부위 × 20명 환자 × 반복 측정
- 결과: results/t0_epidermis_by_region.png, results/t0_epidermis_stats.csv
"""

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

RESULT_DIR = Path(__file__).parent / "results"
INPUT      = RESULT_DIR / "epidermis_analysis.csv"
OUT_PNG    = RESULT_DIR / "t0_epidermis_by_region.png"
OUT_CSV    = RESULT_DIR / "t0_epidermis_stats.csv"

SOUND_SPEED_MM_US = 1.54

def main():
    df = pd.read_csv(INPUT)

    # 부위별 epi_mm 평균으로 정렬
    order = df.groupby("position")["epi_mm"].mean().sort_values().index.tolist()
    n_pos = len(order)

    # ── 통계 계산 ─────────────────────────────────────────────────────
    stats = (df.groupby("position")[["t0_us", "epi_mm"]]
               .agg(["count", "mean", "std", "min", "max"])
               .round(4))
    stats.columns = ["_".join(c) for c in stats.columns]
    stats = stats.loc[order]

    print("[세부 부위별 T0 및 표피두께]")
    print(f"{'부위':28s}  {'n':>3}  {'T0 mean':>8}  {'T0 std':>7}  "
          f"{'epi mean':>9}  {'epi std':>8}  {'epi min':>8}  {'epi max':>8}")
    print("-" * 90)
    for pos, row in stats.iterrows():
        print(f"{pos:28s}  {int(row['epi_mm_count']):3d}  "
              f"{row['t0_us_mean']:8.3f}μs  {row['t0_us_std']:7.3f}  "
              f"{row['epi_mm_mean']:8.3f}mm  {row['epi_mm_std']:7.3f}  "
              f"{row['epi_mm_min']:7.3f}  {row['epi_mm_max']:7.3f}")

    overall = df["epi_mm"]
    print(f"\n{'전체':28s}  {len(df):3d}  "
          f"{df['t0_us'].mean():8.3f}μs  {df['t0_us'].std():7.3f}  "
          f"{overall.mean():8.3f}mm  {overall.std():7.3f}  "
          f"{overall.min():7.3f}  {overall.max():7.3f}")

    stats.to_csv(OUT_CSV, encoding="utf-8-sig")
    print(f"\n통계 저장: {OUT_CSV}")

    # ── 시각화 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_pos))

    # 1) 부위별 표피두께 박스플롯 (얇은 순 정렬)
    ax = axes[0, 0]
    data_epi = [df[df["position"] == p]["epi_mm"].values for p in order]
    bp = ax.boxplot(data_epi, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.axhline(overall.mean(), color="red", ls="--", lw=1.5,
               label=f"전체 평균 {overall.mean():.3f}mm")
    ax.set_title("부위별 표피두께 분포 (얇은 순)", fontsize=11, fontweight="bold")
    ax.set_ylabel("표피두께 (mm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 2) 부위별 T0 박스플롯
    ax = axes[0, 1]
    data_t0 = [df[df["position"] == p]["t0_us"].values for p in order]
    bp2 = ax.boxplot(data_t0, patch_artist=True, notch=False,
                     medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.axhline(df["t0_us"].mean(), color="red", ls="--", lw=1.5,
               label=f"전체 평균 {df['t0_us'].mean():.3f}μs")
    ax.set_title("부위별 T0 분포 (표면 반사 시간)", fontsize=11, fontweight="bold")
    ax.set_ylabel("T0 (μs)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 3) 환자별 × 부위별 히트맵 (표피두께)
    ax = axes[1, 0]
    patients = sorted(df["name"].unique())
    hmap = np.full((len(order), len(patients)), np.nan)
    for j, pat in enumerate(patients):
        for i, pos in enumerate(order):
            vals = df[(df["name"] == pat) & (df["position"] == pos)]["epi_mm"].values
            if len(vals): hmap[i, j] = vals.mean()
    im = ax.imshow(hmap, aspect="auto", cmap="RdYlGn",
                   vmin=overall.quantile(0.05), vmax=overall.quantile(0.95))
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels(patients, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    plt.colorbar(im, ax=ax, label="epi_mm")
    ax.set_title("환자 × 부위 표피두께 히트맵", fontsize=11, fontweight="bold")

    # 4) 표피두께 vs T0 산점도 (부위별 색상)
    ax = axes[1, 1]
    cmap_pos = plt.cm.tab20(np.linspace(0, 1, n_pos))
    for i, pos in enumerate(order):
        sub = df[df["position"] == pos]
        ax.scatter(sub["t0_us"], sub["epi_mm"],
                   c=[cmap_pos[i]], label=pos, alpha=0.5, s=18)
    ax.set_xlabel("T0 (μs, 표면 반사 시간)")
    ax.set_ylabel("표피두께 (mm)")
    ax.set_title("T0 vs 표피두께 (부위별)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=5, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"표피 시작점(T0)·두께 부위별 분석  "
        f"(전체 {len(df)}건 / 20명 / 15부위)\n"
        f"전체 평균: T0={df['t0_us'].mean():.3f}±{df['t0_us'].std():.3f}μs  "
        f"epi={overall.mean():.3f}±{overall.std():.3f}mm",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"그래프 저장: {OUT_PNG}")

    # ── 핵심 요약 ─────────────────────────────────────────────────────
    print("\n[부위별 표피두께 요약 - 얇은 순]")
    for pos in order:
        m = df[df["position"] == pos]["epi_mm"].mean()
        s = df[df["position"] == pos]["epi_mm"].std()
        bar = "█" * int(m * 40)
        print(f"  {pos:30s}  {m:.3f}±{s:.3f}mm  {bar}")


if __name__ == "__main__":
    main()
