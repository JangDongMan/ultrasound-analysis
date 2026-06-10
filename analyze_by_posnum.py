"""
번호별(pos_num 1-28) 전체 환자 분석
- 각 번호의 표피두께(epi_mm)·T0 분포 시각화
- 개인별 데이터 오버레이 (strip plot)
- 공간 진행(이마→목) 흐름 확인
- 결과: results/posnum_analysis.png, results/posnum_stats.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=_font_path).get_name()

import numpy as np
import pandas as pd

RESULT_DIR = Path(__file__).parent / "results"
INPUT      = RESULT_DIR / "epidermis_analysis.csv"
OUT_PNG    = RESULT_DIR / "posnum_analysis.png"
OUT_CSV    = RESULT_DIR / "posnum_stats.csv"


def main():
    df = pd.read_csv(INPUT)

    # 각 pos_num 대표 이름 (최빈값)
    pos_label = (df.groupby("pos_num")["position"]
                   .agg(lambda x: x.mode()[0])
                   .to_dict())

    pos_nums = sorted(df["pos_num"].unique())
    n = len(pos_nums)   # 28

    # ── 통계 ─────────────────────────────────────────────────────────
    stats = (df.groupby("pos_num")[["epi_mm", "t0_us"]]
               .agg(["count", "mean", "std", "min", "max"])
               .round(4))
    stats.columns = ["_".join(c) for c in stats.columns]
    stats.index.name = "pos_num"
    stats["position"] = stats.index.map(pos_label)
    stats.to_csv(OUT_CSV, encoding="utf-8-sig")

    print("[번호별 표피두께 & T0 통계]")
    print(f"{'번호':>4}  {'부위':28s}  {'n':>3}  "
          f"{'epi mean':>9}  {'epi std':>8}  "
          f"{'T0 mean':>8}  {'T0 std':>7}")
    print("-" * 80)
    for pn in pos_nums:
        r = stats.loc[pn]
        print(f"  {pn:2d}  {r['position']:28s}  "
              f"{int(r['epi_mm_count']):3d}  "
              f"{r['epi_mm_mean']:8.3f}mm  {r['epi_mm_std']:7.3f}  "
              f"{r['t0_us_mean']:8.3f}μs  {r['t0_us_std']:7.3f}")

    # ── 시각화 ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    gs  = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.35)

    x = np.arange(n)
    epi_means = [stats.loc[p, "epi_mm_mean"] for p in pos_nums]
    epi_stds  = [stats.loc[p, "epi_mm_std"]  for p in pos_nums]
    t0_means  = [stats.loc[p, "t0_us_mean"]  for p in pos_nums]
    t0_stds   = [stats.loc[p, "t0_us_std"]   for p in pos_nums]
    short_labels = [f"#{p}" for p in pos_nums]
    full_labels  = [f"#{p} {pos_label[p]}" for p in pos_nums]

    # 색상: 이마(1-7) → 관자/광대(8-13) → 볼/귀(14-23) → 턱/목(24-28)
    zone_colors = []
    for p in pos_nums:
        if   p <= 7:  zone_colors.append("#e06060")  # 이마/관자놀이
        elif p <= 13: zone_colors.append("#e0a030")  # 눈/광대
        elif p <= 23: zone_colors.append("#50b050")  # 볼/귀
        else:         zone_colors.append("#4080c0")  # 턱/목

    # ── 패널 1: 번호별 표피두께 막대 + 오차막대 ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(x, epi_means, yerr=epi_stds, capsize=3,
                   color=zone_colors, alpha=0.80, ecolor="gray", lw=0.8)
    # 개인별 스트립 오버레이
    patients = sorted(df["name"].unique())
    cmap_pat = plt.cm.tab20(np.linspace(0, 1, len(patients)))
    for j, pat in enumerate(patients):
        sub = df[df["name"] == pat]
        for _, row in sub.iterrows():
            xi = pos_nums.index(row["pos_num"])
            ax1.scatter(xi, row["epi_mm"],
                        color=cmap_pat[j], s=18, alpha=0.55, zorder=5)
    # 전체 평균선
    overall_mean = df["epi_mm"].mean()
    ax1.axhline(overall_mean, color="red", ls="--", lw=1.4,
                label=f"전체 평균 {overall_mean:.3f}mm")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=8)
    ax1.set_ylabel("표피두께 (mm)", fontsize=10)
    ax1.set_title("번호별 표피두께 (epi_mm) – 막대: 평균±std, 점: 개인값",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25, axis="y")
    # 구간 배경
    ax1.axvspan(-0.5,  6.5, color="#e06060", alpha=0.05)
    ax1.axvspan( 6.5, 12.5, color="#e0a030", alpha=0.05)
    ax1.axvspan(12.5, 22.5, color="#50b050", alpha=0.05)
    ax1.axvspan(22.5, 27.5, color="#4080c0", alpha=0.05)
    # 범례 패치
    leg_patches = [
        mpatches.Patch(color="#e06060", alpha=0.6, label="#1-7  이마/관자놀이"),
        mpatches.Patch(color="#e0a030", alpha=0.6, label="#8-13 눈/광대"),
        mpatches.Patch(color="#50b050", alpha=0.6, label="#14-23 볼/귀/이하선"),
        mpatches.Patch(color="#4080c0", alpha=0.6, label="#24-28 턱/목"),
    ]
    ax1.legend(handles=leg_patches + [ax1.get_lines()[0]], fontsize=8, loc="upper right")

    # ── 패널 2: T0 추이 ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.errorbar(x, t0_means, yerr=t0_stds, fmt="-o",
                 color="#3070b0", ecolor="lightsteelblue", capsize=3,
                 markersize=5, lw=1.5, label="T0 평균±std")
    for j, pat in enumerate(patients):
        sub = df[df["name"] == pat]
        for _, row in sub.iterrows():
            xi = pos_nums.index(row["pos_num"])
            ax2.scatter(xi, row["t0_us"],
                        color=cmap_pat[j], s=14, alpha=0.45, zorder=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=8)
    ax2.set_ylabel("T0 (μs, 표면 반사 시간)", fontsize=10)
    ax2.set_title("번호별 T0 (표면 반사 시작점) 추이",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.axvspan(-0.5,  6.5, color="#e06060", alpha=0.05)
    ax2.axvspan( 6.5, 12.5, color="#e0a030", alpha=0.05)
    ax2.axvspan(12.5, 22.5, color="#50b050", alpha=0.05)
    ax2.axvspan(22.5, 27.5, color="#4080c0", alpha=0.05)

    # ── 패널 3: 박스플롯 (epi_mm) ────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    data_epi = [df[df["pos_num"] == p]["epi_mm"].values for p in pos_nums]
    bp = ax3.boxplot(data_epi, patch_artist=True, notch=False,
                     medianprops=dict(color="black", lw=2), widths=0.65)
    for patch, c in zip(bp["boxes"], zone_colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax3.set_xticklabels(short_labels, fontsize=7, rotation=45, ha="right")
    ax3.axhline(overall_mean, color="red", ls="--", lw=1.2,
                label=f"전체 평균 {overall_mean:.3f}mm")
    ax3.set_ylabel("표피두께 (mm)"); ax3.set_title("번호별 표피두께 박스플롯",
                                                fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.25, axis="y")

    # ── 패널 4: 환자별 히트맵 (epi_mm) ──────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    hmap = np.full((n, len(patients)), np.nan)
    for i, pn in enumerate(pos_nums):
        for j, pat in enumerate(patients):
            vals = df[(df["pos_num"] == pn) & (df["name"] == pat)]["epi_mm"].values
            if len(vals): hmap[i, j] = vals[0]
    vlo, vhi = np.nanquantile(hmap, 0.05), np.nanquantile(hmap, 0.95)
    im = ax4.imshow(hmap, aspect="auto", cmap="RdYlGn", vmin=vlo, vmax=vhi)
    ax4.set_xticks(range(len(patients)))
    ax4.set_xticklabels([p[:4] for p in patients], rotation=45, ha="right", fontsize=6)
    ax4.set_yticks(range(n))
    ax4.set_yticklabels(short_labels, fontsize=7)
    plt.colorbar(im, ax=ax4, label="epi_mm")
    ax4.set_title("번호×환자 표피두께 히트맵", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"얼굴 번호별(#1–28) 전체 분석  "
        f"(총 {len(df)}건 / {len(patients)}명)\n"
        f"전체 평균: epi={df['epi_mm'].mean():.3f}±{df['epi_mm'].std():.3f}mm  "
        f"T0={df['t0_us'].mean():.3f}±{df['t0_us'].std():.3f}μs",
        fontsize=13, fontweight="bold"
    )

    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\n그래프 저장: {OUT_PNG}")
    print(f"통계 저장:  {OUT_CSV}")

    # ── 텍스트 요약 ───────────────────────────────────────────────────
    print("\n[번호별 표피두께 요약 (공간 순서)]")
    for p in pos_nums:
        m = stats.loc[p, "epi_mm_mean"]
        s = stats.loc[p, "epi_mm_std"]
        bar = "█" * int(m * 35)
        print(f"  #{p:2d} {pos_label[p]:30s}  {m:.3f}±{s:.3f}mm  {bar}")


if __name__ == "__main__":
    main()
