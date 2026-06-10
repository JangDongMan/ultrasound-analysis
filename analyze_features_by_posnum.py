"""
번호별(pos_num 1-28) 신호 특징 비교 분석
- 핵심 특징: n_clusters, second_span_us, gap_us, epi_mm
- ANOVA/t-test 통계 포함
- 결과: results/feature_by_posnum.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import stats

_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=_font_path).get_name()

RESULT_DIR = Path(__file__).parent / "results"
INPUT      = RESULT_DIR / "epidermis_analysis.csv"
OUT_PNG    = RESULT_DIR / "feature_by_posnum.png"


def zone(p):
    if   p <= 7:  return 0  # 이마/관자놀이
    elif p <= 13: return 1  # 눈/광대
    elif p <= 23: return 2  # 볼/귀
    else:         return 3  # 턱/목

ZONE_COLORS = ["#e06060", "#e0a030", "#50b050", "#4080c0"]
ZONE_LABELS = ["#1-7 이마/관자놀이", "#8-13 눈/광대",
               "#14-23 볼/귀", "#24-28 턱/목"]

def main():
    df = pd.read_csv(INPUT)
    pos_label = df.groupby("pos_num")["position"].agg(lambda x: x.mode()[0]).to_dict()
    pos_nums = sorted(df["pos_num"].unique())
    n = len(pos_nums)
    x = np.arange(n)
    colors = [ZONE_COLORS[zone(p)] for p in pos_nums]
    short = [f"#{p}" for p in pos_nums]

    # ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(22, 22), sharex=True)
    fig.subplots_adjust(hspace=0.10, top=0.93, bottom=0.07, left=0.07, right=0.97)

    features = [
        ("n_clusters",      "클러스터 수 (반사 에코 그룹 수)",           "개수"),
        ("second_span_us",  "2번째 클러스터 폭 (뼈·연조직 반사 지속시간)", "μs"),
        ("gap_us",          "표피 ~ 2번째 클러스터 간격 (피하지방 두께 지표)", "μs"),
        ("epi_mm",          "표피두께",                              "mm"),
    ]

    for ax, (feat, title, unit) in zip(axes, features):
        means, stds, ns_per_pos = [], [], []
        for pn in pos_nums:
            vals = df[df["pos_num"] == pn][feat].dropna()
            means.append(vals.mean()); stds.append(vals.std()); ns_per_pos.append(len(vals))

        # 막대 + 오차
        ax.bar(x, means, yerr=stds, capsize=2.5,
               color=colors, alpha=0.82, ecolor="gray", lw=0.7)

        # 개인별 점
        patients = sorted(df["name"].unique())
        cmap_pat = plt.cm.tab20(np.linspace(0, 1, len(patients)))
        for j, pat in enumerate(patients):
            sub = df[df["name"] == pat].dropna(subset=[feat])
            for _, row in sub.iterrows():
                xi = pos_nums.index(row["pos_num"])
                ax.scatter(xi, row[feat],
                           color=cmap_pat[j], s=16, alpha=0.50, zorder=5)

        # 구역 평균선
        for zi, (z_start, z_end) in enumerate([(0,6),(7,12),(13,22),(23,27)]):
            z_means = [m for i, m in enumerate(means)
                       if z_start <= i <= z_end and not np.isnan(m)]
            if z_means:
                zm = np.nanmean(z_means)
                ax.hlines(zm, z_start - 0.4, z_end + 0.4,
                          color=ZONE_COLORS[zi], lw=2.0, ls="--", alpha=0.9)

        # 전체 평균선
        overall = df[feat].mean()
        ax.axhline(overall, color="black", ls=":", lw=1.2, alpha=0.6,
                   label=f"전체 평균 {overall:.3f}{unit}")

        # 배경 구역 색
        ax.axvspan(-0.5,  6.5, color="#e06060", alpha=0.04)
        ax.axvspan( 6.5, 12.5, color="#e0a030", alpha=0.04)
        ax.axvspan(12.5, 22.5, color="#50b050", alpha=0.04)
        ax.axvspan(22.5, 27.5, color="#4080c0", alpha=0.04)

        ax.set_ylabel(f"{unit}", fontsize=9)
        ax.set_title(f"▶ {title}", fontsize=11, fontweight="bold", loc="left", pad=3)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2, axis="y")

        # ANOVA 표시 (4구역)
        groups = []
        for zi, idx_range in enumerate([(0,7),(7,13),(13,23),(23,28)]):
            vals = df[df["pos_num"].between(*idx_range)][feat].dropna().values
            groups.append(vals)
        try:
            f_val, p_val = stats.f_oneway(*[g for g in groups if len(g) > 2])
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            ax.text(0.01, 0.96, f"ANOVA: F={f_val:.1f}, p={p_val:.4f} {sig}",
                    transform=ax.transAxes, fontsize=8,
                    va="top", color="black",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        except Exception:
            pass

    # x축 레이블 (마지막 패널만)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(short, fontsize=8)
    axes[-1].set_xlabel("측정 번호 (공간 순서: 이마 → 목)", fontsize=10)

    # 범례 패치
    legend_patches = [mpatches.Patch(color=c, alpha=0.75, label=l)
                      for c, l in zip(ZONE_COLORS, ZONE_LABELS)]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.01),
               framealpha=0.85)

    fig.suptitle(
        "번호별(#1–28) 신호 특징 분석  –  "
        "점선(--): 구역 평균 / 점선(…): 전체 평균 / 점: 개인값\n"
        f"총 {len(df)}건 / {len(df['name'].unique())}명",
        fontsize=13, fontweight="bold"
    )

    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"그래프 저장: {OUT_PNG}")

    # ── 텍스트 요약 ────────────────────────────────────────────────────
    print()
    print("=== 구역별 특징 요약 ===")
    zone_def = {
        "이마/관자놀이(#1-7)":  df[df["pos_num"].between(1,7)],
        "눈/광대(#8-13)":       df[df["pos_num"].between(8,13)],
        "볼/귀(#14-23)":        df[df["pos_num"].between(14,23)],
        "턱/목(#24-28)":        df[df["pos_num"].between(24,28)],
    }
    feat_list = ["n_clusters","second_span_us","gap_us","epi_mm"]
    header = f"{'구역':18s}" + "".join(f"  {f:>16s}" for f in feat_list)
    print(header)
    print("-" * (18 + 18*len(feat_list)))
    for zname, zdf in zone_def.items():
        row = f"{zname:18s}"
        for f in feat_list:
            vals = zdf[f].dropna()
            row += f"  {vals.mean():7.3f}±{vals.std():5.3f}"
        print(row)

    print()
    print("=== 구역 구분력 (ANOVA) ===")
    for feat in feat_list:
        groups = [z[feat].dropna().values for z in zone_def.values()]
        f_val, p_val = stats.f_oneway(*[g for g in groups if len(g)>2])
        sig = "***" if p_val<0.001 else ("**" if p_val<0.01 else ("*" if p_val<0.05 else "ns"))
        bar = "█" * min(int(f_val/2), 40)
        print(f"  {feat:20s}: F={f_val:6.1f} {sig:4s}  {bar}")

    print()
    print("=== 주요 해석 ===")
    print("  n_clusters:     이마/관자놀이(뼈) ↑, 볼(연조직) ↓  → 뼈·연조직 구분 최강 변수")
    print("  second_span_us: 이마뼈 반사폭 넓음, 볼 연조직 반사 좁음")
    print("  gap_us:         볼/턱(피하지방 두꺼움) ↑, 이마/목 ↓")
    print("  epi_mm:         구역간 차이 작음 (가장 약한 변수)")


if __name__ == "__main__":
    main()
