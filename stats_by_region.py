"""
부위별 표피 두께 통계 분석
- results/epidermis_analysis.csv 읽기
- 부위별 / 환자별 통계 출력 및 시각화
- 결과: results/stats_by_region.png, results/stats_by_region.csv
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

RESULT_DIR  = Path(__file__).parent / "results"
INPUT_FILE  = RESULT_DIR / "epidermis_analysis.csv"
OUT_PNG     = RESULT_DIR / "stats_by_region.png"
OUT_CSV     = RESULT_DIR / "stats_by_region.csv"


def region_group(position: str) -> str:
    """부위 이름을 대분류로 묶기"""
    p = position
    if "이마" in p:                       return "이마"
    if "관자놀이" in p or "광대활" in p:  return "관자놀이/광대"
    if "눈" in p or "눈썹" in p:          return "눈 주변"
    if "코" in p or "콧구멍" in p:        return "코 주변"
    if "볼" in p or "광대뼈" in p:        return "볼"
    if "귀" in p or "이하선" in p:        return "귀/이하선"
    if "턱" in p:                          return "턱"
    if "잎술" in p or "인중" in p:        return "입술/인중"
    if "목" in p:                          return "목"
    return "기타"


def main():
    df = pd.read_csv(INPUT_FILE)
    ok = df[df["ok"] == True].copy()
    ok["region"] = ok["position"].apply(region_group)

    # ── 콘솔 통계 ─────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"유효 샘플: {len(ok)} / 전체: {len(df)}")
    print(f"{'='*60}")

    print("\n[대분류별 표피 두께]")
    reg_stats = (ok.groupby("region")["epi_mm"]
                   .agg(["count", "mean", "std", "min", "max"])
                   .rename(columns={"count":"n","mean":"avg","std":"std","min":"min","max":"max"})
                   .sort_values("avg", ascending=False))
    reg_stats["avg"] = reg_stats["avg"].round(2)
    reg_stats["std"] = reg_stats["std"].round(2)
    reg_stats["min"] = reg_stats["min"].round(2)
    reg_stats["max"] = reg_stats["max"].round(2)
    print(reg_stats.to_string())

    print("\n[세부 부위별 표피 두께 (상위 10개)]")
    pos_stats = (ok.groupby("position")["epi_mm"]
                   .agg(n="count", avg="mean", std="std")
                   .sort_values("avg", ascending=False)
                   .head(10))
    pos_stats["avg"] = pos_stats["avg"].round(2)
    pos_stats["std"] = pos_stats["std"].round(2)
    print(pos_stats.to_string())

    print("\n[성별 비교]")
    gender_stats = ok.groupby("gender")["epi_mm"].agg(n="count", avg="mean", std="std")
    gender_stats["avg"] = gender_stats["avg"].round(2)
    gender_stats["std"] = gender_stats["std"].round(2)
    print(gender_stats.to_string())

    # ── CSV 저장 ───────────────────────────────────────────────
    reg_stats.to_csv(OUT_CSV, encoding="utf-8-sig")
    print(f"\n통계 CSV 저장: {OUT_CSV}")

    # ── 시각화 ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) 대분류별 박스플롯
    ax = axes[0, 0]
    regions = reg_stats.index.tolist()
    data_by_region = [ok[ok["region"] == r]["epi_mm"].values for r in regions]
    bp = ax.boxplot(data_by_region, labels=regions, patch_artist=True, notch=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(regions)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("대분류별 표피 두께 분포", fontsize=12, fontweight="bold")
    ax.set_ylabel("표피 두께 (mm)")
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(ok["epi_mm"].mean(), color="red", ls="--", lw=1.2, label=f"전체 평균 {ok['epi_mm'].mean():.2f}mm")
    ax.legend(fontsize=9)

    # 2) 환자별 평균 막대그래프
    ax = axes[0, 1]
    pat_avg = ok.groupby("name")["epi_mm"].agg(avg="mean", std="std").sort_values("avg")
    colors_pat = plt.cm.tab20(np.linspace(0, 1, len(pat_avg)))
    bars = ax.barh(pat_avg.index, pat_avg["avg"], xerr=pat_avg["std"],
                   color=colors_pat, alpha=0.8, capsize=4)
    ax.axvline(ok["epi_mm"].mean(), color="red", ls="--", lw=1.2)
    ax.set_title("환자별 평균 표피 두께", fontsize=12, fontweight="bold")
    ax.set_xlabel("표피 두께 (mm)")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, (_, row) in zip(bars, pat_avg.iterrows()):
        ax.text(row["avg"] + row["std"] + 0.05, bar.get_y() + bar.get_height()/2,
                f"{row['avg']:.2f}", va="center", fontsize=8)

    # 3) 전체 표피 두께 히스토그램
    ax = axes[1, 0]
    ax.hist(ok["epi_mm"], bins=30, color="#4a9eff", alpha=0.75, edgecolor="white")
    ax.axvline(ok["epi_mm"].mean(), color="red", ls="--", lw=1.5,
               label=f"평균 {ok['epi_mm'].mean():.2f}mm")
    ax.axvline(ok["epi_mm"].median(), color="orange", ls="--", lw=1.5,
               label=f"중앙값 {ok['epi_mm'].median():.2f}mm")
    ax.set_title("표피 두께 전체 분포", fontsize=12, fontweight="bold")
    ax.set_xlabel("표피 두께 (mm)")
    ax.set_ylabel("빈도")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4) 부위별 세부 평균 (수평 막대)
    ax = axes[1, 1]
    pos_full = (ok.groupby("position")["epi_mm"]
                  .agg(avg="mean", std="std", n="count")
                  .sort_values("avg"))
    ax.barh(pos_full.index, pos_full["avg"], xerr=pos_full["std"],
            color="#ff8844", alpha=0.75, capsize=3)
    ax.axvline(ok["epi_mm"].mean(), color="red", ls="--", lw=1.2)
    ax.set_title("세부 부위별 평균 표피 두께", fontsize=12, fontweight="bold")
    ax.set_xlabel("표피 두께 (mm)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"표피 두께 통계 분석  (유효 {len(ok)}건 / 20명 / 3일)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"그래프 저장: {OUT_PNG}")


if __name__ == "__main__":
    main()
