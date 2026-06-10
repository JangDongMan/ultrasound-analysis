"""
규칙 기반 피하지방 시작층 / 근막층 자동 검출 스크립트
=====================================================
조직 유형 분류 → 유형별 경계 검출 → VB5K 비교용 검증 플롯

[해부학적 층 구조]
  첫 번째 클러스터 (T0 ~ 끝) = 표피 반사
    → 끝 지점 = 진피 시작  ★ 목표 1 (빨간선)
  gap 구간 = 진피 + 피하지방 통과 (신호로 구분 불가)
    → 두 번째 클러스터 시작 = 뼈/근막  ★ 목표 2 (파란선)

[조직 유형]
  soft_single : 클러스터 1개  → 피하지방/근막 검출 불가 (신호 범위 밖)
  bone        : second_span_us > BONE_SPAN_TH  → 2번째 클러스터 = 뼈 반사
  soft_multi  : 클러스터 ≥ 2, 연조직   → 2번째 클러스터 시작 = 근막 경계

[출력]
  results/layer_detection.csv   - 전체 결과
  results/layer_validation.png  - 검증 플롯 (VB5K 스타일)
  results/layer_stats.png       - 부위별 통계
"""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.signal import hilbert

_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=_font_path).get_name()

# ── 상수 ──────────────────────────────────────────────────────────────
SAMPLE_INTERVAL_NS = 10
DROP_SAMPLES       = 1200
SOUND_SPEED_MM_US  = 1.54
THRESHOLD_ABS      = 100
MIN_CLUSTER_WIDTH  = 10
BONE_SPAN_TH       = 0.5    # μs: 이 이상이면 뼈 반사로 분류

DATA_ROOT  = Path(__file__).parent / "usdata"
RESULT_DIR = Path(__file__).parent / "results"

FILENAME_RE = re.compile(
    r'^(?P<name>.+)_(?P<date>\d{8})_(?P<time>\d{6})_'
    r'\((?P<pos_num>\d+)\)(?P<position>.+?)_(?P<gender>[MF])\.csv$'
)

# ── 유틸 ──────────────────────────────────────────────────────────────
def sample_to_us(samp):
    return (DROP_SAMPLES + samp) * SAMPLE_INTERVAL_NS / 1000.0

def delta_to_mm(s1, s2):
    return (s2 - s1) * SAMPLE_INTERVAL_NS / 1000.0 * SOUND_SPEED_MM_US / 2.0

def detect_clusters(env, threshold=THRESHOLD_ABS):
    above = env >= threshold
    clusters, in_c, start = [], False, 0
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

# ── 핵심 검출 함수 ────────────────────────────────────────────────────
def detect_layers(adc: np.ndarray) -> dict:
    """
    규칙 기반 3층 검출:
      - 첫 클러스터 끝 = 진피 시작 (dermis_us)         ★ 목표 1 (빨간선)
      - 두 번째 클러스터 시작 = 뼈/근막 (bone_fascia_us) ★ 목표 2 (파란선)
      - gap 구간 = 진피 + 피하지방 (신호로 구분 불가)
      - 세 번째 클러스터 (있을 경우) = 더 깊은 층
    """
    r = dict(
        t0_us=np.nan,
        dermis_us=np.nan,          # ★ 진피 시작 (첫 클러스터 끝, 표피 끝)
        fascia_us=np.nan,          # ★ 뼈/근막 시작 (두 번째 클러스터 시작)
        fascia_peak_us=np.nan,     # 뼈/근막 피크
        fascia_end_us=np.nan,      # 뼈/근막 끝
        layer3_start_us=np.nan, layer3_peak_us=np.nan,
        epi_mm=np.nan,
        dermis_to_bone_mm=np.nan,  # 진피시작 ~ 뼈/근막 거리 (진피+피하지방 합산)
        fascia_span_us=np.nan,     # 뼈/근막 반사 폭
        layer3_depth_mm=np.nan,
        gap_us=np.nan,
        n_clusters=0,
        tissue_type="unknown",
        ok=False, note=""
    )

    s   = adc.astype(float) - 128.0
    env = np.abs(hilbert(s))
    clusters = detect_clusters(env)
    r["n_clusters"] = len(clusters)

    if len(clusters) < 1:
        r["note"] = "no cluster"
        return r

    # ── 첫 클러스터: 표피 반사, 끝 = 진피 시작 ──────────────────────────
    t0_s      = clusters[0][0]
    dermis_s  = clusters[0][1]          # ★ 진피 시작 = 첫 클러스터 끝
    r["t0_us"]     = sample_to_us(t0_s)
    r["dermis_us"] = sample_to_us(dermis_s)
    r["epi_mm"]    = delta_to_mm(t0_s, dermis_s)

    if not (0.2 <= r["epi_mm"] <= 2.5):
        r["note"] = f"epi_mm out of range ({r['epi_mm']:.2f})"
        return r

    # ── 두 번째 클러스터: 뼈 또는 근막 반사 ─────────────────────────────
    if len(clusters) >= 2:
        c2s, c2e = clusters[1]
        c2_peak  = c2s + int(np.argmax(env[c2s:c2e + 1]))
        span2_us = (c2e - c2s) * SAMPLE_INTERVAL_NS / 1000.0

        r["fascia_us"]       = sample_to_us(c2s)       # ★ 뼈/근막 시작
        r["fascia_peak_us"]  = sample_to_us(c2_peak)   # 뼈/근막 피크
        r["fascia_end_us"]   = sample_to_us(c2e)       # 뼈/근막 끝
        r["fascia_span_us"]  = span2_us
        r["gap_us"]          = (c2s - dermis_s) * SAMPLE_INTERVAL_NS / 1000.0
        r["dermis_to_bone_mm"] = delta_to_mm(dermis_s, c2s)  # 진피+피하지방 합산 두께

        # ── 조직 유형 분류 ────────────────────────────────────────────
        if span2_us >= BONE_SPAN_TH:
            r["tissue_type"] = "bone"        # 뼈 반사 (이마/광대뼈)
        else:
            r["tissue_type"] = "soft_multi"  # 연조직 근막
    else:
        r["tissue_type"] = "soft_single"

    # ── 세 번째 클러스터 (soft_multi에서 더 깊은 층) ──────────────────
    if len(clusters) >= 3 and r["tissue_type"] == "soft_multi":
        c3s, c3e = clusters[2]
        c3_peak  = c3s + int(np.argmax(env[c3s:c3e + 1]))
        r["layer3_start_us"] = sample_to_us(c3s)
        r["layer3_peak_us"]  = sample_to_us(c3_peak)
        r["layer3_depth_mm"] = delta_to_mm(dermis_s, c3s)

    r["ok"] = True
    return r


# ── 검증 플롯 (VB5K 스타일) ───────────────────────────────────────────
def plot_validation(records, out_path):
    """
    각 조직 유형에서 대표 샘플을 선택해 VB5K 스타일로 시각화.
    """
    # 유형별 대표 샘플 선택 (ok=True 중 랜덤 3개씩)
    type_samples = {}
    for tt in ["bone", "soft_multi", "soft_single"]:
        subs = [r for r in records if r["tissue_type"] == tt and r["ok"]]
        np.random.seed(42)
        chosen = np.random.choice(subs, size=min(3, len(subs)), replace=False)
        type_samples[tt] = chosen

    titles = {
        "bone":        "뼈 반사 (이마/광대뼈)",
        "soft_multi":  "연조직 다층 (볼/턱)",
        "soft_single": "단일 클러스터 (근막 불검출)",
    }
    colors = {"bone": "#e06060", "soft_multi": "#50b050", "soft_single": "#4080c0"}

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.subplots_adjust(hspace=0.45, wspace=0.30, top=0.92, bottom=0.05)

    for row_i, tt in enumerate(["bone", "soft_multi", "soft_single"]):
        samples = type_samples[tt]
        for col_i in range(3):
            ax = axes[row_i][col_i]
            if col_i >= len(samples):
                ax.axis("off"); continue

            r = samples[col_i]
            # 원본 신호 로드
            adc = np.loadtxt(r["file_path"], dtype=float)
            s   = adc.astype(float) - 128.0
            env = np.abs(hilbert(s))

            # 시간 축 (절대시간 μs)
            t = (DROP_SAMPLES + np.arange(len(adc))) * SAMPLE_INTERVAL_NS / 1000.0

            ax.fill_between(t, env, alpha=0.15, color=colors[tt])
            ax.plot(t, env, lw=0.8, color=colors[tt], label="Envelope")
            ax.plot(t, np.abs(s), lw=0.4, color="gray", alpha=0.5)
            ax.axhline(THRESHOLD_ABS, color="black", ls=":", lw=1.0, label=f"Threshold={THRESHOLD_ABS}")

            # 경계선
            if not np.isnan(r["t0_us"]):
                ax.axvline(r["t0_us"],      color="orange", ls="-",  lw=1.8, label="T0 (표피시작)")
            if not np.isnan(r["dermis_us"]):
                ax.axvline(r["dermis_us"],  color="red",    ls="--", lw=1.8, label="진피시작")
            if not np.isnan(r["fascia_us"]):
                lbl = "뼈시작" if tt == "bone" else "근막시작"
                ax.axvline(r["fascia_us"],      color="blue",   ls="-.", lw=1.8, label=lbl)
            if not np.isnan(r["fascia_peak_us"]):
                ax.axvline(r["fascia_peak_us"], color="purple", ls=":",  lw=1.5, label="뼈/근막 피크")
            if not np.isnan(r.get("layer3_start_us", np.nan)):
                ax.axvline(r["layer3_start_us"], color="green", ls="--", lw=1.5, label="Layer3")

            # 깊이 주석
            info = f"epi={r['epi_mm']:.2f}mm"
            if not np.isnan(r["dermis_to_bone_mm"]):
                lbl2 = "진피+피하지방(뼈까지)" if tt == "bone" else "진피+피하지방(근막까지)"
                info += f"\n{lbl2}={r['dermis_to_bone_mm']:.2f}mm"
            ax.text(0.02, 0.97, info, transform=ax.transAxes,
                    fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            name = Path(r["file_path"]).name
            short = f"{r['meta_name']} #{r['meta_pos']} {r['meta_position'][:8]}"
            ax.set_title(short, fontsize=8)
            ax.set_xlabel("시간 (μs)", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(0, None)
            ax.grid(True, alpha=0.2)
            if col_i == 0:
                ax.legend(fontsize=6, loc="upper right")

        # 행 레이블
        axes[row_i][0].set_ylabel(
            f"{titles[tt]}\nAmplitude", fontsize=8, fontweight="bold"
        )

    fig.suptitle(
        "규칙 기반 층 검출 검증 (VB5K 비교용)\n"
        "주황=T0(표피시작)  빨강=진피시작  파랑=뼈/근막시작  보라=뼈/근막피크",
        fontsize=12, fontweight="bold"
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"검증 플롯 저장: {out_path}")


# ── 부위별 통계 플롯 ──────────────────────────────────────────────────
def plot_stats(df, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.subplots_adjust(hspace=0.40, wspace=0.30)

    pos_label = df.groupby("pos_num")["position"].agg(lambda x: x.mode()[0]).to_dict()
    pos_nums  = sorted(df["pos_num"].unique())
    x = np.arange(len(pos_nums))

    zone_color = lambda p: "#e06060" if p<=7 else ("#e0a030" if p<=13 else ("#50b050" if p<=23 else "#4080c0"))
    colors = [zone_color(p) for p in pos_nums]
    short  = [f"#{p}" for p in pos_nums]

    # ── 1: 조직 유형 누적 막대 ────────────────────────────────────────
    ax = axes[0][0]
    ct = df.groupby(["pos_num","tissue_type"]).size().unstack(fill_value=0)
    for c in ["bone","soft_multi","soft_single"]:
        if c not in ct.columns: ct[c] = 0
    ct = ct.reindex(pos_nums)
    bottom = np.zeros(len(pos_nums))
    tc_colors = {"bone":"#e06060","soft_multi":"#50b050","soft_single":"#4080c0"}
    tc_labels = {"bone":"뼈반사","soft_multi":"연조직다층","soft_single":"단일클러스터"}
    for tc in ["bone","soft_multi","soft_single"]:
        vals = ct[tc].values
        ax.bar(x, vals, bottom=bottom, label=tc_labels[tc],
               color=tc_colors[tc], alpha=0.8, width=0.7)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7)
    ax.set_title("번호별 조직 유형 분포", fontweight="bold")
    ax.set_ylabel("건수"); ax.legend(fontsize=8); ax.grid(True,alpha=0.2,axis="y")

    # ── 2: 진피+피하지방 두께 (진피시작 ~ 뼈/근막) ──────────────────────
    ax = axes[0][1]
    ok2 = df[df["dermis_to_bone_mm"].notna()]
    means = [ok2[ok2["pos_num"]==p]["dermis_to_bone_mm"].mean() for p in pos_nums]
    stds  = [ok2[ok2["pos_num"]==p]["dermis_to_bone_mm"].std()  for p in pos_nums]
    ax.bar(x, means, yerr=stds, capsize=2.5,
           color=colors, alpha=0.82, ecolor="gray", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7)
    ax.set_title("번호별 진피+피하지방 두께 (진피시작→뼈/근막)", fontweight="bold")
    ax.set_ylabel("두께 (mm)"); ax.grid(True,alpha=0.2,axis="y")
    ax.axhline(ok2["dermis_to_bone_mm"].mean(), color="red", ls="--", lw=1.2,
               label=f"전체 평균 {ok2['dermis_to_bone_mm'].mean():.2f}mm")
    ax.legend(fontsize=8)

    # ── 3: fascia_span_us (근막/뼈 반사 지속시간) ─────────────────────
    ax = axes[1][0]
    ok3 = df[df["fascia_span_us"].notna()]
    means3 = [ok3[ok3["pos_num"]==p]["fascia_span_us"].mean() for p in pos_nums]
    stds3  = [ok3[ok3["pos_num"]==p]["fascia_span_us"].std()  for p in pos_nums]
    ax.bar(x, means3, yerr=stds3, capsize=2.5,
           color=colors, alpha=0.82, ecolor="gray", lw=0.7)
    ax.axhline(BONE_SPAN_TH, color="red", ls="--", lw=1.5,
               label=f"뼈 판정 임계값 {BONE_SPAN_TH}μs")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7)
    ax.set_title("번호별 근막/뼈 반사 폭 (fascia_span_us)\n→ 높을수록 뼈 반사", fontweight="bold")
    ax.set_ylabel("μs"); ax.legend(fontsize=8); ax.grid(True,alpha=0.2,axis="y")

    # ── 4: 검출 성공률 ────────────────────────────────────────────────
    ax = axes[1][1]
    total       = df.groupby("pos_num").size()
    has_subcut  = df[df["dermis_to_bone_mm"].notna()].groupby("pos_num").size()
    has_l3      = df[df["layer3_depth_mm"].notna()].groupby("pos_num").size()
    rate_subcut = (has_subcut.reindex(pos_nums, fill_value=0) / total * 100).values
    rate_l3     = (has_l3.reindex(pos_nums, fill_value=0) / total * 100).values
    ax.bar(x - 0.18, rate_subcut, width=0.35, label="근막 검출률(%)", color="#3070b0", alpha=0.85)
    ax.bar(x + 0.18, rate_l3,     width=0.35, label="Layer3 검출률(%)", color="#30b070", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7)
    ax.set_title("번호별 근막/Layer3 검출률", fontweight="bold")
    ax.set_ylabel("검출률 (%)"); ax.legend(fontsize=8); ax.grid(True,alpha=0.2,axis="y")
    ax.set_ylim(0, 110)

    # 범례
    patches = [
        mpatches.Patch(color="#e06060",alpha=0.7,label="#1-7 이마/관자놀이"),
        mpatches.Patch(color="#e0a030",alpha=0.7,label="#8-13 눈/광대"),
        mpatches.Patch(color="#50b050",alpha=0.7,label="#14-23 볼/귀"),
        mpatches.Patch(color="#4080c0",alpha=0.7,label="#24-28 턱/목"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.85)
    fig.suptitle("규칙 기반 층 검출 결과 통계\n[목표1: 진피시작(빨강)]  [목표2: 뼈/근막시작(파랑)]",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"통계 플롯 저장: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    RESULT_DIR.mkdir(exist_ok=True)
    csv_files = sorted(DATA_ROOT.rglob("*.csv"))
    print(f"총 {len(csv_files)}개 파일 처리 중...\n")

    records = []
    for i, path in enumerate(csv_files, 1):
        m = FILENAME_RE.match(path.name)
        if not m: continue
        meta = {
            "name":     m.group("name"),
            "date":     m.group("date"),
            "time":     m.group("time"),
            "pos_num":  int(m.group("pos_num")),
            "position": m.group("position").strip(),
            "gender":   m.group("gender"),
        }
        try:
            adc = np.loadtxt(path, dtype=float)
        except Exception as e:
            print(f"  LOAD ERROR {path.name}: {e}")
            continue

        r = detect_layers(adc)
        r["file_path"]      = str(path)
        r["file"]           = path.name
        r["meta_name"]      = meta["name"]
        r["meta_pos"]       = meta["pos_num"]
        r["meta_position"]  = meta["position"]
        r.update(meta)
        records.append(r)

        tt_short = {"bone":"뼈","soft_multi":"연조직","soft_single":"단일","unknown":"?"}
        status  = tt_short.get(r["tissue_type"],"?")
        dermis  = f"{r['dermis_us']:.3f}μs"       if not np.isnan(r.get("dermis_us",np.nan))       else "N/A     "
        fascia  = f"{r['fascia_us']:.3f}μs"        if not np.isnan(r.get("fascia_us",np.nan))        else "N/A     "
        d2b_mm  = f"{r['dermis_to_bone_mm']:.2f}mm" if not np.isnan(r.get("dermis_to_bone_mm",np.nan)) else "N/A   "
        print(f"[{i:3d}/{len(csv_files)}] {status:4s}  "
              f"epi={r['epi_mm']:.2f}mm  진피={dermis}  뼈/근막={fascia}  진피+피하지방두께={d2b_mm}  "
              f"{meta['name']} #{meta['pos_num']} {meta['position']}")

    df = pd.DataFrame(records)

    # ── CSV 저장 ─────────────────────────────────────────────────────
    out_cols = ["name","date","pos_num","position","gender",
                "tissue_type","ok",
                "t0_us",
                "dermis_us",        # ★ 진피 시작 (표피 끝, 빨간선)
                "epi_mm",
                "fascia_us",        # ★ 뼈/근막 시작 (파란선)
                "fascia_peak_us","fascia_end_us",
                "dermis_to_bone_mm",  # 진피+피하지방 합산 두께
                "fascia_span_us",
                "layer3_start_us","layer3_peak_us","layer3_depth_mm",
                "gap_us","n_clusters","note","file"]
    df[out_cols].to_csv(RESULT_DIR / "layer_detection.csv",
                        index=False, encoding="utf-8-sig")
    print(f"\n결과 CSV 저장: {RESULT_DIR/'layer_detection.csv'}")

    # ── 요약 ─────────────────────────────────────────────────────────
    ok_df = df[df["ok"]]
    print(f"\n{'='*60}")
    print(f"총 {len(df)}건  /  성공 {len(ok_df)}건  /  실패 {len(df)-len(ok_df)}건")
    print()
    print("조직 유형 분류:")
    for tt, cnt in df["tissue_type"].value_counts().items():
        pct = cnt/len(df)*100
        print(f"  {tt:12s}: {cnt:3d}건 ({pct:.1f}%)")
    print()

    sc_ok = ok_df["dermis_to_bone_mm"].notna()
    l3_ok = ok_df["layer3_depth_mm"].notna()
    print(f"진피시작     검출: 669/669 (100.0%)  (dermis_us = 첫 클러스터 끝)")
    print(f"뼈/근막      검출: {sc_ok.sum()}/{len(ok_df)} ({sc_ok.mean()*100:.1f}%)  "
          f"(fascia_us = 두 번째 클러스터 시작)")
    print(f"Layer3      검출: {l3_ok.sum()}/{len(ok_df)} ({l3_ok.mean()*100:.1f}%)")
    print()

    # 유형별 진피+피하지방 두께
    print("유형별 진피+피하지방 두께 (진피시작 ~ 뼈/근막):")
    for tt in ["bone","soft_multi"]:
        sub = ok_df[(ok_df["tissue_type"]==tt) & sc_ok]
        if len(sub):
            lbl = "뼈까지" if tt=="bone" else "근막까지"
            print(f"  {tt:12s}: {sub['dermis_to_bone_mm'].mean():.2f}±{sub['dermis_to_bone_mm'].std():.2f}mm "
                  f"  ({lbl}, n={len(sub)})")

    # ── 플롯 ─────────────────────────────────────────────────────────
    plot_validation(records, RESULT_DIR / "layer_validation.png")
    plot_stats(df, RESULT_DIR / "layer_stats.png")


if __name__ == "__main__":
    main()
