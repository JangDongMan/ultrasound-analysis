"""
얼굴 부위 분류기 학습
- 입력: epi_mm, gap_us, second_span_us, n_clusters (신호 특징 4가지)
- 출력: 7개 대분류 부위 예측
- 모델: Random Forest (교차 검증 + 혼동 행렬 + 특징 중요도)
- 저장: results/region_classifier.pkl, results/region_classifier.png
"""

import pickle
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

RESULT_DIR  = Path(__file__).parent / "results"
DATA_ROOT   = Path(__file__).parent / "usdata"
INPUT_FILE  = RESULT_DIR / "epidermis_analysis.csv"
OUT_MODEL   = RESULT_DIR / "region_classifier.pkl"
OUT_PNG     = RESULT_DIR / "region_classifier.png"

THRESHOLD_ABS     = 100
MIN_CLUSTER_WIDTH = 10
DROP_SAMPLES      = 1200
DT_US             = 0.01   # 10ns
MAX_CLUSTERS      = 5      # 최대 클러스터 수

# ── 대분류 정의 ───────────────────────────────────────────────────────
def region_group(pos: str) -> str:
    if "이마" in pos:                           return "이마"
    if "관자놀이" in pos or "광대활" in pos:    return "관자놀이/광대"
    if "눈" in pos or "눈썹" in pos:            return "눈 주변"
    if "코" in pos or "콧구멍" in pos:          return "코 주변"
    if "볼" in pos or "광대뼈" in pos:          return "볼"
    if "귀" in pos or "이하선" in pos:          return "귀/이하선"
    if "턱" in pos or "잎술" in pos or "인중" in pos: return "턱/입술"
    if "목" in pos:                             return "목"
    return "기타"

# ── 신호에서 풍부한 특징 추출 ────────────────────────────────────────
def extract_signal_features(path: Path) -> dict:
    """원시 신호에서 다중 클러스터 특징 추출."""
    adc = np.loadtxt(path, dtype=float)
    s   = adc - 128.0
    env = np.abs(hilbert(s))

    # 클러스터 검출
    above = env >= THRESHOLD_ABS
    clusters = []
    in_c, start = False, 0
    for i, v in enumerate(above):
        if v and not in_c:   in_c, start = True, i
        elif not v and in_c:
            in_c = False
            if i-1-start >= MIN_CLUSTER_WIDTH: clusters.append((start, i-1))
    if in_c and len(env)-1-start >= MIN_CLUSTER_WIDTH:
        clusters.append((start, len(env)-1))

    feats = {"n_clusters": len(clusters)}

    # 클러스터별 특징 (최대 MAX_CLUSTERS개)
    for k in range(MAX_CLUSTERS):
        pfx = f"c{k}_"
        if k < len(clusters):
            s_i, e_i = clusters[k]
            peak_idx  = s_i + int(np.argmax(env[s_i:e_i+1]))
            start_us  = (DROP_SAMPLES + s_i) * DT_US
            end_us    = (DROP_SAMPLES + e_i) * DT_US
            peak_us   = (DROP_SAMPLES + peak_idx) * DT_US
            feats[pfx+"start_us"]  = start_us
            feats[pfx+"end_us"]    = end_us
            feats[pfx+"peak_us"]   = peak_us
            feats[pfx+"span_us"]   = end_us - start_us
            feats[pfx+"peak_amp"]  = float(env[peak_idx])
            feats[pfx+"mean_amp"]  = float(env[s_i:e_i+1].mean())
        else:
            # 클러스터 없음 → 패딩
            feats[pfx+"start_us"]  = 24.5   # 데이터 끝
            feats[pfx+"end_us"]    = 24.5
            feats[pfx+"peak_us"]   = 24.5
            feats[pfx+"span_us"]   = 0.0
            feats[pfx+"peak_amp"]  = 0.0
            feats[pfx+"mean_amp"]  = 0.0

    # 클러스터 간 gap 특징
    for k in range(1, MAX_CLUSTERS):
        pfx = f"gap{k-1}{k}_"
        if k < len(clusters):
            gap = (clusters[k][0] - clusters[k-1][1]) * DT_US
        else:
            gap = 12.5
        feats[pfx+"us"] = gap

    # 전체 신호 에너지 분포 (4구간)
    seg = len(env) // 4
    for q in range(4):
        feats[f"energy_q{q}"] = float(env[q*seg:(q+1)*seg].mean())

    return feats


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        path = next(DATA_ROOT.rglob(row["file"]), None)
        if path is None:
            continue
        feats = extract_signal_features(path)
        feats["_idx"] = row.name
        rows.append(feats)
    feat_df = pd.DataFrame(rows).set_index("_idx")
    return feat_df


def main():
    df = pd.read_csv(INPUT_FILE)
    df["region"] = df["position"].apply(region_group)

    print("신호 특징 추출 중...")
    X = build_feature_matrix(df)
    y = df.loc[X.index, "region"]
    print(f"특징 수: {X.shape[1]}")


    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_

    print(f"샘플 수: {len(X)}")
    print(f"클래스: {list(class_names)}")
    print(f"\n[클래스별 샘플 수]")
    print(y.value_counts().to_string())

    # ── 모델 학습 (5-fold CV) ────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y_enc, cv=cv, scoring="accuracy")
    y_pred_cv = cross_val_predict(rf, X, y_enc, cv=cv)

    print(f"\n[5-fold CV 정확도]")
    print(f"  {scores.round(3)}  →  평균 {scores.mean():.3f} ± {scores.std():.3f}")

    print(f"\n[분류 리포트 (CV 예측)]")
    print(classification_report(y_enc, y_pred_cv, target_names=class_names))

    # ── 전체 데이터로 최종 모델 학습 ─────────────────────────────────
    rf.fit(X, y_enc)
    feature_names = list(X.columns)
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"[상위 특징 중요도]")
    print(importances.head(10).round(4).to_string())

    # ── 모델 저장 ────────────────────────────────────────────────────
    with open(OUT_MODEL, "wb") as f:
        pickle.dump({"model": rf, "label_encoder": le, "features": feature_names}, f)
    print(f"\n모델 저장: {OUT_MODEL}")

    # ── 시각화 ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # 1) 혼동 행렬
    cm = confusion_matrix(y_enc, y_pred_cv)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"혼동 행렬 (5-fold CV)\n평균 정확도: {scores.mean():.1%}", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)

    # 2) 특징 중요도 (상위 15개)
    top_imp = importances.head(15)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_imp)))
    axes[1].barh(top_imp.index, top_imp.values, color=colors)
    axes[1].set_title("특징 중요도 상위 15 (Random Forest)", fontsize=11)
    axes[1].set_xlabel("Importance")
    for i, (name, val) in enumerate(top_imp.items()):
        axes[1].text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="x")

    # 3) 부위별 특징 산점도 (c0_span_us vs gap01_us)
    colors_cls = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    color_map  = dict(zip(class_names, colors_cls))
    Xv = X.copy(); Xv["region"] = y
    for region, grp in Xv.groupby("region"):
        axes[2].scatter(grp["c0_span_us"], grp["gap01_us"],
                       c=[color_map[region]], label=region, alpha=0.6, s=25)
    axes[2].set_xlabel("표피 폭 (c0_span_us, mm×0.77)")
    axes[2].set_ylabel("gap 0→1 (μs)")
    axes[2].set_title("부위별 특징 분포\n(표피폭 vs gap)", fontsize=11)
    axes[2].legend(fontsize=7, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("얼굴 부위 분류기 학습 결과", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"그래프 저장: {OUT_PNG}")


if __name__ == "__main__":
    main()
