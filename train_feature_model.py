"""
train_feature_model.py
신호 특징 기반 진피/근막 깊이 예측 모델 학습

입력 특징 (4개):
  epi_mm          - 표피 두께
  gap_us          - 표피 끝 ~ 두 번째 클러스터 시작 간격
  second_span_us  - 두 번째 클러스터 폭
  n_clusters      - 전체 클러스터 수

라벨 (2개):
  dermis_mm  - 진피 깊이 (피하지방시작)
  fascia_mm  - 근막 깊이

모델: Random Forest + MLP (비교)
분할: 환자 단위 Leave-One-Patient-Out CV
결과: results/feature_model_cv.png, results/feature_model.pkl
"""

import re
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import hilbert
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

# ── 경로 ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
USDATA     = ROOT / "usdata" / "data"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

# ── 신호 파라미터 ─────────────────────────────────────────────────────────
TRIM_START        = 1200
TRIM_COUNT        = 1250
DISPLAY_OFFSET_US = 12.00
SAMPLE_NS         = 10
THRESHOLD         = 100.0
MIN_CLUSTER_W     = 10
SOUND_SPEED       = 1540.0   # m/s

FILENAME_RE = re.compile(
    r'^(?P<name>.+)_(?P<date>\d{8})_(?P<time>\d{6})_'
    r'\((?P<pos_num>\d+)\)(?P<position>.+?)_(?P<gender>[MF])\.csv$'
)

# ── 특징 추출 ─────────────────────────────────────────────────────────────
def load_adc(csv_path):
    adc = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
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
    return adc


def detect_clusters(env, threshold=THRESHOLD, min_w=MIN_CLUSTER_W):
    above = env >= threshold
    clusters, in_c, start = [], False, 0
    for i, v in enumerate(above):
        if v and not in_c:
            in_c, start = True, i
        elif not v and in_c:
            in_c = False
            if i - 1 - start >= min_w:
                clusters.append((start, i - 1))
    if in_c and len(env) - 1 - start >= min_w:
        clusters.append((start, len(env) - 1))
    return clusters


def extract_features(adc):
    """ADC 배열 → 특징 딕셔너리"""
    s   = adc - 128.0
    env = np.abs(hilbert(s))
    dt  = SAMPLE_NS / 1000.0   # μs/sample

    clusters = detect_clusters(env)
    n_clusters = len(clusters)

    if n_clusters < 1:
        return None

    # 첫 번째 클러스터: 표피
    c0_start, c0_end = clusters[0]
    t0_us     = c0_start * dt + DISPLAY_OFFSET_US
    epi_end_us = c0_end  * dt + DISPLAY_OFFSET_US
    epi_mm    = (epi_end_us - t0_us) * SOUND_SPEED / 2 / 1000

    # 두 번째 클러스터 정보
    if n_clusters >= 2:
        c1_start, c1_end = clusters[1]
        gap_us        = (c1_start - c0_end) * dt
        second_span_us = (c1_end - c1_start) * dt
    else:
        gap_us        = np.nan
        second_span_us = np.nan

    return {
        "t0_us":         t0_us,
        "epi_mm":        epi_mm,
        "gap_us":        gap_us,
        "second_span_us": second_span_us,
        "n_clusters":    n_clusters,
    }


def load_label(csv_path):
    """같은 디렉토리의 _positions.json 에서 dermis_mm, fascia_mm 읽기"""
    json_path = csv_path.with_suffix("").with_suffix("") \
                    if csv_path.stem.endswith("_positions") \
                    else csv_path.with_name(csv_path.stem + "_positions.json")
    if not json_path.exists():
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    dermis_mm = fascia_mm = None
    for pos in data.get("positions", []):
        name = pos.get("position_name", "")
        mm   = pos.get("thickness_mm")
        if name in ("피하지방시작", "Dermis") and mm is not None:
            dermis_mm = mm
        elif name == "Fascia" and mm is not None:
            fascia_mm = mm

    if dermis_mm is None or fascia_mm is None:
        return None
    return {"dermis_mm": dermis_mm, "fascia_mm": fascia_mm}


# ── 데이터셋 구성 ─────────────────────────────────────────────────────────
def build_dataset():
    rows = []
    for csv_path in sorted(USDATA.rglob("*.csv")):
        # _positions.json 파일 자체는 제외
        if "_positions" in csv_path.name:
            continue
        m = FILENAME_RE.match(csv_path.name)
        if not m:
            continue

        label = load_label(csv_path)
        if label is None:
            continue

        adc = load_adc(csv_path)
        if len(adc) < 100:
            continue

        feat = extract_features(adc)
        if feat is None:
            continue

        rows.append({
            "patient":        m.group("name"),
            "pos_num":        int(m.group("pos_num")),
            "position":       m.group("position").strip(),
            "gender":         m.group("gender"),
            **feat,
            **label,
        })

    return pd.DataFrame(rows)


# ── 환자 단위 Leave-One-Out CV ────────────────────────────────────────────
FEATURES = ["epi_mm", "gap_us", "second_span_us", "n_clusters"]
TARGETS  = ["dermis_mm", "fascia_mm"]


def patient_loo_cv(df, model_fn):
    """
    환자 한 명씩 test set으로 사용하는 Leave-One-Patient-Out CV
    Returns: per-patient MAE DataFrame
    """
    patients = sorted(df["patient"].unique())
    records  = []

    for pat in patients:
        train_df = df[df["patient"] != pat].copy()
        test_df  = df[df["patient"] == pat].copy()

        # 결측 제거
        train_df = train_df.dropna(subset=FEATURES + TARGETS)
        test_df  = test_df.dropna(subset=FEATURES + TARGETS)

        if len(train_df) < 10 or len(test_df) == 0:
            continue

        X_train = train_df[FEATURES].values
        y_train = train_df[TARGETS].values
        X_test  = test_df[FEATURES].values
        y_test  = test_df[TARGETS].values

        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae_dermis = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        mae_fascia = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

        records.append({
            "patient":    pat,
            "n_test":     len(test_df),
            "mae_dermis": mae_dermis,
            "mae_fascia": mae_fascia,
        })

    return pd.DataFrame(records)


# ── 최종 모델 학습 (전체 데이터) ──────────────────────────────────────────
def train_final_model(df, model_fn):
    clean = df.dropna(subset=FEATURES + TARGETS)
    X = clean[FEATURES].values
    y = clean[TARGETS].values
    model = model_fn()
    model.fit(X, y)
    return model


# ── 시각화 ────────────────────────────────────────────────────────────────
def plot_cv_results(cv_results: dict, df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Feature-Based Model — Leave-One-Patient-Out CV", fontsize=14)

    model_names = list(cv_results.keys())
    colors      = ["#4c72b0", "#dd8452", "#55a868"]

    # ── 상단: 모델별 MAE 분포 (박스플롯) ─────────────────────────
    for col, target in enumerate(["mae_dermis", "mae_fascia"]):
        ax = axes[0][col]
        data = [cv_results[m][target].values for m in model_names]
        bp = ax.boxplot(data, labels=model_names, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0.3, color="red", linestyle="--", linewidth=1, label="목표 0.3mm")
        ax.set_title(f"{'진피(Dermis)' if 'dermis' in target else '근막(Fascia)'} MAE 분포")
        ax.set_ylabel("MAE (mm)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4)

    # ── 상단 오른쪽: 평균 MAE 막대 ───────────────────────────────
    ax = axes[0][2]
    x = np.arange(len(model_names))
    w = 0.35
    means_d = [cv_results[m]["mae_dermis"].mean() for m in model_names]
    means_f = [cv_results[m]["mae_fascia"].mean() for m in model_names]
    ax.bar(x - w/2, means_d, w, label="진피", color="#4c72b0", alpha=0.8)
    ax.bar(x + w/2, means_f, w, label="근막", color="#dd8452", alpha=0.8)
    ax.axhline(0.3, color="red", linestyle="--", linewidth=1, label="목표")
    ax.set_xticks(x); ax.set_xticklabels(model_names)
    ax.set_ylabel("평균 MAE (mm)"); ax.set_title("모델 비교 (평균 MAE)")
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    for xi, (d, f) in enumerate(zip(means_d, means_f)):
        ax.text(xi - w/2, d + 0.01, f"{d:.3f}", ha="center", fontsize=8)
        ax.text(xi + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)

    # ── 하단: 최적 모델의 예측 vs 실제 산점도 ────────────────────
    best_name = min(model_names, key=lambda m:
                    cv_results[m]["mae_dermis"].mean() + cv_results[m]["mae_fascia"].mean())

    # 전체 데이터로 최종 예측 (학습용 — 참고용 시각화)
    clean  = df.dropna(subset=FEATURES + TARGETS)
    models = {
        "Random Forest":   _make_rf(),
        "Gradient Boost":  _make_gb(),
        "MLP":             _make_mlp(),
    }
    best_model = models[best_name]
    best_model.fit(clean[FEATURES].values, clean[TARGETS].values)
    y_pred = best_model.predict(clean[FEATURES].values)
    y_true = clean[TARGETS].values

    for col, (ti, label) in enumerate(zip([0, 1], ["진피 (mm)", "근막 (mm)"])):
        ax = axes[1][col]
        ax.scatter(y_true[:, ti], y_pred[:, ti], alpha=0.4, s=20, color=colors[ti])
        lim = [min(y_true[:, ti].min(), y_pred[:, ti].min()) - 0.2,
               max(y_true[:, ti].max(), y_pred[:, ti].max()) + 0.2]
        ax.plot(lim, lim, "r--", linewidth=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"실제 {label}"); ax.set_ylabel(f"예측 {label}")
        ax.set_title(f"{best_name} — {label} 예측 (train fit)")
        ax.grid(alpha=0.3)

    # ── 하단 오른쪽: 특징 중요도 (RF) ────────────────────────────
    ax = axes[1][2]
    rf = _make_rf()
    rf.fit(clean[FEATURES].values, clean[TARGETS].values)
    importances = np.mean([est.feature_importances_ for est in rf.estimators_], axis=0)
    ax.barh(FEATURES, importances, color="#55a868", alpha=0.8)
    ax.set_title("Random Forest 특징 중요도")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.4)

    fig.tight_layout()
    out = RESULT_DIR / "feature_model_cv.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"그래프 저장: {out}")


# ── 모델 팩토리 ──────────────────────────────────────────────────────────
def _make_rf():
    return MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=8,
                              min_samples_leaf=3, random_state=42, n_jobs=-1)
    )

def _make_gb():
    return MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, random_state=42)
    )

def _make_mlp():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPRegressor(hidden_layer_sizes=(64, 32, 16),
                                activation="relu", max_iter=1000,
                                early_stopping=True, validation_fraction=0.1,
                                random_state=42)),
    ])


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Feature-Based Model Training")
    print("=" * 60)

    # 1. 데이터셋 구성
    print("\n[1/4] 데이터셋 구성 중...")
    df = build_dataset()
    print(f"  총 샘플: {len(df)}개  /  환자: {df['patient'].nunique()}명"
          f"  /  부위: {df['pos_num'].nunique()}개")
    print(f"  특징 결측: {df[FEATURES].isna().sum().to_dict()}")
    print(f"  라벨 범위: dermis {df['dermis_mm'].min():.2f}~{df['dermis_mm'].max():.2f} mm"
          f"  /  fascia {df['fascia_mm'].min():.2f}~{df['fascia_mm'].max():.2f} mm")

    # 2. Leave-One-Patient-Out CV
    print("\n[2/4] Leave-One-Patient-Out CV 실행 중...")
    models_fns = {
        "Random Forest":  _make_rf,
        "Gradient Boost": _make_gb,
        "MLP":            _make_mlp,
    }
    cv_results = {}
    for name, fn in models_fns.items():
        cv = patient_loo_cv(df, fn)
        cv_results[name] = cv
        print(f"  {name:<18s}  "
              f"진피 MAE={cv['mae_dermis'].mean():.3f}±{cv['mae_dermis'].std():.3f} mm  "
              f"근막 MAE={cv['mae_fascia'].mean():.3f}±{cv['mae_fascia'].std():.3f} mm")

    # 3. 최적 모델 선택 및 최종 학습
    print("\n[3/4] 최종 모델 학습 (전체 데이터)...")
    best_name = min(models_fns, key=lambda m:
                    cv_results[m]["mae_dermis"].mean() + cv_results[m]["mae_fascia"].mean())
    print(f"  선택된 모델: {best_name}")

    final_model = train_final_model(df, models_fns[best_name])

    model_path = RESULT_DIR / "feature_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":      final_model,
            "model_name": best_name,
            "features":   FEATURES,
            "targets":    TARGETS,
            "cv_results": cv_results,
        }, f)
    print(f"  모델 저장: {model_path}")

    # 4. 시각화
    print("\n[4/4] 결과 시각화 저장 중...")
    plot_cv_results(cv_results, df)

    # 5. 최종 요약
    print("\n" + "=" * 60)
    print("  결과 요약")
    print("=" * 60)
    for name, cv in cv_results.items():
        print(f"  {name:<18s}  "
              f"진피 {cv['mae_dermis'].mean():.3f} mm  "
              f"근막 {cv['mae_fascia'].mean():.3f} mm")
    best_cv = cv_results[best_name]
    print(f"\n  최적 모델: {best_name}")
    print(f"  진피 MAE: {best_cv['mae_dermis'].mean():.3f} mm  "
          f"(목표 < 0.30 mm  {'✓' if best_cv['mae_dermis'].mean() < 0.3 else '✗'})")
    print(f"  근막 MAE: {best_cv['mae_fascia'].mean():.3f} mm  "
          f"(목표 < 0.30 mm  {'✓' if best_cv['mae_fascia'].mean() < 0.3 else '✗'})")
    print("=" * 60)


if __name__ == "__main__":
    main()
