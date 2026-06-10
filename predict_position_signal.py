#!/usr/bin/env python3
"""
predict_position_signal.py
ADC 신호 파형 특징 + 진피층 특징 → 얼굴 부위 예측

신호 특징:
  epi_mm          : 표피 두께 (T0→피하지방시작)
  dermis_mm       : 피하지방 두께
  fascia_mm       : T0→Fascia 전체 깊이
  t0_us           : T0 절대 시각
  max_env         : 최대 envelope 값
  n_clusters      : envelope≥100 구간 수
  gap_us          : 1번째~2번째 클러스터 간 갭 (μs)
  second_span_us  : 2번째 클러스터 폭 (뼈=넓음)
  third_span_us   : 3번째 클러스터 폭 (없으면 0)
  post_epi_energy : 표피 이후 신호 에너지
  dom_freq_mhz    : 지배 주파수 (FFT)
  spectral_cent   : 스펙트럼 무게중심
"""

import json, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from scipy.signal import hilbert, find_peaks
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# ── 한글 폰트 ─────────────────────────────────────────────────────────────
def _setup_korean_font():
    candidates = {'Linux': ['NanumGothic', 'NanumBarunGothic', 'UnDotum'],
                  'Windows': ['Malgun Gothic'], 'Darwin': ['AppleGothic']}
    for name in candidates.get(platform.system(), []):
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return
_setup_korean_font()

ROOT    = Path(__file__).resolve().parent
USDATA  = ROOT / "usdata" / "data"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_INTERVAL_NS = 10          # 10ns/sample → 100MHz
THRESHOLD          = 100         # envelope 클러스터 기준
MIN_CLUSTER_LEN    = 5           # 최소 클러스터 길이(샘플)

FILENAME_RE = re.compile(
    r'^(?P<name>.+)_(?P<date>\d{8})_(?P<time>\d{6})_'
    r'\((?P<pos_num>\d+)\)(?P<position>.+?)_(?P<gender>[MF])_positions\.json$'
)

REGION_MAP = {
    1: '이마', 2: '이마', 3: '이마',
    4: '관자놀이', 5: '관자놀이', 6: '관자놀이',
    7: '관자놀이', 8: '관자놀이', 9: '관자놀이',
    10: '광대', 11: '광대', 12: '광대', 13: '광대',
    14: '볼', 15: '볼', 16: '볼',
    17: '볼', 18: '볼', 19: '볼',
    20: '턱선', 21: '턱선', 22: '턱선', 23: '턱선',
    24: '턱', 25: '턱', 26: '턱',
    27: '목', 28: '목',
}
REGION_ORDER = ['이마', '관자놀이', '광대', '볼', '턱선', '턱', '목']


# ── ADC 로드 ──────────────────────────────────────────────────────────────
def load_adc(csv_path):
    adc = []
    with open(csv_path, encoding='utf-8', errors='replace') as f:
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
    return np.array(adc, dtype=np.float32)


# ── 클러스터 추출 ─────────────────────────────────────────────────────────
def get_clusters(env, threshold=THRESHOLD, min_len=MIN_CLUSTER_LEN):
    """envelope >= threshold 인 연속 구간 반환 [(start, end), ...]"""
    above = (env >= threshold).astype(int)
    clusters = []
    i = 0
    while i < len(above):
        if above[i] == 1:
            j = i
            while j < len(above) and above[j] == 1:
                j += 1
            if j - i >= min_len:
                clusters.append((i, j))
            i = j
        else:
            i += 1
    return clusters


# ── 신호 특징 추출 ────────────────────────────────────────────────────────
def extract_signal_features(adc, drop_us=12.0):
    """ADC 배열 → 특징 dict"""
    feat = {}

    adc_c = adc - 128.0
    env   = np.abs(hilbert(adc_c))

    dt_us = SAMPLE_INTERVAL_NS / 1000.0   # 0.01 μs/sample

    # ── 기본 envelope 통계 ──────────────────────────────────────────────
    feat['max_env']  = float(np.max(env))
    feat['mean_env'] = float(np.mean(env))
    feat['std_env']  = float(np.std(env))

    # ── 클러스터 분석 ───────────────────────────────────────────────────
    clusters = get_clusters(env)
    feat['n_clusters'] = len(clusters)

    # T0: 첫 클러스터 시작
    if len(clusters) >= 1:
        c0_s, c0_e = clusters[0]
        t0_us = drop_us + c0_s * dt_us
        feat['t0_us']       = t0_us
        feat['epi_span_us'] = (c0_e - c0_s) * dt_us  # 1번 클러스터 폭
    else:
        feat['t0_us']       = 0.0
        feat['epi_span_us'] = 0.0

    # 2번째 클러스터 (진피/뼈 반사)
    if len(clusters) >= 2:
        c1_s, c1_e = clusters[1]
        gap_us          = (c1_s - clusters[0][1]) * dt_us if len(clusters) >= 1 else 0.0
        second_span_us  = (c1_e - c1_s) * dt_us
        feat['gap_us']         = gap_us
        feat['second_span_us'] = second_span_us
        feat['second_max_env'] = float(np.max(env[c1_s:c1_e]))
    else:
        feat['gap_us']         = 0.0
        feat['second_span_us'] = 0.0
        feat['second_max_env'] = 0.0

    # 3번째 클러스터
    if len(clusters) >= 3:
        c2_s, c2_e = clusters[2]
        feat['third_span_us'] = (c2_e - c2_s) * dt_us
        feat['third_max_env'] = float(np.max(env[c2_s:c2_e]))
    else:
        feat['third_span_us'] = 0.0
        feat['third_max_env'] = 0.0

    # 표피 이후 에너지 (클러스터 2번~ 합산)
    if len(clusters) >= 2:
        post_start = clusters[1][0]
        feat['post_epi_energy'] = float(np.sum(env[post_start:] ** 2))
    else:
        feat['post_epi_energy'] = 0.0

    # ── 주파수 특징 ─────────────────────────────────────────────────────
    n    = len(adc_c)
    fft  = np.abs(np.fft.rfft(adc_c))
    freq = np.fft.rfftfreq(n, d=SAMPLE_INTERVAL_NS * 1e-9) / 1e6  # MHz
    valid = (freq >= 1.0) & (freq <= 20.0)
    if valid.any():
        fft_v = fft[valid]
        freq_v = freq[valid]
        feat['dom_freq_mhz']  = float(freq_v[np.argmax(fft_v)])
        feat['spectral_cent'] = float(np.sum(freq_v * fft_v) / (np.sum(fft_v) + 1e-9))
        feat['spectral_bw']   = float(np.sqrt(
            np.sum(fft_v * (freq_v - feat['spectral_cent'])**2) / (np.sum(fft_v) + 1e-9)))
    else:
        feat['dom_freq_mhz']  = 0.0
        feat['spectral_cent'] = 0.0
        feat['spectral_bw']   = 0.0

    return feat


# ── 데이터 로드 ───────────────────────────────────────────────────────────
def load_all_data():
    records = []
    skipped = 0

    for jf in sorted(USDATA.rglob('*_positions.json')):
        m = FILENAME_RE.match(jf.name)
        if not m:
            continue

        # 대응 CSV
        csv_name = jf.name.replace('_positions.json', '.csv')
        csv_path = jf.parent / csv_name
        if not csv_path.exists():
            skipped += 1
            continue

        with open(jf, encoding='utf-8') as f:
            d = json.load(f)

        pos_num  = int(m.group('pos_num'))
        position = m.group('position')
        patient  = m.group('name')
        gender   = m.group('gender')

        # 진피 마커
        markers = {p['position_name']: p for p in d.get('positions', [])}
        p1 = markers.get('피하지방시작')
        p2 = markers.get('Fascia')
        if p1 is None or p2 is None:
            skipped += 1
            continue

        epi_mm    = p1['depth_end_mm']
        fascia_mm = p2['depth_end_mm']
        dermis_mm = fascia_mm - epi_mm

        # ADC 신호 특징
        adc = load_adc(csv_path)
        if len(adc) < 100:
            skipped += 1
            continue

        drop_us = d.get('start_point_us', 12.0)
        sig_feat = extract_signal_features(adc, drop_us)

        rec = {
            'patient':   patient,
            'gender':    gender,
            'pos_num':   pos_num,
            'position':  position,
            'region':    REGION_MAP.get(pos_num, '기타'),
            'epi_mm':    round(epi_mm, 4),
            'dermis_mm': round(dermis_mm, 4),
            'fascia_mm': round(fascia_mm, 4),
        }
        rec.update(sig_feat)
        records.append(rec)

    df = pd.DataFrame(records)
    print(f"로드: {len(df)}개 샘플  |  건너뜀: {skipped}개")
    return df


# ── LOGO-CV 분류 ─────────────────────────────────────────────────────────
ALL_FEATS = [
    'epi_mm', 'dermis_mm', 'fascia_mm',
    't0_us', 'epi_span_us', 'max_env', 'mean_env', 'std_env',
    'n_clusters', 'gap_us', 'second_span_us', 'second_max_env',
    'third_span_us', 'third_max_env', 'post_epi_energy',
    'dom_freq_mhz', 'spectral_cent', 'spectral_bw',
]

DERMIS_ONLY = ['epi_mm', 'dermis_mm', 'fascia_mm']

def run_logo_cv(df, feats, target, clf):
    X      = df[feats].values
    y      = df[target].values
    groups = df['patient'].values
    logo   = LeaveOneGroupOut()

    y_true, y_pred = [], []
    for tr, te in logo.split(X, y, groups):
        clf.fit(X[tr], y[tr])
        y_pred.extend(clf.predict(X[te]))
        y_true.extend(y[te])
    return np.array(y_true), np.array(y_pred)


# ── 시각화 ───────────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, labels, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('예측')
    ax.set_ylabel('실제')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=120)
    plt.close()
    print(f"저장: results/{fname}")


def plot_feature_importance(clf, feats, title, fname):
    imp = pd.Series(clf.feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#e15759' if v > imp.median() else '#4e79a7' for v in imp]
    imp.plot.barh(ax=ax, color=colors)
    ax.axvline(imp.median(), color='gray', linestyle='--', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('중요도')
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=120)
    plt.close()
    print(f"저장: results/{fname}")


def plot_signal_features_by_region(df):
    """부위별 신호 특징 박스플롯 (4개 핵심 특징)"""
    key_feats = [
        ('gap_us',         '갭 폭 (μs): 표피→진피 갭'),
        ('second_span_us', '2번 클러스터 폭 (μs): 뼈=넓음'),
        ('n_clusters',     '클러스터 수'),
        ('spectral_cent',  '스펙트럼 무게중심 (MHz)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (feat, title) in zip(axes.flat, key_feats):
        data  = [df[df['region'] == r][feat].values for r in REGION_ORDER]
        bp = ax.boxplot(data, labels=REGION_ORDER, patch_artist=True)
        colors = ['#4e79a7','#59a14f','#f28e2b','#e15759','#76b7b2','#edc948','#b07aa1']
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(feat)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle('부위별 신호 특징 분포', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'signal_features_by_region.png', dpi=120)
    plt.close()
    print("저장: results/signal_features_by_region.png")


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    print("=== 신호 특징 추출 중... ===")
    df = load_all_data()
    df.to_csv(OUT_DIR / 'signal_features_by_position.csv', index=False, encoding='utf-8-sig')
    print("저장: results/signal_features_by_position.csv\n")

    # 부위별 핵심 특징 통계
    print("[부위별 신호 특징 평균]")
    show_cols = ['gap_us', 'second_span_us', 'n_clusters', 'spectral_cent', 'dom_freq_mhz']
    stats = df.groupby('region')[show_cols].mean()
    stats = stats.reindex([r for r in REGION_ORDER if r in stats.index])
    print(stats.round(3).to_string())

    plot_signal_features_by_region(df)

    clf_rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                     min_samples_leaf=2, random_state=42)
    clf_gb = GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                         learning_rate=0.05, random_state=42)

    # ── Region 분류 비교 ────────────────────────────────────────────────
    print("\n[Region 분류 비교]")
    labels_r = [r for r in REGION_ORDER if r in df['region'].unique()]

    yt, yp = run_logo_cv(df, DERMIS_ONLY, 'region', clf_rf)
    acc_dermis = accuracy_score(yt, yp)
    print(f"  진피만    (RF): {acc_dermis*100:.1f}%")

    yt, yp = run_logo_cv(df, ALL_FEATS, 'region', clf_rf)
    acc_all_rf = accuracy_score(yt, yp)
    print(f"  신호+진피 (RF): {acc_all_rf*100:.1f}%")
    plot_confusion(yt, yp, labels_r,
                   f'Region 예측 혼동행렬 — 신호+진피 RF ({acc_all_rf*100:.1f}%)',
                   'signal_confusion_region.png')
    print(classification_report(yt, yp, target_names=labels_r, zero_division=0))

    yt, yp = run_logo_cv(df, ALL_FEATS, 'region', clf_gb)
    acc_all_gb = accuracy_score(yt, yp)
    print(f"  신호+진피 (GB): {acc_all_gb*100:.1f}%")

    # feature importance
    clf_rf.fit(df[ALL_FEATS].values, df['region'].values)
    plot_feature_importance(clf_rf, ALL_FEATS,
                             'Region 분류 — 특징 중요도 (RF)',
                             'signal_feature_importance_region.png')

    # ── pos_num 분류 ────────────────────────────────────────────────────
    print("\n[pos_num 직접 분류 (1~28)]")
    yt, yp = run_logo_cv(df, ALL_FEATS, 'pos_num', clf_rf)
    acc_pos_rf = accuracy_score(yt, yp)
    print(f"  신호+진피 (RF): {acc_pos_rf*100:.1f}%")

    yt, yp = run_logo_cv(df, ALL_FEATS, 'pos_num', clf_gb)
    acc_pos_gb = accuracy_score(yt, yp)
    print(f"  신호+진피 (GB): {acc_pos_gb*100:.1f}%")

    # pos_num confusion matrix (최고 모델)
    best_clf = clf_rf if acc_pos_rf >= acc_pos_gb else clf_gb
    yt, yp = run_logo_cv(df, ALL_FEATS, 'pos_num', best_clf)
    pos_labels = sorted(df['pos_num'].unique())
    plot_confusion(yt, yp, pos_labels,
                   f'pos_num 예측 혼동행렬 ({max(acc_pos_rf, acc_pos_gb)*100:.1f}%)',
                   'signal_confusion_posnum.png')

    print(f"\n{'='*55}")
    print(f"  진피만     Region 정확도 : {acc_dermis*100:.1f}%")
    print(f"  신호+진피  Region 정확도 : {max(acc_all_rf, acc_all_gb)*100:.1f}%"
          f"  ({'RF' if acc_all_rf >= acc_all_gb else 'GB'})")
    print(f"  신호+진피  pos_num 정확도: {max(acc_pos_rf, acc_pos_gb)*100:.1f}%"
          f"  ({'RF' if acc_pos_rf >= acc_pos_gb else 'GB'})")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
