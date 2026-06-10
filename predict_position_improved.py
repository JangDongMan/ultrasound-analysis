#!/usr/bin/env python3
"""
predict_position_improved.py
개선된 얼굴 부위 예측:
  1) 환자 내 정규화 (개인차 제거)
  2) 세밀한 파형 특징 추가
  3) 좌우 대칭 지점 병합 (28 → 16 클래스)
"""

import json, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import platform
import matplotlib.font_manager as fm
from scipy.signal import hilbert
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ── 한글 폰트 ──────────────────────────────────────────────────────────────
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

SAMPLE_NS  = 10
THRESHOLD  = 100
MIN_CLUST  = 5

FILENAME_RE = re.compile(
    r'^(?P<name>.+)_(?P<date>\d{8})_(?P<time>\d{6})_'
    r'\((?P<pos_num>\d+)\)(?P<position>.+?)_(?P<gender>[MF])_positions\.json$'
)

# ── 좌우 병합 맵 (28 → 16 클래스) ─────────────────────────────────────────
MERGE_MAP = {
    1:  '이마중앙',
    2:  '이마옆',      3:  '이마옆',
    4:  '이마끝관자',  5:  '이마끝관자',
    6:  '관자놀이위',  7:  '관자놀이위',
    8:  '눈썹옆관자',  9:  '눈썹옆관자',
    10: '광대활',      11: '광대활',
    12: '눈아래광대',  13: '눈아래광대',
    14: '귀쪽볼',      19: '귀쪽볼',
    15: '광대아래볼',  18: '광대아래볼',
    16: '콧구멍옆볼',  17: '콧구멍옆볼',
    20: '귀아래턱선',  23: '귀아래턱선',
    21: '인중옆볼',    22: '인중옆볼',
    24: '아래잎술턱선', 26: '아래잎술턱선',
    25: '턱중앙',
    27: '목',          28: '목',
}
MERGE_ORDER = [
    '이마중앙', '이마옆', '이마끝관자',
    '관자놀이위', '눈썹옆관자',
    '광대활', '눈아래광대',
    '귀쪽볼', '광대아래볼', '콧구멍옆볼',
    '귀아래턱선', '인중옆볼',
    '아래잎술턱선', '턱중앙',
    '목',
]

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


# ── ADC 로드 ───────────────────────────────────────────────────────────────
def load_adc(path):
    adc = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            v = line.strip()
            if not v:
                continue
            try:
                n = int(v)
                if 0 <= n <= 255:
                    adc.append(n)
            except ValueError:
                pass
    return np.array(adc, dtype=np.float32)


# ── 클러스터 ───────────────────────────────────────────────────────────────
def get_clusters(env, thr=THRESHOLD, min_len=MIN_CLUST):
    above = (env >= thr).astype(int)
    result = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if j - i >= min_len:
                result.append((i, j))
            i = j
        else:
            i += 1
    return result


# ── 신호 특징 (확장) ────────────────────────────────────────────────────────
def extract_features(adc, drop_us=12.0):
    dt = SAMPLE_NS / 1000.0   # μs/sample
    adc_c = adc - 128.0
    env   = np.abs(hilbert(adc_c))
    clusters = get_clusters(env)
    n_c = len(clusters)
    feat = {}

    # ── 전체 신호 통계 ────────────────────────────────────────────────────
    feat['max_env']   = float(np.max(env))
    feat['mean_env']  = float(np.mean(env))
    feat['std_env']   = float(np.std(env))
    feat['n_clusters'] = n_c

    # ── 클러스터 1 (표피) ─────────────────────────────────────────────────
    if n_c >= 1:
        s0, e0 = clusters[0]
        feat['t0_us']        = drop_us + s0 * dt
        feat['epi_span_us']  = (e0 - s0) * dt
        feat['epi_max_env']  = float(np.max(env[s0:e0]))
        feat['epi_mean_env'] = float(np.mean(env[s0:e0]))
        # 표피 envelope 기울기 (앞쪽 절반)
        half = (s0 + e0) // 2
        feat['epi_rise_slope'] = float(np.polyfit(np.arange(half - s0),
                                                    env[s0:half], 1)[0]) if half > s0 else 0.0
    else:
        for k in ['t0_us', 'epi_span_us', 'epi_max_env', 'epi_mean_env', 'epi_rise_slope']:
            feat[k] = 0.0

    # ── 클러스터 2 (첫 번째 반사: 뼈 or 진피) ────────────────────────────
    if n_c >= 2:
        s1, e1 = clusters[1]
        feat['gap_us']          = (s1 - clusters[0][1]) * dt
        feat['second_span_us']  = (e1 - s1) * dt
        feat['second_max_env']  = float(np.max(env[s1:e1]))
        feat['second_mean_env'] = float(np.mean(env[s1:e1]))
        feat['amp_ratio_12']    = feat['second_max_env'] / (feat['epi_max_env'] + 1e-9)
        feat['energy_2']        = float(np.sum(env[s1:e1] ** 2))
        # 뼈 지표: 두 번째 클러스터 안에서 피크 수
        from scipy.signal import find_peaks as fp
        peaks, _ = fp(env[s1:e1], height=THRESHOLD * 0.7, distance=5)
        feat['peaks_in_c2'] = len(peaks)
    else:
        for k in ['gap_us', 'second_span_us', 'second_max_env', 'second_mean_env',
                  'amp_ratio_12', 'energy_2', 'peaks_in_c2']:
            feat[k] = 0.0

    # ── 클러스터 3 ────────────────────────────────────────────────────────
    if n_c >= 3:
        s2, e2 = clusters[2]
        feat['third_span_us']  = (e2 - s2) * dt
        feat['third_max_env']  = float(np.max(env[s2:e2]))
        feat['gap2_us']        = (s2 - clusters[1][1]) * dt
        feat['amp_ratio_23']   = feat['third_max_env'] / (feat['second_max_env'] + 1e-9)
    else:
        for k in ['third_span_us', 'third_max_env', 'gap2_us', 'amp_ratio_23']:
            feat[k] = 0.0

    # ── 표피 이후 에너지 비율 ─────────────────────────────────────────────
    total_e = float(np.sum(env ** 2)) + 1e-9
    if n_c >= 2:
        post_e = float(np.sum(env[clusters[1][0]:] ** 2))
        feat['post_epi_energy_ratio'] = post_e / total_e
    else:
        feat['post_epi_energy_ratio'] = 0.0

    # ── 주파수 특징 ───────────────────────────────────────────────────────
    fft  = np.abs(np.fft.rfft(adc_c))
    freq = np.fft.rfftfreq(len(adc_c), d=SAMPLE_NS * 1e-9) / 1e6
    valid = (freq >= 1.0) & (freq <= 15.0)
    if valid.any():
        fv, ff = fft[valid], freq[valid]
        feat['dom_freq_mhz']  = float(ff[np.argmax(fv)])
        sc = float(np.sum(ff * fv) / (np.sum(fv) + 1e-9))
        feat['spectral_cent'] = sc
        feat['spectral_bw']   = float(np.sqrt(np.sum(fv * (ff - sc)**2) / (np.sum(fv) + 1e-9)))
        # 저주파(<4MHz) vs 고주파(>6MHz) 에너지 비
        low  = float(np.sum(fv[ff < 4.0] ** 2))
        high = float(np.sum(fv[ff > 6.0] ** 2))
        feat['freq_low_high_ratio'] = low / (high + 1e-9)
    else:
        feat['dom_freq_mhz'] = feat['spectral_cent'] = feat['spectral_bw'] = feat['freq_low_high_ratio'] = 0.0

    return feat


# ── 데이터 로드 ────────────────────────────────────────────────────────────
def load_data():
    records = []
    for jf in sorted(USDATA.rglob('*_positions.json')):
        m = FILENAME_RE.match(jf.name)
        if not m:
            continue
        csv_path = jf.parent / jf.name.replace('_positions.json', '.csv')
        if not csv_path.exists():
            continue

        with open(jf, encoding='utf-8') as f:
            d = json.load(f)

        markers = {p['position_name']: p for p in d.get('positions', [])}
        p1, p2 = markers.get('피하지방시작'), markers.get('Fascia')
        if not p1 or not p2:
            continue

        pos_num  = int(m.group('pos_num'))
        adc      = load_adc(csv_path)
        if len(adc) < 100:
            continue

        drop_us  = d.get('start_point_us', 12.0)
        sig      = extract_features(adc, drop_us)

        epi_mm    = p1['depth_end_mm']
        fascia_mm = p2['depth_end_mm']
        rec = {
            'patient':   m.group('name'),
            'gender':    m.group('gender'),
            'pos_num':   pos_num,
            'position':  m.group('position'),
            'region':    REGION_MAP.get(pos_num, '기타'),
            'merged':    MERGE_MAP.get(pos_num, '기타'),
            'epi_mm':    round(epi_mm, 4),
            'dermis_mm': round(fascia_mm - epi_mm, 4),
            'fascia_mm': round(fascia_mm, 4),
        }
        rec.update(sig)
        records.append(rec)

    df = pd.DataFrame(records)
    print(f"로드: {len(df)}개  |  {df['patient'].nunique()}명  |  "
          f"{df['pos_num'].nunique()}개 지점")
    return df


# ── 환자 내 정규화 ─────────────────────────────────────────────────────────
SIGNAL_FEATS = [
    'epi_mm', 'dermis_mm', 'fascia_mm',
    't0_us', 'epi_span_us', 'epi_max_env', 'epi_mean_env', 'epi_rise_slope',
    'max_env', 'mean_env', 'std_env', 'n_clusters',
    'gap_us', 'second_span_us', 'second_max_env', 'second_mean_env',
    'amp_ratio_12', 'energy_2', 'peaks_in_c2',
    'third_span_us', 'third_max_env', 'gap2_us', 'amp_ratio_23',
    'post_epi_energy_ratio',
    'dom_freq_mhz', 'spectral_cent', 'spectral_bw', 'freq_low_high_ratio',
]

def patient_normalize(df, feats):
    df_n = df.copy()
    for pat in df['patient'].unique():
        idx = df['patient'] == pat
        sub = df.loc[idx, feats]
        mu, sigma = sub.mean(), sub.std().replace(0, 1)
        df_n.loc[idx, feats] = (sub - mu) / sigma
    return df_n


# ── LOGO-CV ────────────────────────────────────────────────────────────────
def logo_cv(df, feats, target, clf):
    X      = df[feats].fillna(0).values
    y      = df[target].values
    groups = df['patient'].values
    yt, yp = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        clf.fit(X[tr], y[tr])
        yp.extend(clf.predict(X[te]))
        yt.extend(y[te])
    return np.array(yt), np.array(yp)


# ── 시각화 ─────────────────────────────────────────────────────────────────
def plot_confusion(yt, yp, labels, title, fname, figsize=(10, 8)):
    cm = confusion_matrix(yt, yp, labels=labels)
    # 행별 정규화 (recall 시각화)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, data, fmt, subtitle in zip(
            axes, [cm, cm_norm], ['d', '.2f'], ['개수', '정규화(행별 recall)']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={'size': 7})
        ax.set_xlabel('예측', fontsize=9)
        ax.set_ylabel('실제', fontsize=9)
        ax.set_title(subtitle, fontsize=10)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12, weight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=130)
    plt.close()
    print(f"  저장: results/{fname}")


def plot_importance(clf, feats, title, fname):
    imp = pd.Series(clf.feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = ['#e15759' if v > imp.quantile(0.75) else
              '#f28e2b' if v > imp.median() else '#4e79a7' for v in imp]
    imp.plot.barh(ax=ax, color=colors)
    ax.axvline(imp.median(), color='gray', linestyle='--', alpha=0.6, label='중앙값')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('중요도')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=120)
    plt.close()
    print(f"  저장: results/{fname}")


def plot_accuracy_comparison(results):
    labels = [r['label'] for r in results]
    accs   = [r['acc'] * 100 for r in results]
    colors = ['#4e79a7' if '진피만' in l else
              '#f28e2b' if '정규화 없음' in l else '#e15759' for l in labels]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels, accs, color=colors, alpha=0.85)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel('정확도 (%)')
    ax.set_title('방법별 부위 예측 정확도 비교', fontsize=12, weight='bold')
    ax.axvline(100/7, color='gray', linestyle=':', alpha=0.6, label='랜덤(7클래스)')
    ax.axvline(100/15, color='gray', linestyle='--', alpha=0.6, label='랜덤(15클래스)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'accuracy_comparison.png', dpi=120)
    plt.close()
    print("  저장: results/accuracy_comparison.png")


# ── 메인 ───────────────────────────────────────────────────────────────────
def main():
    print("=== 데이터 로드 ===")
    df    = load_data()
    df_n  = patient_normalize(df, SIGNAL_FEATS)   # 환자 내 정규화

    clf_rf = RandomForestClassifier(
        n_estimators=300, max_depth=12,
        min_samples_leaf=2, max_features='sqrt', random_state=42)

    results = []

    # ── 1) 기존 방법 재현 (진피만, 정규화 없음) ───────────────────────────
    print("\n[1] 진피만 / 정규화 없음 / Region (7)")
    yt, yp = logo_cv(df, ['epi_mm', 'dermis_mm', 'fascia_mm'], 'region', clf_rf)
    acc = accuracy_score(yt, yp)
    results.append({'label': '진피만 / 정규화없음 / Region(7)', 'acc': acc})
    print(f"    정확도: {acc*100:.1f}%")

    # ── 2) 신호+진피, 정규화 없음, Region ────────────────────────────────
    print("\n[2] 신호+진피 / 정규화 없음 / Region (7)")
    yt, yp = logo_cv(df, SIGNAL_FEATS, 'region', clf_rf)
    acc = accuracy_score(yt, yp)
    results.append({'label': '신호+진피 / 정규화없음 / Region(7)', 'acc': acc})
    print(f"    정확도: {acc*100:.1f}%")

    # ── 3) 신호+진피, 환자 내 정규화, Region ─────────────────────────────
    print("\n[3] 신호+진피 / 환자 내 정규화 / Region (7)")
    yt, yp = logo_cv(df_n, SIGNAL_FEATS, 'region', clf_rf)
    acc3 = accuracy_score(yt, yp)
    results.append({'label': '신호+진피 / 환자정규화 / Region(7)', 'acc': acc3})
    print(f"    정확도: {acc3*100:.1f}%")
    labels_r = [r for r in REGION_ORDER if r in set(yt)]
    print(classification_report(yt, yp, target_names=labels_r, zero_division=0))
    plot_confusion(yt, yp, labels_r,
                   f'Region 예측 — 환자정규화+신호 ({acc3*100:.1f}%)',
                   'improved_confusion_region.png', figsize=(12, 5))
    clf_rf.fit(df_n[SIGNAL_FEATS].fillna(0).values, df_n['region'].values)
    plot_importance(clf_rf, SIGNAL_FEATS, 'Region 분류 특징 중요도', 'improved_importance_region.png')

    # ── 4) 신호+진피, 환자 내 정규화, 좌우병합(15클래스) ─────────────────
    print("\n[4] 신호+진피 / 환자 내 정규화 / 좌우병합 (15클래스)")
    yt, yp = logo_cv(df_n, SIGNAL_FEATS, 'merged', clf_rf)
    acc4 = accuracy_score(yt, yp)
    results.append({'label': '신호+진피 / 환자정규화 / 좌우병합(15)', 'acc': acc4})
    print(f"    정확도: {acc4*100:.1f}%")
    labels_m = [r for r in MERGE_ORDER if r in set(yt)]
    print(classification_report(yt, yp, target_names=labels_m, zero_division=0))
    plot_confusion(yt, yp, labels_m,
                   f'좌우병합 예측 — 환자정규화+신호 ({acc4*100:.1f}%)',
                   'improved_confusion_merged.png', figsize=(14, 6))
    clf_rf.fit(df_n[SIGNAL_FEATS].fillna(0).values, df_n['merged'].values)
    plot_importance(clf_rf, SIGNAL_FEATS, '좌우병합 분류 특징 중요도', 'improved_importance_merged.png')

    # ── 5) 신호+진피, 환자 내 정규화, pos_num(28) ────────────────────────
    print("\n[5] 신호+진피 / 환자 내 정규화 / pos_num (28)")
    yt, yp = logo_cv(df_n, SIGNAL_FEATS, 'pos_num', clf_rf)
    acc5 = accuracy_score(yt, yp)
    results.append({'label': '신호+진피 / 환자정규화 / pos_num(28)', 'acc': acc5})
    print(f"    정확도: {acc5*100:.1f}%")

    # ── 결과 비교 시각화 ──────────────────────────────────────────────────
    print("\n=== 정확도 비교 ===")
    plot_accuracy_comparison(results)
    for r in results:
        print(f"  {r['label']:40s}  {r['acc']*100:.1f}%")

    print(f"\n{'='*55}")
    print(f"  최고 Region(7)  정확도: {acc3*100:.1f}%  (환자정규화)")
    print(f"  최고 병합(15)   정확도: {acc4*100:.1f}%  (환자정규화+좌우병합)")
    print(f"  pos_num(28)     정확도: {acc5*100:.1f}%")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
