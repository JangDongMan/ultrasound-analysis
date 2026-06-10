#!/usr/bin/env python3
"""
predict_position_from_dermis.py
마킹된 진피층 데이터로 얼굴 부위 예측

특징:
  - epi_mm       : T0 → 피하지방시작 깊이 (표피+진피 두께)
  - dermis_mm    : 피하지방 두께 (피하지방시작 → Fascia)
  - fascia_mm    : T0 → Fascia 전체 깊이

목표:
  - pos_num (1~28) 직접 예측
  - 또는 region 그룹 예측 (이마/관자놀이/광대/볼/턱선/턱/목)
"""

import json, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

ROOT     = Path(__file__).resolve().parent
USDATA   = ROOT / "usdata" / "data"
OUT_DIR  = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

FILENAME_RE = re.compile(
    r'^(?P<name>.+)_(?P<date>\d{8})_(?P<time>\d{6})_'
    r'\((?P<pos_num>\d+)\)(?P<position>.+?)_(?P<gender>[MF])_positions\.json$'
)

# 28개 지점 → 7개 region 매핑
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


# ── 데이터 로드 ───────────────────────────────────────────────────────────
def load_data():
    records = []
    for jf in sorted(USDATA.rglob('*_positions.json')):
        m = FILENAME_RE.match(jf.name)
        if not m:
            continue
        with open(jf, encoding='utf-8') as f:
            d = json.load(f)

        pos_num  = int(m.group('pos_num'))
        position = m.group('position')
        patient  = m.group('name')
        gender   = m.group('gender')

        # 마커 추출
        markers = {p['position_name']: p for p in d.get('positions', [])}
        p1 = markers.get('피하지방시작')
        p2 = markers.get('Fascia')
        if p1 is None or p2 is None:
            continue

        epi_mm    = p1['depth_end_mm']                      # T0→피하지방시작
        fascia_mm = p2['depth_end_mm']                      # T0→Fascia
        dermis_mm = fascia_mm - epi_mm                      # 피하지방 두께

        records.append({
            'patient':   patient,
            'gender':    gender,
            'pos_num':   pos_num,
            'position':  position,
            'region':    REGION_MAP.get(pos_num, '기타'),
            'epi_mm':    round(epi_mm, 4),
            'dermis_mm': round(dermis_mm, 4),
            'fascia_mm': round(fascia_mm, 4),
        })

    df = pd.DataFrame(records)
    print(f"총 {len(df)}개 샘플  |  {df['patient'].nunique()}명  |  {df['pos_num'].nunique()}개 지점")
    return df


# ── 특징 분포 시각화 ──────────────────────────────────────────────────────
def plot_feature_distribution(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    features = [('epi_mm', '표피 깊이 (mm)'),
                ('dermis_mm', '피하지방 두께 (mm)'),
                ('fascia_mm', 'Fascia 전체 깊이 (mm)')]

    colors = plt.cm.tab10(np.linspace(0, 1, len(REGION_ORDER)))
    color_map = dict(zip(REGION_ORDER, colors))

    for ax, (feat, title) in zip(axes, features):
        for region in REGION_ORDER:
            sub = df[df['region'] == region][feat]
            ax.hist(sub, bins=20, alpha=0.55, label=region,
                    color=color_map[region], density=True)
        ax.set_title(title)
        ax.set_xlabel('mm')
        ax.set_ylabel('밀도')
        ax.legend(fontsize=8)

    fig.suptitle('부위별 특징 분포', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'position_feature_dist.png', dpi=120)
    plt.close()
    print("저장: results/position_feature_dist.png")


# ── 부위별 통계 ───────────────────────────────────────────────────────────
def print_region_stats(df):
    print("\n[부위별 평균 특징]")
    stats = df.groupby('region')[['epi_mm', 'dermis_mm', 'fascia_mm']].mean()
    stats = stats.reindex([r for r in REGION_ORDER if r in stats.index])
    print(stats.round(3).to_string())


# ── LOGO-CV: region 분류 ──────────────────────────────────────────────────
def classify_region(df):
    print("\n[Region 분류 — LOGO-CV (환자 단위)]")
    FEATS = ['epi_mm', 'dermis_mm', 'fascia_mm']

    X      = df[FEATS].values
    y      = df['region'].values
    groups = df['patient'].values

    logo = LeaveOneGroupOut()
    clf  = RandomForestClassifier(n_estimators=200, random_state=42)

    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        clf.fit(X[train_idx], y[train_idx])
        y_pred_all.extend(clf.predict(X[test_idx]))
        y_true_all.extend(y[test_idx])

    acc = accuracy_score(y_true_all, y_pred_all)
    print(f"  정확도: {acc:.3f} ({acc*100:.1f}%)")
    print()
    print(classification_report(y_true_all, y_pred_all,
                                 target_names=[r for r in REGION_ORDER
                                               if r in set(y_true_all)]))

    # confusion matrix
    labels = [r for r in REGION_ORDER if r in set(y_true_all)]
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('예측')
    ax.set_ylabel('실제')
    ax.set_title(f'부위 예측 혼동 행렬 (정확도 {acc*100:.1f}%)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'position_confusion_region.png', dpi=120)
    plt.close()
    print("저장: results/position_confusion_region.png")

    # feature importance (전체 데이터로 재학습)
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=FEATS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    imp.plot.barh(ax=ax, color='steelblue')
    ax.set_title('특징 중요도 (Region 분류)')
    ax.set_xlabel('중요도')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'position_feature_importance.png', dpi=120)
    plt.close()
    print("저장: results/position_feature_importance.png")

    return acc


# ── LOGO-CV: pos_num 분류 ─────────────────────────────────────────────────
def classify_posnum(df):
    print("\n[pos_num 직접 분류 (1~28) — LOGO-CV]")
    FEATS = ['epi_mm', 'dermis_mm', 'fascia_mm']

    X      = df[FEATS].values
    y      = df['pos_num'].values
    groups = df['patient'].values

    logo = LeaveOneGroupOut()
    clf  = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)

    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        clf.fit(X[train_idx], y[train_idx])
        y_pred_all.extend(clf.predict(X[test_idx]))
        y_true_all.extend(y[test_idx])

    acc = accuracy_score(y_true_all, y_pred_all)
    print(f"  정확도: {acc:.3f} ({acc*100:.1f}%)")

    # top-3 accuracy
    from sklearn.calibration import CalibratedClassifierCV
    clf2 = CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
        cv=3
    )
    top3_correct = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        clf2.fit(X[train_idx], y[train_idx])
        proba = clf2.predict_proba(X[test_idx])
        classes = clf2.classes_
        for i, true_label in enumerate(y[test_idx]):
            top3_idx = np.argsort(proba[i])[-3:]
            if true_label in classes[top3_idx]:
                top3_correct += 1
    top3_acc = top3_correct / len(y)
    print(f"  Top-3 정확도: {top3_acc:.3f} ({top3_acc*100:.1f}%)")

    return acc, top3_acc


# ── 부위별 박스플롯 ───────────────────────────────────────────────────────
def plot_boxplot(df):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    features = [('epi_mm', '표피 깊이 (mm)'),
                ('dermis_mm', '피하지방 두께 (mm)'),
                ('fascia_mm', 'Fascia 전체 깊이 (mm)')]

    for ax, (feat, title) in zip(axes, features):
        data_by_pos = [df[df['pos_num'] == p][feat].values
                       for p in range(1, 29)]
        labels = [f"({p:02d})\n{df[df['pos_num']==p]['position'].iloc[0][:6]}"
                  if len(df[df['pos_num']==p]) > 0 else ''
                  for p in range(1, 29)]
        bp = ax.boxplot(data_by_pos, labels=labels, patch_artist=True)

        # region 색상
        region_colors = {'이마': '#4e79a7', '관자놀이': '#59a14f',
                         '광대': '#f28e2b', '볼': '#e15759',
                         '턱선': '#76b7b2', '턱': '#edc948', '목': '#b07aa1'}
        for i, patch in enumerate(bp['boxes']):
            pos_num = i + 1
            region = REGION_MAP.get(pos_num, '기타')
            patch.set_facecolor(region_colors.get(region, 'gray'))
            patch.set_alpha(0.7)

        ax.set_title(title)
        ax.set_ylabel('mm')
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(True, alpha=0.3)

    # legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=r, alpha=0.7)
                       for r, c in {'이마': '#4e79a7', '관자놀이': '#59a14f',
                                    '광대': '#f28e2b', '볼': '#e15759',
                                    '턱선': '#76b7b2', '턱': '#edc948',
                                    '목': '#b07aa1'}.items()]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9)
    fig.suptitle('28개 지점별 특징 분포 (박스플롯)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'position_boxplot_28.png', dpi=120)
    plt.close()
    print("저장: results/position_boxplot_28.png")


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    df.to_csv(OUT_DIR / 'dermis_by_position.csv', index=False, encoding='utf-8-sig')
    print("저장: results/dermis_by_position.csv")

    print_region_stats(df)
    plot_feature_distribution(df)
    plot_boxplot(df)

    region_acc      = classify_region(df)
    posnum_acc, top3 = classify_posnum(df)

    print(f"\n{'='*50}")
    print(f"  Region 분류 (7개 그룹) 정확도 : {region_acc*100:.1f}%")
    print(f"  pos_num 분류 (28개)  정확도   : {posnum_acc*100:.1f}%")
    print(f"  pos_num Top-3 정확도          : {top3*100:.1f}%")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
