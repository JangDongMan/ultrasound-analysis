#!/usr/bin/env python3
"""
train_region_clf.py
얼굴 부위 6개 분류기 — 1D-CNN (raw ADC signal → region)

카테고리: 이마, 관자놀이, 볼, 입술주위, 광대뼈, 턱선목

사용법:
  python train_region_clf.py                   # 5-fold CV + 최종 모델 학습
  python train_region_clf.py --no_cv           # CV 없이 전체 학습만
  python train_region_clf.py --epochs 200      # 에포크 수 지정
  python train_region_clf.py --predict x.csv   # 단일 파일 예측
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

# ── 한국어 폰트 ──────────────────────────────────────────────────
_ko_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_ko_path):
    fm.fontManager.addfont(_ko_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=_ko_path).get_name(), 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 ─────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
CLASS_DIR   = PROJECT_DIR / "classification"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 신호 파라미터 ─────────────────────────────────────────────────
TRIM_START  = 1850
TRIM_COUNT  = 2050

# ── 카테고리 ─────────────────────────────────────────────────────
CATEGORIES  = ["이마", "관자놀이", "볼", "입술주위", "광대뼈", "턱", "턱선목"]
NUM_CLASSES = len(CATEGORIES)
CAT2IDX     = {c: i for i, c in enumerate(CATEGORIES)}


# ════════════════════════════════════════════════════════════════
# 데이터 로드
# ════════════════════════════════════════════════════════════════

def load_adc(filepath: Path):
    """CSV → float32 array (TRIM_COUNT 샘플). 실패 시 None."""
    vals = []
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    v = int(line)
                    if 0 <= v <= 255:
                        vals.append(v)
                except ValueError:
                    pass
    except IOError:
        return None

    if not vals:
        return None

    adc = np.array(vals, dtype=np.float32)

    # 원시(~3900샘플) 파일 트림
    if len(adc) > TRIM_COUNT:
        end = TRIM_START + TRIM_COUNT
        adc = adc[TRIM_START:end] if len(adc) >= end else adc[TRIM_START:]

    if len(adc) < 512:
        return None

    # TRIM_COUNT 길이 맞추기 (패드 or 크롭)
    if len(adc) < TRIM_COUNT:
        adc = np.pad(adc, (0, TRIM_COUNT - len(adc)))
    else:
        adc = adc[:TRIM_COUNT]

    return adc


def load_dataset():
    """classification/ 에서 전체 데이터 로드.

    Returns:
        X        : (N, TRIM_COUNT) float32
        y        : (N,) int  — 카테고리 인덱스
        patients : (N,) str  — 환자명 (fold 분리용)
    """
    X, y, patients = [], [], []
    skipped = 0

    for cat in CATEGORIES:
        cat_dir = CLASS_DIR / cat
        if not cat_dir.exists():
            print(f"  WARNING: {cat_dir} 없음 — 스킵")
            continue

        label = CAT2IDX[cat]
        for link in sorted(cat_dir.glob("*.csv")):
            actual = link.resolve()
            adc = load_adc(actual)
            if adc is None:
                skipped += 1
                continue

            X.append(adc)
            y.append(label)

            # 심볼릭 링크명: {date}__{patient}__xxx.csv
            parts = link.stem.split("__")
            patients.append(parts[1] if len(parts) >= 2 else "unknown")

    if skipped:
        print(f"  (로드 실패 {skipped}개 스킵)")

    return np.stack(X), np.array(y), np.array(patients)


# ════════════════════════════════════════════════════════════════
# 모델
# ════════════════════════════════════════════════════════════════

class RegionCNN(nn.Module):
    """
    1D-CNN 부위 분류기.
    Input  : (B, TRIM_COUNT) float32  — 이미 (X-128)/64 정규화
    Output : (B, NUM_CLASSES) logit
    """

    def __init__(self, n_filters: int = 32, dropout: float = 0.4):
        super().__init__()
        self.conv = nn.Sequential(
            # Layer 1: 2050 → 512
            nn.Conv1d(1, n_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(n_filters), nn.ReLU(),
            nn.MaxPool1d(4),

            # Layer 2: 512 → 128
            nn.Conv1d(n_filters, n_filters * 2, kernel_size=11, padding=5),
            nn.BatchNorm1d(n_filters * 2), nn.ReLU(),
            nn.MaxPool1d(4),

            # Layer 3: 128 → 32
            nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(n_filters * 4), nn.ReLU(),
            nn.MaxPool1d(4),

            # Layer 4 + global pool → (B, n_filters*4, 8)
            nn.Conv1d(n_filters * 4, n_filters * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_filters * 4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        feat_dim = n_filters * 4 * 8   # 32*4*8 = 1024

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (B, 1, L)
        x = self.conv(x).flatten(1)
        return self.head(x)

    @staticmethod
    def param_count(n_filters=32):
        m = RegionCNN(n_filters)
        return sum(p.numel() for p in m.parameters())


# ════════════════════════════════════════════════════════════════
# Dataset / DataLoader
# ════════════════════════════════════════════════════════════════

class ADCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy((X - 128.0) / 64.0).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loader(X, y, batch_size=32, shuffle=True, weighted=False):
    ds = ADCDataset(X, y)
    sampler = None
    if weighted and shuffle:
        counts  = np.bincount(y, minlength=NUM_CLASSES).astype(float)
        weights = 1.0 / np.maximum(counts[y], 1)
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler, num_workers=2, pin_memory=True)


# ════════════════════════════════════════════════════════════════
# 학습 루프
# ════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            n          += len(y)
    return total_loss / n, correct / n


def train_model(X_tr, y_tr, X_val, y_val, args, device, verbose=True, tag=""):
    counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(float)
    weights = torch.tensor(counts.sum() / (NUM_CLASSES * np.maximum(counts, 1)),
                           dtype=torch.float32).to(device)

    tr_loader  = make_loader(X_tr, y_tr, args.batch, shuffle=True, weighted=True)
    val_loader = make_loader(X_val, y_val, args.batch, shuffle=False)

    model     = RegionCNN(args.n_filters, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_state, patience_cnt = 0.0, None, 0
    history = {"tr_loss": [], "val_loss": [], "tr_acc": [], "val_acc": []}

    for ep in range(1, args.epochs + 1):
        tr_loss,  tr_acc  = run_epoch(model, tr_loader,  criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        history["tr_loss"].append(tr_loss);  history["val_loss"].append(val_loss)
        history["tr_acc"].append(tr_acc);    history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if verbose and ep % 10 == 0:
            marker = " ★" if patience_cnt == 0 else ""
            print(f"  {tag}ep {ep:4d} | "
                  f"tr {tr_acc:.3f}({tr_loss:.4f}) | "
                  f"val {val_acc:.3f}({val_loss:.4f}) | "
                  f"best {best_acc:.3f}{marker}")

        if patience_cnt >= args.patience:
            if verbose:
                print(f"  Early stop ep={ep}  best_val_acc={best_acc:.4f}")
            break

    model.load_state_dict(best_state)
    return model, best_acc, history


@torch.no_grad()
def predict_all(model, X, device, batch_size=256):
    model.eval()
    ds = ADCDataset(X, np.zeros(len(X), dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    for xb, _ in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ════════════════════════════════════════════════════════════════
# 시각화
# ════════════════════════════════════════════════════════════════

def plot_confusion(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CATEGORIES, fontsize=11)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CATEGORIES, fontsize=11)
    ax.set_xlabel("예측", fontsize=12); ax.set_ylabel("실제", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax)
    thresh = cm.max() * 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  저장: {save_path}")


def plot_history(history, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    eps = range(1, len(history["tr_loss"]) + 1)
    ax1.plot(eps, history["tr_loss"],  label="Train")
    ax1.plot(eps, history["val_loss"], label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(eps, history["tr_acc"],  label="Train")
    ax2.plot(eps, history["val_acc"], label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  저장: {save_path}")


# ════════════════════════════════════════════════════════════════
# 단일 파일 예측
# ════════════════════════════════════════════════════════════════

def predict_file(csv_path: str):
    model_path = RESULTS_DIR / "region_clf.pth"
    if not model_path.exists():
        print("ERROR: 모델 없음. 먼저 학습을 실행하세요.")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu")
    model = RegionCNN(ckpt["n_filters"], ckpt["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    adc = load_adc(Path(csv_path))
    if adc is None:
        print(f"ERROR: 파일 로드 실패: {csv_path}")
        sys.exit(1)

    x = torch.from_numpy((adc - 128.0) / 64.0).float().unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0]
        probs  = torch.softmax(logits, dim=0).numpy()

    cats = ckpt.get("categories", CATEGORIES)
    pred_idx = int(probs.argmax())
    print(f"\n예측 결과: {cats[pred_idx]}  ({probs[pred_idx]*100:.1f}%)")
    print("\n전체 확률:")
    for cat, p in sorted(zip(cats, probs), key=lambda x: -x[1]):
        bar = "█" * int(p * 30)
        print(f"  {cat:<6} {p*100:5.1f}%  {bar}")


# ════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=150)
    parser.add_argument("--lr",        type=float, default=5e-4)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--patience",  type=int,   default=25)
    parser.add_argument("--n_filters", type=int,   default=32)
    parser.add_argument("--dropout",   type=float, default=0.4)
    parser.add_argument("--folds",     type=int,   default=5)
    parser.add_argument("--no_cv",     action="store_true")
    parser.add_argument("--loo",       action="store_true",
                        help="Leave-One-Patient-Out CV (환자 단위 일반화 측정)")
    parser.add_argument("--predict",   type=str,   default=None,
                        help="단일 CSV 파일 예측 (학습된 모델 사용)")
    args = parser.parse_args()

    # ── 단일 예측 모드 ───────────────────────────────────────
    if args.predict:
        predict_file(args.predict)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : RegionCNN (n_filters={args.n_filters}, "
          f"dropout={args.dropout})  "
          f"params={RegionCNN.param_count(args.n_filters):,}\n")

    # ── 데이터 로드 ──────────────────────────────────────────
    print("데이터 로드...")
    X, y, patients = load_dataset()
    print(f"총 {len(X)}개 로드 완료\n")
    for i, cat in enumerate(CATEGORIES):
        print(f"  {cat:<6}: {(y==i).sum():4d}개")
    print()

    # ── Leave-One-Patient-Out CV ─────────────────────────────
    if args.loo:
        unique_patients = sorted(set(patients))
        n_patients = len(unique_patients)
        print(f"{'─'*60}")
        print(f"Leave-One-Patient-Out CV  ({n_patients}명)")
        print(f"{'─'*60}")

        all_preds    = np.zeros(len(y), dtype=np.int64)
        patient_accs = {}

        for pi, test_pat in enumerate(unique_patients, 1):
            tr_idx  = np.where(patients != test_pat)[0]
            val_idx = np.where(patients == test_pat)[0]

            # 훈련 세트에서 등장하지 않는 클래스가 있으면 스킵 (희귀 케이스)
            if len(np.unique(y[tr_idx])) < NUM_CLASSES:
                print(f"  [{pi:2d}/{n_patients}] {test_pat:<12} — 훈련셋 클래스 부족, 스킵")
                continue

            print(f"\n[{pi:2d}/{n_patients}] Test: {test_pat}  "
                  f"(train {len(tr_idx)}, test {len(val_idx)})")

            model, best_acc, _ = train_model(
                X[tr_idx], y[tr_idx], X[val_idx], y[val_idx],
                args, device, verbose=True, tag=f"P{pi:02d} ")

            preds = predict_all(model, X[val_idx], device)
            all_preds[val_idx] = preds
            acc = (preds == y[val_idx]).mean()
            patient_accs[test_pat] = acc
            print(f"  → {test_pat} accuracy: {acc:.4f}  "
                  f"({int(acc*len(val_idx))}/{len(val_idx)})")

        # 결과 집계
        valid_accs = list(patient_accs.values())
        loo_mean = np.mean(valid_accs)
        loo_std  = np.std(valid_accs)

        print(f"\n{'='*60}")
        print(f"LOO-Patient CV 결과")
        print(f"{'='*60}")
        print(f"평균 정확도: {loo_mean:.4f} ± {loo_std:.4f}\n")

        # 환자별 정확도 테이블
        print(f"{'환자':<12} {'정확도':>8}  {'샘플':>6}")
        print("─" * 32)
        for pat in sorted(patient_accs, key=lambda p: patient_accs[p], reverse=True):
            n_samples = (patients == pat).sum()
            print(f"{pat:<12} {patient_accs[pat]:8.4f}  {n_samples:6d}")

        # 전체 predicted vs true (LOO로 채워진 부분만)
        valid_mask = np.array([p in patient_accs for p in patients])
        y_loo      = y[valid_mask]
        p_loo      = all_preds[valid_mask]

        print(f"\n분류 리포트:")
        print(classification_report(y_loo, p_loo, target_names=CATEGORIES))

        cm = confusion_matrix(y_loo, p_loo)
        plot_confusion(
            cm,
            f"LOO-Patient CV  acc={loo_mean:.3f}±{loo_std:.3f}",
            RESULTS_DIR / "region_clf_loo_confusion.png"
        )

        # 환자별 정확도 바 차트
        fig, ax = plt.subplots(figsize=(12, 4))
        pats_sorted = sorted(patient_accs, key=lambda p: patient_accs[p])
        accs_sorted = [patient_accs[p] for p in pats_sorted]
        colors = ["#e74c3c" if a < 0.5 else "#f39c12" if a < 0.7 else "#2ecc71"
                  for a in accs_sorted]
        ax.barh(pats_sorted, accs_sorted, color=colors)
        ax.axvline(loo_mean, color="navy", linestyle="--",
                   label=f"평균 {loo_mean:.3f}")
        ax.set_xlim(0, 1); ax.set_xlabel("정확도")
        ax.set_title("LOO-Patient CV — 환자별 정확도")
        ax.legend()
        plt.tight_layout()
        bar_path = RESULTS_DIR / "region_clf_loo_by_patient.png"
        plt.savefig(bar_path, dpi=120); plt.close()
        print(f"  저장: {bar_path}")

        # 최종 모델은 이후 전체 학습에서 저장
        print()

    # ── 5-Fold Stratified CV ─────────────────────────────────
    if not args.no_cv and not args.loo:
        print(f"{'─'*55}")
        print(f"{args.folds}-Fold Stratified CV")
        print(f"{'─'*55}")
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        fold_accs  = []
        all_preds  = np.zeros(len(y), dtype=np.int64)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{args.folds}  "
                  f"(train {len(tr_idx)}, val {len(val_idx)})")
            model, best_acc, _ = train_model(
                X[tr_idx], y[tr_idx], X[val_idx], y[val_idx],
                args, device, tag=f"F{fold} ")
            preds = predict_all(model, X[val_idx], device)
            all_preds[val_idx] = preds
            acc = (preds == y[val_idx]).mean()
            fold_accs.append(acc)
            print(f"  → Fold {fold} val accuracy: {acc:.4f}")

        cv_mean = np.mean(fold_accs)
        cv_std  = np.std(fold_accs)
        print(f"\n{'='*55}")
        print(f"CV 정확도: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"{'='*55}\n")
        print(classification_report(y, all_preds, target_names=CATEGORIES))

        cm = confusion_matrix(y, all_preds)
        plot_confusion(
            cm,
            f"부위 분류 혼동행렬  ({args.folds}-fold CV,  acc={cv_mean:.3f}±{cv_std:.3f})",
            RESULTS_DIR / "region_clf_confusion.png"
        )

    # ── 전체 데이터 최종 모델 학습 ───────────────────────────
    print(f"\n{'─'*55}")
    print("최종 모델 학습 (전체 데이터, 10% hold-out 모니터링)")
    print(f"{'─'*55}")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_val = max(NUM_CLASSES, int(len(X) * 0.1))
    val_idx_f, tr_idx_f = idx[:n_val], idx[n_val:]

    final_model, final_acc, history = train_model(
        X[tr_idx_f], y[tr_idx_f],
        X[val_idx_f], y[val_idx_f],
        args, device, verbose=True, tag="Final ")

    plot_history(history, "최종 모델 학습 곡선", RESULTS_DIR / "region_clf_history.png")

    # ── 모델 저장 ────────────────────────────────────────────
    model_path = RESULTS_DIR / "region_clf.pth"
    torch.save({
        "model_state": final_model.state_dict(),
        "categories":  CATEGORIES,
        "n_filters":   args.n_filters,
        "dropout":     args.dropout,
        "trim_start":  TRIM_START,
        "trim_count":  TRIM_COUNT,
    }, model_path)
    print(f"\n모델 저장: {model_path}")
    print("\n완료.")


if __name__ == "__main__":
    main()
