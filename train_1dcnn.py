"""
train_1dcnn.py
1D-CNN 기반 진피/근막 깊이 예측 모델 (PyTorch)
NPU 배포(MCXNx4x INT8) 를 고려한 소형 구조

입력: 1250샘플 정규화 ADC 신호 → [-1, +1]
출력: [dermis_mm, fascia_mm]
분할: Leave-One-Patient-Out CV
결과: results/cnn_model_cv.png, results/best_1dcnn.pth

사용법:
  python train_1dcnn.py                          # 기본 학습
  python train_1dcnn.py --lr 5e-4 --batch 16     # 하이퍼파라미터 지정
  python train_1dcnn.py --search                 # 그리드 서치
  python train_1dcnn.py --search --quick         # 빠른 탐색 (에폭 줄임)
"""

import re, json, pickle, argparse, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── 경로 ─────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
USDATA     = ROOT / "usdata" / "data"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 신호 파라미터 ─────────────────────────────────────────────
TRIM_START     = 1200
TRIM_COUNT     = 1250
DISPLAY_OFFSET = 12.00
SAMPLE_NS      = 10
SOUND_SPEED    = 1540.0

FILENAME_RE = re.compile(
    r'^(?P<name>.+)_(?P<date>\d{8})_(?P<time>\d{6})_'
    r'\((?P<pos_num>\d+)\)(?P<position>.+?)_(?P<gender>[MF])\.csv$'
)

# ── 데이터 로드 ───────────────────────────────────────────────
def load_adc(path):
    adc = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                v = int(line)
                if 0 <= v <= 255: adc.append(v)
            except ValueError: pass
    adc = np.array(adc, dtype=np.float32)
    if len(adc) > TRIM_COUNT:
        end = TRIM_START + TRIM_COUNT
        adc = adc[TRIM_START:end] if len(adc) >= end else adc[TRIM_START:]
    if len(adc) < TRIM_COUNT:
        adc = np.pad(adc, (0, TRIM_COUNT - len(adc)))
    return adc[:TRIM_COUNT]


def load_label(csv_path):
    json_path = csv_path.with_name(csv_path.stem + "_positions.json")
    if not json_path.exists(): return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    dermis_mm = fascia_mm = None
    for pos in data.get("positions", []):
        name = pos.get("position_name", "")
        mm   = pos.get("thickness_mm")
        if name in ("피하지방시작", "Dermis") and mm is not None:
            dermis_mm = float(mm)
        elif name == "Fascia" and mm is not None:
            fascia_mm = float(mm)
    if dermis_mm is None or fascia_mm is None: return None
    return dermis_mm, fascia_mm


def build_dataset():
    rows = []
    for csv_path in sorted(USDATA.rglob("*.csv")):
        if "_positions" in csv_path.name: continue
        m = FILENAME_RE.match(csv_path.name)
        if not m: continue
        label = load_label(csv_path)
        if label is None: continue
        adc = load_adc(csv_path)
        rows.append({
            "patient":  m.group("name"),
            "pos_num":  int(m.group("pos_num")),
            "adc":      adc,
            "dermis":   label[0],
            "fascia":   label[1],
        })
    return rows


# ── 정규화 ────────────────────────────────────────────────────
def normalize_signal(adc: np.ndarray) -> np.ndarray:
    return (adc - 128.0) / 128.0


# ── Dataset ───────────────────────────────────────────────────
class UltrasoundDataset(Dataset):
    def __init__(self, rows):
        self.x = torch.tensor(
            np.stack([normalize_signal(r["adc"]) for r in rows], axis=0),
            dtype=torch.float32
        ).unsqueeze(1)
        self.y = torch.tensor(
            [[r["dermis"], r["fascia"]] for r in rows],
            dtype=torch.float32
        )

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


# ── 모델 ─────────────────────────────────────────────────────
class SkinCNN(nn.Module):
    """
    소형 1D-CNN — NPU INT8 양자화 고려
    n_filters: 기본 채널 수 (16 → 작음, 32 → 중간)
    dropout: 헤드 드롭아웃
    """
    def __init__(self, n_filters=16, dropout=0.3):
        super().__init__()
        f = n_filters
        self.encoder = nn.Sequential(
            # Block 1: 1250 → 625
            nn.Conv1d(1,   f,   kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(f),   nn.ReLU(),
            # Block 2: 625 → 313
            nn.Conv1d(f,   f*2, kernel_size=7,  stride=2, padding=3),
            nn.BatchNorm1d(f*2), nn.ReLU(),
            # Block 3: 313 → 157
            nn.Conv1d(f*2, f*4, kernel_size=5,  stride=2, padding=2),
            nn.BatchNorm1d(f*4), nn.ReLU(),
            # Block 4: 157 → 79
            nn.Conv1d(f*4, f*4, kernel_size=5,  stride=2, padding=2),
            nn.BatchNorm1d(f*4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f*4, f*2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(f*2, 2),
        )

    def forward(self, x):
        return self.head(self.encoder(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 학습 루프 ─────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        total_loss += criterion(pred, y).item() * len(x)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae_d = float(np.mean(np.abs(preds[:, 0] - trues[:, 0])))
    mae_f = float(np.mean(np.abs(preds[:, 1] - trues[:, 1])))
    return total_loss / len(loader.dataset), mae_d, mae_f, preds, trues


def train_model(train_rows, val_rows,
                epochs=150, patience=20,
                lr=1e-3, batch=32,
                n_filters=16, dropout=0.3,
                verbose=True, prefix=""):
    """
    단일 train/val 학습
    verbose=True  → tqdm 진행바 + 에폭별 상세 출력
    verbose=False → 진행바 없음 (LOO-CV 내부 호출용)
    """
    train_ds = UltrasoundDataset(train_rows)
    val_ds   = UltrasoundDataset(val_rows)
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,  drop_last=len(train_rows)>batch)
    val_ld   = DataLoader(val_ds,   batch_size=batch)

    model     = SkinCNN(n_filters=n_filters, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    best_val   = float("inf")
    best_state = None
    no_improve = 0
    history    = {"train": [], "val": [], "mae_d": [], "mae_f": []}

    epoch_iter = range(1, epochs + 1)
    if verbose and HAS_TQDM:
        epoch_iter = tqdm(epoch_iter, desc=f"{prefix}학습", unit="ep",
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

    for ep in epoch_iter:
        tr_loss = train_epoch(model, train_ld, optimizer, criterion)
        val_loss, mae_d, mae_f, _, _ = eval_epoch(model, val_ld, criterion)
        scheduler.step()

        history["train"].append(tr_loss)
        history["val"].append(val_loss)
        history["mae_d"].append(mae_d)
        history["mae_f"].append(mae_f)

        improved = val_loss < best_val
        if improved:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # 진행 상황 업데이트
        if verbose and HAS_TQDM:
            epoch_iter.set_postfix(
                tr=f"{tr_loss:.4f}",
                val=f"{val_loss:.4f}",
                D=f"{mae_d:.3f}",
                F=f"{mae_f:.3f}",
                best=f"{best_val:.4f}",
                pat=f"{no_improve}/{patience}",
                refresh=True
            )
        elif verbose and ep % 10 == 0:
            # tqdm 없을 때 10에폭마다 출력
            marker = "*" if improved else " "
            print(f"  Ep {ep:4d}/{epochs}  {marker}"
                  f"  tr={tr_loss:.4f}  val={val_loss:.4f}"
                  f"  D={mae_d:.3f}mm  F={mae_f:.3f}mm"
                  f"  (patience {no_improve}/{patience})")

        if no_improve >= patience:
            if verbose and HAS_TQDM:
                epoch_iter.set_description(f"{prefix}조기종료 ep={ep}")
            elif verbose:
                print(f"  조기 종료: epoch {ep}")
            break

    model.load_state_dict(best_state)
    return model, history


# ── Leave-One-Patient-Out CV ──────────────────────────────────
def patient_loo_cv(all_rows, hp: dict, quick=False):
    patients = sorted({r["patient"] for r in all_rows})
    records  = []
    epochs   = 80 if quick else 150

    for i, pat in enumerate(patients):
        train_rows = [r for r in all_rows if r["patient"] != pat]
        test_rows  = [r for r in all_rows if r["patient"] == pat]

        prefix = f"[{i+1:2d}/{len(patients)}] {pat:<10s} "
        print(f"  {prefix}train={len(train_rows):3d}  test={len(test_rows):2d}", end=" ", flush=True)

        model, _ = train_model(
            train_rows, test_rows,
            epochs=epochs, patience=20,
            lr=hp["lr"], batch=hp["batch"],
            n_filters=hp["n_filters"], dropout=hp["dropout"],
            verbose=True, prefix=prefix
        )

        _, mae_d, mae_f, preds, trues = eval_epoch(
            model, DataLoader(UltrasoundDataset(test_rows), batch_size=32),
            nn.L1Loss()
        )
        print(f"\n        → 진피 {mae_d:.3f}mm  근막 {mae_f:.3f}mm")
        records.append({
            "patient": pat, "n_test": len(test_rows),
            "mae_dermis": mae_d, "mae_fascia": mae_f,
            "preds": preds, "trues": trues
        })

    return pd.DataFrame(records)


# ── 하이퍼파라미터 그리드 서치 ────────────────────────────────
SEARCH_GRID = {
    "lr":       [1e-3, 5e-4, 2e-4],
    "batch":    [16, 32],
    "n_filters":[16, 32],
    "dropout":  [0.2, 0.3, 0.5],
}

def hyperparameter_search(all_rows, quick=True):
    """
    랜덤/그리드 하이퍼파라미터 탐색
    LOO-CV 대신 마지막 환자를 검증셋으로 고정 (속도)
    """
    patients   = sorted({r["patient"] for r in all_rows})
    val_pat    = patients[-1]          # 마지막 환자 = 검증
    train_rows = [r for r in all_rows if r["patient"] != val_pat]
    val_rows   = [r for r in all_rows if r["patient"] == val_pat]

    keys   = list(SEARCH_GRID.keys())
    combos = list(itertools.product(*[SEARCH_GRID[k] for k in keys]))
    epochs = 60 if quick else 120

    print(f"\n  총 {len(combos)}개 조합 탐색  (검증 환자: {val_pat})")
    print(f"  에폭: {epochs}  (--quick={quick})")
    print(f"  {'#':>3}  {'lr':>8}  {'batch':>6}  {'filters':>8}  {'dropout':>8}"
          f"  {'D_MAE':>8}  {'F_MAE':>8}  {'합계':>8}")
    print("  " + "-"*65)

    results = []
    best_total = float("inf")
    best_hp    = None

    for ci, values in enumerate(combos, 1):
        hp = dict(zip(keys, values))
        model, _ = train_model(
            train_rows, val_rows,
            epochs=epochs, patience=15,
            lr=hp["lr"], batch=hp["batch"],
            n_filters=hp["n_filters"], dropout=hp["dropout"],
            verbose=True,
            prefix=f"[{ci}/{len(combos)}] "
        )
        _, mae_d, mae_f, _, _ = eval_epoch(
            model, DataLoader(UltrasoundDataset(val_rows), batch_size=32), nn.L1Loss()
        )
        total = mae_d + mae_f
        mark  = " <-- 최적" if total < best_total else ""
        if total < best_total:
            best_total = total
            best_hp    = hp.copy()

        results.append({**hp, "mae_d": mae_d, "mae_f": mae_f, "total": total})
        print(f"\n  {ci:>3}  {hp['lr']:>8.0e}  {hp['batch']:>6d}"
              f"  {hp['n_filters']:>8d}  {hp['dropout']:>8.2f}"
              f"  {mae_d:>8.4f}  {mae_f:>8.4f}  {total:>8.4f}{mark}")

    # 결과 정렬
    df = pd.DataFrame(results).sort_values("total").reset_index(drop=True)
    print("\n  ─── 상위 5 조합 ───────────────────────────────────────────")
    print(df.head().to_string(index=False))

    return best_hp, df


# ── 최종 모델 학습 ────────────────────────────────────────────
def train_final(all_rows, hp: dict, epochs=200):
    patients   = sorted({r["patient"] for r in all_rows})
    val_pat    = patients[-1]
    train_rows = [r for r in all_rows if r["patient"] != val_pat]
    val_rows   = [r for r in all_rows if r["patient"] == val_pat]
    model, history = train_model(
        train_rows, val_rows,
        epochs=epochs, patience=30,
        lr=hp["lr"], batch=hp["batch"],
        n_filters=hp["n_filters"], dropout=hp["dropout"],
        verbose=True, prefix="[최종] "
    )
    return model, history


# ── 시각화 ───────────────────────────────────────────────────
def plot_results(cv_df, rf_mae, history, hp: dict):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    hp_str = f"lr={hp['lr']:.0e}  batch={hp['batch']}  f={hp['n_filters']}  drop={hp['dropout']}"
    fig.suptitle(f"1D-CNN — Leave-One-Patient-Out CV\n({hp_str})", fontsize=13)

    # 1. MAE 박스플롯 (진피)
    ax = axes[0][0]
    ax.boxplot([cv_df["mae_dermis"].values], tick_labels=["1D-CNN"], patch_artist=True,
               boxprops=dict(facecolor="#4c72b0", alpha=0.7))
    ax.axhline(rf_mae["dermis"], color="orange", linestyle="--",
               linewidth=1.5, label=f"RF baseline {rf_mae['dermis']:.3f}mm")
    ax.axhline(0.3, color="red", linestyle=":", linewidth=1, label="목표 0.3mm")
    ax.set_title("Dermis MAE 분포")
    ax.set_ylabel("MAE (mm)"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4)

    # 2. MAE 박스플롯 (근막)
    ax = axes[0][1]
    ax.boxplot([cv_df["mae_fascia"].values], tick_labels=["1D-CNN"], patch_artist=True,
               boxprops=dict(facecolor="#dd8452", alpha=0.7))
    ax.axhline(rf_mae["fascia"], color="orange", linestyle="--",
               linewidth=1.5, label=f"RF baseline {rf_mae['fascia']:.3f}mm")
    ax.axhline(0.3, color="red", linestyle=":", linewidth=1, label="목표 0.3mm")
    ax.set_title("Fascia MAE 분포")
    ax.set_ylabel("MAE (mm)"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4)

    # 3. 환자별 MAE 막대
    ax = axes[0][2]
    x = np.arange(len(cv_df))
    ax.bar(x - 0.2, cv_df["mae_dermis"], 0.4, label="Dermis", color="#4c72b0", alpha=0.8)
    ax.bar(x + 0.2, cv_df["mae_fascia"], 0.4, label="Fascia",  color="#dd8452", alpha=0.8)
    ax.axhline(0.3, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(cv_df["patient"], rotation=45, ha="right", fontsize=7)
    ax.set_title("환자별 MAE"); ax.set_ylabel("MAE (mm)")
    ax.legend(); ax.grid(axis="y", alpha=0.4)

    # 4. 손실 곡선 (최종 모델)
    ax = axes[1][0]
    ax.plot(history["train"], label="Train Loss", color="#4c72b0")
    ax.plot(history["val"],   label="Val Loss",   color="#dd8452")
    ax.plot(history["mae_d"], label="Val Dermis MAE", color="green",  linestyle="--")
    ax.plot(history["mae_f"], label="Val Fascia MAE", color="purple", linestyle="--")
    ax.set_title("최종 모델 학습 곡선"); ax.set_xlabel("Epoch"); ax.set_ylabel("mm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 5. 예측 vs 실제 (전체 CV 합산)
    all_preds = np.concatenate([r["preds"] for r in cv_df.to_dict("records")])
    all_trues = np.concatenate([r["trues"] for r in cv_df.to_dict("records")])
    for col, (ti, label, color) in enumerate(
            zip([0, 1], ["Dermis (mm)", "Fascia (mm)"], ["#4c72b0", "#dd8452"])):
        ax = axes[1][col + 1]
        ax.scatter(all_trues[:, ti], all_preds[:, ti], alpha=0.4, s=15, color=color)
        lim = [min(all_trues[:, ti].min(), all_preds[:, ti].min()) - 0.1,
               max(all_trues[:, ti].max(), all_preds[:, ti].max()) + 0.1]
        ax.plot(lim, lim, "r--", linewidth=1)
        mae = float(np.mean(np.abs(all_preds[:, ti] - all_trues[:, ti])))
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"실제 {label}"); ax.set_ylabel(f"예측 {label}")
        ax.set_title(f"{label}  CV MAE={mae:.3f}mm")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = RESULT_DIR / "cnn_model_cv.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  그래프 저장: {out}")


# ── argparse ─────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="1D-CNN 피부층 깊이 예측 학습")
    p.add_argument("--lr",       type=float, default=1e-3,  help="학습률 (default: 1e-3)")
    p.add_argument("--batch",    type=int,   default=32,    help="배치 크기 (default: 32)")
    p.add_argument("--epochs",   type=int,   default=150,   help="최대 에폭 (default: 150)")
    p.add_argument("--patience", type=int,   default=20,    help="조기 종료 patience (default: 20)")
    p.add_argument("--n_filters",type=int,   default=16,    help="기본 채널 수 (default: 16)")
    p.add_argument("--dropout",  type=float, default=0.3,   help="드롭아웃 비율 (default: 0.3)")
    p.add_argument("--search",   action="store_true",       help="하이퍼파라미터 그리드 서치 실행")
    p.add_argument("--quick",    action="store_true",       help="빠른 탐색 (에폭 줄임)")
    p.add_argument("--no_cv",    action="store_true",       help="LOO-CV 건너뛰고 최종 모델만 학습")
    return p.parse_args()


# ── 메인 ─────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 60)
    print("  1D-CNN Model Training  (PyTorch)")
    print(f"  Device: {DEVICE}")
    if HAS_TQDM:
        print("  Progress: tqdm 진행바 사용")
    else:
        print("  Progress: 10에폭마다 출력  (pip install tqdm 으로 진행바 활성화)")
    print("=" * 60)

    # 1. 데이터셋
    print("\n[1] 데이터셋 구성 중...")
    all_rows = build_dataset()
    patients = sorted({r["patient"] for r in all_rows})
    print(f"  총 샘플: {len(all_rows)}개  /  환자: {len(patients)}명")

    # 2. 하이퍼파라미터 결정
    if args.search:
        print("\n[2] 하이퍼파라미터 그리드 서치...")
        best_hp, search_df = hyperparameter_search(all_rows, quick=args.quick)
        search_df.to_csv(RESULT_DIR / "cnn_search_results.csv", index=False)
        print(f"\n  최적 하이퍼파라미터: {best_hp}")
        hp = best_hp
    else:
        hp = {
            "lr":       args.lr,
            "batch":    args.batch,
            "n_filters":args.n_filters,
            "dropout":  args.dropout,
        }

    dummy = SkinCNN(n_filters=hp["n_filters"], dropout=hp["dropout"])
    print(f"\n  하이퍼파라미터: lr={hp['lr']:.0e}  batch={hp['batch']}"
          f"  n_filters={hp['n_filters']}  dropout={hp['dropout']}")
    print(f"  모델 파라미터: {dummy.count_params():,}개")

    # 3. LOO-CV
    cv_df = None
    if not args.no_cv:
        print(f"\n[3] Leave-One-Patient-Out CV  (epochs={args.epochs if not args.quick else 80})...")
        cv_df = patient_loo_cv(all_rows, hp, quick=args.quick)
        print(f"\n  ── CV 결과 ──────────────────────────────")
        print(f"  진피 MAE: {cv_df['mae_dermis'].mean():.3f} ± {cv_df['mae_dermis'].std():.3f} mm")
        print(f"  근막 MAE: {cv_df['mae_fascia'].mean():.3f} ± {cv_df['mae_fascia'].std():.3f} mm")
    else:
        print("\n[3] LOO-CV 건너뜀 (--no_cv)")

    # 4. 최종 모델
    final_epochs = 200 if not args.quick else 100
    print(f"\n[4] 최종 모델 학습  (epochs={final_epochs})...")
    final_model, history = train_final(all_rows, hp, epochs=final_epochs)

    model_path = RESULT_DIR / "best_1dcnn.pth"
    save_dict = {
        "model_state":  final_model.state_dict(),
        "model_class":  "SkinCNN",
        "hyperparams":  hp,
        "input_size":   TRIM_COUNT,
    }
    if cv_df is not None:
        save_dict["cv_mae_dermis"] = float(cv_df["mae_dermis"].mean())
        save_dict["cv_mae_fascia"] = float(cv_df["mae_fascia"].mean())
    torch.save(save_dict, model_path)
    print(f"  모델 저장: {model_path}")

    # 5. 시각화
    if cv_df is not None:
        print("\n[5] 결과 시각화...")
        rf_pkl = RESULT_DIR / "feature_model.pkl"
        rf_mae = {"dermis": 0.125, "fascia": 0.068}
        if rf_pkl.exists():
            with open(rf_pkl, "rb") as f:
                rf_data = pickle.load(f)
            rf_cv = rf_data.get("cv_results", {}).get("Random Forest")
            if rf_cv is not None:
                rf_mae = {"dermis": float(rf_cv["mae_dermis"].mean()),
                          "fascia": float(rf_cv["mae_fascia"].mean())}
        plot_results(cv_df, rf_mae, history, hp)

        # 최종 요약
        print("\n" + "=" * 60)
        print("  최종 비교")
        print("=" * 60)
        print(f"  {'모델':<22s}  {'진피 MAE':>10s}  {'근막 MAE':>10s}")
        print(f"  {'-'*46}")
        print(f"  {'Random Forest (기준)':<22s}  {rf_mae['dermis']:>8.3f}mm  {rf_mae['fascia']:>8.3f}mm")
        cnn_d = cv_df["mae_dermis"].mean()
        cnn_f = cv_df["mae_fascia"].mean()
        print(f"  {'1D-CNN':<22s}  {cnn_d:>8.3f}mm  {cnn_f:>8.3f}mm")
        winner = "1D-CNN" if (cnn_d + cnn_f) < (rf_mae["dermis"] + rf_mae["fascia"]) \
                 else "Random Forest"
        print(f"\n  최적 모델: {winner}")
        goal_d = "OK" if cnn_d < 0.3 else "NG"
        goal_f = "OK" if cnn_f < 0.3 else "NG"
        print(f"  진피 목표 <0.30mm: {goal_d}  근막 목표 <0.30mm: {goal_f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
