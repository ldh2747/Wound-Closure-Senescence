# -*- coding: utf-8 -*-
"""
===============================================================================
 P5 vs P15 분류 — 시간 3시점을 채널로 쌓는 ResNet18 전이학습 (단일 파일 버전)
===============================================================================

[데이터 흐름]
  원본 TIFF 1장 : (1944, 2592, 3)  <- 여기서 3 = R,G,B (색)
      |
      | mean(axis=2)          brightfield 라 R≈G≈B. 색은 촬영일 지문이므로 버린다.
      v
  grayscale     : (1944, 2592)
      |
      | resize + flat-field + CLAHE
      v
  전처리 1장    : (384, 512)
      |
      | 0h / 24h / 48h 세 장을 np.stack
      v
  모델 입력     : (3, 384, 512)   <- 여기서 3 = 0h,24h,48h (시간)

  => 채널 축의 '의미'만 색에서 시간으로 바꿔치기. ResNet 첫 conv 가 3채널을
     기대하므로 ImageNet 사전학습 가중치를 그대로 재사용할 수 있다.

[검증]
  같은 plate 의 샘플들은 서로 매우 비슷하므로, 랜덤 분할하면 성능이 부풀려진다.
  -> plate 를 group 으로 묶는 StratifiedGroupKFold 사용.

[실행]
  python train_p5p15.py                # tri, diff, zero 전부
  python train_p5p15.py tri            # 하나만
===============================================================================
"""
import os
import re
import sys
import json

import numpy as np
import cv2
import tifffile
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
from config import BASE
WORK = os.path.join(BASE, "work")
ROOTS = {
    "P5":  os.path.join(BASE, "p5"),                                   # label 0
    "P15": os.path.join(BASE, "p15/26.09.29 hdf p 15 migration"),     # label 1
}
TPS = ["0h", "24h", "48h"]

IMG_W, IMG_H = 512, 384       # 모델 입력 (원본 4:3 유지)
EPOCHS = 25
BATCH = 16
LR = 3e-4
N_FOLDS = 4
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = os.path.join(WORK, "cache_512.npz")
META = os.path.join(WORK, "cache_meta.json")


# ============================================================================
# 1. 데이터 수집 — 0h/24h/48h 가 모두 존재하는 (plate, position) 만 트리플로 인정
# ============================================================================
def find_triplets(root):
    found = {}
    for tp in TPS:
        for d in os.scandir(os.path.join(root, tp)):
            if not d.is_dir():
                continue
            # 폴더명이 '1번 plate' 도 있고 '4번' 도 있어서 앞 숫자만 뽑는다
            plate = re.match(r"(\d+)", d.name).group(1)
            for f in os.scandir(d.path):
                if f.name.lower().endswith(".tif"):
                    pos = int(os.path.splitext(f.name)[0])
                    found.setdefault((plate, pos), {})[tp] = f.path
    return {k: v for k, v in found.items() if len(v) == 3}


# ============================================================================
# 2. 전처리 — RGB 버리고 grayscale, 조명 보정, 대비 강화
# ============================================================================
def preprocess(path):
    a = tifffile.imread(path)                     # (1944, 2592, 3) uint8

    # (1) 색 축 제거. 색은 그룹별 batch 지문이라 반드시 버린다.
    g = a.mean(2).astype(np.float32) if a.ndim == 3 else a.astype(np.float32)

    # (2) 리사이즈
    g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

    # (3) flat-field 보정 — 현미경 vignetting(가장자리 어두움) 제거.
    #     큰 sigma 로 흐린 배경을 만들어 빼면 저주파 조명 성분만 사라진다.
    bg = cv2.GaussianBlur(g, (0, 0), 30)
    f = np.clip(g - bg + 128, 0, 255).astype(np.uint8)

    # (4) CLAHE — 원본 대비가 std 15~21 수준으로 매우 낮아 세포가 잘 안 보임
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(f)


def build_cache():
    """전체 트리플을 (N, 3, H, W) 로 캐시. 3 = 시간축."""
    X, y, meta = [], [], []
    for label, (grp, root) in enumerate(ROOTS.items()):
        tri = find_triplets(root)
        for (plate, pos) in sorted(tri, key=lambda k: (k[0], k[1])):
            paths = tri[(plate, pos)]

            # ★ 핵심 한 줄 — 0h/24h/48h grayscale 3장을 채널로 쌓는다
            X.append(np.stack([preprocess(paths[t]) for t in TPS]))   # (3, H, W)

            y.append(label)
            meta.append({"group": grp, "plate": plate, "pos": pos,
                         "plate_id": f"{grp}_p{plate}"})
        print(f"  {grp}: {len(tri)} triplets")

    X = np.stack(X)                       # (N, 3, H, W) uint8
    y = np.array(y, dtype=np.int64)
    os.makedirs(WORK, exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y)
    json.dump(meta, open(META, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return X, y, meta


def load_data():
    if os.path.exists(CACHE) and os.path.exists(META):
        d = np.load(CACHE)
        return d["X"], d["y"], json.load(open(META, encoding="utf-8"))
    print("캐시 생성 중...")
    return build_cache()


# ============================================================================
# 3. 채널 구성 모드
# ============================================================================
def make_channels(x, mode):
    """x: (B, 3, H, W) float32 [0,1]. 채널 = [0h, 24h, 48h]"""
    if mode == "tri":       # 세 시점 그대로
        return x
    if mode == "diff":      # 변화량 강조: [0h, 24h-0h, 48h-24h]
        return torch.stack([x[:, 0], x[:, 1] - x[:, 0], x[:, 2] - x[:, 1]], dim=1)
    if mode == "zero":      # ★대조군: 0h 만 3번 복제.
                            #  0h 에는 이동 정보가 물리적으로 없으므로 성능이
                            #  높게 나오면 = 모델이 이동이 아닌 다른 걸 보고 있다는 증거
        return x[:, 0:1].repeat(1, 3, 1, 1)
    raise ValueError(f"unknown mode: {mode}")


# ============================================================================
# 4. Dataset — 증강은 3채널에 '동일하게' 걸려야 한다 (시점 간 정렬 유지)
# ============================================================================
class TripletDataset(Dataset):
    def __init__(self, X, y, train):
        self.X, self.y, self.train = X, y, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i].astype(np.float32) / 255.0)   # (3, H, W)
        if self.train:
            # flip 은 텐서 전체에 걸리므로 세 시점에 자동으로 동일 적용된다
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2])          # 좌우 (스크래치가 세로라 안전)
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[1])          # 상하
            # 밝기/대비 jitter — batch 조명 차이에 둔감해지도록 강하게
            x = x * (0.8 + 0.4 * torch.rand(1)) + (torch.rand(1) - 0.5) * 0.2
            x = x.clamp(0, 1)
        return x, self.y[i]


# ============================================================================
# 5. 모델
# ============================================================================
def build_model():
    try:
        m = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(f"    [경고] pretrained 로드 실패({type(e).__name__}) -> random init")
        m = torchvision.models.resnet18(weights=None)
    # in_chans 를 건드리지 않는다. 3채널 그대로 쓰되 의미만 시간축으로 재해석.
    m.fc = nn.Linear(512, 1)          # 이진분류 -> logit 1개 + BCEWithLogits
    return m.to(DEV)


# ============================================================================
# 6. 학습 + plate 단위 교차검증
# ============================================================================
def run(mode, X, y, groups):
    print(f"\n{'='*64}")
    print(f" MODE = {mode}   N={len(y)}  (P5={(y==0).sum()}, P15={(y==1).sum()})")
    print(f" plate 수 = {len(set(groups))}  | device = {DEV}")
    print(f"{'='*64}")

    oof = np.zeros(len(y), dtype=np.float32)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups)):
        torch.manual_seed(SEED + fold)
        model = build_model()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        crit = nn.BCEWithLogitsLoss()

        dl_tr = DataLoader(TripletDataset(X[tr_idx], y[tr_idx], True),
                           batch_size=BATCH, shuffle=True, drop_last=True)
        dl_te = DataLoader(TripletDataset(X[te_idx], y[te_idx], False),
                           batch_size=BATCH, shuffle=False)

        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl_tr:
                xb = make_channels(xb, mode).to(DEV)
                yb = yb.float().to(DEV)
                loss = crit(model(xb).squeeze(1), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in dl_te:
                xb = make_channels(xb, mode).to(DEV)
                preds.append(torch.sigmoid(model(xb).squeeze(1)).cpu().numpy())
        oof[te_idx] = np.concatenate(preds)

        held = sorted(set(groups[te_idx]))
        yt = y[te_idx]
        line = (f"  fold{fold}  n={len(te_idx):3d}  "
                f"acc={accuracy_score(yt, oof[te_idx] > 0.5):.3f}")
        if len(set(yt)) == 2:
            line += f"  auc={roc_auc_score(yt, oof[te_idx]):.3f}"
        else:
            line += "  auc=N/A(한 클래스만)"
        print(line + f"   test plates={held}")

    auc = roc_auc_score(y, oof)
    acc = accuracy_score(y, oof > 0.5)
    cm = confusion_matrix(y, oof > 0.5)
    print(f"\n  >>> OOF   AUC = {auc:.4f}   ACC = {acc:.4f}")
    print(f"      confusion:  P5 ->[P5 {cm[0,0]}, P15 {cm[0,1]}]")
    print(f"                 P15 ->[P5 {cm[1,0]}, P15 {cm[1,1]}]")
    np.save(os.path.join(WORK, f"oof_{mode}.npy"), oof)
    return auc, acc


def main():
    X, y, meta = load_data()
    groups = np.array([m["plate_id"] for m in meta])
    print(f"데이터: X={X.shape} {X.dtype}  (채널축 = 0h/24h/48h)")

    modes = sys.argv[1:] or ["tri", "diff", "zero"]
    results = {m: run(m, X, y, groups) for m in modes}

    print(f"\n{'='*64}\n 요약")
    for m, (a, c) in results.items():
        tag = "  <- 대조군. 높으면 이동 외 정보로 맞히는 중" if m == "zero" else ""
        print(f"   {m:5s}  AUC={a:.4f}  ACC={c:.4f}{tag}")


if __name__ == "__main__":
    main()
