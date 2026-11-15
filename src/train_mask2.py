# -*- coding: utf-8 -*-
"""
CNN 종합 실험.

 캐시 2종 x 데이터셋 2종:
   raw  : grayscale + flat-field + CLAHE 3채널 (기존 방식)
   mask : 상처 이진 마스크 3채널  <- 밝기/선명도 정보가 아예 없음

 평가 2종:
   (a) 같은 데이터셋 내부, plate 그룹 교차검증
   (b) 데이터셋 교차 (NEW 로 학습 -> OLD 로 검증, 반대도) = 독립 batch 검증

실행: python train_crossds.py
"""
import os, json, itertools
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from train_p5p15 import (TripletDataset, make_channels, build_model,
                         WORK, DEV, EPOCHS, BATCH, LR, SEED)

MODE = "tri"


def load(ds, kind):
    d = np.load(os.path.join(WORK, f"cache_{ds}_{kind}.npz"))
    meta = json.load(open(os.path.join(WORK, f"cache_{ds}_{kind}_meta.json"),
                          encoding="utf-8"))
    return d["X"], d["y"], np.array([m["plate_id"] for m in meta])


def train(X, y, seed=SEED):
    torch.manual_seed(seed)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.BCEWithLogitsLoss()
    dl = DataLoader(TripletDataset(X, y, True), batch_size=BATCH,
                    shuffle=True, drop_last=len(y) > BATCH)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            loss = crit(model(make_channels(xb, MODE).to(DEV)).squeeze(1),
                        yb.float().to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return model


def predict(model, X, y):
    model.eval(); out = []
    with torch.no_grad():
        for xb, _ in DataLoader(TripletDataset(X, y, False), batch_size=BATCH):
            out.append(torch.sigmoid(
                model(make_channels(xb, MODE).to(DEV)).squeeze(1)).cpu().numpy())
    return np.concatenate(out)


results = {}

print("=" * 74)
print(" (a) 데이터셋 내부 — plate 그룹 교차검증")
print("=" * 74)
for ds, kind in itertools.product(["NEW", "OLD"], ["mask2"]):
    X, y, g = load(ds, kind)
    nf = 2 if len(set(g)) <= 4 else 4
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(nf, shuffle=True,
                                       random_state=SEED).split(X, y, g):
        oof[te] = predict(train(X[tr], y[tr]), X[te], y[te])
    auc, acc = roc_auc_score(y, oof), accuracy_score(y, oof > .5)
    results[(ds, kind, "within")] = auc
    print(f"  {ds:4s} {kind:5s}  n={len(y):3d} plate={len(set(g)):2d} folds={nf}  "
          f"AUC={auc:.4f}  ACC={acc:.4f}")
    np.save(os.path.join(WORK, f"oof_{ds}_{kind}.npy"), oof)

print("\n" + "=" * 74)
print(" (b) 데이터셋 교차 — 독립 batch 검증")
print("=" * 74)
for kind in ["mask2"]:
    for a, b in [("NEW", "OLD"), ("OLD", "NEW")]:
        Xa, ya, _ = load(a, kind)
        Xb, yb, _ = load(b, kind)
        p = predict(train(Xa, ya), Xb, yb)
        auc, acc = roc_auc_score(yb, p), accuracy_score(yb, p > .5)
        results[(a, kind, f"->{b}")] = auc
        print(f"  {kind:5s}  train={a:4s}(n={len(ya):3d}) -> test={b:4s}(n={len(yb):3d})"
              f"   AUC={auc:.4f}  ACC={acc:.4f}")

print("\n" + "=" * 74)
print(" 요약")
print("=" * 74)
print(f"{'':22s}{'mask2':>10s}")
for row, keyf in [("NEW 내부 CV", lambda k: ("NEW", k, "within")),
                  ("OLD 내부 CV", lambda k: ("OLD", k, "within")),
                  ("NEW -> OLD", lambda k: ("NEW", k, "->OLD")),
                  ("OLD -> NEW", lambda k: ("OLD", k, "->NEW"))]:
    print(f"{row:22s}" + "".join(f"{results[keyf(k)]:10.4f}" for k in ["mask2"]))
print("\n 참고: closure_48h 단독(tabular)  NEW->OLD 0.879,  OLD->NEW 0.751")
print("       구 데이터 원본 raw 3채널 (오염됨)  AUC 1.000")
