# -*- coding: utf-8 -*-
"""상처 폭 feature 분류"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, accuracy_score

from config import WORK
VIZ = os.path.join(WORK, "viz")
FEATS = ["width_0h", "width_24h", "width_48h",
         "closure_24h", "closure_48h", "rate_1", "rate_2"]

df = pd.read_csv(os.path.join(WORK, "wound_width.csv"))
df = df.dropna(subset=FEATS).copy()
df["y"] = (df.group == "P15").astype(int)


def fit_eval(tr, te, feats=FEATS):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    m.fit(tr[feats], tr.y)
    p = m.predict_proba(te[feats])[:, 1]
    auc = roc_auc_score(te.y, p) if te.y.nunique() == 2 else float("nan")
    return m, p, auc, accuracy_score(te.y, p > .5)


print("=" * 72); print(" (a) 데이터셋 내부 — Leave-One-Plate-Out"); print("=" * 72)
oof_store = {}
for ds in ["NEW", "OLD"]:
    d = df[df.dataset == ds].reset_index(drop=True)
    oof = np.zeros(len(d))
    for tr_i, te_i in LeaveOneGroupOut().split(d, d.y, d.plate_id):
        _, p, _, _ = fit_eval(d.iloc[tr_i], d.iloc[te_i])
        oof[te_i] = p
    auc, acc = roc_auc_score(d.y, oof), accuracy_score(d.y, oof > .5)
    oof_store[ds] = (d, oof)
    print(f"  {ds:4s}  n={len(d):3d}  plate={d.plate_id.nunique():2d}  "
          f"AUC={auc:.4f}  ACC={acc:.4f}")

print("\n" + "=" * 72)
print(" (b) 데이터셋 교차 — 완전 독립 batch 검증")
print("=" * 72)
cross = {}
for a, b in [("NEW", "OLD"), ("OLD", "NEW")]:
    tr, te = df[df.dataset == a], df[df.dataset == b]
    _, p, auc, acc = fit_eval(tr, te)
    cross[(a, b)] = (te, p, auc, acc)
    print(f"  train={a:4s} (n={len(tr):3d})  ->  test={b:4s} (n={len(te):3d})   "
          f"AUC={auc:.4f}  ACC={acc:.4f}")

print("\n 단일 feature 만으로 교차 검증 (train NEW -> test OLD)")
tr, te = df[df.dataset == "NEW"], df[df.dataset == "OLD"]
for f in FEATS:
    _, p, auc, acc = fit_eval(tr, te, [f])
    print(f"   {f:14s} AUC={auc:.4f}")

# 계수
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
m.fit(df[FEATS], df.y)
print("\n 전체 데이터 로지스틱 계수 (표준화, +는 P15 쪽)")
for f, c in sorted(zip(FEATS, m[-1].coef_[0]), key=lambda x: -abs(x[1])):
    print(f"   {f:14s} {c:+.3f}")

# figure
fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.5), constrained_layout=True)
TP = [0, 24, 48]
for c, ds in enumerate(["NEW", "OLD"]):
    d = df[df.dataset == ds]
    # 1) closure 곡선
    for grp, col in [("P5", "#1a6fcc"), ("P15", "#cc3311")]:
        s = d[d.group == grp]
        y = [0, s.closure_24h.mean(), s.closure_48h.mean()]
        e = [0, s.closure_24h.sem(), s.closure_48h.sem()]
        ax[c, 0].errorbar(TP, y, yerr=e, marker="o", ms=8, lw=2.5, capsize=5,
                          color=col, label=f"{grp} (n={len(s)})")
    ax[c, 0].set_title(f"{ds} data — wound closure", fontsize=13, fontweight="bold")
    ax[c, 0].set_xlabel("hours"); ax[c, 0].set_ylabel("closure fraction")
    ax[c, 0].set_xticks(TP); ax[c, 0].legend(); ax[c, 0].grid(alpha=.3)
    ax[c, 0].set_ylim(-0.05, 1.0)

    # 2) 폭 곡선
    for grp, col in [("P5", "#1a6fcc"), ("P15", "#cc3311")]:
        s = d[d.group == grp]
        y = [s.width_0h.mean(), s.width_24h.mean(), s.width_48h.mean()]
        e = [s.width_0h.sem(), s.width_24h.sem(), s.width_48h.sem()]
        ax[c, 1].errorbar(TP, y, yerr=e, marker="s", ms=8, lw=2.5, capsize=5,
                          color=col, label=grp)
    ax[c, 1].set_title(f"{ds} data — wound width", fontsize=13, fontweight="bold")
    ax[c, 1].set_xlabel("hours"); ax[c, 1].set_ylabel("width (px)")
    ax[c, 1].set_xticks(TP); ax[c, 1].legend(); ax[c, 1].grid(alpha=.3)

    # 3) closure_48h 박스 + plate 평균
    data = [d[d.group == g].closure_48h.values for g in ["P5", "P15"]]
    bp = ax[c, 2].boxplot(data, labels=["P5", "P15"], widths=.55,
                          patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], ["#1a6fcc", "#cc3311"]):
        patch.set_facecolor(col); patch.set_alpha(.35)
    for i, g in enumerate(["P5", "P15"]):
        s = d[d.group == g]
        ax[c, 2].scatter(np.random.normal(i+1, .055, len(s)), s.closure_48h,
                         s=13, alpha=.45, color="#1a6fcc" if g == "P5" else "#cc3311")
        pm = s.groupby("plate_id").closure_48h.mean()
        ax[c, 2].scatter([i+1]*len(pm), pm, s=130, marker="D", zorder=5,
                         edgecolor="k", linewidth=1.4,
                         color="#1a6fcc" if g == "P5" else "#cc3311")
    a_ = d[d.group == "P5"].closure_48h; b_ = d[d.group == "P15"].closure_48h
    u = sum(1 for x in a_ for z in b_ if x > z)/(len(a_)*len(b_))
    ax[c, 2].set_title(f"{ds} — closure at 48h   AUC={max(u,1-u):.3f}",
                       fontsize=13, fontweight="bold")
    ax[c, 2].set_ylabel("closure fraction"); ax[c, 2].grid(alpha=.3, axis="y")

fig.suptitle("Scratch-assay quantification — P5 (young) vs P15 (old)\n"
             "diamonds = plate means (the valid unit for statistics)", fontsize=15)
out = os.path.join(VIZ, "closure_result.png")
fig.savefig(out, dpi=100)
print(f"\nsaved {out}")
