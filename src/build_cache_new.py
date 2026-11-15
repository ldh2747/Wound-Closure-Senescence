# -*- coding: utf-8 -*-
"""캐시 생성 (raw, mask)"""
import os, re, json
import numpy as np, cv2, tifffile

from config import BASE
WORK = os.path.join(BASE, "work")
ADD = os.path.join(BASE, "20261013,4 AI hDF P5,15 migration(additional_data)")
DATASETS = {
    "NEW": {"P5":  os.path.join(ADD, "20261013 AI hDF P5 migration"),
            "P15": os.path.join(ADD, "20261014 AI hdf P15 migrtion")},
    "OLD": {"P5":  os.path.join(BASE, "p5"),
            "P15": os.path.join(BASE, "p15/26.09.29 hdf p 15 migration")},
}
TPS = ["0h", "24h", "48h"]
W, H = 512, 384
SEG_W, SEG_H = 648, 486
ENV_W, ENV_H = 421, 81
FRAC, SMOOTH = 0.60, 13


def norm_pos(s): return int(re.findall(r"(\d+)", s)[-1])


def find_triplets(root):
    found = {}
    for tp in TPS:
        td = os.path.join(root, tp)
        for d in sorted(os.listdir(td)):
            dp = os.path.join(td, d)
            if not os.path.isdir(dp): continue
            plate = re.search(r"(\d+)", d).group(1)
            for f in sorted(os.listdir(dp)):
                if f.lower().endswith(".tif"):
                    found.setdefault((plate, norm_pos(os.path.splitext(f)[0])),
                                     {})[tp] = os.path.join(dp, f)
    return {k: v for k, v in found.items() if len(v) == 3}


def gray_full(path):
    g = tifffile.imread(path)
    return g.mean(2).astype(np.float32) if g.ndim == 3 else g.astype(np.float32)


def raw_chan(path):
    g = cv2.resize(gray_full(path), (W, H), interpolation=cv2.INTER_AREA)
    f = np.clip(g - cv2.GaussianBlur(g, (0, 0), 30) + 128, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(3.0, (8, 8)).apply(f)


def mask_chan(path):
    g = gray_full(path)
    hp = g - cv2.GaussianBlur(g, (0, 0), 6)
    m = cv2.blur(hp, (121, 121))
    t = np.sqrt(np.maximum(cv2.blur(hp * hp, (121, 121)) - m * m, 0))
    t = cv2.resize(t, (SEG_W, SEG_H), interpolation=cv2.INTER_AREA)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (ENV_W, ENV_H))
    r = t / (cv2.GaussianBlur(cv2.dilate(t, ker), (0, 0), 40) + 1e-6)
    r = cv2.GaussianBlur(r, (0, 0), SMOOTH)
    mk = (r < FRAC).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, k)
    mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, k)
    return cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)


def build(ds_list, kind):
    fn = raw_chan if kind == "raw" else mask_chan
    X, y, meta = [], [], []
    for ds in ds_list:
        for label, grp in enumerate(["P5", "P15"]):
            tri = find_triplets(DATASETS[ds][grp])
            for k in sorted(tri):
                X.append(np.stack([fn(tri[k][t]) for t in TPS]))
                y.append(label)
                meta.append({"dataset": ds, "group": grp, "plate": k[0], "pos": k[1],
                             "plate_id": f"{ds}_{grp}_p{k[0]}"})
            print(f"  {ds} {grp} ({kind}): {len(tri)}")
    X = np.stack(X); y = np.array(y, dtype=np.int64)
    tag = "_".join(ds_list)
    np.savez_compressed(os.path.join(WORK, f"cache_{tag}_{kind}.npz"), X=X, y=y)
    json.dump(meta, open(os.path.join(WORK, f"cache_{tag}_{kind}_meta.json"),
                         "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  -> cache_{tag}_{kind}.npz  X={X.shape}  y={np.bincount(y)}\n")


if __name__ == "__main__":
    for kind in ["raw", "mask"]:
        for ds_list in [["NEW"], ["OLD"]]:
            build(ds_list, kind)
