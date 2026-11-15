# -*- coding: utf-8 -*-
"""
행 단위(row-wise) 상처 마스크 캐시 — wound_width.py 와 동일한 로직.

기존 cache_*_mask.npz 는 2D 임계값만 써서 세포 영역까지 얼룩덜룩 잡혔다.
여기서는 '각 행에 상처는 하나의 연속 구간' 이라는 기하 제약을 적용해
깨끗한 띠 모양 마스크를 만든다.

출력: cache_{ds}_mask2.npz
"""
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
SW, SH = 648, 486            # 분할 해상도
W, H = 512, 384              # 캐시 해상도
ENV_W, ENV_H = 421, 81
FRAC, DEPTH, SMOOTH = 0.60, 0.78, 13


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


def row_mask(path):
    g = tifffile.imread(path)
    g = g.mean(2).astype(np.float32) if g.ndim == 3 else g.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 6)
    m = cv2.blur(hp, (121, 121))
    t = np.sqrt(np.maximum(cv2.blur(hp * hp, (121, 121)) - m * m, 0))
    t = cv2.resize(t, (SW, SH), interpolation=cv2.INTER_AREA)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (ENV_W, ENV_H))
    r = cv2.GaussianBlur(t / (cv2.GaussianBlur(cv2.dilate(t, ker), (0, 0), 40) + 1e-6),
                         (0, 0), SMOOTH)
    mk = np.zeros((SH, SW), np.uint8)
    for yy in range(SH):
        p = r[yy]
        hi = np.percentile(p, 90)
        j = int(np.argmin(p))
        if p[j] > hi * DEPTH or p[j] >= hi * FRAC:
            continue
        thr = hi * FRAC
        a = j
        while a > 0 and p[a - 1] < thr: a -= 1
        b = j
        while b < SW - 1 and p[b + 1] < thr: b += 1
        mk[yy, a:b + 1] = 255
    return cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)


for ds in ["NEW", "OLD"]:
    X, y, meta = [], [], []
    for label, grp in enumerate(["P5", "P15"]):
        tri = find_triplets(DATASETS[ds][grp])
        for k in sorted(tri):
            X.append(np.stack([row_mask(tri[k][t]) for t in TPS]))
            y.append(label)
            meta.append({"dataset": ds, "group": grp, "plate": k[0], "pos": k[1],
                         "plate_id": f"{ds}_{grp}_p{k[0]}"})
        print(f"  {ds} {grp}: {len(tri)}")
    X = np.stack(X); y = np.array(y, dtype=np.int64)
    np.savez_compressed(os.path.join(WORK, f"cache_{ds}_mask2.npz"), X=X, y=y)
    json.dump(meta, open(os.path.join(WORK, f"cache_{ds}_mask2_meta.json"),
                         "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  -> cache_{ds}_mask2.npz  X={X.shape}  y={np.bincount(y)}\n")
