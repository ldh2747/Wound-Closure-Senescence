# -*- coding: utf-8 -*-
"""전처리 캐시 생성"""
import os, re, json
import numpy as np
import tifffile, cv2

from config import BASE
OUT  = os.path.join(BASE, "work")
ROOTS = {
    "P5":  os.path.join(BASE, "p5"),
    "P15": os.path.join(BASE, "p15/26.09.29 hdf p 15 migration"),
}
TPS = ["0h", "24h", "48h"]
W, H = 512, 384


def list_positions(root):
    found = {}
    for tp in TPS:
        for d in os.scandir(os.path.join(root, tp)):
            if not d.is_dir():
                continue
            plate = re.match(r"(\d+)", d.name).group(1)
            for f in os.scandir(d.path):
                if f.name.lower().endswith(".tif"):
                    pos = int(os.path.splitext(f.name)[0])
                    found.setdefault((plate, pos), {})[tp] = f.path
    return {k: v for k, v in found.items() if len(v) == 3}


def preprocess(path):
    a = tifffile.imread(path)
    g = a.mean(2).astype(np.float32) if a.ndim == 3 else a.astype(np.float32)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(g, (0, 0), 30)
    f = np.clip(g - bg + 128, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(f)


def main():
    X, y, meta = [], [], []
    for gi, (grp, root) in enumerate(ROOTS.items()):
        pos_map = list_positions(root)
        for (plate, pos) in sorted(pos_map, key=lambda k: (k[0], k[1])):
            paths = pos_map[(plate, pos)]
            X.append(np.stack([preprocess(paths[t]) for t in TPS]))
            y.append(gi)
            meta.append({"group": grp, "plate": plate, "pos": pos,
                         "plate_id": f"{grp}_p{plate}"})
        print(f"{grp}: {len(pos_map)} triplets")

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "cache_512.npz"), X=X, y=y)
    with open(os.path.join(OUT, "cache_meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=1)
    print("X:", X.shape, X.dtype, " y:", np.bincount(y))
    print("saved ->", os.path.join(OUT, "cache_512.npz"))


if __name__ == "__main__":
    main()
