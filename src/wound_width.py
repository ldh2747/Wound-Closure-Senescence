# -*- coding: utf-8 -*-
"""행 단위 상처 폭 측정"""
import os, re, sys
import numpy as np, cv2, tifffile, pandas as pd

from config import BASE
WORK = os.path.join(BASE, "work"); VIZ = os.path.join(WORK, "viz")
ADD = os.path.join(BASE, "20261013,4 AI hDF P5,15 migration(additional_data)")
DATASETS = {
    "NEW": {"P5":  os.path.join(ADD, "20261013 AI hDF P5 migration"),
            "P15": os.path.join(ADD, "20261014 AI hdf P15 migrtion")},
    "OLD": {"P5":  os.path.join(BASE, "p5"),
            "P15": os.path.join(BASE, "p15/26.09.29 hdf p 15 migration")},
}
TPS = ["0h", "24h", "48h"]
W, H = 648, 486
FRAC, DEPTH, SMOOTH = 0.60, 0.78, 13
ENV_W, ENV_H = 421, 81


def norm_pos(s):
    return int(re.findall(r"(\d+)", s)[-1])


def find_triplets(root):
    found = {}
    for tp in TPS:
        td = os.path.join(root, tp)
        for d in sorted(os.listdir(td)):
            dp = os.path.join(td, d)
            if not os.path.isdir(dp):
                continue
            plate = re.search(r"(\d+)", d).group(1)
            for f in sorted(os.listdir(dp)):
                if f.lower().endswith(".tif"):
                    found.setdefault((plate, norm_pos(os.path.splitext(f)[0])),
                                     {})[tp] = os.path.join(dp, f)
    return {k: v for k, v in found.items() if len(v) == 3}


def texture(path):
    g = tifffile.imread(path)
    g = g.mean(2).astype(np.float32) if g.ndim == 3 else g.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 6)
    m = cv2.blur(hp, (121, 121))
    t = np.sqrt(np.maximum(cv2.blur(hp * hp, (121, 121)) - m * m, 0))
    t = cv2.resize(t, (W, H), interpolation=cv2.INTER_AREA)

    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (ENV_W, ENV_H))
    env = cv2.GaussianBlur(cv2.dilate(t, ker), (0, 0), 40)
    return t / (env + 1e-6)


def row_wound(t):
    ts = cv2.GaussianBlur(t, (0, 0), SMOOTH)
    xL = np.full(H, np.nan); xR = np.full(H, np.nan); wd = np.zeros(H)
    for y in range(H):
        p = ts[y]
        hi = np.percentile(p, 90)
        j = int(np.argmin(p))
        if p[j] > hi * DEPTH:
            wd[y] = 0.0
            continue
        thr = hi * FRAC
        if p[j] >= thr:
            wd[y] = 0.0
            continue
        a = j
        while a > 0 and p[a - 1] < thr:
            a -= 1
        b = j
        while b < W - 1 and p[b + 1] < thr:
            b += 1
        xL[y], xR[y], wd[y] = a, b, (b - a + 1)
    return xL, xR, wd


def measure(paths):
    out = {}
    for tp in TPS:
        t = texture(paths[tp])
        xL, xR, wd = row_wound(t)
        out[tp] = dict(xL=xL, xR=xR, wd=wd,
                       width=float(np.mean(wd)),
                       frac=float(np.mean(wd) / W),
                       open_rows=float(np.mean(wd > 0)),
                       roughL=float(np.nanstd(xL)) if np.any(~np.isnan(xL)) else np.nan,
                       roughR=float(np.nanstd(xR)) if np.any(~np.isnan(xR)) else np.nan)
    return out


def qc(ds="NEW", n=3):
    from PIL import Image
    rows = []
    for grp, root in DATASETS[ds].items():
        tri = find_triplets(root)
        for k in sorted(tri)[:n]:
            res = measure(tri[k])
            panel = []
            for tp in TPS:
                g = tifffile.imread(tri[k][tp]).mean(2)
                g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
                d = np.clip(g - cv2.GaussianBlur(g, (0, 0), 40) + 128, 0, 255).astype(np.uint8)
                rgb = np.stack([cv2.createCLAHE(2.5, (8, 8)).apply(d)] * 3, -1)
                xL, xR = res[tp]["xL"], res[tp]["xR"]
                for y in range(H):
                    if not np.isnan(xL[y]):
                        a, b = int(xL[y]), int(xR[y])
                        rgb[y, a:b+1] = (0.6 * rgb[y, a:b+1] +
                                         0.4 * np.array([255, 60, 60])).astype(np.uint8)
                        rgb[y, max(a-1,0):a+2] = [255, 255, 0]
                        rgb[y, max(b-1,0):b+2] = [255, 255, 0]
                sm = cv2.resize(rgb, (W // 2, H // 2))
                cv2.putText(sm, f"{grp} p{k[0]} pos{k[1]} {tp}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
                cv2.putText(sm, f"width={res[tp]['width']:.0f}px "
                                f"({res[tp]['frac']*100:.0f}%) open={res[tp]['open_rows']*100:.0f}%",
                            (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
                panel.append(sm)
            rows.append(np.hstack(panel))
    sheet = np.vstack(rows)
    p = os.path.join(VIZ, f"qc_width_{ds}.png")
    Image.fromarray(sheet).save(p)
    print("saved", p, sheet.shape)
    print("노란선 = 검출된 두 전선, 빨강 = 상처 구간. 0h 넓고 48h 좁아야 정상.")


def main():
    recs = []
    for ds, roots in DATASETS.items():
        for grp, root in roots.items():
            tri = find_triplets(root)
            for i, k in enumerate(sorted(tri)):
                r = measure(tri[k])
                w0, w24, w48 = (r[t]["width"] for t in TPS)
                recs.append(dict(
                    dataset=ds, group=grp, plate=k[0], pos=k[1],
                    plate_id=f"{ds}_{grp}_p{k[0]}",
                    width_0h=w0, width_24h=w24, width_48h=w48,
                    frac_0h=r["0h"]["frac"], frac_24h=r["24h"]["frac"],
                    frac_48h=r["48h"]["frac"],
                    open_0h=r["0h"]["open_rows"], open_24h=r["24h"]["open_rows"],
                    open_48h=r["48h"]["open_rows"],
                    closure_24h=(w0 - w24) / w0 if w0 > 1 else np.nan,
                    closure_48h=(w0 - w48) / w0 if w0 > 1 else np.nan,
                    rate_1=(w0 - w24) / 24.0, rate_2=(w24 - w48) / 24.0,
                    roughL_0h=r["0h"]["roughL"], roughR_0h=r["0h"]["roughR"]))
                if (i + 1) % 25 == 0:
                    print(f"  {ds} {grp}: {i+1}/{len(tri)}")
            print(f"{ds} {grp} 완료 ({len(tri)})")
    df = pd.DataFrame(recs)
    df["decel"] = df.rate_2 / df.rate_1.replace(0, np.nan)
    df.to_csv(os.path.join(WORK, "wound_width.csv"), index=False, encoding="utf-8-sig")
    print(f"\n저장: wound_width.csv ({len(df)} rows)\n")
    g = df.groupby(["dataset", "group"])[
        ["width_0h", "width_24h", "width_48h", "closure_24h", "closure_48h"]]
    print(g.mean().round(3))


if __name__ == "__main__":
    qc() if "--qc" in sys.argv else main()
