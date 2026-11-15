# -*- coding: utf-8 -*-
"""
새 데이터 wound segmentation (+ QC).

[설계 — 구 데이터 실패에서 얻은 교훈]
 1) 텍스처는 '원본 해상도' 에서 계산한다. 축소 후 계산하면 섬유아세포의
    가는 결이 평균화되어 사라진다.
 2) 조명 기울기를 먼저 제거한다.
 3) 임계값은 '같은 위치의 0h' 에서 Otsu 로 구해 24h/48h 에 그대로 쓴다.
    시점마다 Otsu 를 돌리면, 상처가 거의 닫힌 이미지에서 멀쩡한 세포 영역을
    억지로 둘로 가른다. (새 데이터는 시점 간 광학 조건이 일정해 성립)

사용:
  python seg_new.py --qc     QC 오버레이만
  python seg_new.py          전체 정량 -> closure_new.csv
"""
import os, re, sys
import numpy as np, cv2, tifffile, pandas as pd

from config import BASE
WORK = os.path.join(BASE, "work"); VIZ = os.path.join(WORK, "viz")
ADD = os.path.join(BASE, "20261013,4 AI hDF P5,15 migration(additional_data)")
ROOTS = {"P5": os.path.join(ADD, "20261013 AI hDF P5 migration"),
         "P15": os.path.join(ADD, "20261014 AI hdf P15 migrtion")}
TPS = ["0h", "24h", "48h"]
OUT_W, OUT_H = 648, 486          # 마스크 해상도
HP_SIGMA, WIN = 6, 121           # 원본 해상도 텍스처 파라미터


def norm_pos(stem):
    return int(re.findall(r"(\d+)", stem)[-1])


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
    """원본 해상도 고주파 -> 국소 표준편차 -> 축소 -> 조명 기울기 제거"""
    g = tifffile.imread(path)
    g = g.mean(2).astype(np.float32) if g.ndim == 3 else g.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), HP_SIGMA)
    m = cv2.blur(hp, (WIN, WIN))
    t = np.sqrt(np.maximum(cv2.blur(hp * hp, (WIN, WIN)) - m * m, 0))
    t = cv2.resize(t, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    return t / (cv2.GaussianBlur(t, (0, 0), 150) + 1e-6)   # 상대 텍스처


def otsu_thr(t):
    lo, hi = np.percentile(t, 0.5), np.percentile(t, 99.5)
    u8 = (np.clip((t - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return lo + thr / 255.0 * (hi - lo)


def wound_mask(t, thr):
    m = (t < thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stt, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stt[i, cv2.CC_STAT_AREA] >= 0.005 * m.size:
            keep[lab == i] = 1
    return keep


def measure(paths):
    tex = {t: texture(paths[t]) for t in TPS}
    thr = otsu_thr(tex["0h"])
    return {t: wound_mask(tex[t], thr) for t in TPS}, thr


def qc(n=4):
    from PIL import Image
    rows = []
    for grp, root in ROOTS.items():
        tri = find_triplets(root)
        for k in sorted(tri)[:n]:
            masks, _ = measure(tri[k])
            panel = []
            for t in TPS:
                g = tifffile.imread(tri[k][t]).mean(2)
                g = cv2.resize(g, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
                d = np.clip(g - cv2.GaussianBlur(g, (0, 0), 30) + 128, 0, 255).astype(np.uint8)
                d = cv2.createCLAHE(2.5, (8, 8)).apply(d)
                rgb = np.stack([d] * 3, -1)
                s = masks[t].astype(bool)
                rgb[s] = (0.55 * rgb[s] + 0.45 * np.array([255, 60, 60])).astype(np.uint8)
                rgb = cv2.resize(rgb, (OUT_W // 2, OUT_H // 2))
                cv2.putText(rgb, f"{grp} p{k[0]} pos{k[1]} {t} wound={masks[t].mean()*100:.0f}%",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
                panel.append(rgb)
            rows.append(np.hstack(panel))
    sheet = np.vstack(rows)
    p = os.path.join(VIZ, "qc_seg_new.png")
    Image.fromarray(sheet).save(p)
    print("saved", p, sheet.shape)
    print("빨강 = wound. 0h 에 넓고 48h 에 좁아야 정상.")


def main():
    recs = []
    for grp, root in ROOTS.items():
        tri = find_triplets(root)
        for i, k in enumerate(sorted(tri)):
            masks, thr = measure(tri[k])
            half = OUT_W // 2
            for side, sl in [("L", slice(0, half)), ("R", slice(half, OUT_W))]:
                a = {t: float(masks[t][:, sl].mean()) for t in TPS}
                recs.append(dict(
                    group=grp, plate=k[0], pos=k[1], half=side,
                    plate_id=f"{grp}_p{k[0]}", thr=thr,
                    area_0h=a["0h"], area_24h=a["24h"], area_48h=a["48h"],
                    closure_24h=(a["0h"]-a["24h"])/a["0h"] if a["0h"] > 1e-6 else np.nan,
                    closure_48h=(a["0h"]-a["48h"])/a["0h"] if a["0h"] > 1e-6 else np.nan,
                    rate_1=(a["0h"]-a["24h"])/24.0, rate_2=(a["24h"]-a["48h"])/24.0))
            if (i+1) % 20 == 0: print(f"  {grp}: {i+1}/{len(tri)}")
        print(f"{grp} 완료 ({len(tri)} triplets)")
    df = pd.DataFrame(recs)
    df["decel"] = df.rate_2 / df.rate_1.replace(0, np.nan)
    df.to_csv(os.path.join(WORK, "closure_new.csv"), index=False, encoding="utf-8-sig")
    print(f"\n저장: closure_new.csv ({len(df)} rows)")
    print(df.groupby("group")[["area_0h","area_24h","area_48h",
                               "closure_24h","closure_48h"]].mean().round(3))


if __name__ == "__main__":
    qc() if "--qc" in sys.argv else main()
