# -*- coding: utf-8 -*-
"""
===============================================================================
 Scratch assay 정량 — 행 단위 상처 폭 측정 (row-wise wound width)
===============================================================================
[왜 면적 임계값 대신 행 단위인가]
  텍스처 절대 레벨이 시점마다 올라간다(세포 증가 + 조명). 0h 에서 뽑은 절대
  임계값을 24h/48h 에 쓰면 깨지고, 시점마다 Otsu 를 돌리면 닫힌 이미지를
  억지로 가른다. 반면 스크래치는 '세로 한 줄' 이라는 강한 기하 제약이 있다.

[방법] 각 행 y 에 대해
  1. 가로 텍스처 프로파일 p(x) 를 구한다
  2. 그 행의 세포 수준 hi = p 의 90 백분위
  3. 가장 낮은 지점에서 양옆으로 p < hi*FRAC 인 동안 확장 -> 그 구간이 상처
  4. 깊이가 얕으면(p_min > hi*DEPTH) 닫힌 것으로 보고 폭 0

  -> 시점별 절대 레벨에 영향받지 않는다(그 행의 세포 수준으로 자동 정규화).
  -> 좌/우 경계 x_L, x_R 이 곧 두 전선 => '가로로 반' 분할과 동일한 정보.

사용:  python wound_width.py --qc      확인용 오버레이
       python wound_width.py          전체 정량 -> wound_width.csv
===============================================================================
"""
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
ENV_W, ENV_H = 421, 81      # 세포 수준 포락선 창 (상처 최대폭보다 넓게)


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
    """국소 텍스처 -> '세포 수준 포락선' 으로 나눠 조명/초점 기울기 제거.

    포락선 = 넓은 가로 창의 국소 최대값(grey dilation).
    창이 상처 폭보다 넓으므로 상처 한가운데서도 양옆 세포의 텍스처 값을
    집어온다 -> 결과적으로 '주변 세포 대비 몇 %인가' 가 된다.
    닫힌 이미지에서는 어디서나 비가 1 에 가까워 상처가 0 으로 나온다.
    """
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
    """행마다 상처 구간을 찾는다. 반환: xL, xR, width (길이 H, 닫힌 행은 NaN)"""
    ts = cv2.GaussianBlur(t, (0, 0), SMOOTH)
    xL = np.full(H, np.nan); xR = np.full(H, np.nan); wd = np.zeros(H)
    for y in range(H):
        p = ts[y]
        hi = np.percentile(p, 90)
        j = int(np.argmin(p))
        if p[j] > hi * DEPTH:          # 골이 얕다 = 닫힘
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
                       width=float(np.mean(wd)),           # 평균 폭 (px)
                       frac=float(np.mean(wd) / W),        # 화면 대비 비율
                       open_rows=float(np.mean(wd > 0)),   # 아직 열린 행 비율
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
