# -*- coding: utf-8 -*-
"""
===============================================================================
 블러가 상처 폐쇄율 측정을 얼마나 왜곡하는가
===============================================================================
리뷰어 질문: "흐린 쪽에서 세포가 덜 검출되어 P15 가 덜 닫힌 것처럼 보이는 것
             아닌가?"

구 데이터를 보정하는 대신, 정답을 아는 깨끗한 새 데이터에 구 데이터와 같은
아티팩트를 인위적으로 만들어 측정값이 어떻게 변하는지 직접 잰다.

  A  새 데이터 원본                        기준값
  B  새 P15 의 24h/48h 만 블러             차이가 부풀려지나?
  C  새 P5  의 24h/48h 만 블러  ★핵심      차이가 줄거나 뒤집히나?
  D  구 데이터 전체를 공통 수준으로 정합    폐쇄율 차이가 남나?

C 가 핵심이다. 블러가 '덜 닫힌 것처럼' 보이게 만든다면 P5 를 흐리게 했을 때
P5 의 폐쇄율이 떨어져야 하고, 그 변화량이 곧 아티팩트의 크기다.

실행:  python exp_blur_bias.py
===============================================================================
"""
import os, re, json, sys
import numpy as np, cv2, tifffile, pandas as pd
from scipy import stats as st

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
W, H = 648, 486
ENV_W, ENV_H = 421, 81
FRAC, DEPTH, SMOOTH = 0.60, 0.78, 13


# ---------------------------------------------------------------- 데이터 탐색
def norm_pos(s): return int(re.findall(r"(\d+)", s)[-1])


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


# ---------------------------------------------------------------- 선명도 / 블러
def sharpness(g):
    """미세/거친 스케일 에너지 비. 흐릴수록 작다. 밝기·대비에 불변."""
    fine = cv2.GaussianBlur(g, (0, 0), 1.4) - cv2.GaussianBlur(g, (0, 0), 2.8)
    coarse = cv2.GaussianBlur(g, (0, 0), 11) - cv2.GaussianBlur(g, (0, 0), 22)
    return float(np.abs(fine).mean() / (np.abs(coarse).mean() + 1e-8))


def blur_to(g, target, hi=16.0, iters=14):
    """sharpness(블러한 g) == target 이 되는 sigma 를 이분탐색으로 찾아 적용"""
    if sharpness(g) <= target:
        return g, 0.0
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if sharpness(cv2.GaussianBlur(g, (0, 0), mid)) > target:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    return cv2.GaussianBlur(g, (0, 0), s), s


# ---------------------------------------------------------------- 상처 폭 측정
def gray_full(path):
    a = tifffile.imread(path)
    return a.mean(2).astype(np.float32) if a.ndim == 3 else a.astype(np.float32)


def width_of(g):
    """wound_width.py 와 동일한 행 단위 측정. 입력은 원본 해상도 grayscale."""
    hp = g - cv2.GaussianBlur(g, (0, 0), 6)
    m = cv2.blur(hp, (121, 121))
    t = np.sqrt(np.maximum(cv2.blur(hp * hp, (121, 121)) - m * m, 0))
    t = cv2.resize(t, (W, H), interpolation=cv2.INTER_AREA)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (ENV_W, ENV_H))
    r = cv2.GaussianBlur(t / (cv2.GaussianBlur(cv2.dilate(t, ker), (0, 0), 40) + 1e-6),
                         (0, 0), SMOOTH)
    wd = np.zeros(H)
    for y in range(H):
        p = r[y]
        hi = np.percentile(p, 90)
        j = int(np.argmin(p))
        if p[j] > hi * DEPTH:
            continue
        thr = hi * FRAC
        if p[j] >= thr:
            continue
        a = j
        while a > 0 and p[a - 1] < thr:
            a -= 1
        b = j
        while b < W - 1 and p[b + 1] < thr:
            b += 1
        wd[y] = b - a + 1
    return float(wd.mean())


# ---------------------------------------------------------------- 목표 선명도
def mean_sharpness(ds, grp, tps, limit=None):
    tri = find_triplets(DATASETS[ds][grp])
    keys = sorted(tri)[:limit] if limit else sorted(tri)
    vals = [sharpness(gray_full(tri[k][tp])) for k in keys for tp in tps]
    return float(np.mean(vals)), float(np.std(vals))


def auc_of(a, b):
    u = st.mannwhitneyu(a, b).statistic / (len(a) * len(b))
    return max(u, 1 - u)


def main():
    print("=" * 76)
    print(" 1단계 — 목표 선명도 결정")
    print("=" * 76)
    tgt_m, tgt_s = mean_sharpness("OLD", "P15", ["24h", "48h"], limit=30)
    print(f"  구 P15 24h/48h 평균 선명도 = {tgt_m:.3f} ± {tgt_s:.3f}   <- 블러 목표")
    for ds, grp, tps in [("NEW", "P5", TPS), ("NEW", "P15", TPS),
                         ("OLD", "P5", ["24h", "48h"])]:
        m, s = mean_sharpness(ds, grp, tps, limit=30)
        print(f"  {ds} {grp:4s} {'/'.join(tps):12s} = {m:.3f} ± {s:.3f}")

    # ---------------------------------------------------------- 새 데이터 측정
    print("\n" + "=" * 76)
    print(" 2단계 — 새 데이터: 조건 A / B / C")
    print("=" * 76)
    rows = []
    for grp in ["P5", "P15"]:
        tri = find_triplets(DATASETS["NEW"][grp])
        for i, k in enumerate(sorted(tri)):
            g = {tp: gray_full(tri[k][tp]) for tp in TPS}
            # 원본 폭
            w_plain = {tp: width_of(g[tp]) for tp in TPS}
            # 24h/48h 만 블러한 폭 (해당 그룹이 블러 대상일 때만 쓰인다)
            w_blur, sig = {}, {}
            for tp in ["24h", "48h"]:
                gb, s = blur_to(g[tp], tgt_m)
                w_blur[tp] = width_of(gb)
                sig[tp] = s
            rows.append(dict(
                group=grp, plate=k[0], pos=k[1], plate_id=f"NEW_{grp}_p{k[0]}",
                w0=w_plain["0h"], w24=w_plain["24h"], w48=w_plain["48h"],
                w24b=w_blur["24h"], w48b=w_blur["48h"],
                sig24=sig["24h"], sig48=sig["48h"]))
            if (i + 1) % 20 == 0:
                print(f"    NEW {grp}: {i+1}/{len(tri)}")
        print(f"  NEW {grp} 완료 ({len(tri)})")

    nd = pd.DataFrame(rows)
    nd["c48_plain"] = (nd.w0 - nd.w48) / nd.w0
    nd["c48_blur"] = (nd.w0 - nd.w48b) / nd.w0
    nd["c24_plain"] = (nd.w0 - nd.w24) / nd.w0
    nd["c24_blur"] = (nd.w0 - nd.w24b) / nd.w0
    nd.to_csv(os.path.join(WORK, "blur_bias_new.csv"), index=False,
              encoding="utf-8-sig")

    print(f"\n  적용 블러 sigma: 24h {nd.sig24.mean():.2f} / 48h {nd.sig48.mean():.2f}")

    def closure(grp, col):
        return nd[nd.group == grp][col].values

    conds = {
        "A  원본 (기준)":        ("c48_plain", "c48_plain"),
        "B  P15 만 블러":        ("c48_plain", "c48_blur"),
        "C  P5 만 블러  ★":      ("c48_blur",  "c48_plain"),
        "(참고) 양쪽 다 블러":   ("c48_blur",  "c48_blur"),
    }
    print("\n" + "-" * 76)
    print(f"{'조건':22s}{'P5':>9s}{'P15':>9s}{'차이':>9s}{'AUC':>8s}{'기준대비':>10s}")
    print("-" * 76)
    base_diff = None
    summary = []
    for name, (c5, c15) in conds.items():
        a, b = closure("P5", c5), closure("P15", c15)
        diff = a.mean() - b.mean()
        if base_diff is None:
            base_diff = diff
        print(f"{name:22s}{a.mean():9.3f}{b.mean():9.3f}{diff:9.3f}"
              f"{auc_of(a,b):8.3f}{diff-base_diff:+10.3f}")
        summary.append(dict(cond=name, P5=round(a.mean(), 3), P15=round(b.mean(), 3),
                            diff=round(diff, 3), auc=round(auc_of(a, b), 3),
                            delta=round(diff - base_diff, 3)))

    print("\n  블러 자체가 폐쇄율을 어떻게 바꾸는가 (같은 그룹 내 전후 비교)")
    for grp in ["P5", "P15"]:
        s = nd[nd.group == grp]
        t = st.wilcoxon(s.c48_plain, s.c48_blur)
        print(f"    {grp:4s}  원본 {s.c48_plain.mean():.3f} -> 블러 후 "
              f"{s.c48_blur.mean():.3f}   변화 {s.c48_blur.mean()-s.c48_plain.mean():+.3f}"
              f"   (Wilcoxon p={t.pvalue:.2e})")

    # ---------------------------------------------------------- 구 데이터 조건 D
    print("\n" + "=" * 76)
    print(" 3단계 — 조건 D: 구 데이터 전체를 공통 선명도로 정합 후 재측정")
    print("=" * 76)
    rows = []
    for grp in ["P5", "P15"]:
        tri = find_triplets(DATASETS["OLD"][grp])
        for i, k in enumerate(sorted(tri)):
            w = {}
            for tp in TPS:
                g = gray_full(tri[k][tp])
                gb, _ = blur_to(g, tgt_m)
                w[tp] = width_of(gb)
            rows.append(dict(group=grp, plate=k[0], pos=k[1],
                             plate_id=f"OLD_{grp}_p{k[0]}",
                             w0=w["0h"], w24=w["24h"], w48=w["48h"]))
            if (i + 1) % 30 == 0:
                print(f"    OLD {grp}: {i+1}/{len(tri)}")
        print(f"  OLD {grp} 완료 ({len(tri)})")

    od = pd.DataFrame(rows)
    od["c48"] = (od.w0 - od.w48) / od.w0
    od["c24"] = (od.w0 - od.w24) / od.w0
    od.to_csv(os.path.join(WORK, "blur_bias_old.csv"), index=False,
              encoding="utf-8-sig")

    a = od[od.group == "P5"].c48.values
    b = od[od.group == "P15"].c48.values
    orig = pd.read_csv(os.path.join(WORK, "wound_width.csv"))
    o = orig[orig.dataset == "OLD"]
    oa = o[o.group == "P5"].closure_48h.dropna().values
    ob = o[o.group == "P15"].closure_48h.dropna().values
    print(f"\n{'':24s}{'P5':>9s}{'P15':>9s}{'차이':>9s}{'AUC':>8s}")
    print("-" * 60)
    print(f"{'구 데이터 원본':24s}{oa.mean():9.3f}{ob.mean():9.3f}"
          f"{oa.mean()-ob.mean():9.3f}{auc_of(oa,ob):8.3f}")
    print(f"{'구 데이터 선명도 정합 후':24s}{a.mean():9.3f}{b.mean():9.3f}"
          f"{a.mean()-b.mean():9.3f}{auc_of(a,b):8.3f}")

    json.dump({"target_sharpness": tgt_m, "new_conditions": summary,
               "old_original": dict(P5=float(oa.mean()), P15=float(ob.mean()),
                                    auc=float(auc_of(oa, ob))),
               "old_matched": dict(P5=float(a.mean()), P15=float(b.mean()),
                                   auc=float(auc_of(a, b)))},
              open(os.path.join(WORK, "blur_bias_result.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n저장: blur_bias_new.csv, blur_bias_old.csv, blur_bias_result.json")


if __name__ == "__main__":
    main()
