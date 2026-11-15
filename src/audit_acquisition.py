# -*- coding: utf-8 -*-
"""촬영조건 감사"""

import os
import re
import numpy as np
import cv2
import tifffile
import pandas as pd

from config import BASE
WORK = os.path.join(BASE, "work")
ROOTS = {
    "P5": os.path.join(BASE, "p5"),
    "P15": os.path.join(BASE, "p15/26.09.29 hdf p 15 migration"),
}
TPS = ["0h", "24h", "48h"]


def stats_of(path):
    a = tifffile.imread(path)
    g = a.mean(2).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 6)
    m = cv2.blur(hp, (121, 121))
    tex = np.sqrt(np.maximum(cv2.blur(hp * hp, (121, 121)) - m * m, 0))
    small = cv2.resize(tex, (648, 486), interpolation=cv2.INTER_AREA)
    return dict(
        bright=float(g.mean()),
        gstd=float(g.std()),
        blue=float(a[..., 2].mean() - a[..., 1].mean()),
        tex_p50=float(np.percentile(small, 50)),
        tex_p90=float(np.percentile(small, 90)),
        tex_p10=float(np.percentile(small, 10)),
        focus=float(cv2.Laplacian(cv2.resize(g, (648, 486)), cv2.CV_32F).var()),
    )


rows = []
for grp, root in ROOTS.items():
    for tp in TPS:
        for d in os.scandir(os.path.join(root, tp)):
            if not d.is_dir():
                continue
            plate = re.match(r"(\d+)", d.name).group(1)
            for f in os.scandir(d.path):
                if f.name.lower().endswith(".tif"):
                    r = stats_of(f.path)
                    r.update(
                        group=grp,
                        tp=tp,
                        plate=plate,
                        pos=int(os.path.splitext(f.name)[0]),
                    )
                    rows.append(r)
        print(
            f"  {grp} {tp} 완료 ({sum(1 for x in rows if x['group']==grp and x['tp']==tp)}장)"
        )

df = pd.DataFrame(rows)
df.to_csv(
    os.path.join(WORK, "acquisition_audit.csv"), index=False, encoding="utf-8-sig"
)

keys = ["bright", "gstd", "blue", "tex_p10", "tex_p50", "tex_p90", "focus"]
print(f"\n{'='*86}")
print(" 그룹 x 시점 평균 (괄호는 표준편차)")
print(f"{'='*86}")
hdr = f"{'':10s}" + "".join(f"{k:>12s}" for k in keys)
print(hdr)
for grp in ["P5", "P15"]:
    for tp in TPS:
        s = df[(df.group == grp) & (df.tp == tp)]
        line = f"{grp+' '+tp:10s}"
        for k in keys:
            line += f"{s[k].mean():8.2f}±{s[k].std():<3.1f}"
        print(line)

print(f"\n{'='*86}")
print(" 시점 내 P5 vs P15 분리도 (단일 지표 AUC, 0.5=구분불가 1.0=완전분리)")
print(f"{'='*86}")
from scipy import stats as st

print(f"{'timepoint':12s}" + "".join(f"{k:>12s}" for k in keys))
for tp in TPS:
    line = f"{tp:12s}"
    for k in keys:
        a = df[(df.group == "P5") & (df.tp == tp)][k].values
        b = df[(df.group == "P15") & (df.tp == tp)][k].values
        u = st.mannwhitneyu(a, b).statistic / (len(a) * len(b))
        line += f"{max(u, 1-u):12.3f}"
    print(line)

print(f"\n저장: {os.path.join(WORK, 'acquisition_audit.csv')}")
