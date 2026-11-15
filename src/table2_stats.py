# -*- coding: utf-8 -*-
"""Table 2 statistics"""
import csv, math, os, statistics as S, sys

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("DATA_ROOT", "."), "work", "wound_width.csv")
rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
EXP = {"Experiment 1": "OLD", "Experiment 2": "NEW"}


def betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-30 else 1e-30); h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c; c = c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c; c = c if abs(c) > 1e-30 else 1e-30
        de = d * c; h *= de
        if abs(de - 1) < 1e-12:
            break
    return h


def t_p(t, df):
    x = df / (df + t * t)
    bt = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - math.lgamma(0.5) + (df / 2) * math.log(x) + 0.5 * math.log(1 - x))
    return bt * betacf(df / 2, 0.5, x) / (df / 2) if x < (df / 2 + 1) / (df / 2 + 1.5) else 1 - bt * betacf(0.5, df / 2, 1 - x) / 0.5


def ttest(a, b):
    na, nb = len(a), len(b)
    sp = ((na - 1) * S.variance(a) + (nb - 1) * S.variance(b)) / (na + nb - 2)
    t = (S.mean(a) - S.mean(b)) / math.sqrt(sp * (1 / na + 1 / nb))
    return t_p(t, na + nb - 2)


def ranks(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); rk = [0.0] * len(v); i = 0; ties = []
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[s[j + 1]] == v[s[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[s[k]] = (i + j) / 2 + 1
        ties.append(j - i + 1); i = j + 1
    return rk, ties


def mwu(a, b):
    n1, n2 = len(a), len(b); N = n1 + n2
    rk, ties = ranks(a + b)
    u = sum(rk[:n1]) - n1 * (n1 + 1) / 2
    sg = math.sqrt(n1 * n2 / 12 * ((N + 1) - sum(t ** 3 - t for t in ties) / (N * (N - 1))))
    z = (abs(u - n1 * n2 / 2) - 0.5) / sg
    auc = u / (n1 * n2)
    return max(auc, 1 - auc), math.erfc(z / math.sqrt(2))


def solve(A, b):
    n = len(A); M = [r[:] + [b[i]] for i, r in enumerate(A)]
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(M[k][i])); M[i], M[p] = M[p], M[i]
        for k in range(i + 1, n):
            c = M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] -= c * M[i][j]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def ols(X, y):
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    return solve(A, [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(k)])


def spearman(a, b):
    ra, rb = ranks(a)[0], ranks(b)[0]; ma, mb = S.mean(ra), S.mean(rb)
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


for name, ds in EXP.items():
    s = [r for r in rows if r["dataset"] == ds]
    w = {g: [float(r["width_0h"]) for r in s if r["group"] == g] for g in ("P5", "P15")}
    c = {g: [float(r["closure_48h"]) for r in s if r["group"] == g] for g in ("P5", "P15")}
    auc, p = mwu(w["P5"], w["P15"])
    pw, pc = {}, {}
    for r in s:
        pw.setdefault((r["group"], r["plate"]), []).append(float(r["width_0h"]))
        pc.setdefault((r["group"], r["plate"]), []).append(float(r["closure_48h"]))
    m = lambda d, g: [S.mean(v) for k, v in sorted(d.items()) if k[0] == g]
    print("== %s (P5 n=%d, P15 n=%d; plates %d/%d)" % (name, len(w["P5"]), len(w["P15"]), len(m(pw, "P5")), len(m(pw, "P15"))))
    print("initial width  set mean P5 %.1f +- %.1f, P15 %.1f +- %.1f | AUC %.3f P %.3g" % (S.mean(w["P5"]), S.stdev(w["P5"]), S.mean(w["P15"]), S.stdev(w["P15"]), auc, p))
    print("               plate mean P5 %.1f / P15 %.1f | plate-level P %.2g" % (S.mean(m(pw, "P5")), S.mean(m(pw, "P15")), ttest(m(pw, "P5"), m(pw, "P15"))))
    print("48 h closure   plate means P5 %s, P15 %s | plate-level P %.2g" % ([round(v, 3) for v in m(pc, "P5")], [round(v, 3) for v in m(pc, "P15")], ttest(m(pc, "P5"), m(pc, "P15"))))
    ca, cb = c["P5"], c["P15"]
    X = [[1, 1.0 if r["group"] == "P5" else 0.0, float(r["width_0h"])] for r in s]
    y = [float(r["closure_48h"]) for r in s]
    print("P5-P15 diff    set unadjusted %.3f, adjusted for initial width %.3f, plate level %.3f" % (S.mean(ca) - S.mean(cb), ols(X, y)[1], S.mean(m(pc, "P5")) - S.mean(m(pc, "P15"))))
    print("Spearman rho (initial width vs 48 h closure) %.2f ; set-level closure AUC %.3f" % (spearman([x[2] for x in X], y), mwu(ca, cb)[0]))
