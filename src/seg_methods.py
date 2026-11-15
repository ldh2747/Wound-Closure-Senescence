# -*- coding: utf-8 -*-
"""분할 방법"""
import numpy as np
import cv2

W, H = 648, 486
ENV_W, ENV_H = 421, 81

# 공통 유틸
def envelope(x, w=ENV_W, h=ENV_H, sigma=40):
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
    return cv2.GaussianBlur(cv2.dilate(x, ker), (0, 0), sigma)


def row_band(mask_like, frac, depth, smooth=13):
    r = cv2.GaussianBlur(mask_like, (0, 0), smooth)
    out = np.zeros((H, W), np.uint8)
    for y in range(H):
        p = r[y]
        hi = np.percentile(p, 90)
        j = int(np.argmin(p))
        if p[j] > hi * depth or p[j] >= hi * frac:
            continue
        thr = hi * frac
        a = j
        while a > 0 and p[a - 1] < thr:
            a -= 1
        b = j
        while b < W - 1 and p[b + 1] < thr:
            b += 1
        out[y, a:b + 1] = 1
    return out


# M1
def m1_texture(g, frac_low=0.60, frac_gap=0.38, depth=0.78):
    hp = g - cv2.GaussianBlur(g, (0, 0), 6)
    m = cv2.blur(hp, (121, 121))
    t = np.sqrt(np.maximum(cv2.blur(hp * hp, (121, 121)) - m * m, 0))
    t = cv2.resize(t, (W, H), interpolation=cv2.INTER_AREA)
    r = t / (envelope(t) + 1e-6)
    return row_band(r, frac_gap, depth), row_band(r, frac_low, depth)


# M2
def ridge_response(g, scales=(2.0, 3.0, 4.5, 6.5)):
    best = None
    for s in scales:
        gs = cv2.GaussianBlur(g, (0, 0), s)
        gxx = cv2.Sobel(gs, cv2.CV_32F, 2, 0, ksize=5)
        gyy = cv2.Sobel(gs, cv2.CV_32F, 0, 2, ksize=5)
        gxy = cv2.Sobel(gs, cv2.CV_32F, 1, 1, ksize=5)
        tmp = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4 * gxy ** 2, 0))
        l1 = 0.5 * (gxx + gyy + tmp)
        l2 = 0.5 * (gxx + gyy - tmp)
        big = np.where(np.abs(l1) >= np.abs(l2), l1, l2)
        sml = np.where(np.abs(l1) >= np.abs(l2), l2, l1)
        rb = np.abs(big) * s ** 2
        aniso = 1.0 - np.abs(sml) / (np.abs(big) + 1e-6)
        resp = rb * np.clip(aniso, 0, 1)
        best = resp if best is None else np.maximum(best, resp)
    return best


def m2_ridge(g, frac_low=0.55, frac_gap=0.30, depth=0.80, win=61):
    resp = ridge_response(g)
    dens = cv2.blur(resp, (win, win))
    dens = cv2.resize(dens, (W, H), interpolation=cv2.INTER_AREA)
    r = dens / (envelope(dens) + 1e-6)
    return row_band(r, frac_gap, depth), row_band(r, frac_low, depth)
