# -*- coding: utf-8 -*-
"""
===============================================================================
 분할 방법 3종 — 공통 인터페이스
===============================================================================
각 방법은 이미지 한 장에서 두 개의 이진 맵을 낸다.

  gap      세포가 하나도 없는 영역        (아직 아무도 도달 못 한 틈)
  lowdens  세포는 있으나 주변 단층보다 성긴 영역

정의상 gap ⊆ lowdens.

M1  texture/envelope 비율  (현재 방법)
M2  ridge 필터 — Hessian 고유값으로 가늘고 긴 세포체를 직접 검출
M3  Cellpose — 학습 기반 세포 분할

파라미터는 사전등록에 따라 0h 이미지에서만 조정한다.
===============================================================================
"""
import os
import numpy as np
import cv2

W, H = 648, 486
ENV_W, ENV_H = 421, 81

# ---------------------------------------------------------------- 공통 유틸
def to_gray_full(path):
    import tifffile
    a = tifffile.imread(path)
    return a.mean(2).astype(np.float32) if a.ndim == 3 else a.astype(np.float32)


def envelope(x, w=ENV_W, h=ENV_H, sigma=40):
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
    return cv2.GaussianBlur(cv2.dilate(x, ker), (0, 0), sigma)


def row_band(mask_like, frac, depth, smooth=13):
    """행마다 가장 낮은 골 하나를 찾아 연속 구간만 취한다 (세로 스크래치 기하)."""
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


# ================================================================ M1
def m1_texture(g, frac_low=0.60, frac_gap=0.38, depth=0.78):
    """현재 방법. 비율 임계값 두 개로 lowdens 와 gap 을 나눈다."""
    hp = g - cv2.GaussianBlur(g, (0, 0), 6)
    m = cv2.blur(hp, (121, 121))
    t = np.sqrt(np.maximum(cv2.blur(hp * hp, (121, 121)) - m * m, 0))
    t = cv2.resize(t, (W, H), interpolation=cv2.INTER_AREA)
    r = t / (envelope(t) + 1e-6)
    return row_band(r, frac_gap, depth), row_band(r, frac_low, depth)


# ================================================================ M2
def ridge_response(g, scales=(2.0, 3.0, 4.5, 6.5)):
    """Hessian 기반 ridge 검출. 세포체처럼 가늘고 긴 구조에 크게 반응한다.

    빈 영역에는 구조 자체가 없으므로 응답이 0 에 가깝고, 성기게 퍼진 세포도
    개별 구조가 남아 있어 응답이 살아 있다. 텍스처 '에너지' 와 달리
    밀도가 낮아도 구조가 있으면 검출된다.
    """
    best = None
    for s in scales:
        gs = cv2.GaussianBlur(g, (0, 0), s)
        gxx = cv2.Sobel(gs, cv2.CV_32F, 2, 0, ksize=5)
        gyy = cv2.Sobel(gs, cv2.CV_32F, 0, 2, ksize=5)
        gxy = cv2.Sobel(gs, cv2.CV_32F, 1, 1, ksize=5)
        tmp = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4 * gxy ** 2, 0))
        l1 = 0.5 * (gxx + gyy + tmp)
        l2 = 0.5 * (gxx + gyy - tmp)
        big = np.where(np.abs(l1) >= np.abs(l2), l1, l2)      # 큰 쪽 고유값
        sml = np.where(np.abs(l1) >= np.abs(l2), l2, l1)
        # ridge: |big| 크고 |sml| 작음  -> 선형 구조
        rb = np.abs(big) * s ** 2
        aniso = 1.0 - np.abs(sml) / (np.abs(big) + 1e-6)
        resp = rb * np.clip(aniso, 0, 1)
        best = resp if best is None else np.maximum(best, resp)
    return best


def m2_ridge(g, frac_low=0.55, frac_gap=0.30, depth=0.80, win=61):
    resp = ridge_response(g)
    # 구조 '존재 밀도' 로 집계: 응답을 윈도우 평균
    dens = cv2.blur(resp, (win, win))
    dens = cv2.resize(dens, (W, H), interpolation=cv2.INTER_AREA)
    r = dens / (envelope(dens) + 1e-6)
    return row_band(r, frac_gap, depth), row_band(r, frac_low, depth)


# ================================================================ M3
_CP = None


def _cellpose():
    global _CP
    if _CP is None:
        from cellpose import models
        import torch
        _CP = models.CellposeModel(gpu=torch.cuda.is_available())
    return _CP


def m3_cellpose(g, diameter=38, frac_low=0.55, frac_gap=0.30, depth=0.80,
                win=81):
    """Cellpose 로 세포를 분할한 뒤, 세포 중심 밀도로 gap / lowdens 를 나눈다."""
    small = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    sm = small - cv2.GaussianBlur(small, (0, 0), 40)
    sm = np.clip(sm + 128, 0, 255).astype(np.uint8)
    sm = cv2.createCLAHE(2.5, (8, 8)).apply(sm)

    masks, _, _ = _cellpose().eval(sm, diameter=diameter,
                                   flow_threshold=0.5, cellprob_threshold=-1.0)
    occ = (masks > 0).astype(np.float32)
    dens = cv2.blur(occ, (win, win))
    r = dens / (envelope(dens) + 1e-6)
    return row_band(r, frac_gap, depth), row_band(r, frac_low, depth), masks


METHODS = {"M1": m1_texture, "M2": m2_ridge, "M3": m3_cellpose}
