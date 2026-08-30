# -*- coding: utf-8 -*-
"""
Crop enhancement before Recognition.

Hezar and the local CRNN both degrade on low-contrast / noisy patches.
We do not binarize by default (that destroys trained appearance); we
amplify local contrast and suppress speckle, then optionally retry with
adaptive threshold when the first decode looks like collapsed garbage.
"""

from collections import Counter

import cv2
import numpy as np


def looks_garbage(text):
    """Repeating one letter (لالالا / ششش) is a classic CTC/Hezar failure mode."""
    compact = "".join(ch for ch in (text or "") if not ch.isspace())
    if len(compact) < 6:
        return False
    counts = Counter(compact)
    top = counts.most_common(1)[0][1]
    return (top / len(compact)) >= 0.42


def enhance_crop(bgr):
    """CLAHE + bilateral denoise + mild sharpen. Keeps 3 channels."""
    if bgr is None or bgr.size == 0:
        return bgr
    img = bgr
    h, w = img.shape[:2]
    if h < 36:
        scale = 36.0 / max(h, 1)
        img = cv2.resize(img, (max(8, int(w * scale)), 36), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.bilateralFilter(img, 5, 40, 40)
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.35, blur, -0.35, 0)
    return img


def binarize_crop(bgr):
    """Fallback for very noisy backgrounds: adaptive threshold as a 3-channel image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    # Prefer dark text on light paper; invert if the crop is mostly black.
    if bw.mean() < 127:
        bw = 255 - bw
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
