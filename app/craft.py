# -*- coding: utf-8 -*-
"""
CRAFT scene-text detection with the real 4-point boxes.

Hezar's default post-process converts rotated CRAFT quads to (x, y, w, h)
using only two corners. That drops rotation and often yields empty/wrong
boxes. We run the same VGG-UNet, then keep minAreaRect quadrilaterals.

Thresholds are lower than Hezar's defaults (0.7) so noisy photos still fire.
Canvas is capped at 1280px so this fits an RTX 2050 4GB.
"""

import numpy as np
import torch
from PIL import Image

from .boxes import order_tl_tr_br_bl, unclip_quad


def _quads_from_logits(logits, ratio, text_th, link_th, low_text, orig_w, orig_h, unclip):
    from hezar.models.text_detection.craft.craft_utils import (
        adjust_result_coordinates,
        get_detection_boxes,
    )
    score_text = logits[:, :, 0]
    score_link = logits[:, :, 1]
    if hasattr(score_text, "cpu"):
        score_text = score_text.detach().cpu().numpy()
        score_link = score_link.detach().cpu().numpy()
    boxes, _polys, _ = get_detection_boxes(
        score_text, score_link, text_th, link_th, low_text, poly=False
    )
    boxes = adjust_result_coordinates(boxes, 1 / ratio, 1 / ratio)
    quads, scores = [], []
    for b in boxes:
        pts = np.array(b, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 4:
            continue
        quad = order_tl_tr_br_bl(pts[:4])
        quad = unclip_quad(quad, unclip)
        quad[:, 0] = np.clip(quad[:, 0], 0, orig_w - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, orig_h - 1)
        w = float(np.linalg.norm(quad[0] - quad[1]))
        h = float(np.linalg.norm(quad[0] - quad[3]))
        if w < 10 or h < 8 or w * h < 100:
            continue
        if w > orig_w * 0.98 and h > orig_h * 0.7:
            continue
        quads.append(quad)
        # Local score: mean heatmap in the box, not the global max (that
        # made every CRAFT box identical so NMS could not rank them).
        hx, wx = score_text.shape[:2]
        x1 = int(np.clip(quad[:, 0].min() / max(orig_w, 1) * wx, 0, wx - 1))
        x2 = int(np.clip(quad[:, 0].max() / max(orig_w, 1) * wx + 1, 1, wx))
        y1 = int(np.clip(quad[:, 1].min() / max(orig_h, 1) * hx, 0, hx - 1))
        y2 = int(np.clip(quad[:, 1].max() / max(orig_h, 1) * hx + 1, 1, hx))
        patch = score_text[y1:y2, x1:x2]
        scores.append(float(patch.mean()) if patch.size else 0.5)
    return quads, scores


def craft_detect(
    model,
    rgb,
    device,
    text_threshold=0.42,
    link_threshold=0.40,
    low_text=0.28,
    square_size=1280,
    unclip=1.22,
):
    """
    rgb uint8 HWC -> (quads, scores).
    Retries a smaller canvas if CUDA runs out of memory.
    """
    image = Image.fromarray(rgb)
    orig_h, orig_w = rgb.shape[:2]
    last_err = None
    for size in (square_size, 960, 768):
        try:
            processed = model.preprocess(image, size=size, device=str(device))
            pixel_values = processed["pixel_values"]
            ratio_values = processed["ratio_values"]
            with torch.no_grad():
                out = model(pixel_values=pixel_values, ratio_values=ratio_values)
            logits = out["logits"][0]
            ratio = float(out["ratio_values"][0])
            return _quads_from_logits(
                logits, ratio, text_threshold, link_threshold, low_text,
                orig_w, orig_h, unclip,
            )
        except RuntimeError as err:
            last_err = err
            if "out of memory" not in str(err).lower():
                raise
            torch.cuda.empty_cache()
            continue
    if last_err:
        raise last_err
    return [], []
