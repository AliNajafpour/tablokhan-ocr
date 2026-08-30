# -*- coding: utf-8 -*-
"""
جعبه متن: استخراج از نقشه، NMS، مرتب‌سازی خواندن فارسی، کراپ پرسپکتیو.

ترتیب خواندن صورت‌مسأله: اول بالا به پایین، داخل هر سطر راست به چپ.
"""

import cv2
import numpy as np


def order_tl_tr_br_bl(pts):
    """چهار نقطه را به ترتیب بالا-چپ، بالا-راست، پایین-راست، پایین-چپ می‌چیند."""
    pts = np.asarray(pts, dtype=np.float32)
    order = np.argsort(pts[:, 1])
    top = pts[order[:2]]
    bot = pts[order[2:]]
    tl = top[np.argmin(top[:, 0])]
    tr = top[np.argmax(top[:, 0])]
    bl = bot[np.argmin(bot[:, 0])]
    br = bot[np.argmax(bot[:, 0])]
    return np.stack([tl, tr, br, bl], axis=0)


def unclip_quad(quad, ratio=1.18):
    """جعبه را از مرکز کمی بزرگ می‌کند تا حرف اول/آخر بریده نشود."""
    c = quad.mean(axis=0, keepdims=True)
    out = c + (quad - c) * ratio
    return out.astype(np.float32)


def aabb_xyxy(quad):
    q = np.asarray(quad, dtype=np.float32)
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def aabb_area(quad):
    x1, y1, x2, y2 = aabb_xyxy(quad)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def overlap_stats(a, b):
    """
    Axis-aligned overlap of two quads.
    Returns (iou, inter / min(area), area_a, area_b).
    IoMin catches a word box nested inside a longer phrase box,
    which standard IoU often misses (union is large).
    """
    ax1, ay1, ax2, ay2 = aabb_xyxy(a)
    bx1, by1, bx2, by2 = aabb_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    iou = inter / union
    iomin = inter / (min(area_a, area_b) + 1e-6)
    return iou, iomin, area_a, area_b


def rect_iou(a, b):
    """IoU دو مستطیل محور-هم‌تراز (برای NMS). a/b شکل (4,2)."""
    return overlap_stats(a, b)[0]


def boxes_conflict(a, b, iou_thresh=0.28, cover_thresh=0.50):
    """True if two boxes cover the same text (overlap or containment)."""
    iou, iomin, _, _ = overlap_stats(a, b)
    return iou >= iou_thresh or iomin >= cover_thresh


def nms_quads(quads, scores, iou_thresh=0.28, cover_thresh=0.50):
    """
    Drop duplicate detections of the same text.

    Standard IoU-NMS is not enough: a small word box inside a longer
    phrase has low IoU but high intersection-over-min. On conflict we
    keep the *larger* box so the full phrase wins over a fragment.
    """
    if not quads:
        return [], []
    areas = [aabb_area(q) for q in quads]
    order = sorted(
        range(len(quads)),
        key=lambda i: (float(scores[i]), areas[i]),
        reverse=True,
    )
    keep = []
    for i in order:
        conflict_at = None
        for k, j in enumerate(keep):
            if boxes_conflict(quads[i], quads[j], iou_thresh, cover_thresh):
                conflict_at = k
                break
        if conflict_at is None:
            keep.append(int(i))
            continue
        j = keep[conflict_at]
        if areas[i] > areas[j] * 1.08:
            keep[conflict_at] = int(i)
    return [quads[i] for i in keep], [float(scores[i]) for i in keep]


def merge_quad_lists(*pairs, iou_thresh=0.28, cover_thresh=0.50):
    """Union of several (quads, scores) lists, then overlap/containment NMS."""
    quads, scores = [], []
    for item in pairs:
        if not item:
            continue
        q, s = item
        quads.extend(q)
        scores.extend(s)
    return nms_quads(quads, scores, iou_thresh=iou_thresh, cover_thresh=cover_thresh)


def map_to_boxes(prob, scale, orig_w, orig_h, img_size, thresh=0.32, min_area=18, unclip=1.18):
    """
    نقشه احتمال -> لیست چهارضلعی در مختصات تصویر اصلی + امتیاز میانگین.
    dilate کوچک تکه‌های شکسته را وصل می‌کند.
    """
    mh, mw = prob.shape
    mask = (prob > thresh).astype(np.uint8) * 255
    # Small dilate reconnects broken strokes without gluing distant signs.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sx = img_size / float(mw)
    sy = img_size / float(mh)
    quads, scores = [], []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        pts = cv2.boxPoints(cv2.minAreaRect(cnt))
        pts[:, 0] = pts[:, 0] * sx / max(scale, 1e-6)
        pts[:, 1] = pts[:, 1] * sy / max(scale, 1e-6)
        pts[:, 0] = np.clip(pts[:, 0], 0, orig_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, orig_h - 1)
        quad = order_tl_tr_br_bl(pts)
        quad = unclip_quad(quad, unclip)
        quad[:, 0] = np.clip(quad[:, 0], 0, orig_w - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, orig_h - 1)
        # امتیاز = میانگین احتمال داخل کانتور روی نقشه
        x, y, w, h = cv2.boundingRect(cnt)
        patch = prob[y:y + h, x:x + w]
        score = float(patch.mean()) if patch.size else 0.0
        quads.append(quad)
        scores.append(score)
    return nms_quads(quads, scores)


def group_line_indices(quads):
    """
    Cluster word boxes into lines (top to bottom).
    Inside each line, indices are right-to-left (Persian reading).
    """
    if not quads:
        return []
    centers = []
    heights = []
    for q in quads:
        cx = float(q[:, 0].mean())
        cy = float(q[:, 1].mean())
        h = float(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]), 8.0))
        centers.append((cx, cy))
        heights.append(h)
    y_tol = max(12.0, float(np.median(heights)) * 0.65)
    remain = sorted(range(len(quads)), key=lambda i: centers[i][1])
    lines = []
    for i in remain:
        placed = False
        for line in lines:
            mean_y = sum(centers[j][1] for j in line) / len(line)
            if abs(centers[i][1] - mean_y) < y_tol:
                line.append(i)
                placed = True
                break
        if not placed:
            lines.append([i])
    for line in lines:
        line.sort(key=lambda i: centers[i][0], reverse=True)
    return lines


def reading_order(quads):
    """
    ایندکس جعبه‌ها به ترتیب خواندن فارسی:
    سطرها از بالا به پایین، داخل سطر از راست به چپ.
    """
    ordered = []
    for line in group_line_indices(quads):
        ordered.extend(line)
    return ordered


def merge_line_quad(quads):
    """Axis-aligned envelope of several word quads (one sentence box)."""
    pts = np.concatenate([np.asarray(q, dtype=np.float32).reshape(-1, 2) for q in quads], axis=0)
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return order_tl_tr_br_bl([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def _quad_geom(quad):
    q = np.asarray(quad, dtype=np.float32)
    cx = float(q[:, 0].mean())
    cy = float(q[:, 1].mean())
    w = float(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2]), 8.0))
    h = float(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]), 8.0))
    x1, x2 = float(q[:, 0].min()), float(q[:, 0].max())
    return cx, cy, w, h, x1, x2


def group_block_indices(quads, gap_ratio=0.55):
    """
    Split each y-aligned row into separate text blocks.

    Words that almost touch (normal spacing) stay in one block.
    Distinct signs / sentences on the same horizontal line — with a large
    empty gap between them — stay as separate boxes. Never merge a whole
    row into one envelope.
    """
    if not quads:
        return []
    geoms = [_quad_geom(q) for q in quads]
    widths = [g[2] for g in geoms]
    heights = [g[3] for g in geoms]
    median_w = float(np.median(widths)) if widths else 20.0
    median_h = float(np.median(heights)) if heights else 16.0
    max_gap = max(14.0, median_w * float(gap_ratio), median_h * 0.70)
    blocks = []
    for line in group_line_indices(quads):
        if not line:
            continue
        ltr = sorted(line, key=lambda i: geoms[i][4])
        cur = [ltr[0]]
        for idx in ltr[1:]:
            conflict = None
            for pos, prev in enumerate(cur):
                if boxes_conflict(quads[idx], quads[prev]):
                    conflict = pos
                    break
            if conflict is not None:
                prev = cur[conflict]
                if aabb_area(quads[idx]) > aabb_area(quads[prev]) * 1.08:
                    cur[conflict] = idx
                continue
            prev = cur[-1]
            gap = geoms[idx][4] - geoms[prev][5]
            if gap > max_gap:
                blocks.append(cur)
                cur = [idx]
            else:
                cur.append(idx)
        blocks.append(cur)

    def block_key(b):
        cy = sum(geoms[i][1] for i in b) / len(b)
        cx = sum(geoms[i][0] for i in b) / len(b)
        return (round(cy / max(median_h, 1.0)), -cx)

    blocks.sort(key=block_key)
    for block in blocks:
        block.sort(key=lambda i: geoms[i][0], reverse=True)
    return blocks


def _compact_text(s):
    return "".join((s or "").split())


def join_unique_texts(parts):
    """
    Concatenate nearby word reads without repeating the same phrase.

    Drops exact duplicates, fragments contained in a longer string, and
    a leading word that already ended the previous piece
    (حفاظت + حفاظت محیط -> حفاظت محیط).
    """
    kept = []
    for raw in parts:
        words = []
        for w in (raw or "").split():
            if words and w == words[-1]:
                continue
            words.append(w)
        piece = " ".join(words)
        if not piece:
            continue
        compact = _compact_text(piece)
        if not compact:
            continue
        if any(compact == _compact_text(k) or compact in _compact_text(k) for k in kept):
            continue
        kept = [k for k in kept if _compact_text(k) not in compact]
        if kept:
            prev_words = kept[-1].split()
            words = piece.split()
            while words and words[0] in prev_words:
                words.pop(0)
            if not words:
                continue
            piece = " ".join(words)
            compact = _compact_text(piece)
            if not compact or any(compact in _compact_text(k) for k in kept):
                continue
        kept.append(piece)
    return kept


def nms_text_items(items, iou_thresh=0.28, cover_thresh=0.50):
    """Drop overlapping report boxes; keep the longer / larger phrase."""
    if len(items) < 2:
        return items
    quads = [np.asarray(it["box"], dtype=np.float32) for it in items]
    areas = [aabb_area(q) for q in quads]
    scores = [
        float(it.get("score", 0.5)) + 0.002 * len(_compact_text(it.get("text") or ""))
        for it in items
    ]
    order = sorted(
        range(len(items)),
        key=lambda i: (scores[i], areas[i]),
        reverse=True,
    )
    keep = []
    for i in order:
        conflict_at = None
        for k, j in enumerate(keep):
            if boxes_conflict(quads[i], quads[j], iou_thresh, cover_thresh):
                conflict_at = k
                break
        if conflict_at is None:
            keep.append(i)
            continue
        j = keep[conflict_at]
        ti = len(_compact_text(items[i].get("text") or ""))
        tj = len(_compact_text(items[j].get("text") or ""))
        if ti > tj or (ti == tj and areas[i] > areas[j]):
            keep[conflict_at] = i
    keep.sort()
    return [items[i] for i in keep]


def warp_crop(bgr, quad, pad_ratio=0.16):
    """
    چهارضلعی را با perspective به یک مستطیل می‌برد.
    بهتر از برش محور-هم‌تراز است (حرف اول خط فارسی کمتر می‌پرد).
    """
    q = np.asarray(quad, dtype=np.float32)
    w = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3]), 8))
    h = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]), 8))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    m = cv2.getPerspectiveTransform(q, dst)
    crop = cv2.warpPerspective(bgr, m, (w, h), flags=cv2.INTER_LINEAR)
    pad = max(1, int(h * pad_ratio))
    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return crop


def quad_to_wordbb(quad):
    """قالب مرحله ۱: [xs چهارتایی, ys چهارتایی]."""
    xs = [round(float(p[0]), 2) for p in quad]
    ys = [round(float(p[1]), 2) for p in quad]
    return [xs, ys]
