# -*- coding: utf-8 -*-
"""Draw numbered polygons on a BGR image for the API/UI overlay."""

import base64

import cv2
import numpy as np


def annotate_bgr(bgr, items):
    """
    Draw each detection as a green quad with a 1-based index.
    The same index is used in the text list under the image.
    """
    vis = bgr.copy()
    for i, item in enumerate(items, 1):
        pts = np.array(item["box"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (50, 210, 80), 2, cv2.LINE_AA)
        x = int(pts[:, 0, 0].min())
        y = int(pts[:, 0, 1].min())
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y0 = max(0, y - th - 6)
        cv2.rectangle(vis, (x, y0), (x + tw + 8, y0 + th + 6), (50, 210, 80), -1)
        cv2.putText(vis, label, (x + 4, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 2, cv2.LINE_AA)
    return vis


def jpeg_b64(bgr, quality=90):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
