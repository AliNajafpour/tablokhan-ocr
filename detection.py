import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
MODEL_NAME = "PP-OCRv6_medium_det"
_model = None


def available_models():
    return [MODEL_NAME]


def order_quad(points):
    points = np.asarray(points, dtype=np.float32)
    top, bottom = np.split(points[np.argsort(points[:, 1])], 2)
    return np.array([
        top[np.argmin(top[:, 0])], top[np.argmax(top[:, 0])],
        bottom[np.argmax(bottom[:, 0])], bottom[np.argmin(bottom[:, 0])],
    ], dtype=np.float32)


def reading_order(quads):
    if not quads:
        return []
    centers = [(float(q[:, 0].mean()), float(q[:, 1].mean())) for q in quads]
    heights = [max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]), 8) for q in quads]
    tolerance = max(12, np.median(heights) * 0.65)
    lines = []
    for index in sorted(range(len(quads)), key=lambda i: centers[i][1]):
        line = next((line for line in lines
                     if abs(centers[index][1] - np.mean([centers[i][1] for i in line])) < tolerance), None)
        if line is None:
            lines.append([index])
        else:
            line.append(index)
    for line in lines:
        line.sort(key=lambda i: centers[i][0], reverse=True)
    return [index for line in lines for index in line]


def detect(image, model_name=MODEL_NAME):
    global _model
    if model_name != MODEL_NAME:
        raise ValueError(f"Unknown detection model: {model_name}")
    if _model is None:
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / "tmp" / "paddlex"))
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import TextDetection
        _model = TextDetection(
            model_name=MODEL_NAME,
            model_dir=str(ROOT / "models" / "detection" / MODEL_NAME),
            device="cpu",
            enable_mkldnn=False,
        )
    result = next(iter(_model.predict(image, batch_size=1))).json["res"]
    quads = [order_quad(box) for box in result["dt_polys"]]
    scores = [float(score) for score in result["dt_scores"]]
    order = reading_order(quads)
    return [quads[i] for i in order], [scores[i] for i in order]
