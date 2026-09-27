from pathlib import Path
import cv2
import numpy as np
import torch
from paddle_runtime import gpu_device


ROOT = Path(__file__).parent
MODEL_NAME = "arabic-train-v1"
MODEL_DIR = ROOT / "models" / "recognition" / MODEL_NAME
_model = None
MIN_ROTATION_CONFIDENCE = 0.55
MIN_ROTATION_GAIN = 0.12
MIN_ROTATION_TEXT_LENGTH = 3
SIDEWAYS_ASPECT_RATIO = 1.5

if torch.cuda.is_available():
    torch.cuda.empty_cache()

def crop(image, quad):
    width = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3]), 8))
    height = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]), 8))
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, target), (width, height))


def load_model():
    global _model
    if _model is None:
        if not (MODEL_DIR / "inference.pdiparams").is_file():
            raise FileNotFoundError(f"Recognition model is missing: {MODEL_DIR}")
        device = gpu_device()
        from paddleocr import TextRecognition
        _model = TextRecognition(
            model_name="arabic_PP-OCRv5_mobile_rec",
            model_dir=str(MODEL_DIR),
            device=device,
            enable_mkldnn=False,
        )
    return _model

model = load_model()

def predict(images, model=model):
    if not images:
        return []
    return [
        (str(result["rec_text"]), float(result["rec_score"]))
        for output in model.predict(images, batch_size=min(16, len(images)))
        for result in [output.json["res"]]
    ]


def _better_orientation(current, candidates):
    _text, confidence = current
    acceptable = [
        candidate for candidate in candidates
        if len(candidate[0].strip()) >= MIN_ROTATION_TEXT_LENGTH
        and candidate[1] >= MIN_ROTATION_CONFIDENCE
        and candidate[1] >= confidence + MIN_ROTATION_GAIN
    ]
    return max(acceptable, key=lambda result: result[1]) if acceptable else current


def recognize(image, quads):
    if not quads:
        return []
    images = [crop(image, quad) for quad in quads]
    results = predict(images)
    upside_down = predict([cv2.rotate(piece, cv2.ROTATE_180) for piece in images])
    results = [
        _better_orientation(current, [alternative])
        for current, alternative in zip(results, upside_down)
    ]
    retries = [
        index for index, (piece, (text, confidence)) in enumerate(zip(images, results))
        if piece.shape[0] / piece.shape[1] >= SIDEWAYS_ASPECT_RATIO
        or not text.strip()
        or (confidence < 0.5 and len(text.strip()) <= 1)
    ]
    if retries:
        rotated = [variant for index in retries for variant in (
            cv2.rotate(images[index], cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(images[index], cv2.ROTATE_90_COUNTERCLOCKWISE),
        )]
        alternatives = predict(rotated)
        for offset, index in enumerate(retries):
            results[index] = _better_orientation(
                results[index], alternatives[offset * 2:offset * 2 + 2]
            )
    return [text for text, _confidence in results]
