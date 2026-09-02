import gc
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).parent
MODELS = ROOT / "models" / "recognition"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_model_name = None


def available_models(root=MODELS):
    models = {}
    if (root / "model_config.yaml").is_file():
        models["default"] = root
    if root.exists():
        models.update({path.parent.name: path.parent for path in root.glob("*/model_config.yaml")})
    return models


def crop(image, quad):
    width = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3]), 8))
    height = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]), 8))
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    output = cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, target), (width, height))
    pad = max(1, int(height * 0.16))
    return cv2.copyMakeBorder(output, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def enhance(image):
    height, width = image.shape[:2]
    if height < 36:
        image = cv2.resize(image, (max(8, int(width * 36 / height)), 36), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(2.6, (8, 8)).apply(light)
    image = cv2.cvtColor(cv2.merge([light, a, b]), cv2.COLOR_LAB2BGR)
    image = cv2.bilateralFilter(image, 5, 40, 40)
    return cv2.addWeighted(image, 1.35, cv2.GaussianBlur(image, (0, 0), 1), -0.35, 0)


def garbage(text):
    text = "".join((text or "").split())
    return len(text) >= 6 and Counter(text).most_common(1)[0][1] / len(text) >= 0.42


def binarize(image):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    output = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    return cv2.cvtColor(output if output.mean() >= 127 else 255 - output, cv2.COLOR_GRAY2BGR)


def output_text(output):
    if hasattr(output, "text"):
        return str(output.text)
    if isinstance(output, dict):
        return str(output.get("text") or output.get("label") or "")
    if isinstance(output, (list, tuple)) and output:
        return output_text(output[0])
    return "" if output is None else str(output)


def load_model(name):
    global _model, _model_name
    from hezar.models import Model

    models = available_models()
    if name not in models:
        raise ValueError(f"Unknown recognition model: {name}")
    if _model_name != name:
        _model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _model = Model.load(str(models[name]))
        _model_name = name
    return _model


def predict(images, model_name):
    model = load_model(model_name)
    with tempfile.TemporaryDirectory() as folder:
        paths = []
        for index, image in enumerate(images):
            path = Path(folder) / f"{index}.png"
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)
            paths.append(str(path))
        outputs = model.predict(paths, device=str(DEVICE))
    if not isinstance(outputs, list):
        outputs = [outputs]
    if len(outputs) == 1 and isinstance(outputs[0], (list, tuple)):
        outputs = list(outputs[0])
    return [output_text(output) for output in outputs]


def recognize(image, quads, model_name="default"):
    if not quads:
        return []
    images = [enhance(crop(image, quad)) for quad in quads]
    texts = predict(images, model_name)
    retry = [index for index, text in enumerate(texts) if not text.strip() or garbage(text)]
    if retry:
        alternatives = predict([binarize(images[index]) for index in retry], model_name)
        for index, text in zip(retry, alternatives):
            if text.strip() and not garbage(text):
                texts[index] = text
    return texts
