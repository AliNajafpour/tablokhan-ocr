import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .boxes import (
    group_block_indices,
    join_unique_texts,
    merge_line_quad,
    nms_quads,
    nms_text_items,
    quad_to_wordbb,
    reading_order,
    warp_crop,
)
from .craft import craft_detect
from .lexicon import correct_texts
from .preprocess import binarize_crop, enhance_crop, looks_garbage
from .settings import DETECTION_MODEL_DIR, RECOGNITION_MODEL_DIR
from .text import hezar_to_text
from .visualize import annotate_bgr, jpeg_b64


class OCRPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.det_path = str(DETECTION_MODEL_DIR)
        self.rec_path = str(RECOGNITION_MODEL_DIR)
        self._detector = None
        self._recognizer = None

    def _load_detector(self):
        if self._detector is None:
            if not (DETECTION_MODEL_DIR / "model.pt").is_file():
                raise FileNotFoundError(f"Detection model not found: {DETECTION_MODEL_DIR}")
            from hezar.models import Model

            self._detector = Model.load(str(DETECTION_MODEL_DIR), device=str(self.device))
        return self._detector

    def _load_recognizer(self):
        if self._recognizer is None:
            if not (RECOGNITION_MODEL_DIR / "model.pt").is_file():
                raise FileNotFoundError(f"Recognition model not found: {RECOGNITION_MODEL_DIR}")
            from hezar.models import Model

            self._recognizer = Model.load(str(RECOGNITION_MODEL_DIR))
            if self.device.type == "cuda":
                try:
                    self._recognizer.to("cuda")
                except Exception:
                    pass
        return self._recognizer

    def detect(self, rgb):
        quads, scores = craft_detect(self._load_detector(), rgb, self.device)
        return nms_quads(quads, scores, iou_thresh=0.28, cover_thresh=0.50)

    def recognize(self, crops):
        if not crops:
            return []
        model = self._load_recognizer()
        enhanced = [enhance_crop(crop) for crop in crops]
        texts = self._predict_crops(model, enhanced)
        retry_indices = [
            index for index, text in enumerate(texts)
            if not text.strip() or looks_garbage(text)
        ]
        if retry_indices:
            alternatives = self._predict_crops(
                model, [binarize_crop(enhanced[index]) for index in retry_indices]
            )
            for index, alternative in zip(retry_indices, alternatives):
                if alternative.strip() and not looks_garbage(alternative):
                    texts[index] = alternative
        return texts

    @staticmethod
    def _predict_crops(model, crops):
        with tempfile.TemporaryDirectory(prefix="hezar_ocr_") as folder:
            paths = []
            for index, crop in enumerate(crops):
                path = Path(folder) / f"{index}.png"
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(path)
                paths.append(str(path))
            try:
                outputs = model.predict(paths)
            except Exception:
                outputs = [model.predict(path) for path in paths]
        if not isinstance(outputs, list):
            outputs = [outputs]
        if len(outputs) == 1 and isinstance(outputs[0], (list, tuple)):
            outputs = list(outputs[0])
        outputs += [""] * (len(crops) - len(outputs))
        return [hezar_to_text(output) for output in outputs[:len(crops)]]

    def run(self, image_bgr):
        if image_bgr.ndim != 3:
            raise ValueError("Image must be a color image")
        started = time.perf_counter()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        quads, scores = self.detect(rgb)
        order = reading_order(quads)
        quads = [quads[index] for index in order]
        scores = [scores[index] for index in order]
        texts = self.recognize([warp_crop(image_bgr, quad) for quad in quads])
        texts = correct_texts(texts)

        words = [
            {
                "text": text,
                "score": round(float(score), 4),
                "box": [[round(float(x), 1), round(float(y), 1)] for x, y in quad],
            }
            for text, score, quad in zip(texts, scores, quads)
        ]
        blocks = []
        for group in group_block_indices(quads):
            parts = join_unique_texts(words[index]["text"] for index in group)
            if not parts:
                continue
            box = quads[group[0]] if len(group) == 1 else merge_line_quad([quads[index] for index in group])
            blocks.append({
                "text": " ".join(parts),
                "score": round(float(np.mean([scores[index] for index in group])), 4),
                "box": [[round(float(x), 1), round(float(y), 1)] for x, y in box],
                "n_words": len(parts),
            })
        items = nms_text_items(blocks) or words
        return {
            "engine": "hezar",
            "det_backend": "craft",
            "rec_backend": "hezar",
            "device": str(self.device),
            "det_path": self.det_path,
            "rec_path": self.rec_path,
            "n_boxes": len(words),
            "n_blocks": len(items),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "reading_order": "top-to-bottom, then right-to-left",
            "items": items,
            "words": words,
            "text": "\n".join(join_unique_texts(item["text"] for item in items)),
            "annotated_jpeg_b64": jpeg_b64(annotate_bgr(image_bgr, items)),
        }

    @staticmethod
    def to_stage1_json(image_name, result):
        source = result.get("words") or result["items"]
        return {
            "imnames": [image_name],
            "txt": [item["text"] for item in source],
            "wordBB": [quad_to_wordbb(np.array(item["box"], dtype=np.float32)) for item in source],
            "charBB": [],
        }
