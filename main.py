import base64
import io
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image


ROOT = Path(__file__).parent
DET_MODEL = ROOT / "models" / "detection"
REC_MODEL = ROOT / "models" / "recognition"


def order_quad(points):
    points = np.asarray(points, dtype=np.float32)
    top, bottom = np.split(points[np.argsort(points[:, 1])], 2)
    return np.array([
        top[np.argmin(top[:, 0])], top[np.argmax(top[:, 0])],
        bottom[np.argmax(bottom[:, 0])], bottom[np.argmin(bottom[:, 0])],
    ], dtype=np.float32)


def bounds(quad):
    return quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max()


def overlap(left, right):
    ax1, ay1, ax2, ay2 = bounds(left)
    bx1, by1, bx2, by2 = bounds(right)
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a, area_b = (ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection + 1e-6), intersection / (min(area_a, area_b) + 1e-6)


def nms(quads, scores):
    areas = [(q[:, 0].max() - q[:, 0].min()) * (q[:, 1].max() - q[:, 1].min()) for q in quads]
    keep = []
    for index in sorted(range(len(quads)), key=lambda i: (scores[i], areas[i]), reverse=True):
        duplicate = None
        for position, kept in enumerate(keep):
            iou, cover = overlap(quads[index], quads[kept])
            if iou >= 0.28 or cover >= 0.50:
                duplicate = position
                break
        if duplicate is None:
            keep.append(index)
        elif areas[index] > areas[keep[duplicate]] * 1.08:
            keep[duplicate] = index
    return [quads[i] for i in keep], [float(scores[i]) for i in keep]


def lines(quads):
    if not quads:
        return []
    centers = [(float(q[:, 0].mean()), float(q[:, 1].mean())) for q in quads]
    heights = [max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]), 8) for q in quads]
    tolerance = max(12, np.median(heights) * 0.65)
    result = []
    for index in sorted(range(len(quads)), key=lambda i: centers[i][1]):
        line = next((line for line in result
                     if abs(centers[index][1] - np.mean([centers[i][1] for i in line])) < tolerance), None)
        if line is None:
            result.append([index])
        else:
            line.append(index)
    for line in result:
        line.sort(key=lambda i: centers[i][0], reverse=True)
    return result


def reading_order(quads):
    return [index for line in lines(quads) for index in line]


def block_groups(quads):
    geometry = []
    for quad in quads:
        x1, _, x2, _ = bounds(quad)
        geometry.append((float(quad[:, 0].mean()), x1, x2,
                         max(np.linalg.norm(quad[0] - quad[3]), 8)))
    groups = []
    for line in lines(quads):
        left_to_right = sorted(line, key=lambda i: geometry[i][1])
        current = [left_to_right[0]]
        for index in left_to_right[1:]:
            gap = geometry[index][1] - geometry[current[-1]][2]
            if gap > max(14, np.median([geometry[i][3] for i in line]) * 0.7):
                groups.append(current)
                current = [index]
            else:
                current.append(index)
        groups.append(current)
    groups.sort(key=lambda group: (np.mean([quads[i][:, 1].mean() for i in group]),
                                   -np.mean([quads[i][:, 0].mean() for i in group])))
    return [sorted(group, key=lambda i: geometry[i][0], reverse=True) for group in groups]


def unique_texts(texts):
    result = []
    for text in texts:
        text = " ".join(dict.fromkeys((text or "").split()))
        compact = text.replace(" ", "")
        if compact and not any(compact in old.replace(" ", "") for old in result):
            result = [old for old in result if old.replace(" ", "") not in compact]
            result.append(text)
    return result


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


class OCR:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = self.recognizer = None

    def detect(self, rgb):
        from hezar.models import Model
        from hezar.models.text_detection.craft.craft_utils import adjust_result_coordinates, get_detection_boxes

        if self.detector is None:
            self.detector = Model.load(str(DET_MODEL), device=str(self.device))
        image = Image.fromarray(rgb)
        last_error = None
        for size in (1280, 960, 768):
            try:
                processed = self.detector.preprocess(image, size=size, device=str(self.device))
                with torch.no_grad():
                    output = self.detector(pixel_values=processed["pixel_values"],
                                           ratio_values=processed["ratio_values"])
                logits, ratio = output["logits"][0], float(output["ratio_values"][0])
                text_map, link_map = logits[:, :, 0].detach().cpu().numpy(), logits[:, :, 1].detach().cpu().numpy()
                boxes, _, _ = get_detection_boxes(text_map, link_map, 0.42, 0.40, 0.28, poly=False)
                boxes = adjust_result_coordinates(boxes, 1 / ratio, 1 / ratio)
                quads, scores = [], []
                height, width = rgb.shape[:2]
                for box in boxes:
                    quad = order_quad(np.asarray(box, dtype=np.float32)[:4])
                    quad = quad.mean(0) + (quad - quad.mean(0)) * 1.22
                    quad[:, 0], quad[:, 1] = np.clip(quad[:, 0], 0, width - 1), np.clip(quad[:, 1], 0, height - 1)
                    if min(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[0] - quad[3])) < 8:
                        continue
                    x1, y1, x2, y2 = bounds(quad)
                    patch = text_map[int(y1 / height * text_map.shape[0]):max(int(y2 / height * text_map.shape[0]), 1),
                                     int(x1 / width * text_map.shape[1]):max(int(x2 / width * text_map.shape[1]), 1)]
                    quads.append(quad)
                    scores.append(float(patch.mean()) if patch.size else 0.5)
                return nms(quads, scores)
            except RuntimeError as error:
                last_error = error
                if "out of memory" not in str(error).lower():
                    raise
                torch.cuda.empty_cache()
        raise last_error

    def _predict(self, images):
        from hezar.models import Model

        if self.recognizer is None:
            self.recognizer = Model.load(str(REC_MODEL))
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for index, image in enumerate(images):
                path = Path(folder) / f"{index}.png"
                Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)
                paths.append(str(path))
            outputs = self.recognizer.predict(paths)
        if not isinstance(outputs, list):
            outputs = [outputs]
        if len(outputs) == 1 and isinstance(outputs[0], (list, tuple)):
            outputs = list(outputs[0])
        return [output_text(output) for output in outputs]

    def recognize(self, images):
        images = [enhance(image) for image in images]
        texts = self._predict(images)
        retry = [index for index, text in enumerate(texts) if not text.strip() or garbage(text)]
        if retry:
            alternatives = self._predict([binarize(images[index]) for index in retry])
            for index, text in zip(retry, alternatives):
                if text.strip() and not garbage(text):
                    texts[index] = text
        return texts

    def run(self, image):
        started = time.perf_counter()
        quads, scores = self.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        order = reading_order(quads)
        quads, scores = [quads[i] for i in order], [scores[i] for i in order]
        texts = self.recognize([crop(image, quad) for quad in quads])
        words = [{"text": text, "score": round(score, 4), "box": np.round(quad, 1).tolist()}
                 for text, score, quad in zip(texts, scores, quads)]
        items = []
        for group in block_groups(quads):
            text = " ".join(unique_texts(words[i]["text"] for i in group))
            if not text:
                continue
            points = np.concatenate([quads[i] for i in group])
            x1, y1 = map(float, (points[:, 0].min(), points[:, 1].min()))
            x2, y2 = map(float, (points[:, 0].max(), points[:, 1].max()))
            items.append({"text": text, "score": round(float(np.mean([scores[i] for i in group])), 4),
                          "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})
        view = image.copy()
        for index, item in enumerate(items, 1):
            points = np.asarray(item["box"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [points], True, (50, 210, 80), 2)
            x, y = points[:, 0, 0].min(), points[:, 0, 1].min()
            cv2.putText(view, str(index), (x, max(y - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, .55, (30, 180, 50), 2)
        _, encoded = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return {"n_boxes": len(words), "n_blocks": len(items),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "items": items, "words": words,
                "text": "\n".join(item["text"] for item in items),
                "annotated_jpeg_b64": base64.b64encode(encoded).decode()}


ocr = OCR()
app = FastAPI(title="Persian OCR")


def read_image(data):
    try:
        return cv2.cvtColor(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as error:
        raise HTTPException(400, "Invalid image") from error


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((ROOT / "index.html").read_text(encoding="utf-8"))


@app.post("/ocr")
async def recognize(file: UploadFile = File(...)):
    return ocr.run(read_image(await file.read()))


@app.post("/ocr/json")
async def recognize_json(file: UploadFile = File(...)):
    result = ocr.run(read_image(await file.read()))
    return {"imnames": [file.filename], "txt": [word["text"] for word in result["words"]],
            "wordBB": [[[point[0] for point in word["box"]], [point[1] for point in word["box"]]]
                       for word in result["words"]], "charBB": []}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)
