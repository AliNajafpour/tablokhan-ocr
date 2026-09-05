import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from detection import detect
from recognition import recognize


ROOT = Path(__file__).parent
app = FastAPI(title="TabloKhan")

RESULTS_DIR = ROOT / "results"
IMAGES_DIR = RESULTS_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def save_image(annotated_jpg, name=None):
    stem = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in Path(name or "image").stem) or "image"
    (IMAGES_DIR / f"{stem}.jpg").write_bytes(annotated_jpg)


def save_run_json(pairs):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (RESULTS_DIR / f"results_{stamp}.json").write_text(
        json.dumps([{"image": name, "texts": [word["text"] for word in words]}
                    for name, words in pairs], ensure_ascii=False, indent=2), encoding="utf-8")


def run_ocr(image, mode="ocr", name=None):

    started = time.perf_counter()
    quads, scores = detect(image)
    texts = [""] * len(quads) if mode == "detection" else recognize(image, quads)
    words = [{"text": text, "score": None if score is None else round(score, 4),
            "box": np.round(quad, 1).tolist()}
            for text, score, quad in zip(texts, scores, quads)]
    view = image.copy()
    for index, word in enumerate(words if mode != "recognition" else [], 1):
        points = np.asarray(word["box"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(view, [points], True, (50, 210, 80), 2)
        x, y = points[:, 0, 0].min(), points[:, 0, 1].min()
        cv2.putText(view, str(index), (x, max(y - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, .55, (30, 180, 50), 2)
    _, encoded = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 90])
    result = {
        "mode": mode,
        "n_boxes": len(words),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "words": words,
        "text": "\n".join(word["text"] for word in words),
        "annotated_jpeg_b64": base64.b64encode(encoded).decode(),
    }
    save_image(encoded.tobytes(), name)
    return result


def read_image(data):
    try:
        return cv2.cvtColor(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as error:
        raise HTTPException(400, "Invalid image") from error


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((ROOT / "index.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})



@app.post("/ocr")
async def ocr(files: list[UploadFile] = File(...), mode: str = Form("ocr")):
    outcomes = []
    for file in files:
        result = run_ocr(read_image(await file.read()), mode, name=file.filename)
        outcomes.append({**result, "filename": file.filename})
    if mode != "detection":
        save_run_json([(file.filename, result["words"]) for file, result in zip(files, outcomes)])
    return outcomes


@app.post("/ocr/json")
async def ocr_json(file: UploadFile = File(...)):
    result = run_ocr(read_image(await file.read()), "ocr", name=file.filename)
    save_run_json([(file.filename, result["words"])])
    return {"imnames": [file.filename], "txt": [word["text"] for word in result["words"]],
            "wordBB": [[[point[0] for point in word["box"]], [point[1] for point in word["box"]]]
                    for word in result["words"]], "charBB": []}