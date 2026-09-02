import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from detection import MODEL_NAME as DETECTION_MODEL, available_models as detection_models, detect
from recognition import available_models as recognition_models, recognize


ROOT = Path(__file__).parent
app = FastAPI(title="Persian OCR")


def run_ocr(image, mode="ocr", detection_model=DETECTION_MODEL, recognition_model="default"):
    if mode not in ("ocr", "detection", "recognition"):
        raise HTTPException(400, f"Unknown mode: {mode}")
    if mode != "recognition" and detection_model not in detection_models():
        raise HTTPException(400, f"Unknown detection model: {detection_model}")
    if mode != "detection" and recognition_model not in recognition_models():
        raise HTTPException(400, f"Unknown recognition model: {recognition_model}")

    started = time.perf_counter()
    if mode == "recognition":
        height, width = image.shape[:2]
        quads = [np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])]
        scores = [None]
    else:
        quads, scores = detect(image, detection_model)
    texts = [""] * len(quads) if mode == "detection" else recognize(image, quads, recognition_model)
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
    return {
        "mode": mode,
        "n_boxes": len(words),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "words": words,
        "text": "\n".join(word["text"] for word in words),
        "annotated_jpeg_b64": base64.b64encode(encoded).decode(),
    }


def read_image(data):
    try:
        return cv2.cvtColor(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as error:
        raise HTTPException(400, "Invalid image") from error


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((ROOT / "index.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


@app.get("/models")
def models():
    return {"detection": detection_models(), "recognition": list(recognition_models())}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), detection_model: str = Form(DETECTION_MODEL),
              recognition_model: str = Form("default"), mode: str = Form("ocr")):
    return run_ocr(read_image(await file.read()), mode, detection_model, recognition_model)


@app.post("/ocr/json")
async def ocr_json(file: UploadFile = File(...), detection_model: str = Form(DETECTION_MODEL),
                   recognition_model: str = Form("default")):
    result = run_ocr(read_image(await file.read()), "ocr", detection_model, recognition_model)
    return {"imnames": [file.filename], "txt": [word["text"] for word in result["words"]],
            "wordBB": [[[point[0] for point in word["box"]], [point[1] for point in word["box"]]]
                       for word in result["words"]], "charBB": []}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)
