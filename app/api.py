import io

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from .pipeline import OCRPipeline
from .settings import API_HOST, API_PORT, STATIC_DIR


pipeline = OCRPipeline()
app = FastAPI(title="Persian OCR", version="1.0")


def read_image(data):
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as error:
        raise HTTPException(status_code=400, detail="فایل تصویر معتبر نیست.") from error


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": pipeline.device.type,
        "engine": "hezar",
        "det": pipeline.det_path,
        "rec": pipeline.rec_path,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


async def run_ocr(file):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="فایل خالی است.")
    try:
        result = pipeline.run(read_image(data))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"OCR شکست خورد: {error}") from error
    result["filename"] = file.filename
    return result


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    return JSONResponse(await run_ocr(file))


@app.post("/ocr/json")
async def ocr_json(file: UploadFile = File(...)):
    result = await run_ocr(file)
    output = pipeline.to_stage1_json(file.filename or "image.jpg", result)
    output.update(elapsed_ms=result["elapsed_ms"], n_boxes=result["n_boxes"])
    return JSONResponse(output)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host=API_HOST, port=API_PORT)
