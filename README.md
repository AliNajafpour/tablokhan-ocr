# Persian Scene-Text OCR

Persian text detection with PP-OCRv6 medium and recognition with selectable Hezar CRNN models.

## Run locally

Use Python 3.11. The download script fetches the default Paddle detector and Hezar recognizer into `models/`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_models.py
python main.py
```

Open <http://127.0.0.1:8000> or send an image to `POST /ocr`.
The page can run full OCR, detection only, or recognition only. Recognition-only expects a cropped text image.

Detection uses `models/detection/PP-OCRv6_medium_det/`. Recognition lists every
Hezar model stored directly in `models/recognition/` or one of its subfolders.

## Project layout

```text
main.py      API and combined OCR pipeline
detection.py PP-OCRv6 text detection
recognition.py Hezar text recognition
index.html   upload page
models/      local model locations; weights are not committed
scripts/     dataset generation helpers
notebooks/   preprocessing and PaddleOCR training notebooks
data/        earlier generated datasets and corpus work
docs/        research and competition documents
tests/       small checks that do not require model weights
```

The historical `sia/` workspace is intentionally excluded from Git.
