# Persian Scene-Text OCR

Persian text detection and recognition using Hezar CRAFT and CRNN models.

## Run locally

Use Python 3.11. The download script fetches the two required Hezar models and saves them under `models/`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_models.py
python main.py
```

Open <http://127.0.0.1:8000> or send an image to `POST /ocr`.

## Project layout

```text
main.py      OCR API and Hezar inference pipeline
index.html   upload page
models/      local model locations; weights are not committed
scripts/     dataset generation helpers
notebooks/   preprocessing and PaddleOCR training notebooks
data/        earlier generated datasets and corpus work
docs/        research and competition documents
tests/       small checks that do not require model weights
```

The historical `sia/` workspace is intentionally excluded from Git.
