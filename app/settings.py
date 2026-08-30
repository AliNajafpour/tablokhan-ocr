import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DETECTION_MODEL_DIR = Path(
    os.getenv("OCR_DETECTION_MODEL", PROJECT_ROOT / "models" / "detection")
)
RECOGNITION_MODEL_DIR = Path(
    os.getenv("OCR_RECOGNITION_MODEL", PROJECT_ROOT / "models" / "recognition")
)
LEXICON_PATH = PROJECT_ROOT / "data" / "processed" / "selected_3grams_10k_clean.csv"

API_HOST = os.getenv("OCR_HOST", "127.0.0.1")
API_PORT = int(os.getenv("OCR_PORT", "8000"))
