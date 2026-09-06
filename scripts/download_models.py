from pathlib import Path
import shutil

from huggingface_hub import snapshot_download
from hezar.models import Model


ROOT = Path(__file__).resolve().parent.parent
detector = ROOT / "models" / "detection" / "PP-OCRv6_medium_det"
snapshot_download("PaddlePaddle/PP-OCRv6_medium_det", local_dir=detector,
                  allow_patterns=["inference.json", "inference.pdiparams", "inference.yml"])
shutil.rmtree(detector / ".cache", ignore_errors=True)
Model.load("hezarai/crnn-base-fa-v2",
           save_path=str(ROOT / "models" / "recognition"))

print("Models saved in", ROOT / "models")
