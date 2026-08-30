from pathlib import Path

from hezar.models import Model


ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "hezarai/CRAFT": ROOT / "models" / "detection",
    "hezarai/crnn-base-fa-v2": ROOT / "models" / "recognition",
}

for model_id, destination in MODELS.items():
    print(f"Downloading {model_id}...")
    Model.load(model_id, save_path=str(destination))

print("Models saved in", ROOT / "models")
