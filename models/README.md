# Model locations

Model weights are intentionally not stored in Git. Download both models with:

```powershell
python scripts/download_models.py
```

The script creates:

```text
models/
├── detection/
│   ├── config.yaml
│   └── model.pt
└── recognition/
    ├── config.yaml
    └── model.pt
```

You can also point to models elsewhere:

```powershell
$env:OCR_DETECTION_MODEL = "D:\path\to\craft"
$env:OCR_RECOGNITION_MODEL = "D:\path\to\crnn-base-fa-v2"
```
