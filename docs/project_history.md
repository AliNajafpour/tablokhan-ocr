# AI-Kibord project understanding

## Goal

Build a Persian natural-scene OCR system that reads text from photos such as signs and storefronts. The final system has two models:

1. Detection finds each text region and returns its four corner points.
2. Recognition reads every detected crop.

Phase 3 will connect both models behind a FastAPI endpoint and return the recognized lines in reading order, top to bottom and right to left.

## Roadmap

| Phase | Work | Current state |
| --- | --- | --- |
| 1. Study and data generation | Prepare the Hamshahri corpus, select varied Persian 3-grams, and generate detection and recognition data | Mostly complete |
| 2. Model training | Fine-tune and compare PaddleOCR detection and recognition models | Detection smoke tests complete; recognition is next |
| 3. End-to-end pipeline | Build the API, measure speed, and test model optimization | Not started |

## Data pipeline

The text source is the Hamshahri corpus under `data/raw/corpus`.

`Hamshahri documents -> cleaning -> sentence tokenization -> word tokenization -> 3-grams -> TF-IDF -> SVD -> MiniBatchKMeans -> representative 3-grams`

Current decisions:

- Documents shorter than 50 words are removed.
- Hazm normalizes and tokenizes Persian text.
- Punctuation noise is removed, but Persian stopwords such as `از`, `را`, and `به` remain because they form realistic phrases.
- The corpus produced about 2,572,272 3-gram occurrences and 1,924,078 unique TF-IDF rows.
- The original sparse TF-IDF matrix has 69,834 features. `TruncatedSVD` reduces it to 128 dimensions before clustering.
- `MiniBatchKMeans` selects the closest phrase to every cluster center.
- We use 1,000 clusters for detection phrases and 10,000 for recognition phrases. Cleaning reduced the saved pools to 946 and 9,428 phrases.

The preprocessing work is in `preproccessing-corpus.ipynb`. Its outputs are:

- `data/processed/selected_3grams_1k_clean.csv`
- `data/processed/selected_3grams_10k_clean.csv`

## Synthetic datasets

The old SynthText and SynthDoG route was dropped because Persian RTL rendering and annotation support caused repeated compatibility problems. The project now uses two small Pillow/OpenCV scripts.

### Detection

`generate_detection.py` places 1 to 10 phrases on a background with random fonts, shadows, rotation, perspective, and occasional occlusion. It transforms the polygon with the image, so the label remains aligned.

PaddleOCR label format:

```text
images/image_0000.jpg\t[{"transcription":"متن فارسی","points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}]
```

Current test set: 100 images, split into 90 training and 10 validation samples under `data/datasets/detection`.

### Recognition

`generate_recognition.py` creates one phrase crop per image. It varies the font, background, contrast, brightness, blur, angle, and JPEG quality. It removes unsupported control characters, replaces Persian digits with Latin digits for a fair V3/V5 comparison, and limits labels to 25 characters.

PaddleOCR label format:

```text
images/image_00000.jpg\tمتن فارسی
```

Current test set: 100 images, split into 90 training and 10 validation samples under `data/datasets/recognition`.

## Detection experiment

The smoke test ran in Colab on a Tesla T4 with PaddlePaddle 3.3.0. Each model used the same small validation set. These numbers prove that the training path works, but 10 validation images are too few for a final model decision.

| Model | Precision before | Recall before | Hmean before | Precision after | Recall after | Hmean after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PP-OCRv3 mobile det | 0.75 | 0.59 | 0.66 | 0.90 | 0.83 | 0.86 |
| PP-OCRv5 mobile det | 0.61 | 0.64 | 0.62 | 0.86 | 0.88 | 0.87 |
| PP-OCRv5 server det | 0.45 | 0.65 | 0.53 | 0.90 | 0.95 | 0.92 |

Precision measures how many predicted boxes are correct. Recall measures how many real boxes the model found. Hmean is the F1 score, the harmonic mean of precision and recall.

The server model has the best current Hmean. The V5 mobile model is close and much smaller, so speed and final accuracy should decide between them after testing on more images. The server run also used a smaller batch, which gave it more optimizer updates per epoch, so this comparison is not yet fully controlled.

## Recognition plan

Use a new Colab notebook and repeat the same baseline, fine-tune, and post-training evaluation flow. Test these models in increasing size:

1. `arabic_PP-OCRv5_mobile_rec`
2. `rec_arabic_PP-OCRv3`
3. `PP-OCRv5_server_rec`

The Arabic mobile checkpoints are the relevant pretrained models for Persian script. The generic V5 server recognition model does not provide the same direct Persian/Arabic starting point, so it is a comparison model rather than the default choice.

Recognition metrics are exact text accuracy, `acc`, and normalized edit similarity, `norm_edit_dis`.

## Repository map

- `Roadmap of the compitision.pdf`: original competition requirements
- `preproccessing-corpus.ipynb`: corpus cleaning and 3-gram selection
- `generate_detection.py`: detection image and polygon generator
- `generate_recognition.py`: recognition crop generator
- `detection-training.ipynb`: Colab detection comparison
- `data/raw`: corpus, fonts, and backgrounds
- `data/processed`: selected 3-gram CSV files
- `data/datasets`: PaddleOCR-ready test datasets
- `research/modern_persian_ocr_tools.md`: tool research and rejected alternatives

## Next work

1. Remove the stray Persian text line near the top of `generate_detection.py`; it currently stops a fresh run.
2. Create a recognition training notebook.
3. Evaluate every recognition model before and after training with `acc` and `norm_edit_dis`.
4. Regenerate larger datasets after the pipeline is stable.
5. Add a manually checked real-photo validation set. Synthetic validation alone will overestimate real-world quality.
6. Select one detection model and one recognition model, then build the FastAPI pipeline.
