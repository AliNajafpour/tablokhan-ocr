# Dataset research for Persian scene-text OCR

## What this project needs

This repository builds a two-stage PaddleOCR system for Persian text in natural photographs. Detection predicts four-point text polygons. Recognition reads cropped Persian lines or phrases, currently capped at 25 characters. The synthetic generator already varies backgrounds, rotation, perspective, blur, contrast, JPEG quality, shadows, and occlusion.

The evaluation examples are harder than ordinary printed-document OCR. They contain sparse and dense Persian text, tiny characters, low contrast, object overlap, mixed sizes, and cluttered photographic backgrounds. A useful dataset must therefore contribute either real Arabic-script scene text, Persian glyph and language coverage for recognition, or script-neutral scene geometry for detector pretraining.

## Short answer on dataset size

`10k` detection images and `500k` recognition crops are reasonable final-generation targets, but sample count is not the main risk.

- For detection, use roughly **10k to 20k synthetic full images**, plus every usable real Persian/Arabic scene image below. Hold out at least **500 manually checked real images** that never enter training. Ten thousand nearly identical synthetic layouts will not beat two thousand varied real photographs.
- For recognition, **300k to 500k synthetic crops** is a strong first final run. Add all available real Persian crops, especially PTDR. Generate by coverage rather than random repetition: every character, Persian digit, punctuation mark, joining form, font family, length bucket, contrast bucket, blur level, and background type needs enough examples.
- Train in stages: broad synthetic pretraining, then a mixed real plus hard-synthetic stage, then a short real-only fine-tune. Plot validation error against 50k, 100k, 250k, and 500k samples. Stop increasing data when the real validation curve flattens.

These sizes assume transfer learning from PaddleOCR's Arabic-script checkpoints. Training from scratch would need much more data and is unnecessary.

## Ranked datasets

| Dataset | Task and script | Size | Annotation | License or access | Fit and recommendation |
| --- | --- | ---: | --- | --- | --- |
| [PTDR](https://github.com/zobeirraisi/PTDR) | Detection and recognition; Persian; real scene and document images plus synthetic recognition | 2,197 real full images, split 1,651 train and 546 test; 18,424 train and 5,882 test word crops; more than 200k synthetic crops | Detection is released in ICDAR text, COCO JSON, YOLO rectangle, and rotated-box formats; recognition is cropped image plus transcription | Public GitHub and Dropbox download; the repository does not state a dataset license, so confirm reuse terms with the authors | **Highest priority.** It is the closest public match to the target language and task. Keep its official test split untouched. Convert its quadrilaterals to PaddleOCR labels and use its real crops for the final recognition stage. Dataset counts and formats come from the [paper](https://link.springer.com/article/10.1007/s42979-025-04059-3). |
| [Persian Scene Text Recognition Dataset](https://github.com/zekavat-ITRC/Persian-scene-text-recognition-Dataset) | Synthetic Persian detection, recognition, and end-to-end OCR | The project provides separate train and validation archives, but does not publish counts in its README | Scene images have word bounding boxes and transcriptions; recognition uses cropped words and a CSV-like `gt.txt` | Public downloads; no explicit dataset license in the repository | **High priority after inspection.** It directly matches Persian scene OCR and its generator includes 384 Persian fonts. Check label quality, duplicate backgrounds, and archive terms before merging. |
| [Persian OCR Garshasp](https://huggingface.co/datasets/AliShafiee2003/persian-ocr-garshasp-70c) | Synthetic Persian line recognition | About 2.6M RGB image-text pairs across 14 splits; 48 by 640 pixels; labels up to 70 characters; 13 rendering and degradation styles | Hugging Face rows with `image`, `text`, and `style` | CC BY 4.0 | **Best large Persian recognition source.** Sample 300k to 500k rather than blindly using all 2.6M. Match the project's 25-character limit and oversample hard styles. Do not use for detector training because it has no scene polygons. |
| [Arshasb](https://github.com/persiandataset/Arshasb) | Persian document detection and recognition | 33k generated pages; 7k pages are free; 40,911 unique words in the 7k release | Per-page image, full text, word and line spreadsheets, with four points for each word | Repository is MIT; 7k download is public, while the full 33k set is paid. Confirm that the MIT notice covers the dataset itself | **Useful secondary source.** Excellent for Persian word and line coverage and dense layouts, but only one reported font and a document-like domain. Use a modest share so it does not dominate natural-scene appearance. |
| [Shotor](https://github.com/amirabbasasadi/Shotor) | Synthetic Persian word recognition | 120,000 grayscale images at 50 by 100 pixels; alphabetic words only | Image plus corresponding word CSV | Free public repository; no explicit license file or dataset terms | **Useful warm-up only.** It adds Persian vocabulary and fonts, but has no digits, punctuation, color, or real backgrounds. Garshasp and the project's generator are stronger final-stage sources. |
| [EvArEST](https://github.com/HGamal11/EvArEST-dataset-for-Arabic-scene-text) | Arabic-English scene detection and recognition; synthetic Arabic recognition | 510 real scene images; 7,232 real cropped words; about 200k synthetic images | Four-point word polygons with language tags; crop filename plus transcription; synthetic segmentation maps | BSD-3-Clause repository | **Highest-priority Arabic source.** The detector data closely matches the project's polygon format. Arabic is not Persian, so normalize shared characters but do not let Arabic-only characters or spelling statistics replace Persian data. |
| [ARASTI](https://doi.org/10.1109/ASAR.2017.8067776) | Real Arabic scene recognition | 1,687 scene-text images derived from 374 natural scenes, including 1,280 cropped words and 2,093 cropped characters | Scene images plus manually segmented word and character images | Paper describes it as public; no clear license was found. Access may require contacting the authors | **Small but valuable real benchmark.** Use shared Arabic-script samples for robustness or evaluation. It is too small to anchor training. |
| [ICDAR 2019 MLT](https://rrc.cvc.uab.es/?ch=15&com=tasks) | Multilingual real and synthetic scene detection, script identification, recognition, and end-to-end OCR; includes Arabic | 10k train, 2k validation, and 10k test real images are reported by the challenge paper; synthetic data covers seven scripts | Word-level quadrilaterals, transcription, and script or language labels | Registration/download through the Robust Reading Competition site; dataset license is not stated on the task page | **Best general multilingual addition.** Select the Arabic subset for both stages and use the remaining scripts mainly for detector geometry. Preserve the official validation/test sets. |
| [SynthText in the Wild](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) | Synthetic English scene-text detection and recognition | 800k images with about 8M word instances | Text string plus word-level and character-level boxes in MATLAB metadata | Oxford terms allow non-commercial research for 12 months and require respecting source-image terms; official download links are currently unavailable | **Optional detector pretraining.** Its scale and scene placement help geometry, but it does not teach Persian recognition. The restrictive terms and dead official download make newer open datasets easier to use. |
| [TextOCR](https://textvqa.org/textocr/dataset/) | Real arbitrary-shaped scene detection and recognition; mostly English | 21,778 train, 3,124 validation, and 3,232 test images; 822,572 public train and validation word annotations | COCO-Text-like JSON with polygon `points`, horizontal box, transcription, and legibility handling | CC BY 4.0; images come from Open Images and retain their source terms | **Strong detector pretraining.** It contributes dense, small, irregular, low-quality text similar to the evaluation examples. Ignore its English labels during Persian recognizer training. |
| [HierText](https://github.com/google-research-datasets/hiertext) | Real word, line, and paragraph detection plus end-to-end OCR; mostly Latin | 11,639 images and about 1.2M words; 8,281 train, 1,724 validation, 1,634 test | JSONL with word polygons and line and paragraph hierarchy | CC BY-SA 4.0 | **Very useful for line detection and reading-order structure.** It is unusually dense, averaging over 100 words per image. Use for detector pretraining, then fine-tune on Persian layouts. Share-alike implications should be reviewed before product use. |
| [COCO-Text v2](https://bgshih.github.io/cocotext/) | Real scene-text detection and recognition; English versus non-English tags | 63,686 images and 239,506 text instances | Word masks, boxes, transcription, legibility, printed or handwritten, and English or non-English attributes | Annotations and images have separate terms through COCO; verify them before redistribution or commercial use | **Useful but lower priority than TextOCR.** Broad background variety helps detection. Text is often incidental and small, which matches several evaluation examples. |
| [Total-Text](https://github.com/cs-chan/Total-Text-Dataset) | Real curved and multi-oriented English word detection and recognition | 1,555 images | Polygon, rectangular box, transcription, and orientation labels; refined train polygons use 10 vertices | BSD-3-Clause repository; authors request contact for commercial use | **Small geometry supplement.** Use if curved or strongly rotated Persian text matters. Its English transcriptions should not train the final recognizer. |

## What not to prioritize

- Persian handwriting, license-plate-only, isolated-character, and clean scanned-document datasets do not match the evaluation domain. They can initialize glyph features, but should not consume much of the final training mix.
- English recognition corpora such as MJSynth do not solve Persian shaping or vocabulary. The Arabic-script PaddleOCR checkpoint plus Persian synthetic data is the shorter route.
- Private datasets such as EASTR-42K cannot support a reproducible final pipeline. Public alternatives above cover the same role.

## Recommended final mix

### Detection

1. Initialize from the chosen PaddleOCR detector.
2. Pretrain or mix with 5k to 20k images sampled from TextOCR, HierText, MLT Arabic, EvArEST, and optionally Total-Text. Detection is largely script-independent.
3. Fine-tune on 10k to 20k project-generated Persian scenes plus PTDR real train images and other usable Persian real data.
4. Make 25 to 40 percent of synthetic images deliberately hard: tiny text, very low contrast, text crossing objects, partial clipping, dense scattered blocks, and pale text on pale regions. These are visible in the prior evaluation set and are underrepresented by the current generator.

### Recognition

1. Start from `arabic_PP-OCRv5_mobile_rec` or the best Arabic-script checkpoint found in the controlled comparison.
2. Train on 300k to 500k sampled Persian crops from Garshasp and the project generator. Add a smaller EvArEST Arabic share for real-scene degradation.
3. Fine-tune on PTDR real Persian crops and manually verified crops from the project's own photographs.
4. Keep character frequencies controlled. Include Persian `پ چ ژ گ ک ی`, both Persian and Latin digits if the test can contain both, ZWNJ variants, punctuation, and mixed Persian-Latin strings.

## How to know the models are good enough

Do not judge on synthetic validation alone. Create one frozen, manually transcribed real-photo test set that includes the 13 known evaluation-style images and new images from the same failure modes. No crop, source photo, near duplicate, background, or text string from this set may appear in training.

Report three layers of results:

- Detection: precision, recall, and Hmean at the competition's polygon IoU rule, plus recall by text-size, contrast, orientation, and occlusion bucket.
- Recognition on ground-truth crops: exact-line accuracy, character error rate, normalized edit distance, and accuracy by length and degradation bucket.
- End to end: a prediction is correct only when the polygon matches and the transcription is correct. Also inspect reading order from top to bottom and right to left.

Set the pass threshold from the competition or last evaluation score if available. Without that number, a practical internal gate is at least 0.90 detection Hmean and 0.90 normalized recognition similarity on the frozen real set, with no severe bucket below 0.80. Those are engineering gates, not claims that the model will win. The last decision should come from a blind rehearsal using images none of the data-generation or model-selection work has seen.

## Licensing note

Repository code licenses do not always license hosted images or labels. Before final training or redistribution, record the exact license and source URL for each downloaded archive. For PTDR, the Persian scene generator, Shotor, ARASTI, and MLT, ask the dataset owner if commercial or competition use is not explicitly granted.
