import csv
import json
import random
from pathlib import Path

import arabic_reshaper
import cv2
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent.parent


IMAGE_COUNT = 100
IMAGE_SIZE = (1280, 960)
TEXTS_PER_IMAGE = (1, 10)
FONT_SIZE = (32, 64)
MAX_ROTATION = 12

TEXT_FILE = ROOT / "data/processed/selected_3grams_1k_clean.csv"
OUTPUT_DIR = ROOT / "data/datasets/detection"
BACKGROUND_DIR = ROOT / "Synthdog-RTL/resources/background"
FONT_DIR = ROOT / "Synthdog-RTL/resources/font/pr"

with open(TEXT_FILE, encoding="utf-8-sig") as f:
    texts = [row["3-gram"] for row in csv.DictReader(f)]

backgrounds = [p for p in BACKGROUND_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
fonts = list(FONT_DIR.glob("*.ttf"))
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
reshaper = arabic_reshaper.ArabicReshaper({"support_ligatures": False})
label_lines = []

for number in range(IMAGE_COUNT):
    image = ImageOps.fit(Image.open(random.choice(backgrounds)).convert("RGB"), IMAGE_SIZE)
    draw = ImageDraw.Draw(image)
    chosen = random.sample(texts, random.randint(*TEXTS_PER_IMAGE))
    annotations = []

    for row, text in enumerate(chosen):
        font = ImageFont.truetype(str(random.choice(fonts)), random.randint(*FONT_SIZE))
        shown = get_display(reshaper.reshape(text))
        box = draw.textbbox((0, 0), shown, font=font, stroke_width=1)
        width, height = box[2] - box[0], box[3] - box[1]
        pad = 20
        patch = Image.new("RGBA", (width + 2 * pad, height + 2 * pad))
        patch_draw = ImageDraw.Draw(patch)
        position = pad - box[0], pad - box[1]
        white = random.choice([True, False])
        patch_draw.text((position[0] + 4, position[1] + 4), shown, font=font, fill=(0, 0, 0, 100))
        patch_draw.text(position, shown, font=font, fill="white" if white else "black",
                        stroke_width=1, stroke_fill="black" if white else "white")

        text_box = np.float32([[pad, pad], [pad + width, pad],
                               [pad + width, pad + height], [pad, pad + height]])

        patch_width, patch_height = patch.size
        source = np.float32([[0, 0], [patch_width - 1, 0],
                             [patch_width - 1, patch_height - 1], [0, patch_height - 1]])
        angle = np.deg2rad(random.randint(-MAX_ROTATION, MAX_ROTATION))
        rotation = np.float32([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
        target = (source - source.mean(0)) @ rotation.T + source.mean(0)
        target += np.float32([[random.randint(-8, 8), random.randint(-8, 8)] for _ in range(4)])
        target -= target.min(0)
        output_width, output_height = np.ceil(target.max(0)).astype(int) + 1
        matrix = cv2.getPerspectiveTransform(source, target)
        warped = cv2.warpPerspective(np.array(patch), matrix, (output_width, output_height))
        warped = Image.fromarray(warped)
        x = random.randint(10, IMAGE_SIZE[0] - output_width - 10)
        y = round((row + 0.5) * IMAGE_SIZE[1] / len(chosen) - output_height / 2)
        y = max(10, min(y, IMAGE_SIZE[1] - output_height - 10))
        image.paste(warped, (x, y), warped)

        transformed = cv2.perspectiveTransform(text_box.reshape(-1, 1, 2), matrix).reshape(-1, 2) + [x, y]
        points = [[round(px), round(py)] for px, py in transformed]
        annotations.append({"transcription": text, "points": points})

        if random.random() < 0.3:
            left, right = min(p[0] for p in points), max(p[0] for p in points)
            middle = round(sum(p[1] for p in points) / 4)
            ImageDraw.Draw(image).line((left, middle, right, middle + random.randint(-5, 5)),
                                       fill=random.choice(["black", "white"]), width=random.randint(2, 6))

    name = f"image_{number:04}"
    assert all(0 <= x <= IMAGE_SIZE[0] and 0 <= y <= IMAGE_SIZE[1]
               for item in annotations for x, y in item["points"])
    image.save(OUTPUT_DIR / "images" / f"{name}.jpg", quality=90)
    label_lines.append(f"images/{name}.jpg\t{json.dumps(annotations, ensure_ascii=False)}")

(OUTPUT_DIR / "labels.txt").write_text("\n".join(label_lines), encoding="utf-8")
print(f"Created {IMAGE_COUNT} images in {OUTPUT_DIR}")
