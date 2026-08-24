import csv
import random
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageStat


ROOT = Path(__file__).parent

# Change this, then press Run in PyCharm.
IMAGE_COUNT = 100
FONT_SIZE = (32, 64)
ANGLE = 3
BRIGHTNESS = (0.75, 1.25)

TEXT_FILE = ROOT / "data/processed/selected_3grams_10k_clean.csv"
OUTPUT_DIR = ROOT / "data/datasets/recognition"
FONT_DIR = ROOT / "data/raw/fonts"
BACKGROUND_DIR = ROOT / "data/raw//backgrounds"

with open(TEXT_FILE, encoding="utf-8-sig") as file:
    texts = [row["3-gram"].replace("\u200c", " ").replace("\u200e", "").translate(
        str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")) for row in csv.DictReader(file)]

texts = [" ".join(text.split()) for text in texts if len(text) <= 25]
random.shuffle(texts)
fonts = list(FONT_DIR.glob("*.ttf"))
backgrounds = [p for p in BACKGROUND_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
assert texts and fonts and backgrounds
reshaper = arabic_reshaper.ArabicReshaper({"support_ligatures": False})
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
labels = []

for number in range(IMAGE_COUNT):
    text = texts[number % len(texts)]
    font = ImageFont.truetype(str(random.choice(fonts)), random.randint(*FONT_SIZE))
    shown = get_display(reshaper.reshape(text))
    box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), shown, font=font, stroke_width=1)
    width, height = box[2] - box[0], box[3] - box[1]
    with Image.open(random.choice(backgrounds)) as background:
        image = ImageOps.fit(background.convert("RGB"), (width + 20, height + 20))
    draw = ImageDraw.Draw(image)
    fill = "black" if ImageStat.Stat(image.convert("L")).mean[0] > 128 else "white"
    stroke = random.choice([0, 0, 0, 1])
    draw.text((10 - box[0], 10 - box[1]), shown, font=font, fill=fill,
              stroke_width=stroke, stroke_fill="white" if fill == "black" else "black")
    image = ImageEnhance.Brightness(image).enhance(random.uniform(*BRIGHTNESS))
    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0, 0.7)))
    background_color = tuple(map(int, ImageStat.Stat(image).mean[:3]))
    image = image.rotate(random.uniform(-ANGLE, ANGLE), Image.Resampling.BICUBIC,
                         expand=True, fillcolor=background_color)
    name = f"image_{number:05}.jpg"
    image.save(OUTPUT_DIR / "images" / name, quality=random.randint(75, 95))
    labels.append(f"images/{name}\t{text}")

(OUTPUT_DIR / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
assert len(labels) == IMAGE_COUNT
print(f"Created {len(labels)} recognition images in {OUTPUT_DIR}")
