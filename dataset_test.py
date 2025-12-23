import os
import math
from PIL import ImageDraw
from datasets import load_dataset

# -----------------------------
# Configuration
# -----------------------------
SPLIT = "train"
OUT_DIR = "coco_with_clean_bboxes"
MAX_IMAGES = 10          # None = full dataset
MIN_AREA = 4             # in pixels

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("rafaelpadilla/coco2017", split=SPLIT)
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# BBox cleaning + integer clamping
# -----------------------------
def clean_bboxes_int(bboxes, img_w, img_h, min_area=1):
    """
    Cleans COCO bboxes [x, y, w, h] and clamps to integer pixels.
    """
    cleaned = []

    for x, y, w, h in bboxes:

        # NaN / Inf check
        if not all(map(math.isfinite, [x, y, w, h])):
            continue

        # Positive size
        if w <= 0 or h <= 0:
            continue

        # Convert to corners
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

        # Clamp to image bounds
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        new_w = x2 - x1
        new_h = y2 - y1

        # Validate after clamping
        if new_w <= 0 or new_h <= 0:
            continue

        if new_w * new_h < min_area:
            continue

        cleaned.append([x1, y1, new_w, new_h])

    return cleaned

# -----------------------------
# Main loop
# -----------------------------
for idx, example in enumerate(dataset):
    image = example["image"].copy()
    img_w, img_h = image.size

    raw_bboxes = example["objects"]["bbox"]
    clean_boxes = clean_bboxes_int(raw_bboxes, img_w, img_h, MIN_AREA)

    draw = ImageDraw.Draw(image)

    for x, y, w, h in clean_boxes:
        draw.rectangle(
            [x, y, x + w, y + h],
            outline="red",
            width=3
        )

    out_path = os.path.join(OUT_DIR, f"{example['image_id']}.png")
    image.save(out_path)
    print(f"Saved {out_path} ({len(clean_boxes)} boxes)")

    if MAX_IMAGES is not None and idx + 1 >= MAX_IMAGES:
        break

print("Done.")