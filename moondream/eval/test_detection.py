from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import ImageDraw
import os
import datasets



coco_classes = [
    "None",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "hair brush",
]

COCO_LABELS = {}

for i, c in enumerate(coco_classes):
    COCO_LABELS[i] = c



def draw_and_save_bboxes(image, pred_boxes, gt_boxes, save_path, label_name):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = image.copy()
    draw = ImageDraw.Draw(img)

    # ---- Ground Truth (GREEN) ----
    for (x1, y1, x2, y2) in gt_boxes:
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="green",
            width=3
        )
        draw.text((x1 + 2, y1 + 2), f"GT: {label_name}", fill="green")

    # ---- Predictions (RED) ----
    for (x1, y1, x2, y2, conf) in pred_boxes:
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=3
        )
        draw.text((x1 + 2, y1 + 16), f"Pred: {label_name}", fill="red")

    img.save(save_path)



def eval_coco_map(model, iou_threshold=0.5, debug=False):
    dataset = datasets.load_dataset(
        "rafaelpadilla/coco2017", split="val[:10]"
    )

    total = 0
    results_by_label = {}
    frequency_by_label = {}

    for row in tqdm(dataset, disable=debug, desc="COCO mAP"):
        width = row["image"].width
        height = row["image"].height
        total += 1

        objects = row["objects"]

        gt_label_to_boxes = {}

        for bbox, label in zip(objects["bbox"], objects["label"]):
            if label not in gt_label_to_boxes:
                gt_label_to_boxes[label] = []
            x1, y1, w, h = bbox
            gt_label_to_boxes[label].append((x1, y1, x1 + w, y1 + h))

        unique_labels = [label for label in set(objects["label"])]

        encoded_image = model.encode_image(row["image"])
            
        for label in unique_labels:

            model_answer = model.detect(encoded_image, COCO_LABELS[label])["objects"]

            moondream_boxes = []

            for box in model_answer:
                moondream_boxes.append(
                    (
                        box["x_min"] * width,
                        box["y_min"] * height,
                        box["x_max"] * width,
                        box["y_max"] * height,
                        1.0,
                    )
                )

            # 🔴 ONLY ADDITION: draw + save
            draw_and_save_bboxes(
                image=row["image"],
                pred_boxes=moondream_boxes,
                gt_boxes=gt_label_to_boxes[label],
                save_path=f"moondream_preds/img_{total}_label_{label}.jpg",
                label_name=COCO_LABELS[label],
            )

if __name__ == "__main__":

    cfg = MoondreamConfig()
    model = MoondreamModel(cfg)
    load_weights_into_model(
        '/scratch/data/asif_rs/mooondream_models/model_25_06_21.safetensors', 
        model
    )
    model = model.to('cuda')
    # model = AutoModelForCausalLM.from_pretrained(
    #     "vikhyatk/moondream2",
    #     revision="2025-06-21",
    #     trust_remote_code=True,
    #     device_map={"": "cuda"}  # ...or 'mps', on Apple Silicon
    # )

    eval_coco_map(model=model)
