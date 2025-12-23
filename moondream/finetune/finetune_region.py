import json
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
from datasets import load_dataset
from safetensors.torch import save_file

from PIL import Image
from tqdm import tqdm
from bitsandbytes.optim import AdamW8bit
import wandb
import random

from ..torch.weights import load_weights_into_model
from ..torch.moondream import MoondreamModel, MoondreamConfig, text_encoder
from ..torch.text import _produce_hidden
from ..torch.region import (
    decode_coordinate,
    decode_size,
    encode_coordinate,
    encode_size,
)


# This is a intended to be a basic starting point. Your optimal hyperparams and data may be different.
MODEL_PATH = "/scratch/data/asif_rs/mooondream_models/model_25_06_21.safetensors"
LR = 5e-6
EPOCHS = 1
GRAD_ACCUM_STEPS = 128

random.seed(111)


def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


def region_loss(
    hidden_states: torch.Tensor,
    w,
    labels: torch.Tensor,
    c_idx: torch.Tensor,
    s_idx: torch.Tensor,
):
    l_idx = torch.arange(len(labels))

    c_idx = c_idx - 1
    c_hidden = hidden_states[:, c_idx, :]
    c_logits = decode_coordinate(c_hidden, w)
    c_labels = labels[(l_idx % 4) < 2]

    c_loss = F.cross_entropy(
        c_logits.view(-1, c_logits.size(-1)),
        c_labels,
    )

    s_idx = s_idx - 1
    s_hidden = hidden_states[:, s_idx, :]
    s_logits = decode_size(s_hidden, w).view(-1, 1024)
    s_labels = labels[(l_idx % 4) >= 2]
    
    s_loss = F.cross_entropy(s_logits, s_labels)
    
    return c_loss + s_loss


class CocoHFDataset(Dataset):
    """
    COCO-style Dataset backed by HuggingFace datasets.
    Outputs YOLO-normalized boxes + integer labels.
    """

    def __init__(self, hf_dataset, transform=None):
        """
        hf_dataset: HuggingFace Dataset split (train / val)
        """
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        image = example["image"].convert("RGB")
        img_w, img_h = image.size

        objects = example["objects"]
        bboxes = objects["bbox"]      # [[x,y,w,h], ...]
        labels = objects["label"]     # [class_id, ...]

        boxes_out = []
        labels_out = []

        for bbox, label in zip(bboxes, labels):
            x, y, w, h = bbox

            # ---- basic bbox validation ----
            if not all(map(math.isfinite, [x, y, w, h])):
                continue
            if w <= 0 or h <= 0:
                continue

            # ---- COCO -> YOLO (normalized) ----
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            bw = w / img_w
            bh = h / img_h

            # Clamp to [0, 1] for safety
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)

            boxes_out.append([cx, cy, bw, bh])
            labels_out.append(label)

        boxes = torch.tensor(boxes_out, dtype=torch.bfloat16)
        # labels = torch.tensor(labels_out, dtype=torch.int64)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "boxes": boxes,        # (N, 4) YOLO format
            "labels": labels,      # (N,) integer class IDs
            "image_id": example['image_id'],
        }


def main():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    wandb.init(
        project="moondream-ft",
        config={
            "EPOCHS": EPOCHS,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
        },
    )

    config = MoondreamConfig()
    model = MoondreamModel(config)
    model.compile()
    load_weights_into_model(MODEL_PATH, model)

    optimizer = AdamW8bit(
        [{"params": model.region.parameters()}],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    hf_dataset = load_dataset(
        'rafaelpadilla/coco2017',
        split='train'
    )

    dataset = CocoHFDataset(
        hf_dataset=hf_dataset
    )

    total_steps = EPOCHS * len(dataset) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)

    i = 0
    for epoch in range(EPOCHS):
        for sample in dataset:
            i += 1

            with torch.no_grad():
                img_emb = model._run_vision_encoder(sample["image"])
                bos_emb = text_encoder(
                    torch.tensor(
                        [[model.config.tokenizer.bos_id]], device=model.device
                    ),
                    model.text,
                )
                eos_emb = text_encoder(
                    torch.tensor(
                        [[model.config.tokenizer.eos_id]], device=model.device
                    ),
                    model.text,
                )

            boxes_by_class = {}
            for box, cls in zip(sample["boxes"], sample["labels"]):
                boxes_by_class.setdefault(cls, []).append(box)

            total_loss = 0.0
            if len(boxes_by_class) == 0:
                continue
            # print(len(boxes_by_class))
            # print(sample['labels'])
            for class_name, boxes_list in boxes_by_class.items():
                with torch.no_grad():
                    instruction = f"\n\nDetect: {class_name}\n\n"
                    instruction_tokens = model.tokenizer.encode(instruction).ids
                    instruction_emb = text_encoder(
                        torch.tensor([[instruction_tokens]], device=model.device),
                        model.text,
                    ).squeeze(0)

                cs_emb = []
                cs_labels = []
                c_idx = []
                s_idx = []
                for bb in boxes_list:
                    l_cs = len(cs_emb)
                    cs_emb.extend(
                        [
                            encode_coordinate(bb[0].unsqueeze(0), model.region),
                            encode_coordinate(bb[1].unsqueeze(0), model.region),
                            encode_size(bb[2:4], model.region),
                        ]
                    )
                    c_idx.extend([l_cs, l_cs + 1])
                    s_idx.append(l_cs + 2)
                    vals = (bb.clamp(0.0, 1.0)* 1023).round()
                    vals = vals.to(torch.int32)
                    vals = torch.clamp(vals, max=1023)
                    cs_labels.extend(vals.tolist())

                # print(vals)
                # print(len(cs_emb))

                if len(cs_emb) == 0:
                    print('Called')
                    continue

                cs_emb = torch.stack(cs_emb)

                inputs_embeds = torch.cat(
                    [bos_emb, img_emb[None], instruction_emb, cs_emb[None], eos_emb],
                    dim=1,
                )
                prefix = inputs_embeds.size(1) - cs_emb.size(0)
                c_idx = torch.tensor(c_idx) + prefix
                s_idx = torch.tensor(s_idx) + prefix

                hidden = _produce_hidden(
                    inputs_embeds=inputs_embeds, w=model.text, config=config.text
                )

                # print(hidden.shape)
                
                loss = region_loss(
                    hidden_states=hidden,
                    w=model.region,
                    labels=torch.tensor(cs_labels, dtype=torch.int64),
                    c_idx=c_idx,
                    s_idx=s_idx,
                )
                # print(loss)
                total_loss += loss
                # print(total_loss)
            
            # print(total_loss)
            # print(total_loss)
            total_loss.backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr_val = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_val
                pbar.set_postfix(
                    {"step": i // GRAD_ACCUM_STEPS, "loss": total_loss.item()}
                )
                pbar.update(1)
                wandb.log(
                    {
                        "loss/train": total_loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
    wandb.finish()

    # Replace with your desired output location.
    save_file(
        model.state_dict(),
        "/scratch/data/asif_rs/mooondream_models/moondream_finetune.safetensors",
    )


if __name__ == "__main__":
    """
    Replace paths with your appropriate paths.
    To run: python -m moondream.finetune.finetune_region

    1 epoch of fine-tuning on the example 'Waste Detection' dataset results in a
    2 percentage point increase in mAP.
    Dataset: https://universe.roboflow.com/waste-detection-l4m9b/waste-detection-ttdir
    """
    main()
