# 📦 Stable-Diffusion-Student-Distillation

Light-weight *Stable Diffusion* students distilled from **SD-1.5** with as little as **65 M** trainable parameters.  
The repository contains three progressively more advanced training pipelines:

| Pipeline | Short name | Key idea(s) | Lines |
|----------|------------|-------------|-------|
| `v1/`    | **V-1**    | progressive DDPM schedule + feature-matching loss | ~600 |
| `v2/`    | **V-2.1**  | noise–matching (*ε*) + OneCycleLR + periodic FID/LPIPS | ~412 |
| `v2/`    | **V-2.2**  | V-2.1 ++ KD term, dynamic α(t) & conditional-dropout | ~310 |
| `v3/`    | **V-3**    | Accelerate, mixed-precision, `sd_small` / `sd_tiny` pruning | - |

> **TL;DR** &nbsp;V-2.2 (student @ 64×64) reaches **FID ≈ 310** on MS-COCO val (25 DDPM steps, CFG 5) – a **2× improvement** over our baseline &lt; 10 h on a single RTX 3060 6 GB.

---

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Quick start](#quick-start)
5. [Directory layout](#directory-layout)


---

## Features
* **Teacher → Student weight projection** (`initialize_student_weights`) – copies matching tensors, slices the rest.
* **Progressive/DDPM scheduler** that gradually shortens the diffusion chain.
* **EMA tracker** for smooth student weights.
* **On-the-fly metrics:** FID & LPIPS every *n* steps (GPU version, no image dumping).
* **Condition-free guidance (CFG)** & prompt dropout during training (V-2.2).
* **Prunable U-Net** (`prepare_unet`) producing `sd_small` / `sd_tiny` models.
* Single-GPU friendly – all configs tested on < 8 GB VRAM with `torch.float16`.

---

## Installation
```bash

pip install \
  torch \
  torchvision \
  diffusers \
  transformers \
  accelerate \
  torchmetrics \
  lpips \
  pytorch-fid \
  pycocotools \
  pillow \
  tqdm
```

---

## Dataset
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d data/coco2017/

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d data/coco2017/

---

## Directory layout
v1/  v1.1.py                # progressive distillation + feature loss\
v2/\
 ├── v2.1.py    # 2.1 main script\
 └── v2.2.py    # 2.2 main script\
v3/ # Accelerate high-end pipeline\
├── coco_dataset.py\
├── data.py\
├── inference.py\
├── prepare_unet_small.py\
├── requirements.txt\
└── train_student_kd.py\
               


