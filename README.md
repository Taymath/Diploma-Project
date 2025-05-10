# 📦 Stable-Diffusion-Student-Distillation

Light-weight *Stable Diffusion* students distilled from **SD-1.5** with as little as **65 M** trainable parameters.  
The repository contains three progressively more advanced training pipelines:

| Pipeline | Short name | Key idea(s) | Lines |
|----------|------------|-------------|-------|
| `v1/`    | **V-1**    | progressive DDPM schedule + feature-matching loss | ~600 |
| `v2/`    | **V-2.1**  | noise–matching (*ε*) + OneCycleLR + periodic FID/LPIPS | ~412 |
| `v2/`    | **V-2.2**  | V-2.1 ++ KD term, dynamic α(t) & conditional-dropout | ~310 |
| `v3/`    | **V-3**    | Accelerate, mixed-precision, `sd_small` / `sd_tiny` pruning | - |

> **TL;DR** &nbsp;V-2.2 (student @ 64×64) reaches **FID ≈ 310** on MS-COCO val (25 DDPM steps, CFG 5) – a **2× improvement** over our baseline &lt; 10 h on a single mobile RTX 3060 6 GB.

---

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Directory layout](#directory-layout)


---

## Features

### V-1 (v1/1.py)
- **Progressive DDPM schedule**  
  Gradually shortens the diffusion chain during training.
- **Feature-matching loss**  
  Combines noise-matching (MSE) with intermediate layer-wise distillation via hooks.
- **Adaptive weight projection** (`ConvProjection`)  
  Copies teacher weights into student, slicing/mapping mismatched channels.
- **EMA tracker**  
  Keeps an exponential moving average of student weights for stability.

---

### V-2.1 (v2/2.1.py – “noise-only”)
- **Teacher-to-student slicing**  
  Directly slices teacher UNet tensors to initialize student model.
- **Standard MSE loss on noise**  
  Simplified objective: match student noise prediction to true noise.
- **OneCycleLR scheduler**  
  Cosine-annealed learning‐rate with warm-up.
- **Periodic FID & LPIPS evaluation**  
  On‐the‐fly metrics every N steps (no image dumping).

---

### V-2.2 (v2/2.2.py – “KD + CFG”)
- **Knowledge Distillation term**  
  Adds an MSE loss between student and teacher noise predictions.
- **Dynamic α-schedule**  
  Linearly decays the KD weight α(t) from 1→0.5 through training.
- **Conditional-dropout (CFG mask)**  
  Gradually increases caption dropout rate to regularize guidance.
- **All V-2.1 features inherited**  
  OneCycleLR, teacher slicing, periodic FID/LPIPS.


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
v1/  1.py                # progressive distillation + feature loss\
v2/\
 ├── 2.1.py    # 2.1 main script\
 └── 2.2.py    # 2.2 main script\
v3/ # Accelerate high-end pipeline\
├── coco_dataset.py\
├── data.py\
├── inference.py\
├── prepare_unet_small.py\
├── requirements.txt\
└── train_student_kd.py\
               


