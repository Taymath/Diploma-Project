#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_student_kd.py

Дополняет исходный pipeline:
 • вводит кастом‑инициализацию,
 • обучает студента на 64×64 (latents 8×8),
 • каждые 500 шагов генерирует sample‑пары teacher/student и мгновенно считает FID,
 • в конце эпохи сохраняет 3 примерных изображения студента,
 • поддержка двух режимов инициализации (random|custom) для честного сравнения.

Расчёт FID/LPIPS ― torchmetrics (GPU) без сохранения
экстра‑графиков, чтобы поместиться на RTX 3060 6 GB.
"""
from prepare_unet_small import slim_sd_small

import os, math, time, shutil, gc, random, argparse
from pathlib import Path
from functools import partial
import prepare_unet_small
import torch, torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from transformers import CLIPFeatureExtractor

from diffusers import (
    StableDiffusionPipeline, UNet2DConditionModel,
    AutoencoderKL, DDPMScheduler,
)
from prepare_unet_small import prepare_unet
from transformers import CLIPTextModel, CLIPTokenizer
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from coco_dataset import get_coco_dataloader

# ------------------------------------------------------------
# 0.  Кастомная инициализация
# ------------------------------------------------------------
def parse_args():
    import argparse, os

    parser = argparse.ArgumentParser(description="Distill student SD model")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace ID или путь к учительской модели",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Ветка/тег/коммит для загрузки модели и feature_extractor",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Путь к папке с изображениями COCO (train2017)",
    )
    parser.add_argument(
        "--caption_file",
        type=str,
        required=True,
        help="Путь к файлу captions_train2017.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="distill_runs",
        help="Куда сохранять чекпоинты и финальную модель",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Число эпох обучения",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Ограничить число примеров из COCO (для отладки)",
    )
    parser.add_argument(
        "--distill_level",
        type=str,
        default="sd_small",
        choices=["sd_small", "sd_tiny"],
        help="Уровень «сжатия» студента",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Ограничить общее число оптимизационных шагов",
    )

    return parser.parse_args()


def initialize_student_weights(teacher_unet_full: UNet2DConditionModel,
                               student_unet: UNet2DConditionModel):
    """
    Проекция/обрезка весов TEACHER‑UNet → STUDENT‑UNet.
    Сохраняет совпадающие тензоры полностью,
    а более «толстые» ‑‑ срезает по каналам.
    """
    teacher_state = teacher_unet_full.state_dict()
    student_state = student_unet.state_dict()
    new_state = {}

    for name, t_w in teacher_state.items():

        if name.startswith("down_blocks.3.") or name.startswith("up_blocks.0."):
            continue          # эти блоки у студента выброшены

        # --- переименование up‑блоков (1→0, 2→1, 3→2)
        if name.startswith("up_blocks"):
            parts = name.split(".")
            idx = int(parts[1])
            if idx == 0:
                continue
            parts[1] = str(idx - 1)
            s_name = ".".join(parts)

        # --- down‑blocks (0,1,2) — индекс совпадает
        elif name.startswith("down_blocks"):
            if int(name.split(".")[1]) >= 3:
                continue
            s_name = name

        else:                 # mid, conv_in/out, time_embed
            s_name = name

        if s_name not in student_state:
            continue

        s_w = student_state[s_name]
        if s_w.shape == t_w.shape:
            new_state[s_name] = t_w.clone()
        else:
            if t_w.ndim == 4:           # Conv [out, in, k, k]
                new_state[s_name] = t_w[: s_w.shape[0], : s_w.shape[1]].clone()
            elif t_w.ndim == 2:         # Linear
                new_state[s_name] = t_w[: s_w.shape[0], : s_w.shape[1]].clone()
            else:                       # Bias / Norm
                new_state[s_name] = t_w[: s_w.shape[0]].clone()

    student_unet.load_state_dict(new_state, strict=False)
    return student_unet


# ------------------------------------------------------------
# 1.  Генерация и FID‑/LPIPS‑метрики
# ------------------------------------------------------------
@torch.no_grad()
def _generate(pipeline: StableDiffusionPipeline,
              prompt: str,
              num_images: int,
              generator: torch.Generator):
    imgs = pipeline(
        prompt,
        negative_prompt=None,
        num_inference_steps=20,
        guidance_scale=7.,
        num_images_per_prompt=num_images,
        height=512, width=512,
        generator=generator,
    ).images
    return imgs


@torch.no_grad()
def compute_metrics(student_pipe, teacher_pipe, prompts, n_img=8, device="cuda"):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    lpips = LPIPS(net_type='alex').to(device)

    for p in prompts:
        gen = torch.Generator(device).manual_seed(42)
        st_imgs = _generate(student_pipe, p, n_img, gen)
        te_imgs = _generate(teacher_pipe, p, n_img, gen)

        st_t = torch.stack([transforms.ToTensor()(i) for i in st_imgs]).to(device)
        te_t = torch.stack([transforms.ToTensor()(i) for i in te_imgs]).to(device)

        fid.update(st_t, real=False)
        fid.update(te_t, real=True)

        lpips.update(st_t, te_t)

    return fid.compute().item(), lpips.compute().item()


# ------------------------------------------------------------
# 2.  Основная тренировочная функция
# ------------------------------------------------------------
def train_student(args, init_mode="custom"):
    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=args.output_dir),
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
    )

    teacher_pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=None,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    teacher_pipe.to(accelerator.device)
    teacher_unet = teacher_pipe.unet.eval().requires_grad_(False)

    student_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    prepare_unet(student_unet, model_type=args.distill_level)

    if init_mode == "custom":
        initialize_student_weights(teacher_unet, student_unet)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    from coco_dataset import get_coco_dataloader   # tiny helper ➜ вернёт DataLoader
    train_loader = get_coco_dataloader(args, tokenizer)

    opt = torch.optim.AdamW(student_unet.parameters(), lr=1e-4)

    student_unet, opt, train_loader = accelerator.prepare(
        student_unet, opt, train_loader
    )

    if accelerator.mixed_precision == "fp16":
        vae.to(accelerator.device)
        vae.half()
        text_encoder.to(accelerator.device)
        text_encoder.half()

    scaler = GradScaler(enabled=accelerator.mixed_precision == "fp16")

    fid_prompts = ["a photo of a cat", "a city skyline at sunset"]

    from tqdm.auto import tqdm
    import torch.nn.functional as F
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="feature_extractor",
        revision=args.revision,
    )
    global_step = 0
    for epoch in range(args.epochs):
        student_unet.train()
        pbar = tqdm(train_loader, desc=f"[{init_mode}] Epoch {epoch}", disable=not accelerator.is_local_main_process)
        for batch in pbar:
            with accelerator.accumulate(student_unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
                input_ids = batch["input_ids"].to(accelerator.device)

                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(input_ids)[0]

                with torch.no_grad():
                    teacher_pred = teacher_unet(noisy_latents, timesteps, encoder_hidden_states).sample
                student_pred = student_unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss_kd = F.mse_loss(student_pred.float(), teacher_pred.float())
                loss_task = F.mse_loss(student_pred.float(), noise.float())
                loss = 0.5 * loss_task + 0.5 * loss_kd

                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if args.max_train_steps and global_step >= args.max_train_steps:
                    done = True
                    break

                pbar.update(1)
                pbar.set_postfix({"loss": loss.item(), "step": global_step})


                if global_step % 500 == 0 and accelerator.is_main_process:
                    student_pipe = StableDiffusionPipeline(
                        vae=vae, text_encoder=text_encoder,
                        tokenizer=tokenizer, unet=accelerator.unwrap_model(student_unet),
                        scheduler=teacher_pipe.scheduler,
                        safety_checker=None, feature_extractor = feature_extractor,

                    ).to(accelerator.device, dtype=torch.float16)

                    fid, lp = compute_metrics(student_pipe, teacher_pipe,
                                              fid_prompts, n_img=4, device=accelerator.device)
                    accelerator.print(f"[{init_mode}] step {global_step}: FID={fid:.2f}, LPIPS={lp:.3f}")

                    del student_pipe
                    torch.cuda.empty_cache()

                if global_step % 1000 == 0 and accelerator.is_main_process:
                    save_dir = Path(args.output_dir, f"{init_mode}_ckpt_{global_step}")
                    accelerator.save_state(save_dir)

        if done: break

        if accelerator.is_main_process:
            student_pipe = StableDiffusionPipeline(
                vae=vae, text_encoder=text_encoder,
                tokenizer=tokenizer, unet=accelerator.unwrap_model(student_unet),
                scheduler=teacher_pipe.scheduler, safety_checker=None,
                feature_extractor=feature_extractor,
            ).to(accelerator.device, dtype=torch.float16)

            rng = torch.Generator(accelerator.device).manual_seed(1234)
            out_imgs = _generate(student_pipe, "highly detailed photo", 3, rng)
            for i, im in enumerate(out_imgs):
                im.save(Path(args.output_dir, f"{init_mode}_epoch{epoch}_{i}.png"))

            del student_pipe
            torch.cuda.empty_cache()

    return accelerator.unwrap_model(student_unet)


# ------------------------------------------------------------
# 3.  Запуск двух инициализаций и итоговое сравнение
# ------------------------------------------------------------
def evaluate_inits(args):
    log = get_logger("distill")
    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=args.output_dir),
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
    )

    log.info(">> RANDOM init")
    rnd_student = train_student(args, init_mode="random")
    torch.cuda.empty_cache() ; gc.collect()

    log.info(">> CUSTOM init")
    custom_student = train_student(args, init_mode="custom")
    torch.cuda.empty_cache() ; gc.collect()

    teacher_pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=torch.float16,
        safety_checker=None).to(accelerator.device)

    tokenizer = teacher_pipe.tokenizer
    text_encoder = teacher_pipe.text_encoder
    vae = teacher_pipe.vae

    def make_pipe(unet):
        return StableDiffusionPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=teacher_pipe.scheduler,
            safety_checker=None).to(accelerator.device, dtype=torch.float16)

    rnd_pipe = make_pipe(rnd_student)
    cus_pipe = make_pipe(custom_student)

    prompts = ["a photo of a dog", "an ancient temple in the jungle"]
    fid_rnd, _  = compute_metrics(rnd_pipe, teacher_pipe, prompts, n_img=8, device=accelerator.device)
    fid_cus, _  = compute_metrics(cus_pipe, teacher_pipe, prompts, n_img=8, device=accelerator.device)

    log.info(f"FINAL  FID random={fid_rnd:.2f}   FID custom={fid_cus:.2f}")
    if fid_cus < fid_rnd:
        log.info("🎉  Custom init превосходит случайную!")

    if accelerator.is_main_process:
        Path(args.output_dir, "final").mkdir(parents=True, exist_ok=True)
        cus_pipe.save_pretrained(Path(args.output_dir, "final"))


# ------------------------------------------------------------
# 4.  CLI
# ------------------------------------------------------------
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_inits(args)



if __name__ == "__main__":
    main()
