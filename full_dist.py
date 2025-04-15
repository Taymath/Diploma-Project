import os, random, math, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as T

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel

from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from pytorch_fid import fid_score

# Установка устройства и dtype (учитель работает в autocast float16; студент – float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

# =========================================
# =========== DATASET CLASS ===============
# =========================================
class MSCOCODataset(Dataset):
    def __init__(self, img_dir, ann_file, image_size=64, max_samples=None):
        super().__init__()
        self.img_dir = img_dir
        self.coco = COCO(annotation_file=ann_file)
        self.image_size = image_size
        self.img_ids = list(self.coco.imgs.keys())
        if max_samples is not None:
            random.shuffle(self.img_ids)
            self.img_ids = self.img_ids[:max_samples]
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])  # [0,1] -> [-1,1]
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_info = self.coco.loadImgs(img_id)[0]
        file_name = ann_info['file_name']
        path = os.path.join(self.img_dir, file_name)
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = anns[0]['caption'] if len(anns) > 0 else ""
        return image, caption


# =========================================
# =========== EMA TRACKER =================
# =========================================
class EMATracker:
    def __init__(self, model, decay=0.9999, device='cpu'):
        self.decay = decay
        self.shadow = {}
        self.device = device
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().float().to(device)

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                cpu_param = p.detach().clone().float().cpu()
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * cpu_param

    @torch.no_grad()
    def apply_to(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.copy_(self.shadow[name].to(p.device).to(p.dtype))

    @torch.no_grad()
    def restore(self, model):
        pass


# =========================================
# =========== MAKE STUDENT UNET ===========
# =========================================
def make_student_unet(teacher_config):
    if hasattr(teacher_config, "to_dict"):
        teacher_config = dict(teacher_config)
    teacher_config.pop("num_attention_heads", None)
    block_out_channels = [max(1, ch // 4) for ch in teacher_config["block_out_channels"]]
    layers_per_block = max(1, teacher_config["layers_per_block"] - 1)
    attn_dim = teacher_config["attention_head_dim"]
    if isinstance(attn_dim, list):
        attn_dim = [max(1, x // 2) for x in attn_dim]
    else:
        attn_dim = max(1, attn_dim // 2)
    teacher_config["cross_attention_dim"] = 768

    student = UNet2DConditionModel(
        sample_size=64,
        in_channels=teacher_config["in_channels"],
        out_channels=teacher_config["out_channels"],
        layers_per_block=layers_per_block,
        cross_attention_dim=teacher_config["cross_attention_dim"],
        block_out_channels=block_out_channels,
        down_block_types=teacher_config["down_block_types"],
        up_block_types=teacher_config["up_block_types"],
        mid_block_scale_factor=teacher_config["mid_block_scale_factor"],
        attention_head_dim=attn_dim,
        use_linear_projection=teacher_config["use_linear_projection"],
        resnet_time_scale_shift=teacher_config["resnet_time_scale_shift"],
        norm_num_groups=16
    )
    return student


def adaptive_copy_teacher_to_student(teacher: nn.Module, student: nn.Module):
    teacher_sd = teacher.state_dict()
    student_sd = student.state_dict()
    new_sd = {}
    for name, param in student_sd.items():
        if name in teacher_sd:
            t_param = teacher_sd[name]
            if t_param.shape == param.shape:
                new_sd[name] = t_param.clone()
            else:
                min_shape = tuple(min(s, t) for s, t in zip(param.shape, t_param.shape))
                new_param = param.clone()
                slices = tuple(slice(0, ms) for ms in min_shape)
                new_param[slices] = t_param[slices]
                new_sd[name] = new_param
        else:
            new_sd[name] = param
    student.load_state_dict(new_sd, strict=False)


# =========================================
# =========== TRAIN STUDENT ===============
# =========================================
def train_student(student_unet, teacher_pipeline, tokenizer, train_loader, device, epochs=1, log_dir="./logs"):

    optimizer = optim.AdamW(student_unet.parameters(), lr=5e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-6)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=log_dir)

    teacher_unet = teacher_pipeline.unet
    teacher_vae = teacher_pipeline.vae
    teacher_text_encoder = teacher_pipeline.text_encoder

    train_scheduler = DDPMScheduler.from_config(teacher_pipeline.scheduler.config)
    train_scheduler.num_train_timesteps = 250

    ema_tracker = EMATracker(student_unet, decay=0.9999, device='cpu')

    try:
        student_unet.enable_xformers_memory_efficient_attention()
        print("Enabled xFormers memory efficient attention for student.")
    except Exception:
        pass
    try:
        student_unet.enable_gradient_checkpointing()
        print("Enabled gradient checkpointing for student.")
    except Exception:
        pass

    teacher_unet.to(device, dtype=torch.float16)
    teacher_vae.to(device, dtype=torch.float16)
    teacher_text_encoder.to(device, dtype=torch.float16)

    os.makedirs("train_vis/teacher", exist_ok=True)
    os.makedirs("train_vis/student", exist_ok=True)
    os.makedirs("train_vis/prompts", exist_ok=True)
    os.makedirs("train_vis_full_samples", exist_ok=True)

    global_step = 0
    for epoch in range(epochs):
        student_unet.train()
        losses = []
        for step, (images, captions) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, dtype=torch.float32)
            text_inputs = tokenizer(captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            with autocast(dtype=torch.float16):
                with torch.no_grad():

                    enc_out = teacher_text_encoder(**text_inputs)
                    encoder_hidden_states = enc_out.last_hidden_state  # [batch, 77, 768]

                    latents = teacher_vae.encode(images).latent_dist.sample() * 0.18215
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, train_scheduler.num_train_timesteps, (bsz,), device=device).long()
                    noise = torch.randn_like(latents)
                    noisy_latents = train_scheduler.add_noise(latents, noise, timesteps)

                    teacher_out = teacher_unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states
                    ).sample
                teacher_pred = teacher_out.float()

                student_out = student_unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                loss = F.mse_loss(student_out, teacher_pred)

            # ===== Визуализация каждые 50 шагов (одношаговое денойзирование) =====
            if (step + 1) % 50 == 0:
                with torch.no_grad():

                    alpha_t = torch.tensor(train_scheduler.alphas_cumprod[timesteps.cpu().numpy()]).view(-1, 1, 1, 1).float().to(device)

                    teacher_denoised = (noisy_latents - torch.sqrt(1 - alpha_t) * teacher_pred) / torch.sqrt(alpha_t)
                    student_denoised = (noisy_latents - torch.sqrt(1 - alpha_t) * student_out) / torch.sqrt(alpha_t)

                    decoded_teacher = teacher_vae.decode((teacher_denoised / 0.18215).to(teacher_vae.dtype)).sample
                    decoded_student = teacher_vae.decode((student_denoised / 0.18215).to(teacher_vae.dtype)).sample

                def save_image(tensor, path):
                    image = (tensor.clamp(-1, 1) + 1) / 2  # приводим к диапазону [0,1]
                    image = (image * 255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
                    img_pil = Image.fromarray(image)
                    img_pil.save(path)

                save_image(decoded_teacher, f"train_vis/teacher/ep{epoch}_step{step}_denoised.png")
                save_image(decoded_student, f"train_vis/student/ep{epoch}_step{step}_denoised.png")
                with open(f"train_vis/prompts/ep{epoch}_step{step}.txt", "w", encoding="utf-8") as f:
                    f.write(captions[0])
                del decoded_teacher, decoded_student
                torch.cuda.empty_cache()

                print(f"[visualization] Saved denoised images and prompt at step {step} (epoch {epoch})")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(student_unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema_tracker.update(student_unet)
            losses.append(loss.item())
            global_step += 1
            if (step + 1) % 50 == 0:
                avg_loss = np.mean(losses[-50:])
                print(f"[epoch {epoch}] step {step}: loss={avg_loss:.4f}")
                writer.add_scalar("Train/Loss", avg_loss, global_step=global_step)
        epoch_loss = np.mean(losses)
        print(f"Epoch {epoch + 1}/{epochs} finished. avg_loss={epoch_loss:.4f}")
        writer.add_scalar("Epoch/Loss", epoch_loss, epoch)
        scheduler.step(epoch_loss)

        # ===== Полноценное многошаговое сэмплирование в конце эпохи =====
        # Для студента мы используем режим single_step=True, т.к. он обучался на одном шаге
        student_unet.eval()
        epoch_sample_dir = os.path.join("train_vis_full_samples", f"epoch_{epoch}")
        os.makedirs(epoch_sample_dir, exist_ok=True)
        sample_prompts = ["A glass of wine sitting on top of a table next to a bottle of wine",
                          "A large screen monitor on a desk hooked up to a laptop",
                          "this bathroom is painted dar orange and has a glass shower"]
        for i, sp in enumerate(sample_prompts):
            img_full = sample_with_cfg(
                unet_model=student_unet,
                pipeline=teacher_pipeline,
                prompt=sp,
                device=device,
                num_inference_steps=50,
                guidance_scale=8,
                height=64,
                width=64,
                single_step=True
            )
            img_full.save(os.path.join(epoch_sample_dir, f"full_sample_{i}.png"))
            del img_full
            torch.cuda.empty_cache()
        student_unet.train()
    writer.close()
    ema_tracker.apply_to(student_unet)
    print("EMA weights applied to student.")
    return epoch_loss


# =========================================
# =========== GENERATION (CFG) ============
# =========================================
def sample_with_cfg(unet_model, pipeline, prompt, device,
                    num_inference_steps=50, guidance_scale=7.5, height=64, width=64, single_step=False):
    """
    Генерация изображения с использованием classifier-free guidance.
    Если single_step=True, выполняется изображение с одним шагом денойзинга (как при обучении студента),
    иначе – полный многошаговый процесс денойзинга (как для учителя).
    """
    with torch.no_grad():

        if single_step:
            num_inference_steps = 1
            scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        else:
            scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        scheduler.set_timesteps(num_inference_steps)

        cond_input = pipeline.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=77, return_tensors="pt"
        ).to(device)
        cond_embeds = pipeline.text_encoder(**cond_input).last_hidden_state

        uncond_input = pipeline.tokenizer(
            [""], padding="max_length", truncation=True,
            max_length=77, return_tensors="pt"
        ).to(device)
        uncond_embeds = pipeline.text_encoder(**uncond_input).last_hidden_state

        encoder_hidden_states = torch.cat([uncond_embeds, cond_embeds], dim=0)

        latents = torch.randn((1, unet_model.config.in_channels, height, width),
                              device=device, dtype=unet_model.dtype)
        if hasattr(scheduler, "init_noise_sigma"):
            latents *= scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            latent_input = torch.cat([latents] * 2)
            with autocast(dtype=unet_model.dtype):
                noise_pred = unet_model(latent_input, t, encoder_hidden_states=encoder_hidden_states).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        image = pipeline.vae.decode((latents / 0.18215).to(pipeline.vae.dtype)).sample
        image = (image.clamp(-1, 1) + 1) / 2
        image = (image * 255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        return Image.fromarray(image)


def generate_images_cfg(unet_model, pipeline, prompt, device,
                        num_images=1, num_inference_steps=50, guidance_scale=7.5,
                        height=64, width=64, out_dir="gen_output", single_step=False):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_images):
        img = sample_with_cfg(
            unet_model=unet_model,
            pipeline=pipeline,
            prompt=prompt,
            device=device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            single_step=single_step
        )
        img.save(os.path.join(out_dir, f"image_{i}.png"))
        del img
        torch.cuda.empty_cache()


def resize_images_in_folder(src_folder, dst_folder, size=(64, 64)):
    os.makedirs(dst_folder, exist_ok=True)
    files = os.listdir(src_folder)
    transform = T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)
    for file in files:
        img = Image.open(os.path.join(src_folder, file))
        img_resized = transform(img)
        img_resized.save(os.path.join(dst_folder, file))


def compute_fid_for_models(teacher_pipe, student_unet, prompt, device,
                           num_images=5, num_inference_steps=50, guidance_scale=7.5,
                           teacher_height=512, teacher_width=512,
                           student_height=64, student_width=64):
    teacher_dir_full = "fid_teacher_full"
    teacher_dir_resized = "fid_teacher_resized"
    student_dir = "fid_student"

    for dir_path in [teacher_dir_full, teacher_dir_resized, student_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    print("[FID] Generating Teacher images at full resolution...")
    for i in range(num_images):
        img = teacher_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=teacher_height,
            width=teacher_width
        ).images[0]
        img.save(os.path.join(teacher_dir_full, f"img_{i}.png"))
        del img
        torch.cuda.empty_cache()

    resize_images_in_folder(teacher_dir_full, teacher_dir_resized, size=(student_height, student_width))

    print("[FID] Generating Student images...")
    student_unet.eval()
    for i in range(num_images):
        img = sample_with_cfg(
            unet_model=student_unet,
            pipeline=teacher_pipe,
            prompt=prompt,
            device=device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=student_height,
            width=student_width,
            single_step=True
        )
        img.save(os.path.join(student_dir, f"img_{i}.png"))
        del img
        torch.cuda.empty_cache()

    fid_value = fid_score.calculate_fid_given_paths(
        [teacher_dir_resized, student_dir],
        batch_size=1,
        device=device,
        dims=2048
    )
    return fid_value


# =========================================
# =========== MAIN EXPERIMENT =============
# =========================================
def main_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading teacher pipeline...")
    teacher_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    teacher_pipe.scheduler = DPMSolverMultistepScheduler.from_config(teacher_pipe.scheduler.config)
    print("Teacher loaded.")

    print("Building dataset...")
    train_dataset = MSCOCODataset(
        img_dir="D:\\MSU\\diploma\\work\\coco2017\\train2017",
        ann_file="D:\\MSU\\diploma\\work\\coco2017\\annotations\\captions_train2017.json",
        image_size=64,
        max_samples=32000
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    print("Dataset ready.")

    teacher_config = dict(teacher_pipe.unet.config)
    teacher_config.pop("num_attention_heads", None)

    print("\n== Creating Student (Adaptive init) ==")
    student_adaptive = make_student_unet(teacher_config).to(device, dtype=torch.float32)
    adaptive_copy_teacher_to_student(teacher_pipe.unet, student_adaptive)
    print("Starting training (adaptive init)...")

    loss_adaptive = train_student(
        student_unet=student_adaptive,
        teacher_pipeline=teacher_pipe,
        tokenizer=teacher_pipe.tokenizer,
        train_loader=train_loader,
        device=device,
        epochs=20,
        log_dir="./logs_adaptive"
    )
    print(f"[Adaptive init] Final loss: {loss_adaptive:.4f}")
    torch.save(student_adaptive.state_dict(), "student_adaptive.pth")
    del student_adaptive
    torch.cuda.empty_cache()

    print("\n== Creating Student (Random init) ==")
    student_random = make_student_unet(teacher_config).to(device, dtype=torch.float32)
    print("Starting training (random init)...")
    loss_random = train_student(
        student_unet=student_random,
        teacher_pipeline=teacher_pipe,
        tokenizer=teacher_pipe.tokenizer,
        train_loader=train_loader,
        device=device,
        epochs=20,
        log_dir="./logs_random"
    )
    print(f"[Random init] Final loss: {loss_random:.4f}")
    torch.save(student_random.state_dict(), "student_random.pth")

    print("\n== Evaluating FID for both students ==")
    prompt = "A street post showing where the stores are"

    student_adaptive_eval = make_student_unet(teacher_config).to(device, dtype=torch.float32)
    student_adaptive_eval.load_state_dict(torch.load("student_adaptive.pth"))
    fid_adaptive = compute_fid_for_models(
        teacher_pipe,
        student_adaptive_eval,
        prompt=prompt,
        device=device,
        num_images=5,
        num_inference_steps=50,
        guidance_scale=8,
        teacher_height=512,
        teacher_width=512,
        student_height=64,
        student_width=64
    )
    print(f"FID (Adaptive) = {fid_adaptive:.2f}")
    del student_adaptive_eval
    torch.cuda.empty_cache()


    student_random_eval = make_student_unet(teacher_config).to(device, dtype=torch.float32)
    student_random_eval.load_state_dict(torch.load("student_random.pth"))
    fid_random = compute_fid_for_models(
        teacher_pipe,
        student_random_eval,
        prompt=prompt,
        device=device,
        num_images=5,
        num_inference_steps=50,
        guidance_scale=8,
        teacher_height=512,
        teacher_width=512,
        student_height=64,
        student_width=64
    )
    print(f"FID (Random) = {fid_random:.2f}")
    torch.cuda.empty_cache()


    example_prompts = [
        "A grop of sheep on the side of a grassy hill",
        "A large clock tower towering over a city",
        "A fancy sandwich on toasted bread and a small salad",
        "A picture of some animals by a tree in the grass",
        "A man swinging a tennis racquet on a tennis court"
    ]
    student_adaptive_eval = make_student_unet(teacher_config).to(device, dtype=torch.float32)
    student_adaptive_eval.load_state_dict(torch.load("student_adaptive.pth"))
    student_random_eval = make_student_unet(teacher_config).to(device, dtype=torch.float32)
    student_random_eval.load_state_dict(torch.load("student_random.pth"))

    os.makedirs("examples_adaptive", exist_ok=True)
    os.makedirs("examples_random", exist_ok=True)
    print("Generating example images for Adaptive student...")
    for i, sp in enumerate(example_prompts):
        img = sample_with_cfg(
            unet_model=student_adaptive_eval,
            pipeline=teacher_pipe,
            prompt=sp,
            device=device,
            num_inference_steps=50,
            guidance_scale=8,
            height=64,
            width=64,
            single_step=True
        )
        img.save(f"examples_adaptive/{i:02d}_adaptive.png")
        del img
        torch.cuda.empty_cache()

    print("Generating example images for Random student...")
    for i, sp in enumerate(example_prompts):
        img = sample_with_cfg(
            unet_model=student_random_eval,
            pipeline=teacher_pipe,
            prompt=sp,
            device=device,
            num_inference_steps=50,
            guidance_scale=8,
            height=64,
            width=64,
            single_step=True
        )
        img.save(f"examples_random/{i:02d}_random.png")
        del img
        torch.cuda.empty_cache()

    print("\n=== SUMMARY ===")
    print(f"Adaptive final loss: {fid_adaptive:.4f}, FID={fid_adaptive:.2f}")
    print(f"Random final loss:   {fid_random:.4f}, FID={fid_random:.2f}")


if __name__ == "__main__":
    main_experiment()
