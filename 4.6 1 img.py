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

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from pytorch_fid import fid_score

# Для отладки модели студента будем работать в float32, а учитель оставляем в float16.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_dtype = torch.float32  # студент
teacher_dtype = torch.float16  # учитель


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
# =========== MAKE STUDENT UNET ===========
# =========================================
def make_student_unet(teacher_config):

    if hasattr(teacher_config, "to_dict"):
        teacher_config = dict(teacher_config)
    teacher_config.pop("num_attention_heads", None)
    block_out_channels = [max(1, ch // 2) for ch in teacher_config["block_out_channels"]]
    layers_per_block = teacher_config["layers_per_block"]
    attn_dim = teacher_config["attention_head_dim"]
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
    """
    Копирует веса из teacher для соответствующих параметров student.
    Если размерности совпадают, копируем напрямую.
    Если размеры teacher больше, выполняем срез, а затем
    после копирования принудительно переводим все параметры student в float32.
    """
    teacher_sd = teacher.state_dict()
    student_sd = student.state_dict()
    new_sd = {}
    for name, param in student_sd.items():
        if name in teacher_sd:
            t_param = teacher_sd[name]
            if t_param.shape == param.shape:
                new_sd[name] = t_param.clone()
            else:
                if all(t >= s for t, s in zip(t_param.shape, param.shape)):
                    slices = tuple(slice(0, s) for s in param.shape)
                    new_param = param.clone()
                    new_param[slices] = t_param[slices]
                    new_sd[name] = new_param
                else:
                    new_sd[name] = param
        else:
            new_sd[name] = param
    student.load_state_dict(new_sd, strict=False)

    for p in student.parameters():
        p.data = p.data.to(torch.float32)


# =========================================
# =========== TRAIN STUDENT ===============
# =========================================
def train_student(student_unet, teacher_pipeline, tokenizer, train_loader, device, epochs=1, log_dir="./logs"):
    optimizer = optim.AdamW(student_unet.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-7)
    writer = SummaryWriter(log_dir=log_dir)

    teacher_unet = teacher_pipeline.unet
    teacher_vae = teacher_pipeline.vae
    teacher_text_encoder = teacher_pipeline.text_encoder

    train_scheduler = DDPMScheduler.from_config(teacher_pipeline.scheduler.config)
    train_scheduler.num_train_timesteps = 250

    print("Training without EMA/AMP for overfit experiment.")

    teacher_unet.to(device, dtype=teacher_dtype)
    teacher_vae.to(device, dtype=teacher_dtype)
    teacher_text_encoder.to(device, dtype=teacher_dtype)
    student_unet.to(device, dtype=student_dtype)

    os.makedirs("train_vis/teacher", exist_ok=True)
    os.makedirs("train_vis/student", exist_ok=True)
    os.makedirs("train_vis/prompts", exist_ok=True)
    os.makedirs("train_vis_full_samples", exist_ok=True)

    global_step = 0
    for epoch in range(epochs):
        student_unet.train()
        losses = []
        for step, (images, captions) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device, dtype=torch.float32)
            text_inputs = tokenizer(captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}


            enc_out = teacher_text_encoder(**text_inputs)

            encoder_hidden_states_teacher = enc_out.last_hidden_state.to(teacher_unet.dtype)
            encoder_hidden_states_student = enc_out.last_hidden_state.to(student_unet.dtype)

            latents = teacher_vae.encode(images.to(teacher_vae.dtype)).latent_dist.sample() * 0.18215
            latents = latents.to(device, dtype=torch.float32)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, train_scheduler.num_train_timesteps, (bsz,), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = train_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                teacher_out = teacher_unet(
                    sample=noisy_latents.to(teacher_unet.dtype),
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states_teacher
                ).sample
            teacher_pred = teacher_out.to(dtype=torch.float32)
            student_out = student_unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states_student
            ).sample
            loss = F.mse_loss(student_out, teacher_pred)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            global_step += 1
            if (step + 1) % 10 == 0:
                avg_loss = np.mean(losses[-10:])
                print(f"[Epoch {epoch}] Step {step}: avg_loss = {avg_loss:.4f}")
                writer.add_scalar("Train/Loss", avg_loss, global_step=global_step)
        epoch_loss = np.mean(losses)
        print(f"Epoch {epoch + 1}/{epochs} finished. avg_loss = {epoch_loss:.4f}")
        writer.add_scalar("Epoch/Loss", epoch_loss, epoch)
        scheduler.step(epoch_loss)

        student_unet.eval()
        epoch_sample_dir = os.path.join("train_vis_full_samples", f"epoch_{epoch}")
        os.makedirs(epoch_sample_dir, exist_ok=True)
        train_example = train_loader.dataset[0]
        sp = train_example[1] if train_example[1] != "" else "A street post showing where the stores are"
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
        img_full.save(os.path.join(epoch_sample_dir, f"full_sample.png"))
        print(f"[Epoch {epoch}] Saved full sample using prompt: {sp}")
        student_unet.train()
    writer.close()
    return epoch_loss


# =========================================
# =========== GENERATION (CFG) ============
# =========================================
def sample_with_cfg(unet_model, pipeline, prompt, device,
                    num_inference_steps=50, guidance_scale=7.5, height=64, width=64, single_step=False):
    """
    Генерация изображения с использованием classifier-free guidance.
    Если single_step=True, используется один шаг денойзинга (как при обучении студента),
    иначе – итеративный процесс (как для учителя).
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

        cond_embeds = pipeline.text_encoder(**cond_input).last_hidden_state.to(unet_model.dtype)
        uncond_input = pipeline.tokenizer(
            [""], padding="max_length", truncation=True,
            max_length=77, return_tensors="pt"
        ).to(device)
        uncond_embeds = pipeline.text_encoder(**uncond_input).last_hidden_state.to(unet_model.dtype)
        encoder_hidden_states = torch.cat([uncond_embeds, cond_embeds], dim=0)

        latents = torch.randn((1, unet_model.config.in_channels, height, width),
                              device=device, dtype=unet_model.dtype)
        if hasattr(scheduler, "init_noise_sigma"):
            latents *= scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            latent_input = torch.cat([latents] * 2)
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

    fid_value = fid_score.calculate_fid_given_paths(
        [teacher_dir_resized, student_dir],
        batch_size=1,
        device=device,
        dims=2048
    )
    return fid_value


# =========================================
# =========== MAIN EXPERIMENT (Overfit) ===
# =========================================
def main_experiment_overfit():
    """
    Эксперимент для переобучения студента на одном изображении.
    Датасет содержит только один пример (batch=1, max_samples=1), число эпох 50.
    Итоговое изображение генерируется по промпту, соответствующему обучаемому примеру.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading teacher pipeline...")
    teacher_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=teacher_dtype,
        safety_checker=None
    ).to(device)
    teacher_pipe.scheduler = DPMSolverMultistepScheduler.from_config(teacher_pipe.scheduler.config)
    print("Teacher loaded.")

    print("Building dataset for overfitting (one image)...")
    train_dataset = MSCOCODataset(
        img_dir="D:\\MSU\\diploma\\work\\coco2017\\train2017",
        ann_file="D:\\MSU\\diploma\\work\\coco2017\\annotations\\captions_train2017.json",
        image_size=64,
        max_samples=1
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    print("Dataset ready (1 image).")

    train_image, train_caption = train_dataset[0]
    print(f"Training on image with caption: {train_caption}")

    teacher_config = dict(teacher_pipe.unet.config)
    teacher_config.pop("num_attention_heads", None)

    print("\n== Creating Student (Adaptive init) for overfit experiment ==")
    student_adaptive = make_student_unet(teacher_config).to(device, dtype=student_dtype)
    adaptive_copy_teacher_to_student(teacher_pipe.unet, student_adaptive)
    print("Starting overfit training ...")
    loss_overfit = train_student(
        student_unet=student_adaptive,
        teacher_pipeline=teacher_pipe,
        tokenizer=teacher_pipe.tokenizer,
        train_loader=train_loader,
        device=device,
        epochs=50,
        log_dir="./logs_overfit"
    )
    print(f"[Overfit] Final loss: {loss_overfit:.4f}")
    torch.save(student_adaptive.state_dict(), "student_overfit.pth")
    del student_adaptive

    print("\nGenerating final overfit sample using the training caption as prompt...")
    student_overfit = make_student_unet(teacher_config).to(device, dtype=student_dtype)
    student_overfit.load_state_dict(torch.load("student_overfit.pth"))
    final_img = sample_with_cfg(
        unet_model=student_overfit,
        pipeline=teacher_pipe,
        prompt=train_caption,
        device=device,
        num_inference_steps=50,
        guidance_scale=8,
        height=64,
        width=64,
        single_step=True
    )
    final_img.save("overfit_sample.png")
    print("Overfit sample saved as overfit_sample.png")


if __name__ == "__main__":
    main_experiment_overfit()
