import os, random, math, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

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
from lpips import LPIPS

from torchmetrics.multimodal.clip_score import CLIPScore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            T.Resize((image_size + 10, image_size + 10), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
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
            if p.requires_grad and name in self.shadow:
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
# =========== PROJECTOR LAYER =============
# =========================================
class ConvProjection(nn.Module):
    """
    Простой 1x1 conv для приведения (in_channels -> out_channels).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(x)


# =========================================
# =========== FEATURE HOOKS ===============
# =========================================
class FeatureHook:
    """
    Сохраняет выходы (активации) выбранных слоёв UNet.
    Если слой возвращает tuple, берём первый элемент.
    """

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.save_hook)
        self.features = None

    def save_hook(self, module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        self.features = out

    def close(self):
        self.hook.remove()


def collect_features(model, input_latents, timesteps, encoder_hidden_states,
                     hook_layers=("down_blocks.0", "down_blocks.1", "down_blocks.2", "up_blocks.0", "up_blocks.1"),
                     return_output=True):
    hooks_dict = {}
    for name, submodule in model.named_modules():
        if any(name.startswith(layer) for layer in hook_layers):
            hooks_dict[name] = FeatureHook(submodule)

    out = model(sample=input_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states).sample

    features_out = {}
    for name, h in hooks_dict.items():
        features_out[name] = h.features
        h.close()

    if return_output:
        return features_out, out
    else:
        return features_out


def match_teacher_feat_to_student_shape(tF, sF, projector=None):
    """
    Приводим tF к форме (B, C_s, H_s, W_s), где sF имеет форму (B, C_s, H_s, W_s).
    1) Если (H_t, W_t) != (H_s, W_s), делаем F.interpolate.
    2) Если C_t != C_s, используем projector (если задан) или создаем 1x1 conv.
    """
    B_s, C_s, H_s, W_s = sF.shape
    B_t, C_t, H_t, W_t = tF.shape

    if (H_t != H_s) or (W_t != W_s):
        tF = F.interpolate(tF, size=(H_s, W_s), mode="bilinear", align_corners=False)

    if C_t != C_s:
        if projector is not None:
            tF = projector(tF)
        else:
            conv = ConvProjection(in_channels=C_t, out_channels=C_s).to(tF.device, dtype=tF.dtype)
            with torch.no_grad():
                nn.init.xavier_normal_(conv.proj.weight)
            tF = conv(tF)

    return tF


# =========================================
# =========== MAKE STUDENT UNET ===========
# =========================================
def make_student_unet(teacher_config, sample_size=64):

    if hasattr(teacher_config, "to_dict"):
        teacher_config = dict(teacher_config)
    teacher_config.pop("num_attention_heads", None)
    block_out_channels = [max(16, ch // 4) for ch in teacher_config["block_out_channels"]]
    layers_per_block = max(1, teacher_config["layers_per_block"] - 1)

    attn_dim = teacher_config["attention_head_dim"]
    if isinstance(attn_dim, list):
        attn_dim = [max(1, x // 2) for x in attn_dim]
    else:
        attn_dim = max(1, attn_dim // 2)

    teacher_config["cross_attention_dim"] = 768

    student = UNet2DConditionModel(
        sample_size=sample_size,
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


# =========================================
# =========== ADAPTIVE COPY ===============
# =========================================
def adaptive_copy_teacher_to_student(teacher: nn.Module, student: nn.Module):
    """
    Копируем совпадающие веса; для несовпадающих по каналам создаем projectors (1x1 conv).
    (Пространственные несоответствия будут корректироваться при feature distillation через F.interpolate.)
    """
    teacher_sd = teacher.state_dict()
    student_sd = student.state_dict()

    new_sd = {}
    projectors = nn.ModuleDict()
    layer_to_projname = {}

    for name, param in student_sd.items():
        if name in teacher_sd:
            t_param = teacher_sd[name]
            if t_param.shape == param.shape:
                new_sd[name] = t_param.clone()
            else:
                if len(param.shape) == 4 and len(t_param.shape) == 4:
                    outC_s, inC_s, kh_s, kw_s = param.shape
                    outC_t, inC_t, kh_t, kw_t = t_param.shape
                    if kh_s == kh_t and kw_s == kw_t:
                        proj_name = f"proj_{name.replace('.', '_')}"
                        proj_layer = ConvProjection(inC_t, outC_s)
                        nn.init.xavier_normal_(proj_layer.proj.weight)
                        projectors[proj_name] = proj_layer
                        layer_to_projname[name] = proj_name
                        new_sd[name] = param
                    else:
                        new_sd[name] = param
                else:
                    new_sd[name] = param
        else:
            new_sd[name] = param

    student.load_state_dict(new_sd, strict=False)
    student.projectors = projectors
    student.layer_to_projname = layer_to_projname
    student.float()


# =========================================
# =========== FULL SAMPLING ===============
# =========================================
def generate_student_sample(student_unet, teacher_pipeline, prompt="a cat", steps=25, out_path="sample.png"):

    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    pipe = StableDiffusionPipeline(
        vae=teacher_pipeline.vae,
        text_encoder=teacher_pipeline.text_encoder,
        tokenizer=teacher_pipeline.tokenizer,
        unet=student_unet,
        scheduler=DPMSolverMultistepScheduler.from_config(teacher_pipeline.scheduler.config),
        safety_checker=None,
        feature_extractor=None
    ).to(device)

    pipe.vae.to(device, dtype=torch.float16)
    pipe.text_encoder.to(device, dtype=torch.float16)
    pipe.unet.to(device, dtype=torch.float32)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass

    with autocast(enabled=True, dtype=torch.float16), torch.inference_mode():
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5, height=64, width=64).images[0]
    image.save(out_path)


# =========================================
# =========== TRAIN STUDENT ===============
# =========================================
def train_student(student_unet, teacher_pipeline, tokenizer, train_loader, device,
                  epochs=1, log_dir="./logs", alpha_feature_target=0.02, warmup_epochs=5, accumulation_steps=2,
                  patience=5):
    """
    Обучение:
      - Teacher: FP16, no_grad.
      - Student: FP32 параметры, forward через AMP (вычисления в half).
      - Feature distillation: если активации 4D, то:
            * Если spatial размеры не совпадают, применяется F.interpolate.
            * Если channels не совпадают, используется projector или 1x1 conv.
         Перед подсчетом MSE нормализуются фичи по L2.
         Вес feature loss динамически увеличивается от 0.01 до target через warmup_epochs.
      - Добавляется L1 в VAE-пространстве и LPIPS.
      - Используется gradient accumulation, checkpointing и early stopping.
    """
    optimizer = optim.AdamW(student_unet.parameters(), lr=1e-4)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.05)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    scaler = GradScaler()
    writer = SummaryWriter(log_dir=log_dir)

    teacher_unet = teacher_pipeline.unet
    teacher_vae = teacher_pipeline.vae
    teacher_text_encoder = teacher_pipeline.text_encoder

    train_scheduler = DDPMScheduler.from_config(teacher_pipeline.scheduler.config)
    train_scheduler.num_train_timesteps = 250

    train_scheduler.beta_schedule = "cosine"

    ema_tracker = EMATracker(student_unet, decay=0.9999, device='cpu')

    try:
        student_unet.enable_xformers_memory_efficient_attention()
    except:
        pass
    try:
        student_unet.enable_gradient_checkpointing()
    except:
        pass

    teacher_unet.to(device, dtype=torch.float16)
    teacher_vae.to(device, dtype=torch.float16)
    teacher_text_encoder.to(device, dtype=torch.float16)
    student_unet.to(device, dtype=torch.float32)

    os.makedirs("train_vis/teacher", exist_ok=True)
    os.makedirs("train_vis/student", exist_ok=True)
    os.makedirs("train_vis/prompts", exist_ok=True)

    hook_layers = (
        "down_blocks.0", "down_blocks.0"
    )

    lpips_model = LPIPS(net='alex').to(device)
    clip_score_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)

    best_loss = np.inf
    no_improve_epochs = 0

    global_step = 0
    for epoch in range(epochs):
        student_unet.train()
        losses = []

        # Расчет динамического alpha_feature через warmup
        if epoch < warmup_epochs:
            alpha_feature = 0.01 + (alpha_feature_target - 0.01) * (epoch / warmup_epochs)
        else:
            alpha_feature = alpha_feature_target

        for step, (images, captions) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, dtype=torch.float32)
            text_inputs = tokenizer(captions, padding="max_length", max_length=77,
                                    truncation=True, return_tensors="pt").to(device)

            with autocast(enabled=True, dtype=torch.float16):
                # Teacher forward (без расчета градиентов)
                with torch.no_grad():
                    enc_out = teacher_text_encoder(**text_inputs)
                    encoder_hidden_states = enc_out.last_hidden_state

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

                with torch.no_grad():
                    teacher_feats = collect_features(
                        model=teacher_unet,
                        input_latents=noisy_latents,
                        timesteps=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        hook_layers=hook_layers,
                        return_output=False
                    )
                teacher_feats = {k: v for k, v in teacher_feats.items() if v is not None}

                # Student forward
                student_feats, student_out = collect_features(
                    model=student_unet,
                    input_latents=noisy_latents,
                    timesteps=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    hook_layers=hook_layers,
                    return_output=True
                )
                student_feats = {k: v for k, v in student_feats.items() if v is not None}

                loss_out = F.mse_loss(student_out, teacher_out)

                feat_loss = 0.0
                for layer_name, tF in teacher_feats.items():
                    sF = student_feats.get(layer_name, None)
                    if sF is None or tF.dim() != 4 or sF.dim() != 4:
                        continue

                    projector = None
                    if hasattr(student_unet, "layer_to_projname") and (layer_name in student_unet.layer_to_projname):
                        proj_name = student_unet.layer_to_projname[layer_name]
                        projector = student_unet.projectors[proj_name]

                    tF_matched = match_teacher_feat_to_student_shape(tF, sF, projector=projector)
                    sF_norm = sF / (sF.pow(2).mean(dim=[1, 2, 3], keepdim=True).sqrt() + 1e-8)
                    tF_norm = tF_matched / (tF_matched.pow(2).mean(dim=[1, 2, 3], keepdim=True).sqrt() + 1e-8)
                    feat_loss_local = F.mse_loss(sF_norm, tF_norm)
                    feat_loss += feat_loss_local

                feat_loss = alpha_feature * feat_loss

                with torch.no_grad():
                    alpha_t = train_scheduler.alphas_cumprod[timesteps.cpu()].view(-1, 1, 1, 1).to(device)
                    alpha_t = alpha_t.half()
                    teacher_denoised = (noisy_latents - torch.sqrt(1 - alpha_t) * teacher_out) / torch.sqrt(alpha_t)
                    teacher_decoded = teacher_vae.decode(teacher_denoised / 0.18215).sample

                if (step + 1) % 50 == 0:
                    student_denoised = (noisy_latents - torch.sqrt(1 - alpha_t) * student_out) / torch.sqrt(alpha_t)
                    student_decoded = teacher_vae.decode(student_denoised / 0.18215).sample

                    with torch.no_grad():
                        teacher_denoised = (noisy_latents - torch.sqrt(1 - alpha_t) * teacher_out) / torch.sqrt(alpha_t)
                        teacher_decoded = teacher_vae.decode(teacher_denoised / 0.18215).sample

                    l1_vae = 0.1 * F.l1_loss(student_decoded, teacher_decoded)
                    student_32 = student_decoded.float()
                    teacher_32 = teacher_decoded.float()
                    lpips_val = lpips_model(student_32, teacher_32).mean() * 0.1
                else:
                    l1_vae = torch.tensor(0.0, device=device)
                    lpips_val = torch.tensor(0.0, device=device)

                total_loss = loss_out + feat_loss + l1_vae + lpips_val

            scaler.scale(total_loss / accumulation_steps).backward()
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(student_unet.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                ema_tracker.update(student_unet)

            losses.append(total_loss.item())

            if (step + 1) % 50 == 0:
                avg_loss = np.mean(losses[-50:])
                print(f"[epoch {epoch}] step {step}: total_loss={avg_loss:.4f}, out={loss_out.item():.4f}, "
                      f"feats={feat_loss.item():.4f}, l1_vae={l1_vae.item():.4f}, lpips={lpips_val.item():.4f}, alpha_feature={alpha_feature:.4f}")
                writer.add_scalar("Train/TotalLoss", avg_loss, global_step=global_step)

                def save_image(tensor, path):
                    img = (tensor.clamp(-1, 1) + 1) / 2
                    arr = (img * 255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
                    Image.fromarray(arr).save(path)

                save_image(teacher_decoded, f"train_vis/teacher/ep{epoch}_step{step}_denoised.png")
                save_image(student_decoded, f"train_vis/student/ep{epoch}_step{step}_denoised.png")
                with open(f"train_vis/prompts/ep{epoch}_step{step}.txt", "w", encoding="utf-8") as f:
                    f.write(captions[0])

        epoch_loss = np.mean(losses)
        print(f"Epoch {epoch + 1}/{epochs} finished. avg_loss={epoch_loss:.4f}")
        writer.add_scalar("Epoch/Loss", epoch_loss, epoch)
        scheduler.step()

        # Checkpointing каждые 2 эпохи
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f"checkpoint_epoch_{epoch + 1}.pth")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

        # Валидация: генерация изображения и вычисление CLIP Score (опционально)
        student_unet.eval()
        gen_img_path = f"train_vis/student_epoch_{epoch}.png"
        generate_student_sample(student_unet, teacher_pipeline,
                                prompt="A dog in a hat", steps=25, out_path=gen_img_path)
        # Если доступны real captions, можно вычислять CLIP Score; здесь примерный вызов:
        # clip_score = clip_score_metric(gen_img, [caption])
        student_unet.train()

    writer.close()
    ema_tracker.apply_to(student_unet)
    print("EMA weights applied to student.")
    return epoch_loss


# =========================================
# =========== CFG SAMPLING ================
# =========================================
def sample_with_cfg(unet_model, pipeline, prompt, device,
                    num_inference_steps=50, guidance_scale=7.5, height=64, width=64):
    from diffusers import DPMSolverMultistepScheduler
    scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    scheduler.set_timesteps(num_inference_steps)

    with torch.inference_mode(), autocast(enabled=True, dtype=torch.float16):
        cond_input = pipeline.tokenizer(prompt, padding="max_length", truncation=True,
                                        max_length=77, return_tensors="pt").to(device)
        cond_embeds = pipeline.text_encoder(**cond_input).last_hidden_state

        uncond_input = pipeline.tokenizer([""], padding="max_length", truncation=True,
                                          max_length=77, return_tensors="pt").to(device)
        uncond_embeds = pipeline.text_encoder(**uncond_input).last_hidden_state

        encoder_hidden_states = torch.cat([uncond_embeds, cond_embeds], dim=0)

        latents = torch.randn((1, unet_model.config.in_channels, height, width),
                              device=device, dtype=torch.float16)
        if hasattr(scheduler, "init_noise_sigma"):
            latents *= scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            latent_input = torch.cat([latents] * 2)
            noise_pred = unet_model(latent_input, t, encoder_hidden_states=encoder_hidden_states).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        image = pipeline.vae.decode(latents / 0.18215).sample
        image = (image.clamp(-1, 1) + 1) / 2
        image = (image * 255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        return Image.fromarray(image)


def compute_fid_for_models(teacher_pipe, student_unet, prompt, device,
                           num_images=5, num_inference_steps=25, guidance_scale=7.5,
                           teacher_height=512, teacher_width=512,
                           student_height=64, student_width=64):
    """
    Генерируем num_images у Teacher (512x512), ресайзим до 64x64,
    и генерируем num_images у студента (64x64). Считаем FID.
    """
    teacher_dir_full = "fid_teacher_full"
    teacher_dir_resized = "fid_teacher_resized"
    student_dir = "fid_student"

    for d in [teacher_dir_full, teacher_dir_resized, student_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    print(f"[FID] Generating Teacher images at {teacher_height}x{teacher_width} ...")
    for i in range(num_images):
        img = teacher_pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=teacher_height,
            width=teacher_width
        ).images[0]
        img.save(os.path.join(teacher_dir_full, f"teacher_{i}.png"))
        del img
        torch.cuda.empty_cache()

    transform_resize = T.Resize((student_height, student_width), interpolation=T.InterpolationMode.BICUBIC)
    for f in os.listdir(teacher_dir_full):
        path_full = os.path.join(teacher_dir_full, f)
        im_pil = Image.open(path_full).convert("RGB")
        im_resized = transform_resize(im_pil)
        im_resized.save(os.path.join(teacher_dir_resized, f))

    print(f"[FID] Generating Student images at {student_height}x{student_width} ...")
    student_unet.eval()
    for i in range(num_images):
        img_stud = sample_with_cfg(
            unet_model=student_unet,
            pipeline=teacher_pipe,
            prompt=prompt,
            device=device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=student_height,
            width=student_width
        )
        img_stud.save(os.path.join(student_dir, f"student_{i}.png"))
        del img_stud
        torch.cuda.empty_cache()

    print("[FID] Computing FID...")
    fid_value = fid_score.calculate_fid_given_paths(
        [teacher_dir_resized, student_dir],
        batch_size=1,
        device=device,
        dims=2048
    )
    return fid_value


def main_experiment():
    print("Loading teacher pipeline...")
    teacher_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    teacher_pipe.scheduler = DPMSolverMultistepScheduler.from_config(teacher_pipe.scheduler.config)
    print("Teacher loaded.")

    # Dataset: 64x64, up to 20k изображений, batch_size=40
    print("Building dataset...")
    train_dataset = MSCOCODataset(
        img_dir="D:\\MSU\\diploma\\work\\coco2017\\train2017",
        ann_file="D:\\MSU\\diploma\\work\\coco2017\\annotations\\captions_train2017.json",
        image_size=64,
        max_samples=10000
    )
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, num_workers=0)
    print("Dataset ready.")

    teacher_config = dict(teacher_pipe.unet.config)

    # ---- 1) Student (Adaptive init) ----
    print("\n=== 1) Student (Adaptive init) ===")
    student_adaptive = make_student_unet(teacher_config, sample_size=64)
    adaptive_copy_teacher_to_student(teacher_pipe.unet, student_adaptive)

    loss_adaptive = train_student(
        student_unet=student_adaptive,
        teacher_pipeline=teacher_pipe,
        tokenizer=teacher_pipe.tokenizer,
        train_loader=train_loader,
        device=device,
        epochs=20,
        log_dir="./logs_adaptive",
        alpha_feature_target=0.05,  # целевой вес feature loss
        warmup_epochs=5,
        accumulation_steps=2,
        patience=5
    )
    torch.save(student_adaptive.state_dict(), "student_adaptive.pth")
    print(f"[Adaptive init] Final training loss: {loss_adaptive:.4f}")

    # ---- 2) Student (Random init) ----
    print("\n=== 2) Student (Random init) ===")
    student_random = make_student_unet(teacher_config, sample_size=64)
    student_random.float()  # random init

    loss_random = train_student(
        student_unet=student_random,
        teacher_pipeline=teacher_pipe,
        tokenizer=teacher_pipe.tokenizer,
        train_loader=train_loader,
        device=device,
        epochs=20,
        log_dir="./logs_random",
        alpha_feature_target=0.05,
        warmup_epochs=5,
        accumulation_steps=2,
        patience=5
    )
    torch.save(student_random.state_dict(), "student_random.pth")
    print(f"[Random init] Final training loss: {loss_random:.4f}")

    print("\n== FID evaluation (Adaptive vs Random) ==")
    prompt_for_fid = "A picture of a mountain with a lake in front"

    fid_adaptive = compute_fid_for_models(
        teacher_pipe, student_adaptive,
        prompt=prompt_for_fid,
        device=device,
        num_images=5,
        num_inference_steps=25,
        guidance_scale=7.5,
        teacher_height=512,
        teacher_width=512,
        student_height=64,
        student_width=64
    )
    print(f"FID (Adaptive) = {fid_adaptive:.2f}")

    fid_random = compute_fid_for_models(
        teacher_pipe, student_random,
        prompt=prompt_for_fid,
        device=device,
        num_images=5,
        num_inference_steps=25,
        guidance_scale=7.5,
        teacher_height=512,
        teacher_width=512,
        student_height=64,
        student_width=64
    )
    print(f"FID (Random) = {fid_random:.2f}")

    # Оценка CLIP Score для генераций (опционально)
    # Для примера, можно вычислять CLIP Score для нескольких сгенерированных изображений
    # clip_score_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)
    # Примерный вызов:
    # generated_img = sample_with_cfg(student_adaptive, teacher_pipe, prompt_for_fid, device, 25, 7.5, 64, 64)
    # clip_score_val = clip_score_metric(generated_img, [prompt_for_fid]).mean().item()
    # print(f"CLIP Score (Adaptive) = {clip_score_val:.4f}")

    print("\nDone main_experiment.")


if __name__ == "__main__":
    main_experiment()
