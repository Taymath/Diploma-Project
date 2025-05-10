# prepare_unet_small.py

import torch, gc

def prepare_unet(unet, model_type: str):
    """
    Сжимает U-Net до 'sd_small' или 'sd_tiny' точно по оригинальному коду HuggingFace.
    """
    assert model_type in ["sd_small", "sd_tiny"]

    # 1) для tiny удаляем mid_block
    if model_type == "sd_tiny":
        unet.mid_block = None

    # 2) DOWN blocks: убираем второй ResNet+Attention в блоках 0–2
    for i in range(3):
        delattr(unet.down_blocks[i].resnets, "1")
        delattr(unet.down_blocks[i].attentions, "1")

    if model_type == "sd_tiny":
        # tiny: удаляем 4-й блок полностью
        delattr(unet.down_blocks, "3")
        # и в новом последнем блоке (2-м) убираем downsamplers
        unet.down_blocks[2].downsamplers = None
    else:
        # small: из 4-го блока убираем только второй ResNet
        delattr(unet.down_blocks[3].resnets, "1")

    # 3) UP blocks: первый блок — копируем resnets[2]→resnets[1], удаляем resnets[2]
    unet.up_blocks[0].resnets[1] = unet.up_blocks[0].resnets[2]
    delattr(unet.up_blocks[0].resnets, "2")

    # 4) UP blocks 1–3: аналогично
    for i in range(1, 4):
        unet.up_blocks[i].resnets[1]    = unet.up_blocks[i].resnets[2]
        unet.up_blocks[i].attentions[1] = unet.up_blocks[i].attentions[2]
        delattr(unet.up_blocks[i].resnets,    "2")
        delattr(unet.up_blocks[i].attentions, "2")

    if model_type == "sd_tiny":
        # tiny: после копирования 0–2, сдвигаем все вверх и удаляем последний
        for i in range(3):
            unet.up_blocks[i] = unet.up_blocks[i+1]
        delattr(unet.up_blocks, "3")

    # очистка
    torch.cuda.empty_cache()
    gc.collect()
