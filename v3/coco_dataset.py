import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms

class CocoTrainDataset(Dataset):
    """
    Локальный COCO2017 Dataset:
      – изображения в <img_root>/<file_name>
      – аннотации (captions) из <ann_file>
    """
    def __init__(self, img_root, ann_file, tokenizer, resolution=64):
        self.img_root = img_root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(resolution, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_root, info["file_name"])
        image = Image.open(path).convert("RGB")
        pixel_values = self.transform(image)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        caption = self.coco.loadAnns(ann_ids)[0]["caption"]
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": pixel_values, "input_ids": input_ids}


def get_coco_dataloader(args, tokenizer):
    """
    Возвращает DataLoader для COCO2017 из локальных файлов:
      args.train_data_dir — папка train2017
      args.caption_file   — путь к captions_train2017.json
    """
    dataset = CocoTrainDataset(
        img_root=args.train_data_dir,
        ann_file=args.caption_file,
        tokenizer=tokenizer,
        resolution=64,
    )

    # Обрезаем до первых N примеров, если нужно
    if args.max_train_samples and args.max_train_samples > 0:
        dataset.ids = dataset.ids[: args.max_train_samples]

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
