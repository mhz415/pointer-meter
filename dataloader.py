import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as F


class MeterDataset(Dataset):
    def __init__(self, root_dir, subset='train', input_size=(512, 512), augment=False):
        """
        args:
            augment: 是否开启数据增强 (训练集True, 验证集False)
        """
        self.root_dir = root_dir
        self.subset = subset
        self.input_size = input_size
        self.augment = augment

        self.images_dir = os.path.join(root_dir, 'images', subset)
        self.masks_dir = os.path.join(root_dir, 'masks', subset)

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"找不到图片目录: {self.images_dir}")

        self.images = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.images)

    def augment_data(self, image, mask):
        # 1. 随机旋转 (仪表指针的核心：全角度覆盖)
        if random.random() > 0.5:
            angle = random.randint(-180, 180)
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

        # 2. 随机缩放裁剪 (MSHNet 与 PP-Net 均强调的多尺度感知)
        if random.random() > 0.5:
            scale = random.uniform(0.7, 1.3)
            new_size = (int(self.input_size[0] * scale), int(self.input_size[1] * scale))
            image = F.resize(image, new_size, Image.BILINEAR)
            mask = F.resize(mask, new_size, Image.NEAREST)
            # 重新裁剪回原大小
            image = F.center_crop(image, self.input_size)
            mask = F.center_crop(mask, self.input_size)

        return image, mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # 兼容不同的 Mask 后缀
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        if not os.path.exists(mask_path):
            # 尝试直接用同名文件
            mask_path = os.path.join(self.masks_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 如果开启增强，先做几何变换，再 Resize
        # 这样可以保留更多旋转后的细节
        if self.augment:
            image, mask = self.augment_data(image, mask)

        # 统一 Resize
        image = image.resize(self.input_size, Image.BILINEAR)
        mask = mask.resize(self.input_size, Image.NEAREST)

        # 归一化 (手动实现，方便控制)
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Mask 转 Tensor
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask