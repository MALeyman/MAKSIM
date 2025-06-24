""" 
Автор: Лейман М.А.
Дата создания: 24.06.2025
"""

import numpy as np
import os
import shutil
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import cv2
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import gc

import torchvision.transforms as T
import random

import random, numpy as np, torch
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF
from torch.utils.data import Subset



# #########################   Датасет
class CityscapesFlatDataset(Dataset):
    """  
        Датасет: загрузка с PIL
    """
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.images_dir = os.path.join(root_dir, 'images')
        self.targets_dir = os.path.join(root_dir, 'targets')
        self.transform = transform
        self.target_transform = target_transform

        self.images = sorted(os.listdir(self.images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Формируем имя маски, заменяя суффикс
        target_name = img_name.replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')
        target_path = os.path.join(self.targets_dir, target_name)

        image = Image.open(img_path).convert('RGB')
        target = Image.open(target_path)
        target_np = np.array(target)
        target_np = target_np + 1
        target_np[target_np == 256] = 0
        target = Image.fromarray(target_np)
        # print(image.size)
        # print(target.size)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target
    


class CityscapesFlatDataset2(Dataset):
    """ 
        Датасет: загрузка с openCV
    """
    def __init__(self, root_dir, transform=None, val_transform=None):
        self.images_dir = os.path.join(root_dir, 'images')
        self.targets_dir = os.path.join(root_dir, 'targets')
        self.transform = transform
        self.val_transform = val_transform

        self.images = sorted(os.listdir(self.images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Формируем имя маски, заменяя суффикс
        target_name = img_name.replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')
        target_path = os.path.join(self.targets_dir, target_name)

        # Загрузка изображения с помощью OpenCV (BGR -> RGB)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Загрузка маски (одноканальное изображение)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target is None:
            raise FileNotFoundError(f"Mask not found or corrupted: {target_path}")

        # Смещаем классы на +1, заменяем 256 на 0 (фон)
        target = target.astype(np.int32) + 1
        target[target == 256] = 0
        target = target.astype(np.uint8)

        image = image.transpose(2, 0, 1)
        # Применяем трансформации, если есть
        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = cv2.resize(
                image.transpose(1, 2, 0), 
                (512, 256), 
                interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)
            target = cv2.resize(
                target, 
                (512, 256), 
                interpolation=cv2.INTER_NEAREST
            )


        image = torch.from_numpy(image).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        image = (image - mean) / std
        
        # Конвертация в тензоры
        # image = torch.from_numpy(image).float() / 255.0  # уже в формате CHW
        target = torch.from_numpy(target).long()

        return image, target


class CustomAugmentationsNumPy:
    def __init__(self, img_size=(256, 512), p_flip=0.3, p_brightness=0.2, p_noise=0.2):
        self.img_size = img_size
        self.p_flip = p_flip
        self.p_brightness = p_brightness
        self.p_noise = p_noise

    def _add_noise(self, img_np, std=5):
        noise = np.random.normal(0, std, img_np.shape).astype(np.float32)
        img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img_np

    def __call__(self, img_np, mask_np):
        # Проверка типов и размеров
        assert isinstance(img_np, np.ndarray), f"Image must be numpy array, got {type(img_np)}"
        assert isinstance(mask_np, np.ndarray), f"Mask must be numpy array, got {type(mask_np)}"
        
        # Вертикальный флип
        if random.random() < self.p_flip:
            img_np = np.flipud(img_np).copy()
            mask_np = np.flipud(mask_np).copy()

        # Изменение яркости
        if random.random() < self.p_brightness:
            factor = random.uniform(0.9, 1.1)
            img_np = np.clip(img_np.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Добавление шума
        if random.random() < self.p_noise:
            img_np = self._add_noise(img_np)

        # Ресайз (для OpenCV формат должен быть HWC)
        try:
            # Для изображения: меняем формат с CHW -> HWC
            img_resized = cv2.resize(
                img_np.transpose(1, 2, 0), 
                (self.img_size[0], self.img_size[1]), 
                interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)  # обратно в CHW
            

            # Для маски: сохраняем HW
            mask_resized = cv2.resize(
                mask_np, 
                (self.img_size[0], self.img_size[1]), 
                interpolation=cv2.INTER_NEAREST
            )


        except Exception as e:
            print(f"Error during resize: {e}")
            print(f"Image shape: {img_np.shape}, Target shape: {mask_np.shape}")
            raise

        return img_resized, mask_resized




 # #########################  Создание даталоадеров

def prepare_cityscapes_loaders(dataset_class, root_dir,
                               size,
                               batch_size,
                               val_ratio=0.2,
                               num_workers=0):
    # Трансформации для train (с аугментациями)
    train_transforms = CustomAugmentationsNumPy(
        img_size=(512, 256),
        p_flip=0.2,
        p_brightness=0.1,
        p_noise=0.1
    )

    # Трансформации для val (без аугментаций, только resize)
    val_transform_img =  CustomAugmentationsNumPy(
        img_size=(512, 256),
        p_flip=0.0,
        p_brightness=0.0,
        p_noise=0.0
    )


    # Создаём полный датасет без трансформаций
    full_dataset = dataset_class(root_dir=root_dir, transform=None, val_transform=None)

    # Разбиваем индексы
    total_size = len(full_dataset)
    indices = list(range(total_size))
    split = int(val_ratio * total_size)

    train_indices = indices[split:]
    val_indices = indices[:split]

    # Создаём train и val датасеты с разными трансформациями
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

 
    train_full = dataset_class(root_dir=root_dir, transform=train_transforms, val_transform=None)
    val_full = dataset_class(root_dir=root_dir, transform=val_transform_img, val_transform=None)

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}')
    return train_loader, val_loader, train_dataset, val_dataset

