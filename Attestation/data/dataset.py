""" 
Автор: Лейман М.А.
Дата создания: 24.06.2025
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
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
# загрузка с PIL
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



# загрузка с openCV
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

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target is None:
            raise FileNotFoundError(f"Mask not found or corrupted: {target_path}")

        # Смещаем классы на +1, заменяем 256 на 0 (фон)
        target = target.astype(np.int32) + 1
        target[target == 256] = 0
        target = target.astype(np.uint8)

        image = image.transpose(2, 0, 1)
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
        
        target = torch.from_numpy(target).long()
        return image, target



class CityscapesFlatDataset3(Dataset):
    """Датасет для новых изображений и масок с одинаковыми именами файлов."""

    def __init__(self, root_dir, transform=None, val_transform=None, img_size=(512, 256)):
        self.images_dir = os.path.join(root_dir, 'images')
        self.targets_dir = os.path.join(root_dir, 'targets')
        self.transform = transform
        self.val_transform = val_transform
        self.img_size = img_size

        # Список файлов изображений
        self.images = sorted(os.listdir(self.images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        target_path = os.path.join(self.targets_dir, img_name)  # маска с тем же именем

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupted: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if target is None:
            raise FileNotFoundError(f"Mask not found or corrupted: {target_path}")

       
        # target = target.astype(np.int32) + 1
        target[target == 20] = 0
        # target = target.astype(np.uint8)

        # Применяем resize к изображению и маске
        image = cv2.resize(image, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)

        # Преобразуем в формат CHW для PyTorch
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0


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

        # Нормализация
        mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        image = (image - mean) / std


        image = torch.from_numpy(image).float()
        target = torch.from_numpy(target).long()

        return image, target



# Аугментации
class CustomAugmentationsNumPy:
    def __init__(
        self,
        img_size=(256, 512),
        p_flip=0.0,
        p_hflip=0.0,
        p_brightness=0.0,
        p_noise=0.0,
        p_swap_channels=0.0,  # Вероятность смены каналов
        p_contrast=0.0,
        p_saturation=0.0

    ):
        """ 
        Аугментации
        """
        self.img_size = img_size
        self.p_flip = p_flip
        self.p_brightness = p_brightness
        self.p_noise = p_noise
        self.p_swap_channels = p_swap_channels  # Новый параметр
        self.p_contrast = p_contrast
        self.p_saturation = p_saturation
        self.p_hflip = p_hflip


    def _add_noise(self, img_np, std=0.01):
        noise = np.random.normal(0, std, img_np.shape).astype(np.float32)
        img_np = np.clip(img_np + noise, 0, 1)
        return img_np


    def adjust_contrast(self, img_np, factor):
        mean = img_np.mean(axis=(1, 2), keepdims=True)
        img_adj = np.clip((img_np - mean) * factor + mean, 0, 1)
        return img_adj


    def adjust_saturation(self, img_np, factor):
        # img_np: CHW, float32, диапазон [0, 1]
        img_hwc = np.transpose(img_np, (1, 2, 0))
        img_hwc = np.clip(img_hwc, 0, 1)
        img_hsv = cv2.cvtColor((img_hwc * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * factor, 0, 255)
        img_hsv = img_hsv.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return np.transpose(img_rgb, (2, 0, 1))


    def __call__(self, img_np, mask_np):
        # Проверка типов и размеров
        assert isinstance(img_np, np.ndarray), f"Image must be numpy array, got {type(img_np)}"
        assert isinstance(mask_np, np.ndarray), f"Mask must be numpy array, got {type(mask_np)}"
        

        if random.random() < self.p_hflip:  # p_hflip — вероятность горизонтального флипа
            img_np = img_np[:, :, ::-1]
            mask_np = mask_np[:, ::-1]

        if random.random() < self.p_noise:  # Добавление шума
            img_np = self._add_noise(img_np)

        if random.random() < self.p_contrast:  # Изменение контраста
            factor = random.uniform(0.5, 3)
            img_np = self.adjust_contrast(img_np, factor)

        if random.random() < self.p_saturation:  # Изменение насыщенности
            factor = random.uniform(0.3, 4)
            img_np = self.adjust_saturation(img_np, factor)
        

        if random.random() < self.p_brightness:   # Изменение яркости
            factor = random.uniform(0.5, 1.5)
            img_np = np.clip(img_np * factor, 0, 1)

        if random.random() < self.p_swap_channels:   # Смена каналов RGB <-> BGR (до ресайза)
            # Для формата CHW (каналы, высота, ширина)
            img_np = img_np[::-1, :, :].copy()  # Инвертирование порядка каналов


        # Ресайз
        try:
            # Для изображения: CHW -> HWC для OpenCV
            img_resized = cv2.resize(
                img_np.transpose(1, 2, 0), 
                (self.img_size[1], self.img_size[0]),  # (width, height)
                interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)  # Обратно в CHW
            
            # Для маски: HW формат
            mask_resized = cv2.resize(
                mask_np, 
                (self.img_size[1], self.img_size[0]), 
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
                               val_ratio=0.1,
                               num_workers=0,
                                p_hflip=0.2,
                                p_brightness=0.2,
                                p_noise=0.2,
                                p_swap_channels=0.01, 
                                p_contrast=0.1,
                                p_saturation=0.1):
    """ 
    Содание даталоадеров 
    """
                            
    # Трансформации для train (с аугментациями)
    train_transforms = CustomAugmentationsNumPy(
        img_size=(256, 512),
        p_flip=0.0,
        p_hflip=p_hflip,
        p_brightness=p_brightness,
        p_noise=p_noise,
        p_swap_channels=p_swap_channels, 
        p_contrast=p_contrast,
        p_saturation=p_saturation
    )

    # Трансформации для val (без аугментаций, только resize)
    val_transform_img =  CustomAugmentationsNumPy(
        img_size=(256, 512),
        p_flip=0.0,
        p_hflip=0.0,
        p_brightness=0.0,
        p_noise=0.0,
        p_swap_channels=0.0, 
        p_contrast=0.0,
        p_saturation=0.0
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    print(f'Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}')
    return train_loader, val_loader, train_dataset, val_dataset

