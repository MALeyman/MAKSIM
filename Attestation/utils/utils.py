


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

import onnxruntime as ort
import numpy as np
from PIL import Image




######### Очистка памяти
def emty_cache():
    """ 
        Очистка памяти
    """
    gc.collect()  
    torch.cuda.empty_cache()



# #########################  Визуализация изображений
def visualize_image_and_mask(image, mask, class_palette):
    """
    Универсальная визуализация изображения и маски.
    Поддерживает входные изображения из OpenCV (NumPy), PIL, PyTorch Tensor.
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # --- Обработка изображения ---
    if isinstance(image, torch.Tensor):
        # PyTorch Tensor: [C, H, W] -> [H, W, C] + денормализация
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3,1,1)
        image = image * std + mean
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
    elif 'PIL' in str(type(image)):
        # PIL Image -> NumPy
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        # OpenCV: BGR -> RGB
        if image.ndim == 3 and image.shape[2] == 3:
            # Обычно OpenCV-изображения в BGR, а matplotlib ожидает RGB
            image = image[..., ::-1]
    else:
        raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")

    # --- Обработка маски ---
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        # PIL Image
        mask_np = np.array(mask)

    mask_vis = mask_np.copy()
    mask_vis[mask_vis == 255] = 19  # ignore -> фон (или другой индекс, если требуется)
    color_mask = class_palette[mask_vis.squeeze()]

    # --- Визуализация ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title('Image')
    axs[0].axis('off')

    axs[1].imshow(color_mask)
    axs[1].set_title('Mask')
    axs[1].axis('off')

    plt.show()


# #################  Визуализация 
def decode_segmap(mask, colormap):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(colormap)):
        color_mask[mask == cls_id] = colormap[cls_id]
    return color_mask

  
# Функция визуализации
def visualize_segmentation(image_pil, pred_mask, colormap, alpha=0.5):
    """
    image_pil: PIL.Image — исходное изображение
    pred_mask: torch.Tensor или numpy.ndarray (H, W) с классами
    colormap: dict с цветами классов
    alpha: прозрачность наложения маски
    """
    # Преобразуем PIL Image в numpy (H, W, 3)
    image_np = np.array(image_pil.convert("RGB"))

    # Если pred_mask — тензор, преобразуем в numpy
    if hasattr(pred_mask, 'cpu'):
        pred_mask = pred_mask.cpu().numpy()

    # Убираем лишние размерности, если есть
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)
    if pred_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {pred_mask.shape}")

    # Получаем цветную маску
    color_mask = decode_segmap(pred_mask, colormap)
    image_resized = image_pil.resize((512, 256), resample=Image.BILINEAR)
    image_np = np.array(image_resized)
    # Накладываем маску на изображение с прозрачностью
    overlay = (image_np * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    # Визуализация
    plt.figure(figsize=(20, 20))

    plt.subplot(1, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(image_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Сегментированные маски")
    plt.imshow(color_mask)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Наложение")
    plt.imshow(overlay)
    plt.axis('off')

    plt.show()




# Объединение датасета
def merge_folders(src_root, dst_folder):
    """ 
       Объединение исходного датасета в общий каталог 
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for split in ['train', 'val']:
        split_path = os.path.join(src_root, split)
        for city in os.listdir(split_path):
            city_path = os.path.join(split_path, city)
            if os.path.isdir(city_path):
                for file_name in os.listdir(city_path):
                    src_file = os.path.join(city_path, file_name)
                    dst_file = os.path.join(dst_folder, file_name)
                    # Если имена файлов могут совпадать, можно добавить префикс или суффикс
                    if os.path.exists(dst_file):
                        base, ext = os.path.splitext(file_name)
                        dst_file = os.path.join(dst_folder, f"{split}_{city}_{base}{ext}")
                    shutil.copy2(src_file, dst_file)



def preprocess_image_onnx(path_img, input_size=(512, 256)):
    img = Image.open(path_img).convert('RGB')
    img = img.resize(input_size)  
    img_np = np.array(img).astype(np.float32) / 255.0

    # Нормализация ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    # HWC -> CHW
    img_np = img_np.transpose(2, 0, 1)

    # Добавляем batch dimension
    img_np = np.expand_dims(img_np, axis=0)

    return img, img_np

def prediction_mask_onnx(path_img, onnx_session):
    img, img_np = preprocess_image_onnx(path_img)

    
    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: img_np})

    # outputs[0] — выход модели, shape (1, num_classes, H, W)
    pred_mask = np.argmax(outputs[0], axis=1)[0]  # (H, W)

    return img, pred_mask

