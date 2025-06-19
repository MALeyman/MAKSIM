""" 
Автор: Лейман Максим  

Дата создания: 18.06.2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from shutil import move
import xml.etree.ElementTree as ET
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


#  Просмотр предсказанных точек
def show_image_with_predictions(path_image, model, device):
    """
    Визуализирует изображение и предсказанные ключевые точки.
    img_tensor: torch.Tensor [C, H, W] или [H, W, C], значения 0...1 или 0...255
    predicted_keypoints: np.array shape [N, 2] или torch.Tensor [N, 2]
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
        transforms.ToTensor(),    # [0,1]
        # transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # # Загрузка и подготовка изображения
    orig_img = Image.open(path_image).convert('RGB')


    img_tensor = transform(orig_img)  # [C, H, W]
    img_tensor = img_tensor.to(device) 
    print(img_tensor.shape)
    # Предсказание
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))  # [1, N*2]
    predicted_keypoints = output.cpu().numpy().reshape(-1, 2)  # [[x1, y1], [x2, y2], ...]

    # Приводим изображение к numpy и нужному формату
    if isinstance(img_tensor, torch.Tensor):
        if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]:
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img = img_tensor.cpu().numpy()
    else:
        img = img_tensor

    # Если изображение нормализовано (0...1), переводим в 0...255 для корректного отображения
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
   
    # Приводим точки к numpy
    if isinstance(predicted_keypoints, torch.Tensor):
        predicted_keypoints = predicted_keypoints.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], c='lime', s=10, label='Predicted')
    plt.axis('off')
    plt.title('Predicted Keypoints')
    plt.legend()
    plt.show()



# Визуализирует изображение и ключевые точки из батча DataLoader
def show_batch_with_keypoints(batch, batch_idx=0):
    """
    Визуализирует изображение и ключевые точки из батча DataLoader.
    batch: батч, полученный из DataLoader (например, next(iter(dataloader)))
    batch_idx: индекс изображения в батче для отображения
    """
    # Получаем изображение и ключевые точки
    images = batch['image']  # shape: [B, H, W, C] или [B, C, H, W]
    keypoints = batch['keypoints']  # shape: [B, N, 2]

    # Если изображение в формате [B, C, H, W], переводим в [B, H, W, C]
    if images.ndim == 4 and images.shape[1] in [1, 3]:
        images = images.permute(0, 2, 3, 1).numpy()
    else:
        images = images.numpy()

   
    img = images[batch_idx]
    kps = keypoints[batch_idx].numpy()

    # Если изображение одноканальное, убираем последний размер
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    # print(img)
    plt.figure(figsize=(6, 6))
    plt.imshow(img.astype(np.uint8) if img.max() > 1.5 else img, cmap='gray')
    plt.scatter(kps[:, 0], kps[:, 1], c='r', s=20)
    plt.axis('off')
    plt.title('Изображение с Keypoints')
    plt.show()


# Функция просмотра датасета с аннотациями
def img_show(path_image='dataset/dataset_1/images', path_annotation='dataset/dataset_1/training_1.csv', idx=0):
    """ 
         Функция просмотра датасета с аннотациями

    """

    # Загрузка аннотаций
    df = pd.read_csv(path_annotation)
    # Получаем имя файла изображения из первого столбца
    image_name = df.loc[idx, 'image_name']
    image_path = os.path.join(path_image, image_name)

    # Загружаем изображение через PIL
    image = np.array(Image.open(image_path).convert('RGB'))
    # Получаем координаты ключевых точек (все столбцы, кроме image_name)
    keypoint_cols = [col for col in df.columns if col != 'image_name']
    keypoints = df.loc[idx, keypoint_cols].values.astype(float).reshape(-1, 2)

    # Создаем фигуру с двумя подграфиками
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Слева — оригинальное изображение
    axs[0].imshow(image)
    axs[0].set_title('Оригинал')
    axs[0].axis('off')

    # Справа — изображение с наложенными ключевыми точками
    axs[1].imshow(image)
    axs[1].scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=10)
    axs[1].set_title('С ключевыми точками')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


def show_original_image_with_predictions(path_image, model, device, model_input_size=(224, 224)):
    """
    orig_img: np.ndarray или PIL.Image (исходное изображение, [H, W, C])
    predicted_keypoints: np.ndarray [N, 2] — координаты в масштабе модели (например, 224x224)
    model_input_size: tuple (ширина, высота), в каком размере работала модель
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
        transforms.ToTensor(),    # [0,1]
        # transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # # Загрузка и подготовка изображения
    orig_img = Image.open(path_image).convert('RGB')

    img_tensor = transform(orig_img)  # [C, H, W]
    img_tensor = img_tensor.to(device) 
    print(img_tensor.shape)
    # Предсказание
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))  # [1, N*2]
    predicted_keypoints = output.cpu().numpy().reshape(-1, 2)  # [[x1, y1], [x2, y2], ...]


    # Получаем размер исходного изображения
    if hasattr(orig_img, 'size'):  # PIL.Image
        orig_w, orig_h = orig_img.size
        img = np.array(orig_img)
    else:  # np.ndarray
        orig_h, orig_w = orig_img.shape[:2]
        img = orig_img

    # пересчитаем  в исходный размер
    model_w, model_h = model_input_size
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    keypoints_orig = predicted_keypoints.copy()
    keypoints_orig[:, 0] *= scale_x
    keypoints_orig[:, 1] *= scale_y

    plt.figure(figsize=(8,  8))  # масштаб
    plt.imshow(img)
    plt.scatter(keypoints_orig[:, 0], keypoints_orig[:, 1], c='lime', s=10, label='Predicted')
    plt.axis('off')
    plt.gca().set_aspect('auto')  # пропорции исходного изображения
    plt.tight_layout(pad=0)
    plt.show()



