
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




######### Очистка памяти
def emty_cache(variables=None):
    """ 
        Очистка памяти
        variables: список переменных
    """
    if variables is not None:
        for v in variables:
            del v
    gc.collect()  
    torch.cuda.empty_cache()


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



class CityscapesFlatDataset3(Dataset):
    """ 
        Датасет: загрузка с openCV
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

        # image = torch.from_numpy(image)
        # target = torch.from_numpy(target)

        # Конвертация в тензор и нормализация вручную
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 
        target = torch.from_numpy(target).long()

        # print(image.shape)
        # print(target.shape)

        # Применяем трансформации, если есть
        if self.transform:
            image = image.unsqueeze(0)  # (1, C, H, W)
            image = self.transform(image)
            image = image.squeeze(0)    # (C, H, W)
        if self.target_transform:
            target = target.unsqueeze(0)  # (1, H, W)
            target = self.target_transform(target)
            target = target.squeeze(0)    # (H, W)

        return image, target



class CustomAugmentationsNumPy:
    def __init__(self, img_size=(256, 512), p_flip=0.3, p_brightness=0.2, p_noise=0.2):
        self.img_size = img_size
        self.p_flip = p_flip
        self.p_brightness = p_brightness
        self.p_noise = p_noise

    def _add_noise(self, img_np, std=10):
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
            factor = random.uniform(0.8, 1.2)
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



 # #########################  Создание даталоадеров


from torch.utils.data import Subset

def prepare_cityscapes_loaders(dataset_class, root_dir,
                               size,
                               batch_size,
                               val_ratio=0.2,
                               num_workers=0):
    # Трансформации для train (с аугментациями)
    train_transforms = CustomAugmentationsNumPy(
        img_size=(512, 256),
        p_flip=0.5,
        p_brightness=0.2,
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






def prepare_cityscapes_loaders2(dataset_class, root_dir,
                                    size,
                                    batch_size,
                                    val_ratio=0.2,
                                    num_workers=0
                                ):
    """ 
        Создание даталоадеров
    """
    # Преобразования для изображений
    transform_img = transforms.Compose([
        transforms.Resize(size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Преобразования для масок
    transform_mask = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
        # transforms.PILToTensor(),  # если нужно получить индексы как uint8
    ])


    train_transforms = CustomAugmentationsNumPy(
        img_size=(256, 512),
        p_flip=0.5,
        p_brightness=0.2,
        p_noise=0.1
    )


    # Создаём датасет
    dataset = dataset_class(
        root_dir=root_dir,
        transform=train_transforms,
        target_transform=transform_mask
    )

    # Разделение на train/val
    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader-
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f'Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}')
    return train_loader, val_loader, train_dataset, val_dataset



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











def visualize_image_and_mask3(image, mask, class_palette):
    """ 
        Визуализвция изображения с масками
    """

        
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        # Если mask PIL Image, конвертируем в numpy
        mask_np = np.array(mask)

    # Создаём копию маски для отображения
    mask_vis = mask_np.copy()
    
    # Заменяем 255 (ignore) на 0 или другой индекс, например фон
    mask_vis[mask_vis == 255] = 19
    
    # Применяем палитру
    color_mask = class_palette[mask_vis]
    
    # Отображение
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title('Image')
    axs[0].axis('off')
    
    axs[1].imshow(color_mask)
    axs[1].set_title('Mask')
    axs[1].axis('off')
    
    plt.show()




def visualize_image_and_mask2(image, mask, class_palette):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(image, torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3,1,1)

        image = image * std + mean  # правильная денормализация
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        # Если mask PIL Image, конвертируем в numpy
        mask_np = np.array(mask)
    mask_vis = mask_np.copy()
    mask_vis[mask_vis == 255] = 19  # ignore -> фон
    color_mask = class_palette[mask_vis.squeeze()]

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


# ###################  Тренировка
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=20, scheduler=None, save_path='best_model.pth'):
    model.to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_ious, val_ious = [], []
    best_val_iou = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    def iou_score(pred, target, n_classes=19):
        # Предполагается, что pred и target — тензоры (N,H,W) с классами
        ious = []
        pred = pred.view(-1)
        target = target.view(-1)
        for cls in range(n_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()
            if union == 0:
                ious.append(float('nan'))  # класс отсутствует в выборке
            else:
                ious.append(intersection / union)
        # Средний IoU по классам, игнорируя nan
        ious = [iou for iou in ious if not np.isnan(iou)]
        if len(ious) == 0:
            return 0
        return np.mean(ious)

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0
        running_corrects = 0
        running_total = 0
        running_iou = 0
        corrects1 = 0
        corrects2 = 0
        train_batches = len(train_loader)
        val_batches = len(val_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        for images, masks in pbar:
            images = images.to(device)
           
            masks = masks.to(device).long()
            optimizer.zero_grad()
            masks = masks.squeeze(1).long()
            outputs = model(images)  # [B, C, H, W]
            # loss = F.cross_entropy(outputs, masks, ignore_index=0)
            loss = F.cross_entropy(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() 

            preds = torch.argmax(outputs, dim=1)
            corrects1 += (preds == masks).sum().item()
            running_total += masks.numel()

            batch_iou = iou_score(preds.cpu(), masks.cpu())
            running_iou += batch_iou 
            acc=corrects1/running_total
            running_corrects += acc
            pbar.set_postfix(loss=loss.item(), acc=acc, iou=batch_iou)



        epoch_loss = running_loss / train_batches
        epoch_acc = running_corrects / train_batches
        epoch_iou = running_iou / train_batches

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        train_ious.append(epoch_iou)

        # Валидация
        model.eval()
        val_loss = 0
        val_corrects = 0
        val_total = 0
        val_iou_sum = 0

        with torch.no_grad():
            pbar2 = tqdm(val_loader, desc=f"Epoch {epoch} Val")
            for images, masks in pbar2:
                images = images.to(device)
               
                masks = masks.to(device).long()
                masks = masks.squeeze(1).long()

                outputs = model(images)
                loss = F.cross_entropy(outputs, masks)

                val_loss += loss.item() 

                preds = torch.argmax(outputs, dim=1)
                corrects2 += (preds == masks).sum().item()
                val_total += masks.numel()

                batch_iou = iou_score(preds.cpu(), masks.cpu())
                val_iou_sum += batch_iou 
                acc=corrects2/val_total
                val_corrects += acc

                pbar2.set_postfix(loss=loss.item(), acc=acc, iou=batch_iou)

        val_epoch_loss = val_loss / val_batches
        val_epoch_acc = val_corrects / val_batches
        val_epoch_iou = val_iou_sum / val_batches

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        val_ious.append(val_epoch_iou)

        print(f"Epoch {epoch} summary: Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, IoU: {epoch_iou:.4f} | Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}, IoU: {val_epoch_iou:.4f}")
        # Сохраняем модель, если улучшился val IoU
        if val_epoch_iou > best_val_iou:
            best_val_iou = val_epoch_iou
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f"Сохранена лучшая модель {epoch} с val IoU: {best_val_iou:.4f}")
                  
        if scheduler is not None:
            scheduler.step()
            
    # Восстанавливаем лучшие веса
    model.load_state_dict(best_model_wts)
    # Построение графиков
    epochs = range(1, num_epochs+1)

    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.subplot(1,3,2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')

    plt.subplot(1,3,3)
    plt.plot(epochs, train_ious, label='Train IoU')
    plt.plot(epochs, val_ious, label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('IoU over epochs')

    plt.show()



