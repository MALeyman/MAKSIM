


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
