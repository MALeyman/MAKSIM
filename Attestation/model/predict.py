

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




# Предсказание модели
def prediction_mask(path_img, model):
    """ 
        Сегментация изображения
        path_img: путь к изображению
        model: модель
    """
    img = Image.open(path_img)
    preprocess = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0) 

    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_tensor = img_tensor.to(device) 
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1) 
        return img, prediction

