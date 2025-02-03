
'''
    Автор: Лейман М.А.
    Дата создания: 03.02.2025
    Парсинг изображений с яндекс картинок.
'''


import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import shutil

# Настройки
YANDEX_IMAGES_URL = "https://yandex.ru/images/"
SAVE_DIR_THUMBNAILS = "dataset"
CATEGORIES = ["polar bear", "brown bear"]
IMAGES_PER_CATEGORY = 1100


def create_folders():
    '''
        Создание дирректорий для датасета
        thumb_path: миниатюры
    '''
    for category in CATEGORIES:
        thumb_path = os.path.join(SAVE_DIR_THUMBNAILS, category.replace(" ", "_"))
        os.makedirs(thumb_path, exist_ok=True)


def init_driver():
    '''
        Инициализация драйвера
    '''
    options = webdriver.ChromeOptions()
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


import requests

def is_jpg(url):
    """ Проверяет, является ли изображение JPG по HTTP-заголовку Content-Type """
    try:
        response = requests.head(url, timeout=3)
        content_type = response.headers.get("Content-Type", "")
        return "image/jpeg" in content_type
    except requests.RequestException:
        return False

def fetch_images(driver, query, max_images):
    '''
        Поиск миниатюр и полноразмерных изображений (только JPG)
    '''
    driver.get(YANDEX_IMAGES_URL)
    search_box = driver.find_element(By.NAME, "text")
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)
    time.sleep(3)

    thumbnail_urls = []
    full_image_urls = []

    while len(thumbnail_urls) < max_images:
        thumbnails = driver.find_elements(By.CLASS_NAME, "ImagesContentImage-Image")

        for thumb in thumbnails:
            try:
                thumb_url = thumb.get_attribute("src")
                
                # Проверяем формат миниатюры
                if thumb_url and thumb_url.startswith("http") and is_jpg(thumb_url):
                    thumbnail_urls.append(thumb_url)

                if len(thumbnail_urls) >= max_images:
                    break

            except Exception as e:
                print(f"Ошибка: {e}")
                continue

        # Нажимаем "Показать ещё", если кнопка есть
        try:
            more_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "FetchListButton-Button"))
            )
            more_button.click()
            time.sleep(2)
        except:
            print("Кнопка 'Показать ещё' больше не доступна.")
            
        # Прокрутка страницы вниз
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    return list(thumbnail_urls[:1000])


def download_images(image_urls, category, save_dir):
    '''
        Загрузка изображений
    '''
    save_path = os.path.join(save_dir, category.replace(" ", "_"))
    for idx, url in tqdm(enumerate(image_urls), total=len(image_urls), desc=f"Downloading {category}"):
        file_path = os.path.join(save_path, f"{idx:04d}.jpg")
        try:
            response = requests.get(url, stream=True, timeout=5)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
        except Exception as e:
            print(f"Ошибка при загрузке {url}: {e}")


def zip_folder(folder_path, zip_name):
    '''
    Сжимает всю папку в ZIP-архив.
    '''
    zip_path = f"{zip_name}.zip"  # Имя архива
    shutil.make_archive(zip_name, 'zip', folder_path)  # Создание архива
    print(f"Архив создан: {zip_path}")


def main():
    '''
        Основная функция
    '''
    # Получаем путь к папке, где находится сам скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Устанавливаем эту папку как рабочую директорию
    os.chdir(script_dir)
    print("рабочая директория:", os.getcwd())
    
    create_folders()
    driver = init_driver()

    for category in CATEGORIES:
        print(f"Поиск изображений для: {category}")

        thumbnails = fetch_images(driver, category, IMAGES_PER_CATEGORY)
        print(f"Найдено {len(thumbnails)} миниатюр")

        print(f"Загружаем миниатюры для: {category}")
        download_images(thumbnails, category, SAVE_DIR_THUMBNAILS)

        print(f"Загрузка {category} завершена!\n")

    driver.quit()
    folder_path = "dataset" 
    zip_folder(folder_path, "dataset_backup") 


if __name__ == "__main__":
    main()


