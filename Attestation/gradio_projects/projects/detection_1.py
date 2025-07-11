
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import gradio as gr
import sys



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print("ПУТЬ:  ", BASE_DIR)

# Корень проекта — на два уровня выше, т.к. __file__ в gradio_projects/projects
# ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
# Добавляем корень проекта в sys.path, если его там нет
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# print("ПУТЬ:  ", ROOT_DIR)
# Теперь импортируем модуль как абсолютный из корня проекта
from projects.files.utils import predict_jsons1, vis_annotations, gradio_video_processing, onnx_inference
from projects.common.session import ort_session, get_device


image_path = os.path.join(BASE_DIR, "files/3.jpg")
video_path = os.path.join(BASE_DIR, "files/1.mp4")
model_path = os.path.join(BASE_DIR, "files/retinaface_resnet50.onnx")

# session = ort.InferenceSession(model_path)
import gradio as gr
import cv2
import numpy as np
import onnxruntime
import torch



# Загрузка ONNX Runtime с CUDA
# providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# ort_session = onnxruntime.InferenceSession(model_path, providers=providers)

def onnx_inference(image: np.ndarray, confidence_threshold=0.7, nms_threshold=0.4, max_size=1200):
    """
    Функция для инференса ONNX модели на входном изображении.
    image: RGB numpy array
    Возвращает изображение с аннотациями (BGR numpy array)
    """
    # Конвертируем BGR (OpenCV) в RGB, если нужно
    if image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image

    # Запуск предсказания (здесь вызывается ваша функция predict_jsons1)
    annotation = predict_jsons1(
        ort_session,
        img_rgb,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        max_size=max_size,
    )

    # Визуализация аннотаций
    img_vis = vis_annotations(img_rgb, annotation)

    # Конвертируем обратно в BGR для отображения в OpenCV/Gradio
    img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    return img_bgr

# def get_device():
#     providers = ort.get_available_providers()
#     if 'CUDAExecutionProvider' in providers:
#         return "Устройство: GPU (CUDA)"
#     else:
#         return "Устройство: CPU"


def get_detection_tab():
    with gr.Blocks() as demo:
        gr.Markdown("## Детекция лиц с помощью ONNX (Промежуточная аттестация 4)")
        gr.Markdown("---")
        with gr.Row():
            # Левая колонка со слайдерами (общие для обеих вкладок)
            with gr.Column(scale=1):
                # Добавляем метку устройства
                device_label = gr.Label(value=get_device(), label="Работаем на устройстве")
                confidence_slider = gr.Slider(0, 1, value=0.7, label="Порог уверенности")
                nms_slider = gr.Slider(0, 1, value=0.4, label="Порог NMS")
                max_size_slider = gr.Slider(256, 2048, value=1200, step=64, label="Максимальный размер изображения")


            # Правая колонка с вкладками для изображения и видео
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("Изображение"):
                        input_image = gr.Image(type="numpy", value=image_path, label="Загрузите изображение", height=320)
                        output_image = gr.Image(label="Результат с детекцией")
                        btn_img = gr.Button("Запустить детекцию")

                        btn_img.click(
                            onnx_inference,
                            inputs=[input_image, confidence_slider, nms_slider, max_size_slider],
                            outputs=output_image,
                        )

                    with gr.TabItem("Видео"):
                        video_io = gr.Video(
                            label="Загрузите видео или используйте веб-камеру",
                            sources=["upload", "webcam"],  # позволяет выбрать загрузку или веб-камеру
                            value=video_path,
                            height=512
                        )
                        frame_skip_slider = gr.Slider(1, 10, value=4, step=1, label="Обрабатывать каждый n-й кадр")
                        btn_vid = gr.Button("Запустить обработку")

                        btn_vid.click(
                            gradio_video_processing,
                            inputs=[video_io, confidence_slider, nms_slider, max_size_slider, frame_skip_slider],
                            outputs=video_io,
                        )


    return demo






