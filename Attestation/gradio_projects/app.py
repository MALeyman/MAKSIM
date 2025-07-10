
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import gradio as gr


if __name__ == "__main__":
    # =============   Если запускать с "app.py"
    from projects.segmentation_1 import get_segmentation_tab
    from projects.detection_1 import get_detection_tab
    from projects.home_tab import home_tab
else:
    # ===============  Если запускать с ноутбука  "gradio.ipynb"
    from gradio_projects.projects.segmentation_1 import get_segmentation_tab
    from gradio_projects.projects.home_tab import home_tab
    from gradio_projects.projects.detection_1 import get_detection_tab

def main():
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Главная"):
                home_tab()
            with gr.TabItem("Сегментация дорожных сцен"):
                get_segmentation_tab()
            with gr.TabItem("Детекция лиц"):
                 get_detection_tab()
                # detection_demo.render()

    demo.launch()

if __name__ == "__main__":
    main()

