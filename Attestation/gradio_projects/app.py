
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import gradio as gr


if __name__ == "__main__":
    # =============   Если запускать с "app.py"
    from projects.segmentation_1 import get_segmentation_tab
    from projects.home_tab import home_tab
else:
    # ===============  Если запускать с ноутбука  "attestation.ipynb"
    from gradio_projects.projects.segmentation_1 import get_segmentation_tab
    from gradio_projects.projects.home_tab import home_tab

def main():
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Главная"):
                home_tab()
            with gr.TabItem("Сегментация дорожных сцен"):
                get_segmentation_tab()

    demo.launch()

if __name__ == "__main__":
    main()

