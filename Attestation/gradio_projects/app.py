
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import gradio as gr
from projects.segmentation_1 import get_segmentation_tab
from projects.home_tab import home_tab

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

