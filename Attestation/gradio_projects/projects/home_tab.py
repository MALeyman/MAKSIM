

import gradio as gr

def home_tab():
    with gr.Blocks() as tab:
        gr.Markdown(
            """
            # Добро пожаловать на главную страницу!  

            ## __Проекты__:  

            ##  1. Сегментация дорожных сцен  
            ##  2. 

            ---
            *Примечание:* 
            """
        )
    return tab
