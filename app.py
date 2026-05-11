import gradio as gr
from PIL import Image
import numpy as np

def predict(img):
    return img

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(),
    title="Dog Emotion AI"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
