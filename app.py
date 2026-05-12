import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("best_emotion_model.pt")

def predict(img):
    results = model(img)

    names = results[0].names
    probs = results[0].probs.data.cpu().numpy()

    best_idx = np.argmax(probs)
    emotion = names[best_idx]
    confidence = probs[best_idx]

    return f"감정: {emotion} ({confidence:.2f})"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="강아지 감정 분석 AI"
)

demo.launch(server_name="0.0.0.0")
