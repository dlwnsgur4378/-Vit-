import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 모델 불러오기
model = YOLO("best_emotion_model.pt")

# 예측 함수
def predict(img):

    results = model(img)

    # 결과 이미지 생성
    plotted = results[0].plot()

    return plotted

# 웹 인터페이스
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(),
    title="강아지 감정 분석 AI"
)

demo.launch(server_name="0.0.0.0")
