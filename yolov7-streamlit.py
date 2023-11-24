import json
import numpy as np
import os
import io
from PIL import Image, ImageDraw, ImageFont
import requests
import streamlit as st

endpoint = os.getenv('ENDPOINT', '')

CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "car|rot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
    "teddy bear", "hair drier", "toothbrush"]

@st.cache(show_spinner=False, suppress_st_warning=True)  # Avoid redundant API calls for the same image
# def inference(json_data):
#     response = requests.post(endpoint, json=json_data)
#     response_json = json.loads(response.text)
#     pred = response_json['data']['ndarray']
#     return pred

def inference(json_data):
    try:
        response = requests.post(endpoint, json=json_data)
        response.raise_for_status()  # 確保沒有 HTTP 錯誤
    except requests.exceptions.RequestException as e:
        st.error(f"Error making API request: {e}")
        return None

    try:
        response_json = response.json()
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response: {e}")
        return None

    # 檢查 API 回應中是否包含 'data' 和 'ndarray' 欄位
    data_field = response_json.get('data', None)
    ndarray_field = data_field.get('ndarray', None) if data_field is not None else None

    if ndarray_field is None:
        st.error("Missing 'data' or 'ndarray' field in API response")
        return None

    pred = ndarray_field

    return pred


def main():
    st.set_page_config(layout='wide')
    st.title('Yolov7 model Detection.')

    if not endpoint:
        st.error('Environment variable: ENDPOINT not set')
        return

    file = st.sidebar.file_uploader('Upload Images:', type=['png', 'jpeg', 'jpg'], accept_multiple_files=False)

    if file:
        original_image = Image.open(file)
        st.sidebar.text("Original Image")
        st.sidebar.image(original_image)
        json_data = {}
        json_data['data'] = {}
        json_data['data']["ndarray"] = [np.array(original_image).tolist()]
        pred = inference(json_data)
        
        image = original_image
        # font = ImageFont.truetype("/project/phusers/yolov7/Yagora.ttf", 30)
        draw = ImageDraw.Draw(image)
        for x1, y1, x2, y2, conf, class_id in pred:
            draw.rectangle(((x1, y1), (x2, y2)), width=2)
            # draw.text((x1, y1), "{}, {}".format(CLASSES[int(class_id)], round(conf, 3)), font = font)
            draw.text((x1, y1), "{}, {}".format(CLASSES[int(class_id)], round(conf, 3)))
        st.text("Prediction Image")
        st.image(image)
        

if __name__ == "__main__":
    main()
