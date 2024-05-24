import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('models/model-2/yolo200ep.pt')  # Ensure that best.pt is in the correct path or provide the full path

st.title("Определение объектов с помощью YOLOv8")
st.write("Загрузите одно или несколько изображений или укажите их прямые URL-адреса.")

# Upload multiple images
uploaded_files = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Input for multiple URLs
image_urls = st.text_area("Или введите URL-адреса изображений (по одному в строке)...")

images = []

# Load images from uploaded files
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        images.append((uploaded_file.name, image))

# Load images from URLs
if image_urls:
    urls = image_urls.splitlines()
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            images.append((url, image))
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading image from URL {url}: {e}")

if images:
    for image_name, image in images:
        # st.image(image, caption=f'Uploaded Image: {image_name}', use_column_width=True)
        # st.write(f"Running YOLOv8 inference on {image_name}...")

        # Convert the image to a format suitable for YOLOv8
        image_np = np.array(image)

        # Run YOLOv8 inference
        results = model.predict(source=image_np)

        # Access the first result
        result = results[0]

        # Render the result
        rendered_image = result.plot()

        # Display the results
        st.image(rendered_image, caption=f'Обработанное изображение: {image_name}', use_column_width=True)


# Set up the Streamlit app
st.title("Информация об обучении модели YOLOv8")

# Information about the model and training process
st.header("Информация о модели и обучении")
epochs = 200  # Number of epochs
batch_size = 32  # Batch size
imgsz = 640  # Image size
st.write(f"**Количество эпох:** {epochs}")
st.write(f"**Размер партии:** {batch_size}")
st.write(f"**Размер изображения:** {imgsz}")

# Load dataset information
train_size = 2643
val_size = 247
st.write(f"**Кол-во образцов обучения:** {train_size}")
st.write(f"**Кол-во образцов валидации:** {val_size}")

# Metrics
st.header("Метрики модели")

# Display mAP
map50 = 0.7210
map = 0.4300
st.write(f"**mAP 0.5:** {map50:.4f}")
st.write(f"**mAP 0.5:0.95:** {map:.4f}")

# Display PR curve
st.subheader("Precision-Recall Curve")
# Specify the path to your image
st.image('/home/a/ds-phase-2/09-cv/cv_project/images/PR_curve.png')

# Display confusion matrix
st.subheader("Confusion Matrix")
st.image('/images/confusion_matrix.png')

st.subheader("F1_curve")
st.image('/images/F1_curve.png')

st.subheader("PR_curve")
st.image('/images/PR_curve.png')

st.subheader("P_curve")
st.image('/images/P_curve.png')

st.subheader("R_curve")
st.image('/images/R_curve.png')

