import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Load the YOLO model
model = YOLO('best.pt')  # Ensure that best.pt is in the correct path or provide the full path

st.title("YOLOv8 Inference with Streamlit")
st.write("Upload one or more images or provide direct URLs to run YOLOv8 inference.")

# Upload multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Input for multiple URLs
image_urls = st.text_area("Or enter image URLs (one per line)...")

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
        st.image(rendered_image, caption=f'Processed Image: {image_name}', use_column_width=True)
