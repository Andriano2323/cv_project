import streamlit as st
from PIL import Image
import torch
import json
import sys
from pathlib import Path
import requests
from io import BytesIO
import time
import PIL
from PIL import ImageDraw



st.write("# Локализация объектов")
st.write("Загрузите картинку для локализации")
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from models.model_1.model_localization import LocModel

from models.model_1.preprocessing_localization import preprocess


# Загрузка модели и словаря
@st.cache_resource
def load_model():
    device = torch.device('cpu') #установка устройства device на cpu
    model = LocModel() #создаем экземпляр модели
    weights_path = 'models/model_1/weights_localization.pt' #загружаем веса
    state_dict = torch.load(weights_path, map_location=device) #загружаем веса и пуляем их на device
    model.load_state_dict(state_dict)

    # model.clf.load_state_dict(state_dict)
    # model.box.load_state_dict(state_dict)


    model.to(device)
    model.eval()
    return model

model = load_model()

id_class = json.load(open('models/model_1/id_class_localization.json'))
id_class = {int(k): v for k, v in id_class.items()}

# Функция для предсказания класса изображения
def predict(image):
    img = preprocess(image)
    with torch.no_grad():
        start_time = time.time()
        preds = model(img.unsqueeze(0))
        end_time = time.time()
    pred_class = preds[0].argmax(dim=1).item()
    bbox_coords = preds[1].tolist()
    pred_bbox = (bbox_coords[0][0], bbox_coords[0][1], bbox_coords[0][2], bbox_coords[0][3])

# preds - это тензор, вероятностный вывод модели (например, после применения softmax), 
# который содержит предсказанные вероятности принадлежности к различным классам. 
# Каждый элемент тензора представляет вероятность принадлежности соответствующему классу.
# preds.argmax(dim=1) - этот метод возвращает индекс элемента с наибольшим значением в каждой строке тензора preds. 
# То есть для каждого примера модель выбирает индекс класса с наибольшей вероятностью.
# .item() - этот метод преобразует полученный тензор с единственным элементом 
# (индексом класса с наибольшей вероятностью) в Python числовой тип данных.
# pred_class - в результате выполнения данной строки кода переменная pred_class 
# будет содержать индекс класса с наибольшей предсказанной вероятностью для конкретного примера.

    return id_class[pred_class], end_time - start_time, pred_bbox



# Загрузка изображения по ссылке
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Загрузка изображения через загрузку файла или по ссылке
def load_image(image):
    if isinstance(image, BytesIO):
        return Image.open(image)
    else:
        return load_image_from_url(image)

# Загрузка изображений и предсказание класса
def predict_images(images):
    predictions = []
    for img in images:
        image = load_image(img)
        prediction, inference_time, pred_bbox= predict(image)
        predictions.append((image, prediction, inference_time,pred_bbox))

    return predictions



# Отображение изображения и результатов предсказания
def display_results(predictions):
    for img, prediction, inference_time, pred_bbox in predictions:
        st.image(img) 
        st.write(f'Класс картинки: {prediction}')
        st.write(f'Inference Time: {inference_time:.4f} seconds')
        st.write(f'Bbox: {pred_bbox}')

# ТО что новое

def draw_bbox(image, bbox):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red", width=3)
    return image

# Отображение изображения с bbox
def display_results(predictions):
    for img, prediction, inference_time, pred_bbox in predictions:
        img_with_bbox = draw_bbox(img, pred_bbox)
        st.image(img_with_bbox)
        st.write(f'Prediction: {prediction}')
        st.write(f'Inference Time: {inference_time:.4f} seconds')
        st.write(f'Bbox: {pred_bbox}')

# ТО что новое конец



# Загрузка изображений через файлы или ссылки
images = st.file_uploader('Upload file(s)', accept_multiple_files=True)

if not images:
    image_urls = st.text_area('Enter image URLs (one URL per line)', height=100).strip().split('\n')
    images = [url.strip() for url in image_urls if url.strip()]

if images:
    predictions = predict_images(images)
    display_results(predictions)

# Определение корневого каталога проекта и добавление его в sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))




# Информация о первой модели
st.write("Использовалась предобученная модель - ResNet18 с заменой последних двух слоев")
st.write("Модель обучалась на предсказание 3 классов")
st.write("Размер train датасета - 148 картинок")
st.write("Размер valid датасета - 38 картинок")
st.write("Время обучения модели - 70 эпох = 18 минут, batch_size = 32")
st.image(str(project_root / '/Users/daravelikohatko/.ssh/ds-phase-2/cv_project/images/photo_2024-05-23 17.16.53.jpeg'), width=900)

# st.write("Значения метрики f1 на последней эпохе: 0.695-train и 0.840-valid")
# st.write('Confusion matrix')
# st.image(str(project_root / 'images/image2.jpeg'))


