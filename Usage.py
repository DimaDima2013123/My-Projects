import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

DATASET_PATH = "Dataset/"
MODEL_PATH = "image.classifier.keras"
IMAGE_SIZE = (160, 160)  # Должно совпадать с размером из обучения

# Загружаем модель один раз при старте
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Файл модели не найден по пути: {MODEL_PATH}")

# Получаем имена классов из папок
class_names = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])
num_classes = len(class_names)


def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл не найден по пути: {image_path}")
        return

    # Загружаем изображение через OpenCV
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Ошибка: Не удалось прочитать изображение — {image_path}")
        return

    # Переводим BGR в RGB и меняем размер на (160, 160)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE)

    # Добавляем измерение батча: (160, 160, 3) -> (1, 160, 160, 3)
    input_tensor = tf.expand_dims(img_resized, axis=0)

    # Инференс
    prediction = model.predict(input_tensor, verbose=0)

    # Определение класса
    if num_classes == 2:
        predicted_index = int(prediction[0][0] > 0.5)
    else:
        predicted_index = tf.argmax(prediction[0]).numpy()

    predicted_class = class_names[predicted_index]

    print(f"Модель определила: {predicted_class}")

    # Отображение результатов
    plt.imshow(img_rgb)
    plt.title(f"Предсказание: {predicted_class}")
    plt.axis("off")
    plt.show()


predict_image("Dataset/dogs/dogs_0005.jpg")