import os
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# =========================================================
# НАСТРОЙКИ
# =========================================================

DATASET_PATH = "Dataset"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 20
VALIDATION_SPLIT = 0.2

# =========================================================
# ЗАГРУЗКА ДАТАСЕТА (tf.data API)
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

classes = train_ds.class_names
num_classes = len(classes)

print("Классы:", classes)
print("Количество классов:", num_classes)

# Оптимизация производительности: кэширование и подгрузка
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =========================================================
# АУГМЕНТАЦИЯ ДАННЫХ
# =========================================================

augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.15),
    RandomZoom(0.15),
    RandomContrast(0.15),
], name="augmentation")

# =========================================================
# МОДЕЛЬ (TRANSFER LEARNING)
# =========================================================

base_model = EfficientNetB0(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)

# Замораживаем базовую модель на первом этапе
base_model.trainable = False

inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

if num_classes == 2:
    outputs = Dense(1, activation="sigmoid")(x)
    loss_function = "binary_crossentropy"
else:
    outputs = Dense(num_classes, activation="softmax")(x)
    loss_function = "sparse_categorical_crossentropy"

model = tf.keras.Model(inputs, outputs)

# =========================================================
# КОМПИЛЯЦИЯ
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_function,
    metrics=["accuracy"]
)

# =========================================================
# CALLBACKS
# =========================================================

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=0.000001,
        verbose=1
    ),
    ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]

# =========================================================
# ЭТАП 1: ОБУЧЕНИЕ КЛАССИФИКАТОРА
# =========================================================

print("\n--- Этап 1: Обучение верхнего слоя ---")
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================================================
# ЭТАП 2: ТОНКАЯ НАСТРОЙКА (FINE-TUNING)
# =========================================================

print("\n--- Этап 2: Тонкая настройка сверточных слоев ---")
base_model.trainable = True

# Замораживаем нижние слои, размораживаем только верхние 30 слоев
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=loss_function,
    metrics=["accuracy"]
)

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)

# =========================================================
# ОЦЕНКА И СОХРАНЕНИЕ
# =========================================================

test_loss, test_accuracy = model.evaluate(val_ds)

print()
print("====================================")
print(f"Точность модели: {test_accuracy * 100:.2f}%")
print("====================================")

model.save("image.classifier.keras")
print("Модель сохранена как image.classifier.keras")