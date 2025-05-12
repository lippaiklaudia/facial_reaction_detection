import os
import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split

base_dir = "data/drowsiness_processed"
categories = ["Closed", "Open", "yawn", "no_yawn"]
img_size = 64

# Adatok betöltése
def load_data(base_dir, categories, img_size):
    data = []
    for split in ["train", "test"]:
        split_path = os.path.join(base_dir, split)
        for category in categories:
            category_path = os.path.join(split_path, category)
            label = categories.index(category)
            for img_name in os.listdir(category_path):
                try:
                    img_path = os.path.join(category_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    resized_img = cv2.resize(img, (img_size, img_size))
                    data.append([resized_img, label])
                except Exception as e:
                    print(f"Error loading image {img_name}: {e}")
    return data

# Adatok feldolgozása
train_data = load_data(base_dir, categories, img_size)
X, y = zip(*train_data)
X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
y = to_categorical(np.array(y), num_classes=len(categories))

# Adatok felosztása tanító és validációs adatokra
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell létrehozása
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
train_gen = datagen.flow(X_train, y_train, batch_size=32)

# Modell tanítása
history = model.fit(
    train_gen,
    validation_data=(X_val, y_val),
    epochs=14,  # Csak 14 epoch
    verbose=1  # Kimenet a konzolra
)

# Modell mentése
model.save("models/drowsiness_model.h5")
