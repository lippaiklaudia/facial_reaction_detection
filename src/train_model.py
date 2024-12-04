import tensorflow as tf
from src.data_preprocessing import load_data

# Adatok betöltése
data_dir = "../data/fer2013"
train_data, test_data = load_data(data_dir)

# CNN modell építése
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')  # 7 érzelem
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modell tanítása
model.fit(train_data, validation_data=test_data, epochs=25)

# Modell mentése
model.save("../models/emotion_detection_model.h5")
print("Modell betanítva és mentve.")
