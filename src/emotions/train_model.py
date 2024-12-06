import keras._tf_keras.keras.layers as kl
import keras._tf_keras.keras as keras

def train_model(train_data, test_data, model_path):
    # CNN modell építése
    model = keras.Sequential([
        
        kl.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        kl.MaxPooling2D(2, 2),
        kl.Conv2D(64, (3, 3), activation='relu'),
        kl.MaxPooling2D(2, 2),
        kl.Conv2D(128, (3, 3), activation='relu'),
        kl.Flatten(),
        kl.Dense(128, activation='relu'),
        kl.Dense(7, activation='softmax')  # 7 érzelem
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Modell tanítása
    model.fit(train_data, validation_data=test_data, epochs=25)

    # Modell mentése
    model.save(model_path)
    print(f"Modell mentve: {model_path}")

if __name__ == "__main__":
    from data_preprocessing import load_data

    data_dir = "../data/fer2013"
    train_data, test_data = load_data(data_dir)
    train_model(train_data, test_data, "../models/emotion_detection_model.h5")

