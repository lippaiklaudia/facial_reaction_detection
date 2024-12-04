import tensorflow as tf

# Adatok betöltése és előfeldolgozása
def load_data(data_dir, img_size=(48, 48), batch_size=64):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

    train_data = datagen.flow_from_directory(
        directory=f"{data_dir}/train",
        target_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    test_data = datagen.flow_from_directory(
        directory=f"{data_dir}/test",
        target_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_data, test_data

if __name__ == "__main__":
    data_dir = "../data/fer2013"
    train_data, test_data = load_data(data_dir)
    print("Adatok sikeresen betöltve.")
