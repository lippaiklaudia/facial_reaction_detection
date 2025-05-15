import os
import numpy as np
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

def load_data_from_directory(data_dir, target_size=(48, 48), batch_size=64):
    """
    FER2013 képek betöltése a mappastruktúrából
    """
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    test_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=target_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, test_generator

