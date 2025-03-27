from keras._tf_keras.keras import layers, models
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.regularizers import l2

def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model
