import tensorflow as tf
from cnn_model import create_cnn_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

train_dir = "data/drowsy_data_set/train"
test_dir = "data/drowsy_data_set/test"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    label_mode="binary"
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    label_mode="binary"
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.GaussianNoise(0.05)
])

# adatbovites + normalizalas
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# osztalysulyok
class_weights = {0: 1.0, 1: 1.0}  # Alapértelmezett értékek
class_counts = {}

for _, labels in train_ds.unbatch():
    for label in labels.numpy():
        class_counts[label] = class_counts.get(label, 0) + 1

total_samples = sum(class_counts.values())
for class_label, count in class_counts.items():
    class_weights[class_label] = total_samples / (len(class_counts) * count)

print("Class Weights:", class_weights)

# modell letrehozasa
model = create_cnn_model((48, 48, 1))

# tanitas monitorozasa
class TrainingMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}, "
              f"Val Loss = {logs['val_loss']:.4f}, Val Accuracy = {logs['val_accuracy']:.4f}")

from keras._tf_keras.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=40,
    class_weight=class_weights,
    callbacks=[TrainingMonitor(), early_stop]
)


# modell mentese
model.save("models/drowsy_data_model.h5")

# modell tesztelese a tesztadatokon
model = tf.keras.models.load_model("models/drowsy_data_model.h5")

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.round(predictions).flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# eredmenyek
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Drowsy", "Non-Drowsy"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
