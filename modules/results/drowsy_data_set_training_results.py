import matplotlib.pyplot as plt
import numpy as np

# Adatok 26 epochra
epochs = list(range(1, 27))

train_accuracy = [
    0.5441, 0.7052, 0.7236, 0.7443, 0.7600, 0.7772, 0.7866, 0.8062, 0.8169,
    0.8333, 0.8385, 0.8483, 0.8601, 0.8615, 0.8697, 0.8773, 0.8814, 0.8890,
    0.8948, 0.9015, 0.9025, 0.9103, 0.9102, 0.9161, 0.9180, 0.9234
]

val_accuracy = [
    0.4106, 0.5158, 0.5126, 0.5275, 0.5217, 0.5276, 0.5677, 0.6153, 0.5754,
    0.6126, 0.5907, 0.6001, 0.5552, 0.6655, 0.6281, 0.6318, 0.6564, 0.7310,
    0.7151, 0.6573, 0.7170, 0.6193, 0.7124, 0.6883, 0.6808, 0.7357
]

train_loss = [
    0.6718, 0.5585, 0.5202, 0.4894, 0.4692, 0.4431, 0.4357, 0.4155, 0.3951,
    0.3795, 0.3641, 0.3487, 0.3312, 0.3280, 0.3087, 0.2953, 0.2873, 0.2775,
    0.2609, 0.2509, 0.2422, 0.2366, 0.2267, 0.2124, 0.2129, 0.1974
]

val_loss = [
    1.0295, 1.2494, 1.1488, 1.1683, 1.1300, 1.0469, 0.9641, 0.8222, 0.9174,
    0.8590, 0.8411, 0.7849, 0.8323, 0.7411, 0.7649, 0.7904, 0.7703, 0.7291,
    0.7915, 0.8308, 0.7672, 0.8929, 0.7881, 0.8093, 0.7659, 0.7553
]

# Plotolás
plt.figure(figsize=(12, 6))

# Pontosság (Accuracy)
plt.subplot(2, 1, 1)
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o', linestyle='--')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, 27, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.grid(True)

# Veszteség (Loss)
plt.subplot(2, 1, 2)
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o', linestyle='--')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(np.arange(1, 27, 1))
plt.yticks(np.arange(0, 1.4, 0.2))
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
