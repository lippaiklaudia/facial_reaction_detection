import matplotlib.pyplot as plt
import numpy as np

# Adatok
epochs = list(range(1, 15))  # 14 epoch
train_accuracy = [0.4527, 0.7455, 0.7771, 0.7625, 0.7930, 0.7915, 0.7836, 0.7902, 0.8156, 0.8072, 0.8165, 0.8242, 0.7944, 0.8123]
val_accuracy = [0.7328, 0.7517, 0.7241, 0.7603, 0.7897, 0.7414, 0.7914, 0.8103, 0.8103, 0.8103, 0.8207, 0.7810, 0.7845, 0.8224]
train_loss = [1.0753, 0.5439, 0.4440, 0.4334, 0.3925, 0.4037, 0.3833, 0.3954, 0.3686, 0.3576, 0.3491, 0.3426, 0.3835, 0.3371]
val_loss = [0.4952, 0.4319, 0.5147, 0.4327, 0.3680, 0.4420, 0.3566, 0.3448, 0.3347, 0.3364, 0.3154, 0.3441, 0.3570, 0.3150]

plt.figure(figsize=(10, 5))

# Plot - Pontosság
plt.subplot(2, 1, 1)
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o', linestyle='--')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, 15, 1))
plt.yticks(np.arange(0, 1, 0.2))
plt.legend()
plt.grid(True)

# Plot - Veszteség
plt.subplot(2, 1, 2)
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o', linestyle='--')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(1, 15, 1))
plt.yticks(np.arange(0, 1.3, 0.2))
plt.legend()
plt.grid(True)

# Layout optimalizálás és megjelenítés
plt.tight_layout()
plt.show()
