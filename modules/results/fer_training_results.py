import matplotlib.pyplot as plt
import numpy as np

# Adatok
epochs = list(range(1, 26))
train_accuracy = [0.2976, 0.4688, 0.5279, 0.5729, 0.6244, 0.6744, 0.7208, 0.7736, 0.8256, 0.8765, 0.9097, 0.9366, 0.9541, 0.9676, 0.9678, 0.9719, 0.9723, 0.9804, 0.9809, 0.9816, 0.9828, 0.9816, 0.9766, 0.9868, 0.9825]
val_accuracy = [0.4363, 0.4962, 0.5312, 0.5521, 0.5605, 0.5605, 0.5630, 0.5546, 0.5563, 0.5589, 0.5539, 0.5550, 0.5556, 0.5511, 0.5528, 0.5464, 0.5536, 0.5593, 0.5514, 0.5571, 0.5527, 0.5504, 0.5496, 0.5508, 0.5522]
train_loss = [1.7307, 1.3967, 1.2378, 1.1283, 1.0101, 0.8909, 0.7711, 0.6325, 0.5064, 0.3735, 0.2777, 0.2032, 0.1505, 0.1097, 0.1080, 0.0946, 0.0898, 0.0701, 0.0691, 0.0649, 0.0610, 0.0647, 0.0752, 0.0477, 0.0665]
val_loss = [1.4643, 1.3087, 1.2418, 1.1831, 1.1888, 1.2113, 1.2605, 1.3528, 1.4737, 1.6957, 1.9054, 2.1726, 2.5072, 2.5558, 2.8528, 3.1443, 3.1107, 3.1875, 3.3363, 3.4114, 3.6330, 3.5031, 3.6076, 3.7124, 3.7619]

# Diagram létrehozása
plt.figure(figsize=(12, 8))

# Pontosság diagram
plt.subplot(2, 1, 1)
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o', linestyle='--')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1, 26, 1))  # Részletesebb epoch skála
plt.yticks(np.arange(0, 1.1, 0.1))  # Pontosabb accuracy skála
plt.legend()
plt.grid(True)

# Veszteség diagram
plt.subplot(2, 1, 2)
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o', linestyle='--')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(1, 26, 1))  # Részletesebb epoch skála
plt.yticks(np.arange(0, 4.1, 0.5))  # Pontosabb loss skála
plt.legend()
plt.grid(True)

# Layout optimalizálás és megjelenítés
plt.tight_layout()
plt.show()
