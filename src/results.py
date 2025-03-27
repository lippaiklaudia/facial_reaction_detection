import matplotlib.pyplot as plt

epochs = list(range(1, 16))

train_accuracy = [0.5411, 0.6871, 0.7097, 0.7266, 0.7489, 0.7652, 0.7839, 0.8004, 0.8124, 0.8322, 0.8383, 0.8433, 0.8578, 0.8633, 0.8704]
val_accuracy = [0.2897, 0.2739, 0.4240, 0.3904, 0.4713, 0.4628, 0.5205, 0.5230, 0.5315, 0.5477, 0.5480, 0.5809, 0.6079, 0.6126, 0.5736]
train_loss = [0.6711, 0.5828, 0.5522, 0.5346, 0.5073, 0.4835, 0.4633, 0.4318, 0.4158, 0.3909, 0.3745, 0.3602, 0.3396, 0.3262, 0.3171]
val_loss = [1.0318, 0.9786, 0.9578, 0.8794, 0.9501, 0.8841, 0.8529, 0.9148, 0.8636, 0.9391, 0.9252, 0.8772, 0.8835, 0.8838, 0.9950]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, label="Training Accuracy", marker='o')
plt.plot(epochs, val_accuracy, label="Validation Accuracy", marker='o')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label="Training Loss", marker='o')
plt.plot(epochs, val_loss, label="Validation Loss", marker='o')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
