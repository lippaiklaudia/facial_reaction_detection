import cv2
import numpy as np

# hisztogram kiegyenlites + gamma korrekcio
def preprocess_with_lighting_correction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    table = np.array([(i / 255.0) ** (1.0 / 1.2) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(equalized, table)
    return corrected

def preprocess_face_for_model(face_roi):
    corrected = preprocess_with_lighting_correction(face_roi)  # Lighting korrekció
    resized = cv2.resize(corrected, (48, 48))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 48, 48, 1)
    return reshaped