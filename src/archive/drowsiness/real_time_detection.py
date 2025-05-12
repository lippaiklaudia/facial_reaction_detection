import cv2
from keras._tf_keras.keras.models import load_model
import numpy as np
import dlib

# Modell betöltése
model = load_model("models/drowsiness_model.h5")
categories = ["Closed", "Open", "yawn", "no_yawn"]

# Dlib arcdetektor és arcpontok modell betöltése
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_eye_region(face_landmarks, gray_frame, eye_indices):
    points = [face_landmarks.part(i) for i in eye_indices]
    x_min = min(point.x for point in points)
    x_max = max(point.x for point in points)
    y_min = min(point.y for point in points)
    y_max = max(point.y for point in points)
    eye_region = gray_frame[y_min:y_max, x_min:x_max]
    return cv2.resize(eye_region, (64, 64)) / 255.0

# Kamera inicializálása
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Szürkeárnyalatos konverzió
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Arcok detektálása
    faces = face_detector(gray)

    for face in faces:
        # Arcpontok meghatározása
        landmarks = landmark_predictor(gray, face)

        # Kivágás és előfeldolgozás: szemek és száj környéke
        left_eye = extract_eye_region(landmarks, gray, range(36, 42))
        right_eye = extract_eye_region(landmarks, gray, range(42, 48))
        mouth = extract_eye_region(landmarks, gray, range(48, 68))

        inputs = np.array([
            np.expand_dims(left_eye, axis=-1),
            np.expand_dims(right_eye, axis=-1),
            np.expand_dims(mouth, axis=-1)
        ])
        inputs = inputs.reshape(-1, 64, 64, 1)

        # Predikció
        prediction = model.predict(inputs)
        class_idx = np.argmax(prediction)

        if class_idx < len(categories):
            label = categories[class_idx]
        else:
            label = "Unknown"

        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        color = (0, 255, 0) if label in ["Open", "no_yawn"] else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Élőkép megjelenítése
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()