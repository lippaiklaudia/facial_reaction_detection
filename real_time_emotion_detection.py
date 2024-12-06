import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import img_to_array

# Modell betöltése
model = load_model("models/emotion_detection_model.h5")
emotion_labels = ['Düh', 'Undor', 'Félelem', 'Boldogság', 'Szomorúság', 'Meglepetés', 'Semleges']

# Kamera inicializálása
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Az arc detektálása a videóban
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Arc kivágása és előfeldolgozása
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0  # Normalizálás
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        # Érzelem előrejelzés
        preds = model.predict(roi_gray)[0]
        emotion = emotion_labels[np.argmax(preds)]

        # Arc körvonalazása és érzelem kiírása
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Kép megjelenítése
    cv2.imshow("Valós idejű érzelemfelismerés", frame)

    # Kilépés a 'q' gomb megnyomásával
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
