import cv2
import dlib
from deepface import DeepFace

def real_time_emotion_detection_with_dlib(camera_index=0):
    """
    Valós idejű érzelemfelismerés dlib és DeepFace segítségével.

    :param camera_index: Az elérendő kamera indexe. Alapértelmezett: 0 (első kamera).
    """
    # Dlib arcdetektor inicializálása
    detector = dlib.get_frontal_face_detector()

    # Webkamera megnyitása
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Nem sikerült megnyitni a kamerát index: {camera_index}")
        return

    print("Nyomj 'q'-t a kilépéshez.")

    while True:
        # Egy frame olvasása a webkamerából
        ret, frame = cap.read()

        if not ret:
            print("Nem sikerült képkockát olvasni.")
            break

        # Szürkeárnyalatos kép készítése a dlib számára
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Arcok detektálása
        faces = detector(gray)

        for face in faces:
            # Arc koordináták kinyerése
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Győződjünk meg róla, hogy az arc koordinátái nem lépnek túl a képkockán
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue

            # Arc régió kivágása
            face_roi = frame[y:y + h, x:x + w]

            # Arc méretezése DeepFace-hez
            face_roi_resized = cv2.resize(face_roi, (224, 224))

            try:
                # Érzelemfelismerés a kivágott és méretezett arcon
                result = DeepFace.analyze(face_roi_resized, actions=["emotion"], enforce_detection=False)

                if isinstance(result, dict):  # Egyetlen arc eredménye
                    dominant_emotion = result.get("dominant_emotion", "N/A")

                    # Rajzoljuk ki az érzelmeket
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Emotion: {dominant_emotion}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
            except Exception as e:
                print(f"Hiba az érzelemfelismerésnél: {e}")

        # Képkocka megjelenítése
        cv2.imshow("Real-Time Emotion Detection with Dlib", frame)

        # Kilépés 'q' gomb megnyomásával
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Erőforrások felszabadítása
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Állítsd be a megfelelő kamera indexet (0 az alapértelmezett)
    real_time_emotion_detection_with_dlib(camera_index=0)




előző main:

def main(task):
    data_dir = "data/fer2013"

    if task == "train":
        print("Adatok betöltése...")
        train_data, test_data = load_data(data_dir)

        print("Modell betanítása...")
        model_path = "models/emotion_detection_model.h5"
        train_model(train_data, test_data, model_path)

    elif task == "test":
        print("A teszt funkció még nincs implementálva.")
    else:
        print(f"Ismeretlen task: {task}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FER2013 érzelemfelismerő rendszer.")
    parser.add_argument(
        "task",
        type=str,
        choices=["train", "test"],
        help="Feladat: 'train' a modell betanításához, 'test' a teszteléshez."
    )
    args = parser.parse_args()

    main(args.task)

