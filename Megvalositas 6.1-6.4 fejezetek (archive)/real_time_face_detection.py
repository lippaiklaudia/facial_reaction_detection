import cv2
import dlib
from deepface import DeepFace

def real_time_face_and_emotion_detection(camera_index=1):

    detector = dlib.get_frontal_face_detector()

    # Webkamera megnyitása
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Nem sikerült megnyitni a kamerát.")
        return

    print("Nyomj 'q'-t a kilépéshez.")

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Nem sikerült képkockát olvasni.")
            break

        # Szürkeárnyalatos kép az arcdetektáláshoz
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Arc koordináták ellenőrzése
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue

            face_roi = frame[y:y + h, x:x + w]

            # Arc méretezése DeepFace számára
            try:
                face_roi_resized = cv2.resize(face_roi, (224, 224))

                # Érzelemfelismerés DeepFace-szel
                results = DeepFace.analyze(face_roi_resized, actions=["emotion"], enforce_detection=False)

                if isinstance(results, list):
                    for result in results:
                        dominant_emotion = result.get("dominant_emotion", "N/A")
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
                else:
                    dominant_emotion = results.get("dominant_emotion", "N/A")
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
        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
