import cv2
import mediapipe as mp
from deepface import DeepFace

def real_time_landmark_emotion_detection(camera_index=1, analysis_frequency=10):

    # Mediapipe arcfelismerés inicializálása
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # OpenCV kamera inicializálása
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Nem sikerült megnyitni a kamerát.")
        return

    print("Nyomj 'q'-t a kilépéshez.")

    frame_count = 0  # Képkocka számláló
    last_emotion = None  # Az utolsó detektált érzelem tárolása

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nem sikerült képkockát olvasni.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Landmark pontok kirajzolása
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # Arc körvonalainak detektálása
                h, w, _ = frame.shape
                x_min = int(min([landmark.x for landmark in face_landmarks.landmark]) * w)
                y_min = int(min([landmark.y for landmark in face_landmarks.landmark]) * h)
                x_max = int(max([landmark.x for landmark in face_landmarks.landmark]) * w)
                y_max = int(max([landmark.y for landmark in face_landmarks.landmark]) * h)

                if frame_count % analysis_frequency == 0:
                    try:
                        # Arc kivágása érzelemfelismeréshez
                        face_roi = frame[y_min:y_max, x_min:x_max]
                        emotion_results = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)

                        # Több arc eredményeinek kezelése
                        if isinstance(emotion_results, list):
                            for result in emotion_results:
                                last_emotion = result.get("dominant_emotion", "N/A")
                        else:
                            last_emotion = emotion_results.get("dominant_emotion", "N/A")
                    except Exception as e:
                        print(f"Hiba az érzelemfelismerésnél: {e}")

                # Az utolsó detektált érzelem megjelenítése
                if last_emotion:
                    cv2.putText(frame, f"Emotion: {last_emotion}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Arc körvonalának kirajzolása
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Képkocka megjelenítése
        cv2.imshow("Landmark and Emotion Detection", frame)
        frame_count += 1  # Növeld a képkocka számlálót

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


