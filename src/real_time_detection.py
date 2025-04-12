from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from collections import deque
from detection.face_detection import detect_faces
from detection.landmarks import get_landmarks
from keras._tf_keras.keras.models import load_model
import time
from preprocessing import preprocess_face_for_model
from analysis.color_analysis import analyze_color_with_histogram
from analysis.eye_analysis import calculate_ear
from detection.landmarks import draw_landmarks

model = load_model("models/drowsiness_detector.h5")

FRAME_WINDOW = 10
EAR_CALIBRATION_FRAMES = 30
FATIGUE_THRESHOLD = 10
RECOVERY_RATE = 3
MIN_FRAME_FOR_DROWSINESS = 5  # minimum hunyorgas
BLINK_VALIDATION_FRAMES = 3
DROWSY_BLINK_FRAMES = 8
LOG_INTERVAL_SECONDS = 1  # logolas masodpercenkent

ear_history = deque(maxlen=FRAME_WINDOW)
fatigue_frames = 0
blink_count = 0
ear_threshold = 0.2
blink_start_frame = None
blink_detected = False
last_log_time = time.time()

# EAR atlagolasa
def get_avg_ear(ear_history):
    return np.mean(ear_history) if len(ear_history) > 0 else 1.0

# EAR kalibracio (nyitott szem)
def calibrate_ear(cap):
    ear_values = []
    for _ in range(EAR_CALIBRATION_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detect_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            landmarks = get_landmarks(face_roi)
            if landmarks:
                left_eye = landmarks[0][36:42]
                right_eye = landmarks[0][42:48]
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                ear_values.append(avg_ear)

    # kiugro ertekek kiszurese
    valid_ear_values = [ear for ear in ear_values if abs(ear - np.mean(ear_values)) < 2 * np.std(ear_values)]
    if len(valid_ear_values) > 0:
        return np.mean(valid_ear_values) * 0.8
    return 0.2


def real_time_detection():
    global fatigue_frames, blink_count, ear_threshold, blink_start_frame, blink_detected, last_log_time
    fps_counter = []

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Hiba a kamerával!")
        return

    ear_threshold = calibrate_ear(cap)
    print(f"Kalibrált EAR küszöb: {ear_threshold:.2f}")

    with ThreadPoolExecutor(max_workers=3) as executor:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # arcok detektalasa
            future_faces = executor.submit(detect_faces, frame)
            faces = future_faces.result()

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]

                # landmark, borszin elemzese
                future_landmarks = executor.submit(get_landmarks, face_roi)
                future_color_analysis = executor.submit(analyze_color_with_histogram, frame, (x, y, w, h))

                landmarks = future_landmarks.result()
                mean_hue, mean_saturation, mean_value = future_color_analysis.result()

                # szinvizsgalat allpotainak meghatarozasa
                if mean_hue < 20 and mean_saturation < 40 and mean_value < 90:
                    skin_status = "Fatigue"
                elif mean_hue < 30 and mean_saturation < 50 and mean_value < 120:
                    skin_status = "Possibly Fatigued"
                else:
                    skin_status = "Normal"

                eye_status = "No Data"
                overall_status = "Awake"

                if landmarks:
                    draw_landmarks(frame, landmarks, (x, y, w, h))
                    left_eye = landmarks[0][36:42]
                    right_eye = landmarks[0][42:48]
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    ear_history.append(avg_ear)

                    avg_ear_window = get_avg_ear(ear_history)

                    if avg_ear_window < ear_threshold:
                        if not blink_detected:
                            blink_start_frame = time.time()
                            blink_detected = True
                        eye_status = "Closed"
                        blink_count += 1
                    else:
                        if blink_detected:
                            blink_duration = (time.time() - blink_start_frame) * 1000
                            if BLINK_VALIDATION_FRAMES * 100 <= blink_duration <= DROWSY_BLINK_FRAMES * 100:
                                print(f"Blink detected. Duration: {blink_duration:.2f} ms")
                            elif blink_duration > DROWSY_BLINK_FRAMES * 100:
                                print(f"Drowsy blink detected. Duration: {blink_duration:.2f} ms")
                            blink_detected = False
                        eye_status = "Open"

                    # modell predikcio
                    processed_face = preprocess_face_for_model(face_roi)
                    model_prediction = model.predict(processed_face, verbose=0)[0][0]
                    model_label = "Drowsy" if model_prediction >= 0.28 else "Awake"

                    # kombinalt allapot (modell + ear + szinvaltozas)
                    if avg_ear_window >= ear_threshold and skin_status == "Normal":
                        overall_status = "Awake"
                    else:
                        is_drowsy = (fatigue_frames > FATIGUE_THRESHOLD or 
                                    (model_label == "Drowsy" and avg_ear_window < ear_threshold) or 
                                    skin_status == "Fatigue")
                        if is_drowsy:
                            fatigue_frames = min(FATIGUE_THRESHOLD + 5, fatigue_frames + 1)
                        else:
                            fatigue_frames = max(0, fatigue_frames - RECOVERY_RATE)
                        overall_status = "Drowsy" if fatigue_frames > FATIGUE_THRESHOLD else "Awake"

                current_time = time.time()

                if current_time - last_log_time >= LOG_INTERVAL_SECONDS:
                    print(f"Eye Status: {eye_status}, Blink Count: {blink_count}")
                    print(f"Model Prediction: {model_label}")
                    print(f"Skin Status: {skin_status}")
                    print(f"FPS: {np.mean(fps_counter):.2f}")
                    print(f"Mean Hue: {mean_hue:.2f}, Saturation: {mean_saturation:.2f}, Value: {mean_value:.2f}")
                    print("\033[32m" + "=" * 50 + "\033[0m")
                    last_log_time = current_time

                # eredmenyek megjelenitese
                label_eye = f"Eye: {eye_status}"
                label_skin = f"Skin: {skin_status}"
                label_overall = f"Overall: {overall_status}"
                color = (0, 0, 255) if overall_status == "Drowsy" else (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label_eye, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, label_skin, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, label_overall, (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # FPS szamitas
            end_time = time.time()
            fps_counter.append(1 / (end_time - start_time))
            if len(fps_counter) > 30:
                fps_counter.pop(0)

            # elokep
            cv2.imshow("Real-Time Fatigue Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# src/analysis/real_time_detection.py

def analyze_frame(frame, model, ear_threshold, ear_history, fatigue_frames):
    # Ez a függvény egyetlen képkockát elemez
    # és visszaadja a feldolgozott állapotokat és a módosított fatigue_frames értéket

    from detection.face_detection import detect_faces
    from detection.landmarks import get_landmarks
    from analysis.color_analysis import analyze_color_with_histogram
    from analysis.eye_analysis import calculate_ear
    from preprocessing import preprocess_face_for_model

    faces = detect_faces(frame)

    if len(faces) == 0:
        return "No Face", "No Skin", "Unknown", fatigue_frames, frame

    (x, y, w, h) = faces[0]
    face_roi = frame[y:y+h, x:x+w]
    landmarks = get_landmarks(face_roi)
    mean_hue, mean_saturation, mean_value = analyze_color_with_histogram(frame, (x, y, w, h))

    if mean_hue < 20 and mean_saturation < 40 and mean_value < 90:
        skin_status = "Fatigue"
    elif mean_hue < 30 and mean_saturation < 50 and mean_value < 120:
        skin_status = "Possibly Fatigued"
    else:
        skin_status = "Normal"

    eye_status = "No Data"
    overall_status = "Awake"

    if landmarks:
        left_eye = landmarks[0][36:42]
        right_eye = landmarks[0][42:48]
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        ear_history.append(avg_ear)

        avg_ear_window = np.mean(ear_history) if len(ear_history) > 0 else 1.0

        eye_status = "Closed" if avg_ear_window < ear_threshold else "Open"

        processed_face = preprocess_face_for_model(face_roi)
        model_prediction = model.predict(processed_face, verbose=0)[0][0]
        model_label = "Drowsy" if model_prediction >= 0.28 else "Awake"

        if avg_ear_window >= ear_threshold and skin_status == "Normal":
            overall_status = "Awake"
        else:
            is_drowsy = (fatigue_frames > 10 or 
                         (model_label == "Drowsy" and avg_ear_window < ear_threshold) or 
                         skin_status == "Fatigue")
            if is_drowsy:
                fatigue_frames = min(15, fatigue_frames + 1)
            else:
                fatigue_frames = max(0, fatigue_frames - 3)
            overall_status = "Drowsy" if fatigue_frames > 10 else "Awake"

    # Vizuális overlay
    color = (0, 0, 255) if overall_status == "Drowsy" else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"Eye: {eye_status}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"Skin: {skin_status}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"Overall: {overall_status}", (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return eye_status, skin_status, overall_status, fatigue_frames, frame


if __name__ == "__main__":
    real_time_detection()