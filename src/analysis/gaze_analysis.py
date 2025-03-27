import cv2
import mediapipe as mp
import numpy as np
import csv
from datetime import datetime
from collections import deque

# MediaPipe inicializálása
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Szem landmark indexek
LEFT_EYE_LANDMARKS = [33, 133, 160, 158, 153, 144, 163, 7]
RIGHT_EYE_LANDMARKS = [362, 263, 387, 385, 380, 373, 390, 249]

# Tekintet irány simításához
gaze_history = deque(maxlen=5)

# CSV fájl előkészítés
csv_file = open("gaze_data.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "left_gaze", "right_gaze"])

def extract_eye_region(frame, landmarks, eye_indices):
    h, w = frame.shape[:2]
    points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices])
    x, y, w_e, h_e = cv2.boundingRect(points)
    eye_img = frame[y:y + h_e, x:x + w_e]
    return eye_img, x, y, w_e, h_e

def estimate_gaze_from_eye(eye_img):
    gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    _, thresh = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
    moments = cv2.moments(thresh)

    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        pos = cx / eye_img.shape[1]
        return pos
    else:
        return None

def interpret_gaze(pos):
    if pos is None:
        return "Uncertain"
    if pos < 0.35:
        return "Left"
    elif pos > 0.65:
        return "Right"
    else:
        return "Center"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    gaze_text = "No face"

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            left_eye_img, lex, ley, lw, lh = extract_eye_region(frame, landmarks.landmark, LEFT_EYE_LANDMARKS)
            right_eye_img, rex, rey, rw, rh = extract_eye_region(frame, landmarks.landmark, RIGHT_EYE_LANDMARKS)

            left_pos = estimate_gaze_from_eye(left_eye_img)
            right_pos = estimate_gaze_from_eye(right_eye_img)

            # Átlagolás, ha mindkettő elérhető
            if left_pos is not None and right_pos is not None:
                avg_pos = (left_pos + right_pos) / 2
            elif left_pos is not None:
                avg_pos = left_pos
            elif right_pos is not None:
                avg_pos = right_pos
            else:
                avg_pos = None

            # Simítás mozgóátlaggal
            if avg_pos is not None:
                gaze_history.append(avg_pos)
                smoothed_pos = sum(gaze_history) / len(gaze_history)
            else:
                smoothed_pos = None

            gaze_label = interpret_gaze(smoothed_pos)
            gaze_text = gaze_label

            # CSV mentés
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            csv_writer.writerow([timestamp, interpret_gaze(left_pos), interpret_gaze(right_pos)])

            # Vizualizáció
            cv2.putText(frame, f'Gaze: {gaze_label}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 2)

            # Nyilak rajzolása
            center_x = int(lex + lw / 2)
            center_y = int(ley + lh / 2)
            if smoothed_pos is not None:
                offset = int((smoothed_pos - 0.5) * 60)
                cv2.arrowedLine(frame, (center_x, center_y),
                                (center_x + offset, center_y),
                                (0, 0, 255), 2)

            # Szemek köré téglalap
            cv2.rectangle(frame, (lex, ley), (lex + lw, ley + lh), (255, 0, 0), 1)
            cv2.rectangle(frame, (rex, rey), (rex + rw, rey + rh), (255, 0, 0), 1)

    cv2.imshow("Real-Time Gaze Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
