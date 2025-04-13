import cv2
import mediapipe as mp
import numpy as np
import time

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.prev_blink_time = 0
        self.blink_count = 0
        self.blinks_in_last_minute = []
        self.eye_closed = False
        self.pupil_ratios_last_minute = []

    def get_eye_aspect_ratio(self, landmarks, eye_indices):
        eye = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
        width = np.linalg.norm(eye[0] - eye[3])
        height = (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / 2.0
        return height / width

    def get_pupil_ratio(self, landmarks):
        # Bal szem
        left_pupil = np.array([landmarks[468].x, landmarks[468].y])
        left_inner = np.array([landmarks[133].x, landmarks[133].y])   # bal szem külső sarka
        left_outer = np.array([landmarks[33].x, landmarks[33].y])     # bal szem belső sarka
        left_eye_width = np.linalg.norm(left_outer - left_inner)

        # Jobb szem
        right_pupil = np.array([landmarks[473].x, landmarks[473].y])
        right_inner = np.array([landmarks[362].x, landmarks[362].y])  # jobb szem belső sarka
        right_outer = np.array([landmarks[263].x, landmarks[263].y])  # jobb szem külső sarka
        right_eye_width = np.linalg.norm(right_outer - right_inner)

        if left_eye_width == 0 or right_eye_width == 0:
            return None

        left_ratio = np.linalg.norm(left_pupil - left_outer) / left_eye_width
        right_ratio = np.linalg.norm(right_pupil - right_outer) / right_eye_width

        # Átlagoljuk a két szemet
        return (left_ratio + right_ratio) / 2


    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        blink_detected = False
        left_ear = None
        pupil_ratio = None
        blink_interval_stddev = None
        pupil_ratio_delta = None

        if not results.multi_face_landmarks:
            return {
                "timestamp": time.time(),
                "ear": None,
                "blink_detected": False,
                "total_blinks": self.blink_count,
                "blinks_per_minute": len(self.blinks_in_last_minute),
                "pupil_ratio": None,
                "blink_interval_stddev": None,
                "pupil_ratio_delta": None,
                "landmarks": None
            }

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        left_eye_indices = [33, 160, 158, 133, 153, 144]

        left_ear = self.get_eye_aspect_ratio(landmarks, left_eye_indices)
        pupil_ratio = self.get_pupil_ratio(landmarks)

        if pupil_ratio:
            self.pupil_ratios_last_minute.append((time.time(), pupil_ratio))

        if left_ear < 0.20:
            if not self.eye_closed:
                self.blink_count += 1
                blink_detected = True
                self.prev_blink_time = time.time()
                self.blinks_in_last_minute.append(self.prev_blink_time)
                self.eye_closed = True
        else:
            self.eye_closed = False

        current_time = time.time()
        self.blinks_in_last_minute = [t for t in self.blinks_in_last_minute if current_time - t <= 60]
        self.pupil_ratios_last_minute = [(t, r) for t, r in self.pupil_ratios_last_minute if current_time - t <= 60]

        if len(self.blinks_in_last_minute) >= 2:
            intervals = np.diff(self.blinks_in_last_minute)
            blink_interval_stddev = np.std(intervals)

        if len(self.pupil_ratios_last_minute) >= 2:
            ratios = [r for _, r in self.pupil_ratios_last_minute]
            pupil_ratio_delta = max(ratios) - min(ratios)

        return {
            "timestamp": current_time,
            "ear": left_ear,
            "blink_detected": blink_detected,
            "total_blinks": self.blink_count,
            "blinks_per_minute": len(self.blinks_in_last_minute),
            "pupil_ratio": pupil_ratio,
            "blink_interval_stddev": blink_interval_stddev,
            "pupil_ratio_delta": pupil_ratio_delta,
            "landmarks": face_landmarks.landmark 
        }
