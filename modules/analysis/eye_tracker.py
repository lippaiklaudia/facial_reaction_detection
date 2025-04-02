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
        left_pupil = np.array([landmarks[468].x, landmarks[468].y])
        eye_left = np.array([landmarks[33].x, landmarks[33].y])
        eye_right = np.array([landmarks[133].x, landmarks[133].y])
        eye_width = np.linalg.norm(eye_left - eye_right)
        if eye_width == 0:
            return None
        return 1 - (np.linalg.norm(left_pupil - eye_left) / eye_width)

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        blink_detected = False
        left_ear = None
        pupil_ratio = None
        blink_interval_stddev = None
        pupil_ratio_delta = None

        if results.multi_face_landmarks:
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
            "pupil_ratio_delta": pupil_ratio_delta
        }
