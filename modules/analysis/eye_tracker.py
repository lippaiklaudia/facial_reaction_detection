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
        self.gaze_points_last_minute = []

    def get_eye_aspect_ratio(self, landmarks, eye_indices):
        eye = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
        width = np.linalg.norm(eye[0] - eye[3])
        height = (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / 2.0
        return height / width

    def get_pupil_ratio(self, landmarks):
        try:
            left_pupil = np.array([landmarks[468].x, landmarks[468].y])
            left_inner = np.array([landmarks[133].x, landmarks[133].y])
            left_outer = np.array([landmarks[33].x, landmarks[33].y])
            left_eye_width = np.linalg.norm(left_outer - left_inner)

            right_pupil = np.array([landmarks[473].x, landmarks[473].y])
            right_inner = np.array([landmarks[362].x, landmarks[362].y])
            right_outer = np.array([landmarks[263].x, landmarks[263].y])
            right_eye_width = np.linalg.norm(right_outer - right_inner)

            if left_eye_width == 0 or right_eye_width == 0:
                return None

            left_ratio = np.linalg.norm(left_pupil - left_outer) / left_eye_width
            right_ratio = np.linalg.norm(right_pupil - right_outer) / right_eye_width

            return (left_ratio + right_ratio) / 2
        except:
            return None

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        blink_detected = False
        left_ear = None
        pupil_ratio = None
        blink_interval_stddev = None
        pupil_ratio_delta = None
        gaze_instability = 0.0
        avg_gaze = None
        gaze_x = None
        gaze_y = None

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
                "gaze_instability": 0.0,
                "landmarks": None,
                "gaze_x": None,
                "gaze_y": None
            }

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # EAR
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        left_ear = self.get_eye_aspect_ratio(landmarks, left_eye_indices)

        # Pupil ratio
        pupil_ratio = self.get_pupil_ratio(landmarks)
        current_time = time.time()
        if pupil_ratio:
            self.pupil_ratios_last_minute.append((current_time, pupil_ratio))

        # Blink detection
        if left_ear < 0.20:
            if not self.eye_closed:
                self.blink_count += 1
                blink_detected = True
                self.prev_blink_time = current_time
                self.blinks_in_last_minute.append(current_time)
                self.eye_closed = True
        else:
            self.eye_closed = False

        # Cleanup 60s window
        self.blinks_in_last_minute = [t for t in self.blinks_in_last_minute if current_time - t <= 60]
        self.pupil_ratios_last_minute = [(t, r) for t, r in self.pupil_ratios_last_minute if current_time - t <= 60]
        self.gaze_points_last_minute = [(t, p) for t, p in self.gaze_points_last_minute if current_time - t <= 60]

        # Blink STD
        if len(self.blinks_in_last_minute) >= 2:
            intervals = np.diff(self.blinks_in_last_minute)
            blink_interval_stddev = np.std(intervals)

        # Pupil delta
        if len(self.pupil_ratios_last_minute) >= 2:
            ratios = [r for _, r in self.pupil_ratios_last_minute]
            pupil_ratio_delta = max(ratios) - min(ratios)

        # Gaze avg and instability
        try:
            left_pupil = np.array([landmarks[468].x, landmarks[468].y])
            right_pupil = np.array([landmarks[473].x, landmarks[473].y])
            avg_gaze = (left_pupil + right_pupil) / 2
            gaze_x = float(avg_gaze[0])
            gaze_y = float(avg_gaze[1])
            self.gaze_points_last_minute.append((current_time, avg_gaze))

            if len(self.gaze_points_last_minute) >= 5:
                positions = np.array([p for t, p in self.gaze_points_last_minute])
                gaze_instability = float(np.mean(np.std(positions, axis=0))) * 100

                print(f"[Gaze DEBUG] STD x: {np.std(positions[:, 0])}")
                print(f"[Gaze DEBUG] STD y: {np.std(positions[:, 1])}")
                print(f"[Gaze DEBUG] mean STD: {gaze_instability}")

        except Exception as e:
            print("[Gaze ERROR]", e)

        return {
            "timestamp": current_time,
            "ear": left_ear,
            "blink_detected": blink_detected,
            "total_blinks": self.blink_count,
            "blinks_per_minute": len(self.blinks_in_last_minute),
            "pupil_ratio": pupil_ratio,
            "blink_interval_stddev": blink_interval_stddev,
            "pupil_ratio_delta": pupil_ratio_delta,
            "gaze_instability": gaze_instability,
            "landmarks": landmarks,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y
        }
