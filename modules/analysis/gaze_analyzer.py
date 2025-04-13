import numpy as np

class GazeAnalyzer:
    def __init__(self):
        self.prev_gaze = None
        self.gaze_movements = []

    def compute_gaze_direction(self, left_eye, right_eye):
        # Középpont számítása
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        eye_center = (left_center + right_center) / 2

        # Vektor két szem között
        eye_vector = right_center - left_center
        eye_width = np.linalg.norm(eye_vector)

        # Egyszerű horizontális tekintetmozgás becslés
        gaze_vector = eye_vector / eye_width if eye_width > 0 else np.zeros_like(eye_vector)
        return gaze_vector

    def update(self, left_eye, right_eye):
        current_gaze = self.compute_gaze_direction(left_eye, right_eye)

        if self.prev_gaze is not None:
            movement = np.linalg.norm(current_gaze - self.prev_gaze)
            self.gaze_movements.append(movement)
            if len(self.gaze_movements) > 30:  # max 30 frame tárolása (~1 másodperc)
                self.gaze_movements.pop(0)

        self.prev_gaze = current_gaze

    def get_gaze_instability(self):
        if len(self.gaze_movements) < 2:
            return 0.0
        return np.std(self.gaze_movements)
