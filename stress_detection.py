import sys
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import simpleaudio as sa
import pandas as pd
from collections import deque
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox, QDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from modules.analysis.eye_tracker import EyeTracker
import subprocess

STRESS_AU_THRESHOLDS = {
    "AU04_r": 1.0,
    "AU07_r": 0.7,
    "AU17_r": 0.4
}

STRESS_AU_WEIGHTS = {
    "AU04_r": 1.2,
    "AU07_r": 1.5,
    "AU17_r": 0.7
}

def compute_stress_probability(blink_std, pupil_delta, gaze_instability, au_values):
    probs = []

    if blink_std > 2.0:
        probs.append(0.6)
    if pupil_delta > 0.25:
        probs.append(0.5)
    if gaze_instability > 3.5:
        probs.append(0.4)

    # AU komponensekhez rendelés
    if au_values.get("AU04_r", 0.0) > 1.0:
        probs.append(0.65)
    if au_values.get("AU07_r", 0.0) > 0.6:
        probs.append(0.85)
    if au_values.get("AU17_r", 0.0) > 0.4:
        probs.append(0.50)

    if probs:
        return sum(probs) / len(probs)
    else:
        return 0.0

class StressDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Stress Detection")
        self.resize(1300, 600)

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        self.ear_label = QLabel("EAR: -")
        self.blink_label = QLabel("Blinks: -")
        self.blink_std_label = QLabel("Blink STD: -")
        self.pupil_delta_label = QLabel("Pupil Delta: -")
        self.gaze_instability_label = QLabel("Gaze Instability: -")
        self.au_label = QLabel("AU Score: -")
        self.stress_score_label = QLabel("Stress Score: -")

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")

        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)

        info_layout = QVBoxLayout()
        info_layout.addWidget(self.ear_label)
        info_layout.addWidget(self.blink_label)
        info_layout.addWidget(self.blink_std_label)
        info_layout.addWidget(self.pupil_delta_label)
        info_layout.addWidget(self.gaze_instability_label)
        info_layout.addWidget(self.au_label)
        info_layout.addWidget(self.stress_score_label)
        info_layout.addStretch()
        info_layout.addWidget(self.start_button)
        info_layout.addWidget(self.stop_button)

        group_box = QGroupBox("Live Metrics")
        group_box.setLayout(info_layout)

        self.stress_scores = []
        self.ear_values = []
        self.pupil_deltas = []
        self.timestamps = []
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.ax.set_title("Stress Score Over Time")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Metric Values")

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)
        plot_box = QGroupBox("Live Metrics Graph")
        plot_box.setLayout(plot_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addWidget(group_box)
        main_layout.addWidget(plot_box)
        self.setLayout(main_layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.eye_tracker = EyeTracker()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_csv_path = f"output/webcam_{timestamp}.csv"
        os.makedirs("output", exist_ok=True)

        self.openface_process = subprocess.Popen([
            "/Users/lippaiklaudia/git/OpenFace/build/bin/FeatureExtraction",
            "-device", "1",
            "-aus",
            "-of", self.output_csv_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.latest_au_data = {}
        self.au_thread = threading.Thread(target=self.update_au_data_from_csv, daemon=True)
        self.au_thread.start()

        self.frame_counter = 0
        self.log_path = "logs/stress_log.csv"
        os.makedirs("results", exist_ok=True)
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Frame", "EAR", "Blinks", "Blink_STD", "Pupil_Delta",
                "Gaze_Instability", "AU_Score", "AU_StressPoints", "Stress_Score"
            ])

        self.last_alert_time = 0
        self.alert_interval = 10
        self.alert_dialog = None
        self.stress_score_window = deque()
        self.gaze_history = deque(maxlen=30)
        self.gaze_instability = 0.0

    def update_au_data_from_csv(self):
        while True:
            if os.path.exists(self.output_csv_path):
                try:
                    df = pd.read_csv(self.output_csv_path)
                    if df.shape[0] > 1:
                        latest_row = df.iloc[-1]

                        au_values = {
                            col: latest_row[col] for col in latest_row.index
                            if col.startswith("AU") and col.endswith("_r") and not pd.isna(latest_row[col])
                        }
                        self.latest_au_data = {k: float(v) for k, v in au_values.items()}

                        gaze_x = latest_row.get("gaze_angle_x", 0.0)
                        gaze_y = latest_row.get("gaze_angle_y", 0.0)
                        self.gaze_history.append((gaze_x, gaze_y))

                        if len(self.gaze_history) >= 5:
                            gaze_array = np.array(self.gaze_history)
                            std_dev = np.std(gaze_array, axis=0)
                            self.gaze_instability = float(np.mean(std_dev))
                        else:
                            self.gaze_instability = 0.0
                except Exception as e:
                    print("[AU CSV read error]:", e)

            time.sleep(0.3)

    def start_camera(self):
        self.cap = cv2.VideoCapture(1)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        if hasattr(self, "openface_process"):
            self.openface_process.terminate()
        self.image_label.clear()
        self.ax.clear()
        self.canvas.draw()

    def play_alert_sound(self):
        try:
            wave_obj = sa.WaveObject.from_wave_file("assets/alert.wav")
            wave_obj.play()
        except Exception as e:
            print("Hang lejátszása sikertelen:", e)

    def show_timed_alert(self, message):
        def show():
            if self.alert_dialog and self.alert_dialog.isVisible():
                return

            self.alert_dialog = QDialog(self)
            self.alert_dialog.setWindowTitle("Stress Alert")
            layout = QVBoxLayout()
            label = QLabel(message)
            label.setWordWrap(True)
            layout.addWidget(label)
            self.alert_dialog.setLayout(layout)
            self.alert_dialog.setWindowModality(Qt.NonModal)
            self.alert_dialog.resize(300, 100)
            self.alert_dialog.show()
            QTimer.singleShot(5000, self.alert_dialog.close)

        threading.Thread(target=self.play_alert_sound).start()
        QTimer.singleShot(0, show)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        result = self.eye_tracker.detect(frame)
        current_time = time.time()

        blink_std = result["blink_interval_stddev"] or 0
        pupil_delta = result["pupil_ratio_delta"] or 0
        gaze_instability = result["gaze_instability"] or 0

        au_values = self.latest_au_data or {}
        au_score = sum(au_values.values()) / len(au_values) if au_values else 0.0

        au_stress_points = 0.0
        for au_name, threshold in STRESS_AU_THRESHOLDS.items():
            au_val = au_values.get(au_name, 0.0)
            if au_val > threshold:
                au_stress_points += STRESS_AU_WEIGHTS.get(au_name, 1.0)

        stress_prob = compute_stress_probability(blink_std, pupil_delta, self.gaze_instability, au_values)
        stress_score = round(stress_prob * 4)  # skálázás 0–4-ig


        # Visszacsökkentés, ha minden metrika nyugalmi szinten van
        if blink_std < 1.0 and pupil_delta < 0.12 and gaze_instability < 0.008 and au_score < 0.2 and au_stress_points == 0:
            stress_score = max(stress_score - 1, 0)

        self.stress_score_window.append((current_time, stress_prob))
        self.stress_score_window = deque([(t, s) for t, s in self.stress_score_window if current_time - t <= 60])
        avg_stress = sum(s for t, s in self.stress_score_window) / len(self.stress_score_window)

        self.ear_label.setText(f"EAR: {result['ear']:.2f}" if result['ear'] else "EAR: -")
        self.blink_label.setText(f"Blinks: {result['total_blinks']}")
        self.blink_std_label.setText(f"Blink STD: {blink_std:.2f}")
        self.pupil_delta_label.setText(f"Pupil Delta: {pupil_delta:.2f}")
        self.gaze_instability_label.setText(f"Gaze Instability: {gaze_instability:.3f} (STD)")
        self.au_label.setText(f"AU Score: {au_score:.2f} | AU stress: {au_stress_points:.1f}")
        self.stress_score_label.setText(f"Stress Probability: {stress_prob:.2f} → Score: {stress_score}")

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for i, (au_name, au_val) in enumerate(au_values.items()):
            if au_name in STRESS_AU_THRESHOLDS:
                threshold = STRESS_AU_THRESHOLDS[au_name]
                color = (0, 0, 255) if au_val > threshold else (0, 200, 0)
                text = f"{au_name}: {au_val:.2f}"
                y = 30 + i * 20
                cv2.putText(rgb_image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

        avg_stress = sum(s for t, s in self.stress_score_window) / len(self.stress_score_window)
        if avg_stress >= 0.6 and current_time - self.last_alert_time > self.alert_interval:
            self.show_timed_alert("A rendszer fokozott stresszvalószínűséget észlelt. Javasolt rövid szünetet tartani.")


        self.frame_counter += 1
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.frame_counter,
                f"{result['ear']:.2f}" if result['ear'] else "-",
                result['total_blinks'],
                f"{blink_std:.2f}",
                f"{pupil_delta:.2f}",
                f"{gaze_instability:.4f}",
                f"{au_score:.2f}",
                f"{au_stress_points:.1f}",
                stress_score
            ])

        self.timestamps.append(self.frame_counter)
        self.stress_scores.append(stress_score)
        self.ear_values.append(result['ear'] if result['ear'] else 0)
        self.pupil_deltas.append(pupil_delta)

        self.ax.clear()
        self.ax.plot(self.timestamps, self.stress_scores, color='red', label='Stress Score')
        self.ax.plot(self.timestamps, self.ear_values, color='green', label='EAR')
        self.ax.plot(self.timestamps, self.pupil_deltas, color='blue', label='Pupil Δ')
        self.ax.set_facecolor("#f9f9f9")
        self.ax.grid(True, linestyle='--', linewidth=0.5)
        self.ax.set_title("Stress Score Over Time", fontsize=12)
        self.ax.set_xlabel("Frame", fontsize=10)
        self.ax.set_ylabel("Value", fontsize=10)
        self.fig.tight_layout()
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StressDetectionApp()
    window.show()
    sys.exit(app.exec_())