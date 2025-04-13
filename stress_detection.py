import sys
import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import threading
import time
import simpleaudio as sa
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox
from PyQt5.QtWidgets import QMessageBox, QDialog
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer, Qt
from modules.analysis.eye_tracker import EyeTracker
from collections import deque

class GazeAnalyzer:
    def __init__(self):
        self.prev_gaze = None
        self.gaze_movements = deque()

    def compute_gaze_direction(self, landmarks):
        if not landmarks or len(landmarks) < 468:
            return None
        left_eye = np.array([landmarks[33].x, landmarks[33].y])
        right_eye = np.array([landmarks[263].x, landmarks[263].y])
        gaze_vector = right_eye - left_eye
        return gaze_vector

    def update(self, landmarks):
        current_time = time.time()
        gaze_vector = self.compute_gaze_direction(landmarks)
        if gaze_vector is None:
            return

        if self.prev_gaze is not None:
            delta = np.linalg.norm(gaze_vector - self.prev_gaze)
            self.gaze_movements.append((current_time, delta))

        self.prev_gaze = gaze_vector
        self.gaze_movements = deque([(t, d) for t, d in self.gaze_movements if current_time - t <= 1.0])

    def get_gaze_instability(self):
        if not self.gaze_movements:
            return 0.0
        values = [d for t, d in self.gaze_movements]
        return np.std(values)

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
        self.gaze_analyzer = GazeAnalyzer()
        self.frame_counter = 0

        self.log_path = "results/stress_log.csv"
        os.makedirs("results", exist_ok=True)
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "EAR", "Blinks", "Blink_STD", "Pupil_Delta", "Gaze_Instability", "Stress_Score"])

        self.last_alert_time = 0
        self.alert_interval = 10
        self.alert_dialog = None
        self.stress_score_window = deque()

    def start_camera(self):
        self.cap = cv2.VideoCapture(1)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()
        self.ax.clear()
        self.canvas.draw()

    def play_alert_sound(self):
        try:
            wave_obj = sa.WaveObject.from_wave_file("assets/alert.wav")
            play_obj = wave_obj.play()
        except Exception as e:
            print("Hang lejátszása sikertelen:", e)

    def show_timed_alert(self, message):
        def show():
            if self.alert_dialog and self.alert_dialog.isVisible():
                return

            self.alert_dialog = QDialog(self)
            self.alert_dialog.setWindowTitle("Stress Alert")
            self.alert_dialog.setStyleSheet("")
            layout = QVBoxLayout()
            label = QLabel("A rendszer fokozott stresszszintet észlelt. Javasolt rövid szünetet tartani a hatékonyabb regeneráció érdekében.")
            label.setStyleSheet("")
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
        self.gaze_analyzer.update(result.get("landmarks"))
        current_time = time.time()

        blink_std = result["blink_interval_stddev"] or 0
        pupil_delta = result["pupil_ratio_delta"] or 0
        gaze_instability = self.gaze_analyzer.get_gaze_instability()

        stress_score = 0
        if blink_std > 2.0:
            stress_score += 1
        if pupil_delta > 0.25:
            stress_score += 1
        if gaze_instability > 0.015:
            stress_score += 1

        if blink_std < 1.0 and pupil_delta < 0.12 and gaze_instability < 0.008:
            stress_score = max(stress_score - 1, 0)

        self.stress_score_window.append((current_time, stress_score))
        self.stress_score_window = deque([(t, s) for t, s in self.stress_score_window if current_time - t <= 60])

        avg_stress = sum(s for t, s in self.stress_score_window) / len(self.stress_score_window)

        print("[DEBUG] EAR:", result["ear"])
        print("[DEBUG] Blinks:", result["total_blinks"])
        print("[DEBUG] Blink STD:", blink_std)
        print("[DEBUG] Pupil delta:", pupil_delta)
        print("[DEBUG] Gaze instability:", gaze_instability)
        print("[DEBUG] Stress score:", stress_score)
        print("[DEBUG] Avg stress (window):", avg_stress)

        self.ear_label.setText(f"EAR: {result['ear']:.2f}" if result['ear'] else "EAR: -")
        self.blink_label.setText(f"Blinks: {result['total_blinks']}")
        self.blink_std_label.setText(f"Blink STD: {blink_std:.2f}")
        self.pupil_delta_label.setText(f"Pupil Delta: {pupil_delta:.2f}")
        self.gaze_instability_label.setText(f"Gaze Instability: {gaze_instability:.4f}")
        self.stress_score_label.setText(f"Stress Score: {stress_score}")

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

        if avg_stress >= 2.0 and current_time - self.last_alert_time > self.alert_interval:
            self.show_timed_alert("A rendszer fokozott stresszszintet észlelt. Javasolt rövid szünetet tartani a hatékonyabb regeneráció érdekében.")
            self.last_alert_time = current_time

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
        self.ax.set_title("Stress Score Over Time")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Value")
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StressDetectionApp()
    window.show()
    sys.exit(app.exec_())