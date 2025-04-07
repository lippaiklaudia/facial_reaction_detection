import sys
import os
import cv2
import numpy as np
import time
import csv
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QTextEdit, QPushButton)
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtMultimedia import QSoundEffect
from collections import deque
from datetime import datetime

from keras._tf_keras.keras.models import load_model

# Import útvonal bővítés
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from modules.detection.face_detection import *
from modules.detection.landmarks import *
from modules.analysis.eye_analysis import *
from modules.analysis.color_analysis import *
from modules.analysis.eye_tracker import *
from modules.ml.preprocessing import *

class FacialMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Fatigue Detection")
        self.setGeometry(100, 100, 1000, 700)

        self.cap = cv2.VideoCapture(1)
        #self.model = load_model("models/drowsiness_detector.h5")
        self.model = load_model("models/drowsy_data_model.h5")

        # Állapotváltozók
        self.ear_history = deque(maxlen=10)
        self.fatigue_frames = 0
        self.blink_count = 0
        self.ear_threshold = 0.2
        self.mar_threshold = 0.6
        self.blink_start_frame = None
        self.blink_detected = False
        self.drowsy_state = False
        self.last_log_time = time.time()
        self.detection_active = False
        self.csv_log = []

        self.ear_threshold = self.calibrate_ear()

        # Hangjelzés
        self.sound = QSoundEffect()
        alert_path = os.path.abspath("assets/alert.wav")
        self.sound.setSource(QUrl.fromLocalFile(alert_path))
        self.sound.setLoopCount(-1)  # végtelen ciklus
        self.sound.setVolume(0) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # UI elemek
        self.video_label = QLabel("Camera feed loading...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(800, 500)

        self.status_label = QLabel("Status: Not started")
        self.status_label.setFont(QFont("Arial", 14))

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)

        self.save_button = QPushButton("Export CSV")
        self.save_button.clicked.connect(self.export_log)

        self.recalibrate_button = QPushButton("Recalibrate EAR")
        self.recalibrate_button.clicked.connect(self.recalibrate_ear)

        self.last_sound_play = time.time()
        self.sound_restart_interval = 2  # másodpercenként újraindítjuk

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.recalibrate_button)
        button_layout.addWidget(self.save_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        layout.addWidget(self.log)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def calibrate_ear(self):
        ear_values = []
        for _ in range(30):
            ret, frame = self.cap.read()
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
        valid = [v for v in ear_values if abs(v - np.mean(ear_values)) < 2 * np.std(ear_values)]
        return np.mean(valid) * 0.8 if valid else 0.2

    def recalibrate_ear(self):
        self.ear_threshold = self.calibrate_ear()
        self.log.append(f"[INFO] EAR threshold recalibrated to: {self.ear_threshold:.2f}")

    def start_detection(self):
        self.detection_active = True
        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Detection started.")

    def stop_detection(self):
        self.detection_active = False
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Detection stopped.")
        self.sound.stop()

    def save_screenshot(self, frame):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        screenshot_dir = os.path.join(base_dir, "logs", "screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)

        filename = f"drowsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(screenshot_dir, filename)

        cv2.imwrite(filepath, frame)
        self.log.append(f"[INFO] Screenshot saved: {filepath}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        if self.detection_active:
            frame, overall_status = self.analyze_frame(frame)

            if overall_status == "Drowsy":
                now = time.time()
                if not self.drowsy_state:
                    self.drowsy_state = True
                    self.sound.play()
                    self.save_screenshot(frame)
                    self.last_sound_play = now
                elif now - self.last_sound_play >= self.sound_restart_interval:
                    self.sound.stop()
                    self.sound.play()
                    self.last_sound_play = now
            else:
                if self.drowsy_state:
                    self.drowsy_state = False
                    self.sound.stop()

        self.display_image(frame)

    def analyze_frame(self, frame):
        FRAME_WINDOW = 10
        FATIGUE_THRESHOLD = 10
        RECOVERY_RATE = 3
        DROWSY_BLINK_FRAMES = 8
        BLINK_VALIDATION_FRAMES = 3
        LOG_INTERVAL_SECONDS = 1

        faces = detect_faces(frame)
        overall_status = "Unknown"

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            landmarks = get_landmarks(face_roi)
            mean_hue, mean_sat, mean_val = analyze_color_with_histogram(frame, (x, y, w, h))

            if mean_hue < 20 and mean_sat < 40 and mean_val < 90:
                skin_status = "Fatigue"
            elif mean_hue < 30 and mean_sat < 50 and mean_val < 120:
                skin_status = "Possibly Fatigued"
            else:
                skin_status = "Normal"

            eye_status = "No Data"
            model_label = "Awake"

            if landmarks:
                draw_landmarks(frame, landmarks, (x, y, w, h))
                left_eye = landmarks[0][36:42]
                right_eye = landmarks[0][42:48]
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                self.ear_history.append(avg_ear)

                mouth = landmarks[0][60:68]
                mar = calculate_mar(mouth)
                self.log.append(f"[DEBUG] EAR: {avg_ear:.3f}, MAR: {mar:.3f}")

                avg_ear_window = np.mean(self.ear_history) if self.ear_history else 1.0

                if avg_ear_window < self.ear_threshold:
                    if not self.blink_detected:
                        self.blink_start_frame = time.time()
                        self.blink_detected = True
                    eye_status = "Closed"
                    self.blink_count += 1
                else:
                    if self.blink_detected:
                        blink_duration = (time.time() - self.blink_start_frame) * 1000
                        if blink_duration > DROWSY_BLINK_FRAMES * 100:
                            print(f"Drowsy blink detected. Duration: {blink_duration:.2f} ms")
                        self.blink_detected = False
                    eye_status = "Open"

                processed = preprocess_face_for_model(face_roi)
                prediction = self.model.predict(processed, verbose=0)[0][0]
                model_label = "Drowsy" if prediction >= 0.68 else "Awake"

                if avg_ear_window >= self.ear_threshold and skin_status == "Normal":
                    overall_status = "Awake"
                else:
                    is_drowsy = (self.fatigue_frames > FATIGUE_THRESHOLD or
                                 (model_label == "Drowsy" and avg_ear_window < self.ear_threshold) or
                                 (mar > self.mar_threshold) or
                                 skin_status == "Fatigue")
                    if is_drowsy:
                        self.fatigue_frames = min(FATIGUE_THRESHOLD + 5, self.fatigue_frames + 1)
                    else:
                        self.fatigue_frames = max(0, self.fatigue_frames - RECOVERY_RATE)
                    overall_status = "Drowsy" if self.fatigue_frames > FATIGUE_THRESHOLD else "Awake"

            color = (0, 0, 255) if overall_status == "Drowsy" else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Eye: {eye_status}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Skin: {skin_status}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Overall: {overall_status}", (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            now = time.time()
            if now - self.last_log_time >= LOG_INTERVAL_SECONDS:
                log_entry = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), eye_status, skin_status, model_label, overall_status]
                self.csv_log.append(log_entry)
                self.log.append(f"[INFO] {log_entry[0]} | Eye: {eye_status}, Skin: {skin_status}, Model: {model_label}, Overall: {overall_status}")
                self.last_log_time = now

            self.status_label.setText(f"Status: {overall_status}")

        return frame, overall_status

    def display_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def show_alert(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()

    def export_log(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        filename = f"fatigue_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(logs_dir, filename)

        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Eye Status", "Skin Status", "Model Prediction", "Overall Status"])
            writer.writerows(self.csv_log)

        self.log.append(f"[INFO] Log exported to {filepath}")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FacialMonitor()
    window.show()
    sys.exit(app.exec())