import cv2
from analysis.eye_tracker import EyeTracker
from analysis.stress_estimator import StressEstimator
import csv
import os

eye_tracker = EyeTracker()
stress_estimator = StressEstimator()

cap = cv2.VideoCapture(1)
log_file = "stress_log.csv"

if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "ear", "blink_rate", "pupil_ratio", "blink_stddev", "pupil_delta", "stress_score"])

def draw_text_block(frame, text, pos, color, font_scale=0.7, thickness=2, bg_color=(0, 0, 0)):
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    metrics = eye_tracker.detect(frame)
    stress_score = stress_estimator.estimate_stress(
        blink_rate=metrics["blinks_per_minute"],
        ear=metrics["ear"],
        pupil_ratio=metrics["pupil_ratio"],
        blink_stddev=metrics["blink_interval_stddev"],
        pupil_delta=metrics["pupil_ratio_delta"]
    )

    y_pos = 30
    line_space = 35

    draw_text_block(frame, f"EAR: {metrics['ear']:.2f}" if metrics["ear"] else "EAR: --", (10, y_pos), (255, 255, 255))
    y_pos += line_space
    draw_text_block(frame, f"Blinks/min: {metrics['blinks_per_minute']}", (10, y_pos), (0, 255, 0))
    y_pos += line_space
    draw_text_block(frame, f"Pupil Ratio: {metrics['pupil_ratio']:.2f}" if metrics["pupil_ratio"] else "Pupil Ratio: --", (10, y_pos), (255, 255, 0))
    y_pos += line_space
    draw_text_block(frame, f"Blink Stddev: {metrics['blink_interval_stddev']:.2f}" if metrics["blink_interval_stddev"] else "Blink Stddev: --", (10, y_pos), (255, 165, 0))
    y_pos += line_space
    draw_text_block(frame, f"Pupil Delta: {metrics['pupil_ratio_delta']:.2f}" if metrics["pupil_ratio_delta"] else "Pupil Delta: --", (10, y_pos), (200, 200, 0))
    y_pos += line_space

    color = (0, 255, 0) if stress_score <= 1 else (0, 165, 255) if stress_score <= 3 else (0, 0, 255)
    draw_text_block(frame, f"Stress Score: {stress_score}/5", (10, y_pos), color, font_scale=0.8, thickness=2)

    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            metrics["timestamp"],
            metrics["ear"],
            metrics["blinks_per_minute"],
            metrics["pupil_ratio"],
            metrics["blink_interval_stddev"],
            metrics["pupil_ratio_delta"],
            stress_score
        ])

    cv2.imshow("Stress Detection Dashboard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
