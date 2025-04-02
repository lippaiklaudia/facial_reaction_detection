import cv2
import numpy as np

# hisztogram alapu szinanalizis
def analyze_color_with_histogram(frame, bbox):

    roi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[:, :, 0])  # Hue ertekek atlaga
    mean_saturation = np.mean(hsv[:, :, 1])  # Saturation ertekek atlaga
    mean_value = np.mean(hsv[:, :, 2])  # Value ertekek atlaga

    return mean_hue, mean_saturation, mean_value