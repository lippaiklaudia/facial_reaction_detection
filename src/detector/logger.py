import csv
import os
import cv2
from datetime import datetime

"""
Állapotloggolás CSV-be és képernyőkép mentés adott feltételek mellett.
"""
# Fájlnevek
LOG_FILE = "logs/stress_log.csv"
SCREENSHOT_DIR = "logs/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def log_state(ear, mar, frame=None):
    """
    Mentés logfájlba és screenshot, ha küszöb alatt van az érték.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # CSV loggolás
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, f"{ear:.3f}", f"{mar:.3f}"])

    # Screenshot mentés, ha EAR vagy MAR érték alacsony
    if (ear < 0.2 or mar > 0.8) and frame is not None:
        filename = f"{SCREENSHOT_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, frame)
