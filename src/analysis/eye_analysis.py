import numpy as np
import cv2
from detection.landmarks import get_landmarks

def calculate_ear(eye):
    # Eye Aspect Ratio kiszamitasa

    eye = np.array(eye)
    A = np.linalg.norm(eye[1] - eye[5])  
    B = np.linalg.norm(eye[2] - eye[4])  
    C = np.linalg.norm(eye[0] - eye[3]) 
    ear = (A + B) / (2.0 * C)  # EAR keplet
    return ear

