import numpy as np

def calculate_ear(eye):
    # Eye Aspect Ratio kiszamitasa

    eye = np.array(eye)
    A = np.linalg.norm(eye[1] - eye[5])  
    B = np.linalg.norm(eye[2] - eye[4])  
    C = np.linalg.norm(eye[0] - eye[3]) 
    ear = (A + B) / (2.0 * C)  # EAR keplet
    return ear

def calculate_mar(mouth):
    # Tuple -> NumPy array konverzió
    p1 = np.array(mouth[1])
    p2 = np.array(mouth[7])
    p3 = np.array(mouth[2])
    p4 = np.array(mouth[6])
    p5 = np.array(mouth[0])
    p6 = np.array(mouth[4])

    A = np.linalg.norm(p1 - p2)  # 61 - 67
    B = np.linalg.norm(p3 - p4)  # 62 - 66
    C = np.linalg.norm(p5 - p6)  # 60 - 64

    mar = (A + B) / (2.0 * C)
    return mar

