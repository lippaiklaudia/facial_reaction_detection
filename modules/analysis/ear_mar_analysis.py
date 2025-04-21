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
    if len(mouth) != 8:
        raise ValueError("A 'mouth' listának 8 pontot kell tartalmaznia.")

    # NumPy tömbbé alakítás
    mouth = np.array(mouth)

    # Vertikális távolságok kiszámítása
    A = np.linalg.norm(mouth[1] - mouth[5])  # 61 - 67
    B = np.linalg.norm(mouth[2] - mouth[4])  # 62 - 66
    C = np.linalg.norm(mouth[0] - mouth[3])  # 60 - 64

    # MAR kiszámítása
    mar = (A + B) / (2.0 * C)
    return mar

