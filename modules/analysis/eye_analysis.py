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
    # Szájjellemző pontok NumPy tömbbé alakítása
    mouth = np.array(mouth)

    # Vertikális távolságok kiszámítása
    A = np.linalg.norm(mouth[50] - mouth[58])  # 51 - 59
    B = np.linalg.norm(mouth[51] - mouth[57])  # 52 - 58
    C = np.linalg.norm(mouth[52] - mouth[56])  # 53 - 57
    D = np.linalg.norm(mouth[53] - mouth[55])  # 54 - 56
    E = np.linalg.norm(mouth[54] - mouth[54])  # 55 - 55

    # Horizontális távolság kiszámítása
    F = np.linalg.norm(mouth[48] - mouth[54])  # 49 - 55

    # MAR kiszámítása
    mar = (A + B + C + D + E) / (2.0 * F)
    return mar


