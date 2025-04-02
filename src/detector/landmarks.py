import dlib
from scipy.spatial import distance as dist

"""
Landmark pontok, EAR és MAR számítása.
"""

# Dlib shape predictor betöltése
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(gray_frame, face):
    """Landmark pontok lekérése egy detektált archoz."""
    shape = predictor(gray_frame, face)
    coords = [(p.x, p.y) for p in shape.parts()]
    return coords

def compute_ear(eye_points):
    """Eye Aspect Ratio kiszámítása egy szemhez."""
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def compute_mar(mouth_points):
    """Mouth Aspect Ratio kiszámítása a szájhoz."""
    A = dist.euclidean(mouth_points[13], mouth_points[19])
    B = dist.euclidean(mouth_points[14], mouth_points[18])
    C = dist.euclidean(mouth_points[15], mouth_points[17])
    D = dist.euclidean(mouth_points[12], mouth_points[16])
    return (A + B + C) / (3.0 * D)
