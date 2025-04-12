import dlib

"""
Arcok detektálása Dlib segítségével.
"""
face_detector = dlib.get_frontal_face_detector()

def detect_faces(gray_frame):
    """
    Arcok detektálása egy szürkeárnyalatos képen.
    Paraméterek:
        gray_frame: szürkeárnyalatos kép (numpy array)
    return:
        list: detektált arcok
    """
    return face_detector(gray_frame)
