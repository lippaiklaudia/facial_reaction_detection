import cv2
import dlib

# Dlib + landmark detektor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    landmarks_list = []

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        points = [(p.x, p.y) for p in landmarks.parts()]
        landmarks_list.append(points)

    return landmarks_list

def draw_landmarks(frame, landmarks, face_bbox):
    for point in landmarks:
        for (x, y) in point:
            cv2.circle(frame, (x + face_bbox[0], y + face_bbox[1]), 2, (0, 255, 0), -1)
