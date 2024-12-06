from real_time_face_detection import real_time_face_and_emotion_detection
from enhanced_emotion_detection_mediapipe import real_time_landmark_emotion_detection

if __name__ == "__main__":
    # DLib Deepface:
    # real_time_face_and_emotion_detection()

    # Mediapipe DeepFace:
    real_time_landmark_emotion_detection(analysis_frequency=10)