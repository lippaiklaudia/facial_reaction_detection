# Facial Reaction Detection

A comprehensive real-time system for detecting various emotional and physical states such as **drowsiness**, **stress**, and **fatigue** based on facial reactions. The system uses a combination of computer vision techniques (EAR, gaze tracking, facial landmarks) and machine learning models (e.g., CNN) with a PyQt-based GUI.

## Project Objective

This project aims to develop a real-time facial analysis system capable of recognizing:
- **Drowsiness** via Eye Aspect Ratio (EAR), mouth opening (MAR), and blinking detection
- **Stress** using eye movement instability, pupil fluctuation, and facial Action Units (AUs) via OpenFace
- **Fatigue** using a deep learning model trained on labeled datasets

All components are integrated into a modular GUI for real-time visualization and logging.

## Project Structure
(structure.png)

**Used datasets**
Driver Drowsiness Dataset (Ismail Nasri, Kaggle)

FER2013 (for emotion recognition preprocessing – optional module)

Custom image/video input for real-time analysis

**⚙️ Installation**

**Clone the repository:**

git clone https://github.com/lippaiklaudia/facial_reaction_detection.git

cd facial_reaction_detection

**Create and activate a virtual environment:**

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

**Install requirements:**

pip install -r requirements.txt

(Optional) Install and build OpenFace for AU detection.

**▶️ Usage**

**Drowsiness Detection**

python drowsiness_detection.py

**Stress Detection**

python stress_detection.py

**Ensure that:**
A webcam is connected

OpenFace's FeatureExtraction tool is accessible via system PATH (or configured in config.py)

**📈 Features**
Real-time EAR, MAR, blink analysis

Gaze tracking with MediaPipe

Action Unit detection using OpenFace

PyQt GUI with live status indicators and signal plots

Automatic alerting and logging to CSV

Configurable thresholds and model integration

**Future Work**

Machine learning-based stress score estimation (SVM/Random Forest)

Multi-angle detection support

Real-time dashboard enhancements
