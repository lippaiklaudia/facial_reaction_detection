import sys
import os
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtCore import QUrl

class SoundTest(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sound Test")
        self.setGeometry(200, 200, 300, 100)

        sound_path = os.path.abspath("assets/alert.wav")
        print(f"Sound file path: {sound_path}")
        print(f"File exists: {os.path.exists(sound_path)}")

        self.sound = QSoundEffect()
        self.sound.setSource(QUrl.fromLocalFile(sound_path))
        self.sound.setLoopCount(-1)  # végtelen ciklus
        self.sound.setVolume(1.0)
        self.sound.play()

        print("Playing sound...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SoundTest()
    window.show()
    sys.exit(app.exec())
