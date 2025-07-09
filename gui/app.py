import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QTextEdit, QVBoxLayout
from src.recognizer import SpeechRecognizer
from src.translator import english_to_asl_gloss
from src.player import play_glosses

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech → ASL")
        self.rec = SpeechRecognizer("models/vosk-model-en")
        self.btn = QPushButton("Start / Stop")
        self.out = QTextEdit(readOnly=True)
        self.btn.clicked.connect(self.toggle)
        layout = QVBoxLayout(self)
        layout.addWidget(self.btn), layout.addWidget(self.out)
        self.listening = False

    def toggle(self):
        if not self.listening:
            self.btn.setText("Listening… Click to stop")
            self.listening = True
            text = self.rec.listen()
            self.out.setPlainText(text)
            play_glosses(english_to_asl_gloss(text))
            self.btn.setText("Start / Stop")
            self.listening = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Window().show()
    sys.exit(app.exec())
