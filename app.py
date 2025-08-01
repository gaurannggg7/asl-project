import sys
import os
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QPushButton, QTextEdit, QLineEdit,
                             QStatusBar, QMenuBar, QMenu, QAction, QFileDialog,
                             QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QIcon
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import sounddevice as sd
import numpy as np
import io
from scipy.io.wavfile import write


class GlossProcessor:
    def __init__(self):
        import re
        self.gloss_regex = re.compile(r'([A-Z]+)(?:\s|$)')

    def clean_gloss(self, gloss_text):
        """Extract valid gloss tokens from model output"""
        matches = self.gloss_regex.findall(gloss_text)
        return [match for match in matches if match]

    def validate_gloss(self, gloss_tokens, available_signs):
        """Check which gloss tokens have corresponding videos"""
        valid = []
        missing = []

        for token in gloss_tokens:
            if token in available_signs:
                valid.append(token)
            else:
                missing.append(token)

        return valid, missing


class VideoRenderer:
    def __init__(self, media_dir="media/dictionary"):
        self.media_dir = media_dir

    def get_video_path(self, gloss_token):
        return os.path.join(self.media_dir, f"{gloss_token}.mp4")

    def stitch_videos(self, gloss_tokens, output_path="output.mp4"):
        """Combine multiple ASL videos into one"""
        import ffmpeg
        input_streams = []

        for token in gloss_tokens:
            video_path = self.get_video_path(token)
            if os.path.exists(video_path):
                input_streams.append(ffmpeg.input(video_path))

        if not input_streams:
            raise ValueError("No valid video files found for gloss tokens")

        # Concatenate videos
        joined = ffmpeg.concat(*input_streams, v=1, a=0)
        output = ffmpeg.output(joined, output_path)
        ffmpeg.run(output, overwrite_output=True)
        return output_path


class GemmaAudioProcessor:
    def __init__(self, model_path="models/gemma3n_E2B_it"):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        try:
            print(f"Initializing Gemma model from {model_path}...")

            # Configure 4-bit quantization
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16
            )

            print(f"Model successfully loaded on {self.device} with 4-bit quantization")

        except Exception as e:
            print(f"Quantized load failed: {str(e)}. Trying CPU fallback...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    offload_folder="offload",
                    low_cpu_mem_usage=True
                )
                print("Model loaded on CPU with offloading")
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {str(e2)}")

    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio from microphone"""
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype='float32')
        sd.wait()
        return audio, sample_rate

    def audio_to_text(self, audio, sample_rate):
        """Convert audio to text using Gemma"""
        # Convert numpy array to WAV bytes
        wav_io = io.BytesIO()
        write(wav_io, sample_rate, audio)
        wav_io.seek(0)

        # Create prompt with audio
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wav_io},
                    {"type": "text", "text": "Transcribe this audio:"},
                ]
            }
        ]

        # Process and generate
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(inputs, max_new_tokens=128)
        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract just the transcription part
        return text.split("Transcribe this audio:")[-1].strip()

    def text_to_gloss(self, text):
        """Convert text to ASL gloss using Gemma"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Translate this English text to ASL gloss:\nEnglish: {text}\nASL Gloss:"},
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(inputs, max_new_tokens=32)
        gloss = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract just the ASL gloss part
        return gloss.split("ASL Gloss:")[-1].strip()

class ASLTranslationThread(QThread):
    finished = pyqtSignal(str, list, list)  # text, valid_gloss, missing_gloss
    error = pyqtSignal(str)

    def __init__(self, text, audio_processor, parent=None):
        super().__init__(parent)
        self.text = text
        self.audio_processor = audio_processor

    def run(self):
        try:
            # Get gloss translation
            gloss = self.audio_processor.text_to_gloss(self.text)

            # Process gloss tokens
            processor = GlossProcessor()
            gloss_tokens = processor.clean_gloss(gloss)

            # For now, we'll just return all tokens as "valid"
            valid_gloss = gloss_tokens
            missing_gloss = []

            self.finished.emit(gloss, valid_gloss, missing_gloss)
        except Exception as e:
            self.error.emit(f"Error in translation: {str(e)}")


class AudioRecordingThread(QThread):
    finished = pyqtSignal(np.ndarray, int)  # audio, sample_rate
    error = pyqtSignal(str)

    def __init__(self, duration, parent=None):
        super().__init__(parent)
        self.duration = duration

    def run(self):
        try:
            audio, sample_rate = sd.rec(int(self.duration * 16000),
                                        samplerate=16000,
                                        channels=1,
                                        dtype='float32')
            sd.wait()
            self.finished.emit(audio, 16000)
        except Exception as e:
            self.error.emit(f"Recording error: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("ASL Translator")
        self.setWindowIcon(QIcon("icon.png"))  # Add your icon file
        self.setGeometry(100, 100, 1000, 700)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Initialize components
        self.audio_processor = None
        self.renderer = VideoRenderer()
        self.current_video_path = None
        self.is_recording = False
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_timer)
        self.recording_seconds = 0

        # Create UI
        self.create_menu_bar()
        self.create_status_bar()
        self.create_input_section(main_layout)
        self.create_output_section(main_layout)
        self.create_video_section(main_layout)

        # Media player for video playback
        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)
        main_layout.addWidget(self.video_widget)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Recording timer label
        self.recording_timer_label = QLabel("00:00")
        self.recording_timer_label.setAlignment(Qt.AlignCenter)
        self.recording_timer_label.setVisible(False)
        main_layout.addWidget(self.recording_timer_label)

    def create_menu_bar(self):
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)

        # File menu
        file_menu = QMenu("&File", self)
        menu_bar.addMenu(file_menu)

        open_action = QAction("&Open Audio File...", self)
        open_action.triggered.connect(self.open_audio_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Settings menu
        settings_menu = QMenu("&Settings", self)
        menu_bar.addMenu(settings_menu)

        model_action = QAction("&Model Settings...", self)
        model_action.triggered.connect(self.show_model_settings)
        settings_menu.addAction(model_action)

        # Help menu
        help_menu = QMenu("&Help", self)
        menu_bar.addMenu(help_menu)

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def create_input_section(self, layout):
        input_layout = QHBoxLayout()

        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("Enter text or record audio...")
        input_layout.addWidget(self.input_text, stretch=4)

        self.record_button = QPushButton("Record (0:05)")
        self.record_button.clicked.connect(self.toggle_recording)
        input_layout.addWidget(self.record_button, stretch=1)

        self.translate_button = QPushButton("Translate")
        self.translate_button.clicked.connect(self.start_translation)
        input_layout.addWidget(self.translate_button, stretch=1)

        layout.addLayout(input_layout)

    def create_output_section(self, layout):
        output_layout = QHBoxLayout()

        self.gloss_output = QTextEdit()
        self.gloss_output.setReadOnly(True)
        self.gloss_output.setPlaceholderText("ASL gloss will appear here...")
        output_layout.addWidget(self.gloss_output, stretch=1)

        self.missing_tokens = QTextEdit()
        self.missing_tokens.setReadOnly(True)
        self.missing_tokens.setPlaceholderText("Missing signs will appear here...")
        output_layout.addWidget(self.missing_tokens, stretch=1)

        layout.addLayout(output_layout)

    def create_video_section(self, layout):
        video_controls = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_video)
        self.play_button.setEnabled(False)
        video_controls.addWidget(self.play_button)

        self.save_button = QPushButton("Save Video")
        self.save_button.clicked.connect(self.save_video)
        self.save_button.setEnabled(False)
        video_controls.addWidget(self.save_button)

        layout.addLayout(video_controls)

    def initialize_audio_processor(self):
        if self.audio_processor is None:
            self.status_bar.showMessage("Initializing Gemma audio processor...")
            QApplication.processEvents()
            try:
                self.audio_processor = GemmaAudioProcessor()
                self.status_bar.showMessage("Gemma processor initialized successfully")
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize audio processor: {str(e)}")
                self.status_bar.showMessage("Initialization failed")
                return False
        return True

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not self.initialize_audio_processor():
            return

        self.is_recording = True
        self.recording_seconds = 0
        self.record_button.setText("Stop Recording")
        self.recording_timer_label.setText("00:00")
        self.recording_timer_label.setVisible(True)
        self.status_bar.showMessage("Recording...")

        # Disable other controls during recording
        self.translate_button.setEnabled(False)
        self.input_text.setEnabled(False)

        # Start recording thread
        self.recording_thread = AudioRecordingThread(duration=5)
        self.recording_thread.finished.connect(self.recording_complete)
        self.recording_thread.error.connect(self.handle_recording_error)
        self.recording_thread.start()

        # Start timer for UI updates
        self.recording_timer.start(1000)

    def stop_recording(self):
        self.is_recording = False
        self.recording_timer.stop()
        self.recording_timer_label.setVisible(False)
        self.record_button.setText("Record (0:05)")
        self.translate_button.setEnabled(True)
        self.input_text.setEnabled(True)

        # If we have a recording thread, wait for it to finish
        if hasattr(self, 'recording_thread'):
            self.recording_thread.quit()

    def update_recording_timer(self):
        self.recording_seconds += 1
        remaining = max(0, 5 - self.recording_seconds)
        self.recording_timer_label.setText(f"00:{remaining:02d}")

        if remaining == 0:
            self.stop_recording()

    def recording_complete(self, audio, sample_rate):
        try:
            self.status_bar.showMessage("Processing audio...")
            QApplication.processEvents()

            # Convert audio to text
            text = self.audio_processor.audio_to_text(audio, sample_rate)
            self.input_text.setText(text)
            self.status_bar.showMessage("Audio transcribed successfully")

        except Exception as e:
            self.status_bar.showMessage(f"Error processing audio: {str(e)}")
            QMessageBox.critical(self, "Error", f"Audio processing failed: {str(e)}")

        finally:
            self.record_button.setEnabled(True)
            self.translate_button.setEnabled(True)
            self.input_text.setEnabled(True)

    def handle_recording_error(self, error_msg):
        QMessageBox.critical(self, "Recording Error", error_msg)
        self.stop_recording()

    def open_audio_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.mp3 *.wav *.m4a);;All Files (*)"
        )

        if file_name:
            if not self.initialize_audio_processor():
                return

            self.input_text.setText(f"Processing audio file: {file_name}")
            self.process_audio_file(file_name)

    def process_audio_file(self, file_path):
        self.status_bar.showMessage("Transcribing audio...")
        QApplication.processEvents()

        try:
            # Read audio file
            import librosa
            audio, sample_rate = librosa.load(file_path, sr=16000)

            # Convert audio to text
            text = self.audio_processor.audio_to_text(audio, sample_rate)
            self.input_text.setText(text)
            self.status_bar.showMessage("Audio transcribed successfully")

        except Exception as e:
            self.status_bar.showMessage(f"Error processing audio: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to process audio: {str(e)}")

    def start_translation(self):
        text = self.input_text.text().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "Please enter some text to translate")
            return

        if not self.initialize_audio_processor():
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_bar.showMessage("Translating to ASL gloss...")
        self.translate_button.setEnabled(False)
        self.record_button.setEnabled(False)

        # Run translation in background thread
        self.translation_thread = ASLTranslationThread(text, self.audio_processor)
        self.translation_thread.finished.connect(self.translation_complete)
        self.translation_thread.error.connect(self.handle_translation_error)
        self.translation_thread.start()

    def translation_complete(self, gloss, valid_gloss, missing_gloss):
        self.progress_bar.setVisible(False)
        self.translate_button.setEnabled(True)
        self.record_button.setEnabled(True)

        self.gloss_output.setPlainText(gloss)

        if missing_gloss:
            self.missing_tokens.setPlainText("\n".join(missing_gloss))
            QMessageBox.warning(self, "Missing Signs",
                                f"Videos not available for: {', '.join(missing_gloss)}")
        else:
            self.missing_tokens.clear()

        if valid_gloss:
            self.status_bar.showMessage("Generating ASL video...")
            QApplication.processEvents()

            try:
                output_path = self.renderer.stitch_videos(valid_gloss)
                self.current_video_path = output_path
                self.play_button.setEnabled(True)
                self.save_button.setEnabled(True)
                self.status_bar.showMessage("ASL video generated successfully")
            except Exception as e:
                self.status_bar.showMessage(f"Error generating video: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to generate video: {str(e)}")
        else:
            self.status_bar.showMessage("No valid gloss tokens to render")

    def handle_translation_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.translate_button.setEnabled(True)
        self.record_button.setEnabled(True)
        QMessageBox.critical(self, "Translation Error", error_msg)

    def play_video(self):
        if hasattr(self, 'current_video_path') and self.current_video_path:
            media_content = QMediaContent(QUrl.fromLocalFile(self.current_video_path))
            self.media_player.setMedia(media_content)
            self.media_player.play()

    def save_video(self):
        if hasattr(self, 'current_video_path') and self.current_video_path:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Video", "",
                "Video Files (*.mp4);;All Files (*)"
            )

            if file_name:
                try:
                    import shutil
                    shutil.copy2(self.current_video_path, file_name)
                    self.status_bar.showMessage(f"Video saved to {file_name}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save video: {str(e)}")

    def show_model_settings(self):
        if self.audio_processor:
            device = self.audio_processor.device
            dtype = str(self.audio_processor.model.dtype)
            model_path = "models/gemma3n_E2B_it (local)"
        else:
            device = "Not loaded"
            dtype = "Unknown"
            model_path = "Not loaded"

        QMessageBox.information(self, "Model Settings",
                                f"Current Model: {model_path}\n"
                                f"Device: {device}\n"
                                f"Precision: {dtype}")

    def show_about(self):
        about_text = """<b>ASL Translator</b><br><br>
        Version 2.0<br>
        <br>
        An application that translates spoken or written English to American Sign Language (ASL) videos.<br>
        <br>
        Uses Gemma for both speech recognition and ASL gloss translation."""

        QMessageBox.about(self, "About ASL Translator", about_text)


if __name__ == "__main__":
    # Create offload directory if it doesn't exist
    os.makedirs("offload", exist_ok=True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())