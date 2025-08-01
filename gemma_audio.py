import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import sounddevice as sd
import numpy as np
import io
from scipy.io.wavfile import write


class GemmaAudioProcessor:
    def __init__(self, model_id="google/gemma-3n-E4B-it"):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
            device_map="auto"
        )

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