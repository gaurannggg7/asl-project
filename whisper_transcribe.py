# -----------------------------
# File: whisper_transcribe.py
# -----------------------------
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

def transcribe_audio(wav_path: str, device=None) -> str:
    """
    Load a WAV file, run it through Whisper, and return the transcription.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", local_files_only=True).to(device)
    model.eval()

    # Load and preprocess audio
    audio, sr = librosa.load(wav_path, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)

    # Generate transcription
    with torch.no_grad():
        tokens = model.generate(**inputs)
    transcription = processor.batch_decode(tokens, skip_special_tokens=True)[0]
    return transcription
