import torch
from transformers import pipeline

# Load once, on import
device = 0 if torch.cuda.is_available() else -1
asr = pipeline(
    "automatic-speech-recognition",
    model="path/to/gemma-3n-e2b-it",      # or your Drive path
    device=device,
    chunk_length_s=30,                    # adjust for longer audio
)

def transcribe_file(wav_path: str) -> str:
    """
    Transcribe a WAV file to text using Gemma 3n.
    """
    result = asr(wav_path)
    return result["text"]
if __name__ == "__main__":
    text = transcribe_file("tests/hello_borrow_book.wav")
    print("GemmaASR →", text)
