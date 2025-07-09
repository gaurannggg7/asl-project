from recognizer import SpeechRecognizer
from translator import english_to_asl_gloss
from player import play_glosses
import argparse

def main():
    parser = argparse.ArgumentParser(description="Speech-to-ASL Translator")
    parser.add_argument("--model", default="models/vosk-model-en", help="Path to Vosk model")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    args = parser.parse_args()

    recognizer = SpeechRecognizer(model_path=args.model)

    if args.mic:
        print("🎙️ Listening... (Ctrl-C to stop)")
        transcript = recognizer.listen()
    else:
        # fallback: use speech.wav
        import wave, json
        from vosk import KaldiRecognizer

        wf = wave.open("speech.wav", "rb")
        rec = KaldiRecognizer(recognizer.model, wf.getframerate())
        transcript = ""
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                transcript += res.get("text", "") + " "

    print("📝 Transcript:", transcript)

    glosses = english_to_asl_gloss(transcript)
    print("🤟 ASL Gloss:", glosses)

    play_glosses(glosses)

if __name__ == "__main__":
    main()
