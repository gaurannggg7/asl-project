# -----------------------------
# File: demo_cli.py
# -----------------------------
import argparse
import soundfile as sf
import subprocess
import sys

from source.whisper_transcribe import transcribe_audio
from gemma_loader import load_gemma
from source.gloss_builder import build_gloss
from source.renderer import render_gloss


def main():
    parser = argparse.ArgumentParser(description="Core CLI: speech→gloss→video")
    parser.add_argument('input', help='WAV file path or plain text')
    parser.add_argument('--csv', '-c', default='gloss_index.csv', help='Gloss CSV mapping')
    args = parser.parse_args()

    # 1) Whisper transcribe (if WAV)
    if args.input.lower().endswith('.wav'):
        text = transcribe_audio(args.input)
    else:
        text = args.input
    print(f"[Whisper] → {text}")

    # 2) Gemma gloss
    processor, model, device = load_gemma()
    if args.input.lower().endswith('.wav'):
        data, sr = sf.read(args.input)
        tokens, resp = build_gloss(processor, model, device, data, sr)
    else:
        tokens, resp = build_gloss(processor, model, device, text)
    print(f"[Gemma] → {resp}")

    # 3) Render + play
    out = render_gloss(tokens, args.csv)
    print(f"Output video: {out}")

    # Play video
    if sys.platform == 'darwin':
        subprocess.run(['open', out])
    elif sys.platform.startswith('linux'):
        subprocess.run(['ffplay','-autoexit','-nodisp', out])
    else:
        print("Please open the video file manually.")

if __name__ == '__main__':
    main()