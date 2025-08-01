#!/usr/bin/env python3
import os
import sys
import re
import subprocess
import argparse

import pandas as pd
import sounddevice as sd
import wavio
import torch
import torchaudio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# 300-word set
WORDS = set("""accept afternoon after again against age agree allow almost alone along already also always and angry
animal answer any anything apologize appear approach argue arrive ask asl attitude autumn average avoid away
bad basic beautiful because before believe best better big book both bout boy bring brother but buy busy calm can
car change child class clean close close by cold color come comfortable day deaf deep different drink drive early
eat email enough english every everyday everything example excuse false family far fast father feel few fingerspell
find fine finish follow food for forget friend from funny game gift girl give go gone good grow guess happen hard
have he hear hearing hello help here hold home hot house how hurt idea if important improve in include inform
internet interpreter joke keep know later last late learn leave letter life like little live look look for look like
lose lot love make man many mean meet minute money month more morning mother most move movie must my name near need
never new next night no normal not nothing now number okay old only open opposite out outside overlook paper party
pay pen people picture plan play please practice prefer problem question read ready remember rest right room run sad
safe say school see sell send service share she show sick sign since sister sit slow smart some something
sometimes soon sorry spring start stay still stop store story struggle study summer support sure take talk teach thank
than that their they thing think time tired toilet today tomorrow true try understand until use visit wait walk want
warm wash water watch week weekend what when where which who why will winter wish with word work worse write wrong yes
yesterday yet you young your yourself""".split())

def record_audio(duration=5, fs=16000, filename="sample.wav"):
    print(f"🎙️ Recording {duration}s of audio at {fs} Hz...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    print(f"✅ Recording complete, saved to '{filename}'")
    return filename

def transcribe_whisper(audio_path):
    print("🚀 Loading Whisper model on MPS...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("mps")
    # load audio waveform + sampling rate
    waveform, sr = torchaudio.load(audio_path)               # returns Tensor [channels, time]
    mono = waveform.mean(dim=0)                              # mix to mono
    # feature-extractor expects float array + sampling_rate
    inputs = processor(mono.numpy(), sampling_rate=sr, return_tensors="pt").input_features.to("mps")
    generated_ids = model.generate(inputs)
    transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcript

def generate_gloss(text):
    print("🚀 Loading Gemma model...")
    tokenizer = AutoTokenizer.from_pretrained("models/gemma3n_E2B_it")
    gemma = AutoModelForCausalLM.from_pretrained(
        "models/local-gemma-3n-E2B",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    ).to("mps")
    prompt = """You are a sign-language interpreter. Convert English to ASL gloss in ALL CAPS,
drop articles (A, AN, THE), use topic–comment order, separate signs with spaces.
Examples:
English: HELLO HOW ARE YOU?
ASL Gloss: HELLO HOW YOU
English: Hello, my name is Cactus. Chubby face.
ASL Gloss:"""
    inputs = tokenizer(prompt + text, return_tensors="pt").input_ids.to("mps")
    outputs = gemma.generate(inputs, max_new_tokens=32)
    gloss = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    tokens = re.findall(r"[A-Z]+", gloss)
    return tokens

def load_video_mapping(csv_name="gloss_index.csv"):
    script_dir = os.path.dirname(__file__)
    csv_path  = os.path.join(script_dir, csv_name)
    media_root = os.path.join(script_dir, "media", "SG ASL Dictionary")
    df = pd.read_csv(csv_path)
    df["word"] = df["gloss"].str.lower().str.extract(r"^([a-z]+)")[0]
    df = df[df["word"].isin(WORDS)].copy()
    df["path"] = df["path"].apply(lambda p: os.path.join(media_root, p.replace("\\", "/")))
    df["token"] = df["word"].str.upper()
    return df.set_index("token")["path"].to_dict()

def play_clips(tokens, mapping):
    clips = []
    for tok in tokens:
        path = mapping.get(tok)
        if path and os.path.isfile(path):
            clips.append(path)
        else:
            print(f"⚠️ Missing video for token '{tok}'")
    if not clips:
        print("❌ No clips found to play.")
        sys.exit(1)
    for clip in clips:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", clip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

def main():
    parser = argparse.ArgumentParser(
        description="CLI: speech → ASL gloss → video playback"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--record", action="store_true",
        help="Record 5s of mic audio and transcribe"
    )
    group.add_argument(
        "--text", type=str,
        help="Skip recording; use this text for ASR→gloss"
    )
    args = parser.parse_args()

    if args.record:
        wav = record_audio()
        transcript = transcribe_whisper(wav)
    else:
        transcript = args.text
    print("📋 Whisper transcription:", transcript)

    gloss_tokens = generate_gloss(transcript)
    print("🧩 ASL gloss tokens:", gloss_tokens)

    mapping = load_video_mapping()
    play_clips(gloss_tokens, mapping)

if __name__ == "__main__":
    main()
