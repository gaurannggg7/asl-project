#!/usr/bin/env python3
"""
Standalone test script: record 5s from mic → save WAV → transcribe with Whisper → feed transcription to Gemma → print ASL gloss → stitch ASL videos based on tokens.
Each step prints a status message.
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import re
import csv
import os
# Video stitching dependency
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ImportError:
    try:
        import moviepy.editor as mpy
        VideoFileClip = mpy.VideoFileClip
        concatenate_videoclips = mpy.concatenate_videoclips
    except Exception:
        print("⚠️ moviepy.editor not available. Install with: pip install moviepy")
        VideoFileClip = None
        concatenate_videoclips = None

from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoProcessor, AutoModelForImageTextToText, pipeline
)

# 1️⃣ Record audio from mic
def record_and_save(duration_s=5.0, sr=16000, filename="sample.wav"):
    print(f"🎙️ Recording {duration_s}s of audio at {sr} Hz...")
    data = sd.rec(int(duration_s * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = data.squeeze(1)
    print(f"✅ Recording complete, {audio.shape[0]} samples captured.")

    sf.write(filename, audio, sr)
    print(f"💾 Audio saved to '{filename}'\n")
    return filename, audio, sr

# 2️⃣ Transcribe WAV via Whisper
def transcribe_with_whisper(wav_path: str, device=None):
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"🚀 Loading Whisper model on {device}...")
    proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    model.eval()
    print("✅ Whisper loaded. Transcribing...")

    import librosa
    audio, sr = librosa.load(wav_path, sr=16000)
    inputs = proc(audio, sampling_rate=sr, return_tensors="pt").to(device)

    with torch.no_grad():
        tokens_ids = model.generate(**inputs)
    text = proc.batch_decode(tokens_ids, skip_special_tokens=True)[0].strip()
    print(f"📋 Whisper transcription: {text}\n")
    return text

# 3️⃣ Load Gemma for gloss
def load_gemma(model_dir="models/local-gemma-3n-E2B"):
    device, dtype = (
        (torch.device("cuda"), torch.float16) if torch.cuda.is_available() else
        (torch.device("mps"), torch.float16) if torch.backends.mps.is_available() else
        (torch.device("cpu"), torch.float32)
    )
    print(f"🚀 Loading Gemma model from '{model_dir}' on {device}...")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        local_files_only=True
    )
    model.to(device).eval()
    print(f"✅ Gemma loaded (dtype={model.dtype})\n")
    return processor, model, device

# 4️⃣ Get ASL gloss from text via Gemma
def gemma_gloss_from_text(processor, model, device, text: str):
    prompt = (
        "You are a sign-language interpreter. Convert English to ASL gloss in ALL CAPS, "
        "drop articles (A, AN, THE), use topic–comment order, separate signs with spaces.\n"
        "Examples:\n"
        "English: HELLO HOW ARE YOU?\n"
        "ASL Gloss: HELLO HOW YOU\n"
        f"English: {text}\n"
        "ASL Gloss:"
    )
    print(f"✏️ Prompting Gemma with few-shot examples:\n{prompt}\n")

    gloss_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=processor.tokenizer,
        device_map="auto",
        torch_dtype=model.dtype
    )
    output = gloss_pipe(
        prompt,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id
    )[0]["generated_text"]

    gloss = output.split("ASL Gloss:")[-1].strip()
    print(f"📋 ASL Gloss: {gloss}\n")

    tokens = [t for t in re.findall(r"[A-Z()0-9]+", gloss) if len(t) > 1]
    print(f"🧩 ASL gloss tokens: {tokens}\n")
    print(f"✅ Completed: {len(tokens)} token(s).\n")
    return tokens

# 5️⃣ Load gloss-to-video mapping from CSV
def load_video_mapping(csv_path="gloss_index.csv"):
    print(f"📂 Loading video mapping from '{csv_path}'...")
    mapping = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # CSV columns: 'gloss' and 'path'
            gloss_key = (row.get("gloss") or row.get("token") or "").upper()
            video_path = row.get("path") or row.get("video_path")
            if gloss_key and video_path:
                mapping[gloss_key] = video_path
    print(f"✅ Loaded {len(mapping)} mappings.")
    return mapping

# 6️⃣ Stitch tokens into a single video
def stitch_videos(tokens, mapping, output_path="output.mp4"):
    clips = []
    for t in tokens:
        # direct lookup
        path = mapping.get(t)
        if not path:
            # fuzzy match: find any mapping key that starts with token
            # try substring match if prefix fails
            fuzzy_keys = [k for k in mapping.keys() if t in k]
            if fuzzy_keys:
                key = fuzzy_keys[0]
                path = mapping[key]
                print(f"🔍 Fuzzy matched '{t}' to '{key}'")
        if path and VideoFileClip and os.path.isfile(path):
            print(f"▶️ Adding clip for '{t}': {path}")
            clips.append(VideoFileClip(path))
        else:
            print(f"⚠️ Missing video for token '{t}'")
    if not clips:
        print("❌ No clips found to stitch.")
        return None
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_path, codec="libx264")
    print(f"💾 Saved stitched video to {output_path}")
    return output_path

# ───── Main ─────────────────────────────────
if __name__ == '__main__':
    wav_path, audio, sr = record_and_save()
    transcription = transcribe_with_whisper(wav_path)
    processor, model, device = load_gemma()
    tokens = gemma_gloss_from_text(processor, model, device, transcription)
    mapping = load_video_mapping()
    stitch_videos(tokens, mapping)
