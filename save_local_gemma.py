#!/usr/bin/env python3
import os
from gemma_loader import load_gemma

# 1) Load from HF (or cache) just once
processor, model, device = load_gemma()

# 2) Pick an output folder under your top-level models/
out_dir = os.path.join(os.path.dirname(__file__), "models", "local-gemma-3n-E4B")
os.makedirs(out_dir, exist_ok=True)

# 3) Write both processor and model weights/config there
print(f"💾 Saving processor to {out_dir}")
processor.save_pretrained(out_dir)

print(f"💾 Saving model to {out_dir}")
model.save_pretrained(out_dir)

print("✅ Done! You can now call from_pretrained on:\n    ", out_dir)
