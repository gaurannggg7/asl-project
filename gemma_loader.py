from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# 1️⃣ Load from the hub (audio-capable base model)
GEMMA_MODEL_ID = "google/gemma-3n-E2B"   # or “google/gemma-3n-E2B”
processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
model     = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map=None
)

# 2️⃣ Save to disk
OUT_DIR = "models/local-gemma-3n-E2B"
processor.save_pretrained(OUT_DIR)
model.save_pretrained(OUT_DIR)
print(f"✅ Model and processor saved under ./{OUT_DIR}")
