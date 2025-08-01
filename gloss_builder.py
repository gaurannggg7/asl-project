

# -----------------------------
# File: gloss_builder.py
# -----------------------------
import re
import torch

def build_gloss(processor, model, device, text_or_audio, sampling_rate=None):
    """
    Given either a text string or numpy audio array + rate, returns (tokens, raw_response).
    """
    # Construct chat-style prompt
    if isinstance(text_or_audio, str):
        messages = [{"role":"user","content":[{"type":"text","text":text_or_audio}]}]
    else:
        messages = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": text_or_audio, "sampling_rate": sampling_rate},
                {"type": "text",  "text": "Transcribe and provide ASL gloss."}
            ]
        }]

    # Tokenize + generate
    raw = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )
    inputs = {}
    for name, tensor in raw.items():
        # first move everything on‑device
        t = tensor.to(device)
        # then, if it’s a floating tensor, cast to model.dtype (fp16)
        if t.dtype.is_floating_point:
            t = t.to(dtype=model.dtype)
        # leave integer tensors alone
        inputs[name] = t

    # now call generate
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    resp = processor.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    # Extract tokens (e.g. ['HELLO','HOW','ARE','YOU'])
    tokens = re.findall(r"[A-Z()0-9]+", resp)
    return tokens, resp
