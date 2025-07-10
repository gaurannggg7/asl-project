
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GemmaTranslator:
    def __init__(self):
        model_id = "google/gemma-3n-e4b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.model.eval()

    def to_asl_gloss(self, english_text):
        prompt = f"Translate this to ASL gloss: {english_text}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        asl_gloss = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return asl_gloss.replace(prompt, "").strip()
