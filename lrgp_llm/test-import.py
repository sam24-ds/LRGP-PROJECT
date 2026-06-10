import sys
sys.path.insert(0, '.')
from unsloth import FastLanguageModel
from pathlib import Path

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = 'C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\training\\training\\exports\\qwen35-9b-lrgp-lora-v2',
    max_seq_length = 4096,
    load_in_4bit   = False,
)
model.save_pretrained_gguf(
    'C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\training\\training\\exports\\qwen35-9b-lrgp-gguf-v2',
    tokenizer,
    quantization_method = 'q4_k_m',
)
print('GGUF exporté avec succès')