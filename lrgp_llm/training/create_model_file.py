from pathlib import Path
p = Path(r'training\training\exports\qwen35-9b-lrgp-gguf_gguf\Modelfile')
p.write_text(
    'FROM C:\\\\Users\\\\Samir\\\\Documents\\\\LRGP-PROJECT\\\\lrgp_llm\\\\training\\\\training\\\\exports\\\\qwen35-9b-lrgp-gguf_gguf\\\\Qwen3.5-9B.Q4_K_M.gguf\n'
    'PARAMETER stop \"<|endoftext|>\"\n'
    'PARAMETER stop \"<|im_end|>\"\n'
    'PARAMETER stop \"Human:\"\n'
)
print('Modelfile mis à jour')