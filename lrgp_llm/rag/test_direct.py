# test_direct.py
import ollama

r = ollama.chat(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": "Qu'est-ce que K_OV ?"}],
    options={"think": False},
)
print(len(r.message.content), "chars")
print(r.message.content[:300])