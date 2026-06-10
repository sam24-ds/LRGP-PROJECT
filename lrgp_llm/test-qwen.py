# test_qwen.py
import ollama

# Test simple
print("=== Test 1 : avec /no_think ===")
r = ollama.chat(
    model="qwen3.5:27b",
    messages=[
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": "Calcule 2+2. Réponds en un mot."}
    ],
    options={"num_predict": 100},
)
print(f"Réponse ({len(r.message.content)} chars) : '{r.message.content}'")

# Test 2 : sans /no_think
print("\n=== Test 2 : sans /no_think ===")
r = ollama.chat(
    model="qwen3.5:27b",
    messages=[{"role": "user", "content": "Calcule 2+2. Réponds en un mot."}],
    options={"num_predict": 100},
)
print(f"Réponse ({len(r.message.content)} chars) : '{r.message.content}'")

# Test 3 : avec num_predict élevé
print("\n=== Test 3 : num_predict=5000 ===")
r = ollama.chat(
    model="qwen3.5:27b",
    messages=[{"role": "user", "content": "Calcule 2+2. Réponds en un mot."}],
    options={"num_predict": 5000},
)
print(f"Réponse ({len(r.message.content)} chars) : '{r.message.content}'")