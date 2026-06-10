import ollama
import json

print("Test 1 — Sans format JSON...")
try:
    r = ollama.chat(
        model="qwen3.5:27b",
        messages=[{"role": "user", "content": "Réponds par 2+2=4"}],
        think=False,
        options={"temperature": 0.1, "num_predict": 100},
    )
    print(f"OK ({len(r.message.content)} chars) : {r.message.content[:100]}")
except Exception as e:
    print(f"ÉCHEC : {e}")

print("\nTest 2 — Avec format JSON...")
try:
    r = ollama.chat(
        model="qwen3.5:27b",
        messages=[{"role": "user", "content": 'Renvoie {"resultat": 4}'}],
        think=False,
        format="json",
        options={"temperature": 0.1, "num_predict": 200},
    )
    print(f"OK : {r.message.content}")
except Exception as e:
    print(f"ÉCHEC : {e}")

print("\nTest 3 — Avec gros prompt comme le vrai Validator...")
gros_prompt = "x" * 5000  # simule un Blueprint + code
try:
    r = ollama.chat(
        model="qwen3.5:27b",
        messages=[{"role": "user", "content": gros_prompt}],
        think=False,
        format="json",
        options={"temperature": 0.1, "num_predict": 1500},
    )
    print(f"OK : {r.message.content[:100]}")
except Exception as e:
    print(f"ÉCHEC : {e}")