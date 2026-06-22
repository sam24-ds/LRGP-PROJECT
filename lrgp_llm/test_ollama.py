import ollama

prompt = "Explique en 2 phrases le mécanisme solution-diffusion."

# Test 1 : SANS frequency_penalty (devrait marcher)
print("─── Test 1 : sans frequency_penalty ───")
r1 = ollama.chat(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0.3, "num_predict": 500},
    think=False,
)
print(f"Longueur : {len(r1.message.content)}")
print(r1.message.content[:200])

# Test 2 : AVEC frequency_penalty (devrait être vide)
print("\n─── Test 2 : avec frequency_penalty=1.15 ───")
r2 = ollama.chat(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": prompt}],
    options={
        "temperature": 0.3,
        "num_predict": 500,
        "frequency_penalty": 1.15,   # le coupable suspecté
    },
    think=False,
)
print(f"Longueur : {len(r2.message.content)}")
print(repr(r2.message.content[:200]))