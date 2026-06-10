import json
import re
import argparse

input_file = "C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\test.jsonl"
output_file = "C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\output_v4.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Split intelligent basé sur "instruction"
chunks = re.split(r'\n(?=\{"instruction")', content)

valid_objects = []

for chunk in chunks:
    chunk = chunk.strip()

    if not chunk:
        continue

    try:
        obj = json.loads(chunk)
        valid_objects.append(obj)
    except json.JSONDecodeError:
        print("⚠️ Correction tentative...")

        # tentative de réparation simple
        fixed = chunk.replace("\n", "\\n")

        try:
            obj = json.loads(fixed)
            valid_objects.append(obj)
        except:
            print("❌ Toujours invalide")

# Sauvegarde propre
with open(output_file, "w", encoding="utf-8") as f:
    for obj in valid_objects:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ {len(valid_objects)} objets sauvés")