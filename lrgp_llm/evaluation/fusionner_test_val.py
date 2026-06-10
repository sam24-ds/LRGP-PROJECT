"""
fusionner_test_val.py
Fusionne benchmark_val dans benchmark_test.
"""
import json
from pathlib import Path

SPLIT_DIR = Path(__file__).parent.parent / "data/datasets/benchmark/split"

def charger(nom):
    p = SPLIT_DIR / f"benchmark_{nom}.jsonl"
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def sauvegarder(nom, questions):
    p = SPLIT_DIR / f"benchmark_{nom}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"✓ benchmark_{nom}.jsonl — {len(questions)} questions")

# Charger
test = charger("test")
val  = charger("val")

print(f"benchmark_test actuel : {len(test)} questions")
print(f"benchmark_val actuel  : {len(val)} questions")

# Fusionner
nouveau_test = test + val

# Sauvegarder
sauvegarder("test", nouveau_test)

# Vider val — plus utilisé
sauvegarder("val", [])

from collections import Counter
types = Counter(q["type"] for q in nouveau_test)
print(f"\nNouveau benchmark_test : {len(nouveau_test)} questions")
print(f"  CALCUL      : {types.get('CALCUL', 0)}")
print(f"  FACTUEL     : {types.get('FACTUEL', 0)}")
print(f"  COMPARAISON : {types.get('COMPARAISON', 0)}")