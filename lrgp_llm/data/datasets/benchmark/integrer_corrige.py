"""
integrer_corrige.py
Intègre les questions corrigées dans le benchmark.
"""
import json
from pathlib import Path
from collections import Counter

BENCHMARK_PATH = Path("questions.jsonl")
SOURCE_PATH    = Path("reponse_llm_calcul.json")

# Charger
with open(SOURCE_PATH, encoding="utf-8") as f:
    nouvelles = json.load(f)

with open(BENCHMARK_PATH, encoding="utf-8") as f:
    existantes = [json.loads(l) for l in f if l.strip()]

print(f"Benchmark actuel  : {len(existantes)} questions")
print(f"Nouvelles         : {len(nouvelles)} questions")

# Assigner IDs et source
for i, q in enumerate(nouvelles, len(existantes) + 1):
    q["id"]     = f"Q{i:03d}"
    q["source"] = q.get("source", "corpus_lrgp")

existantes.extend(nouvelles)

with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
    for q in existantes:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

# Stats
types    = Counter(q["type"] for q in existantes)
domaines = Counter(q["domaine"] for q in existantes)
niveaux  = Counter(q["difficulté"] for q in existantes)

print(f"\n{'═'*55}")
print(f"  BENCHMARK LRGP — {len(existantes)} questions")
print(f"{'═'*55}")
print(f"\n  Par type :")
for t, n in sorted(types.items()):
    print(f"    {t:<15} {n:2d}  {'█'*n}")
print(f"\n  Par domaine :")
for d, n in sorted(domaines.items(), key=lambda x: -x[1]):
    print(f"    {d:<35} {n:2d}")
print(f"\n  Par niveau :")
for nv, n in sorted(niveaux.items()):
    print(f"    {nv}  {n:2d}  {'█'*n}")
print(f"\n  Progression : {len(existantes)}/80 minimum")
print(f"{'═'*55}\n")