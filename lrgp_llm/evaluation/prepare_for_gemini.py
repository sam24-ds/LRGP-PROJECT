"""
prepare_for_gemini.py
Prépare le JSON formaté pour évaluation manuelle par Gemini 3.1 Pro.
Colle le contenu du fichier généré dans Gemini après ton prompt LLM-Judge.
"""

import json
from pathlib import Path

INPUT  = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\evaluation\\results\\srar_gp_responses.jsonl")
OUTPUT = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\evaluation\\results\\srar_gp_for_gemini.json")

# Charger les réponses SRAR-GP
items = []
with open(INPUT, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            items.append({
                "id":              r["id"],
                "type":            r["type"],
                "question":        r["question"],
                "reponse_reference": r["reference"],
                "reponse_modele":  r["reponse_modele"],  # limiter pour le contexte Gemini
            })

# Sauvegarde JSON formaté
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print(f"✓ {len(items)} entrées préparées dans {OUTPUT}")
print(f"\nProchaine étape :")
print(f"  1. Ouvre {OUTPUT}")
print(f"  2. Copie son contenu")
print(f"  3. Colle dans Gemini 3.1 Pro après ton prompt LLM-Judge")
print(f"  4. Sauvegarde la réponse JSON dans evaluation/results/srar_gp_evaluations.json")