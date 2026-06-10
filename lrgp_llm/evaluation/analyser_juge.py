"""
analyser_judge.py
Analyse les scores LLM-judge retournés par Gemini.
Usage : python evaluation/analyser_judge.py
"""
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\evaluation\\results")
FICHIERS = {
    "Baseline RAG":  "judge_baseline.json",
    "V4 sans RAG":   "judge_v4_no_rag.json",
    "V4 avec RAG":   "judge_v4_rag.json",
}

CRITERES = ["exactitude","rigueur","physique","clarte","sources","score_global"]

print(f"\n{'═'*65}")
print(f"  RÉSULTATS LLM-JUDGE — GEMINI 3.1 PRO")
print(f"{'═'*65}")
print(f"  {'Modèle':<22} {'Exact':>7} {'Rigueur':>8} {'Phys':>6} "
      f"{'Clarté':>7} {'Src':>5} {'GLOBAL':>7} {'N':>4}")
print(f"  {'─'*22} {'─'*7} {'─'*8} {'─'*6} {'─'*7} {'─'*5} {'─'*7} {'─'*4}")

tous_resultats = {}
for label, fichier in FICHIERS.items():
    path = RESULTS_DIR / fichier
    if not path.exists():
        print(f"  {label:<22} — fichier introuvable")
        continue

    with open(path, encoding="utf-8") as f:
        scores = json.load(f)

    tous_resultats[label] = scores

    # Moyennes globales
    moyennes = {}
    for c in CRITERES:
        vals = [s[c] for s in scores if isinstance(s.get(c), (int,float))]
        moyennes[c] = round(sum(vals)/len(vals), 2) if vals else 0

    n = len(scores)
    print(f"  {label:<22} {moyennes['exactitude']:>7.2f} "
          f"{moyennes['rigueur']:>8.2f} {moyennes['physique']:>6.2f} "
          f"{moyennes['clarte']:>7.2f} {moyennes['sources']:>5.2f} "
          f"{moyennes['score_global']:>7.2f} {n:>4}")

# Par type de question
print(f"\n{'─'*65}")
print(f"  SCORE GLOBAL PAR TYPE")
print(f"{'─'*65}")
print(f"  {'Type':<15} {'Baseline':>10} {'V4 no RAG':>11} {'V4 RAG':>8}")
print(f"  {'─'*15} {'─'*10} {'─'*11} {'─'*8}")

for qtype in ["CALCUL","FACTUEL","COMPARAISON"]:
    ligne = f"  {qtype:<15}"
    for label in ["Baseline RAG","V4 sans RAG","V4 avec RAG"]:
        if label not in tous_resultats:
            ligne += f" {'—':>10}"
            continue
        subset = [s for s in tous_resultats[label] if s.get("type")==qtype]
        if subset:
            moy = sum(s["score_global"] for s in subset)/len(subset)
            ligne += f" {moy:>10.2f}"
        else:
            ligne += f" {'—':>10}"
    print(ligne)

# Meilleur et pire par modèle
print(f"\n{'─'*65}")
print(f"  TOP 3 ET PIRE 3 PAR MODÈLE")
print(f"{'─'*65}")
for label, scores in tous_resultats.items():
    tries = sorted(scores, key=lambda x: x.get("score_global",0), reverse=True)
    print(f"\n  {label} :")
    print(f"  Meilleurs :")
    for s in tries[:3]:
        print(f"    [{s['id']}] [{s['type']}] score={s['score_global']} — {s.get('commentaire','')[:60]}")
    print(f"  Pires :")
    for s in tries[-3:]:
        print(f"    [{s['id']}] [{s['type']}] score={s['score_global']} — {s.get('commentaire','')[:60]}")

print(f"\n{'═'*65}\n")