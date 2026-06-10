# temps_reponse.py
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\evaluation\\results")
FICHIERS = {
    "Baseline RAG":  "baseline_responses.jsonl",
    "V4 sans RAG":   "v4_no_rag_responses.jsonl",
    "V4 avec RAG":   "v4_rag_responses.jsonl",
}

print(f"\n{'═'*60}")
print(f"  TEMPS DE RÉPONSE — 41 questions")
print(f"{'═'*60}")

for label, fichier in FICHIERS.items():
    path = RESULTS_DIR / fichier
    if not path.exists():
        print(f"  ⚠ {fichier} introuvable")
        continue

    with open(path, encoding="utf-8") as f:
        resultats = [json.loads(l) for l in f if l.strip()]

    # Extraire les durées
    durees = [r.get("duree_s", 0) for r in resultats if r.get("duree_s")]

    if not durees:
        print(f"\n  {label} — pas de données de temps")
        continue

    # Stats globales
    moy   = sum(durees) / len(durees)
    total = sum(durees)
    mini  = min(durees)
    maxi  = max(durees)
    durees_sorted = sorted(durees)
    p50   = durees_sorted[len(durees_sorted)//2]
    p90   = durees_sorted[int(len(durees_sorted)*0.9)]

    print(f"\n  {label}")
    print(f"  {'─'*45}")
    print(f"  Total         : {total/60:.1f} min")
    print(f"  Moyenne       : {moy:.1f}s / question")
    print(f"  Médiane (P50) : {p50:.1f}s")
    print(f"  P90           : {p90:.1f}s")
    print(f"  Min           : {mini:.1f}s")
    print(f"  Max           : {maxi:.1f}s")

    # Par type de question
    by_type = defaultdict(list)
    for r in resultats:
        if r.get("duree_s"):
            by_type[r.get("type","?")].append(r["duree_s"])

    print(f"\n  Par type :")
    for qtype in ["CALCUL", "FACTUEL", "COMPARAISON"]:
        if qtype in by_type:
            vals = by_type[qtype]
            moy_t = sum(vals)/len(vals)
            print(f"    {qtype:<15} : {moy_t:.1f}s  (n={len(vals)})")

# ── Tableau comparatif final ──────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  TABLEAU COMPARATIF")
print(f"{'─'*60}")
print(f"  {'Modèle':<35} {'Moy':>6} {'Total':>8} {'P90':>6}")
print(f"  {'─'*35} {'─'*6} {'─'*8} {'─'*6}")

for label, fichier in FICHIERS.items():
    path = RESULTS_DIR / fichier
    if not path.exists():
        continue
    with open(path, encoding="utf-8") as f:
        resultats = [json.loads(l) for l in f if l.strip()]
    durees = [r.get("duree_s",0) for r in resultats if r.get("duree_s")]
    if not durees:
        print(f"  {label:<35} {'—':>6} {'—':>8} {'—':>6}")
        continue
    moy   = sum(durees)/len(durees)
    total = sum(durees)/60
    p90   = sorted(durees)[int(len(durees)*0.9)]
    print(f"  {label:<35} {moy:>5.1f}s {total:>6.1f}min {p90:>5.1f}s")

print(f"{'═'*60}\n")