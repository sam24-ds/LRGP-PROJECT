"""
compare_models.py
Analyse comparative des 3 runs d'évaluation.
Usage : python evaluation/compare_models.py
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

RESULTS_DIR = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\evaluation\\results")

FICHIERS = {
    "Baseline RAG (qwen3.5:9b)": "baseline_responses.jsonl",
    "V4 sans RAG":                "v4_no_rag_responses.jsonl",
    "V4 avec RAG":                "v4_rag_responses.jsonl",
}

def charger(nom_fichier):
    path = RESULTS_DIR / nom_fichier
    if not path.exists():
        print(f"  ⚠ Introuvable : {path}")
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def stats_reponses(resultats):
    longueurs  = [len(r.get("reponse_modele","")) for r in resultats]
    erreurs    = sum(1 for r in resultats if r.get("erreur"))
    vides      = sum(1 for r in resultats if len(r.get("reponse_modele","")) < 50)
    types      = Counter(r.get("type","?") for r in resultats)
    return {
        "n_total":   len(resultats),
        "n_erreurs": erreurs,
        "n_vides":   vides,
        "moy_chars": round(sum(longueurs)/max(len(longueurs),1)),
        "min_chars": min(longueurs) if longueurs else 0,
        "max_chars": max(longueurs) if longueurs else 0,
        "types":     dict(types),
    }

# ── Chargement ────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  COMPARAISON MODÈLES — BENCHMARK LRGP 41 questions")
print(f"{'═'*65}")

donnees = {}
for label, fichier in FICHIERS.items():
    resultats = charger(fichier)
    donnees[label] = resultats
    print(f"  {label:<35} : {len(resultats)} réponses")

# ── Stats globales ────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  STATISTIQUES GLOBALES")
print(f"{'─'*65}")
print(f"  {'Modèle':<35} {'Total':>6} {'Erreurs':>8} {'Vides':>6} {'Moy.chars':>10}")
print(f"  {'─'*35} {'─'*6} {'─'*8} {'─'*6} {'─'*10}")

for label, resultats in donnees.items():
    s = stats_reponses(resultats)
    print(f"  {label:<35} {s['n_total']:>6} {s['n_erreurs']:>8} "
          f"{s['n_vides']:>6} {s['moy_chars']:>10}")

# ── Comparaison par type ──────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  LONGUEUR MOYENNE PAR TYPE DE QUESTION")
print(f"{'─'*65}")
print(f"  {'Type':<15} {'Baseline RAG':>15} {'V4 sans RAG':>13} {'V4 avec RAG':>13}")
print(f"  {'─'*15} {'─'*15} {'─'*13} {'─'*13}")

for qtype in ["CALCUL", "FACTUEL", "COMPARAISON"]:
    moyennes = []
    for label, resultats in donnees.items():
        subset = [r for r in resultats if r.get("type") == qtype]
        if subset:
            moy = sum(len(r.get("reponse_modele","")) for r in subset) // len(subset)
            moyennes.append(f"{moy:>13}")
        else:
            moyennes.append(f"{'—':>13}")
    print(f"  {qtype:<15} {'  '.join(moyennes)}")

# ── Comparaison question par question ─────────────────────────────
print(f"\n{'─'*65}")
print(f"  COMPARAISON PAR QUESTION (extrait)")
print(f"{'─'*65}")

# Construire un index par ID
index = defaultdict(dict)
for label, resultats in donnees.items():
    for r in resultats:
        index[r["id"]][label] = r

# Afficher les questions CALCUL
calcul_ids = [
    qid for qid, data in index.items()
    if any(r.get("type") == "CALCUL" for r in data.values())
][:8]

for qid in calcul_ids:
    data = index[qid]
    first = list(data.values())[0]
    print(f"\n  [{qid}] [{first.get('type','?')}] {first.get('question','')[:65]}...")
    for label, r in data.items():
        reponse = r.get("reponse_modele","")
        n_chars = len(reponse)
        apercu  = reponse[:80].replace("\n"," ")
        erreur  = " ⚠ ERREUR" if r.get("erreur") else ""
        print(f"    {label:<35} {n_chars:>6} chars | {apercu}...{erreur}")

# ── Sauvegarder le rapport ────────────────────────────────────────
rapport = []
for qid, data in index.items():
    ligne = {"id": qid}
    first = list(data.values())[0]
    ligne["question"]   = first.get("question","")
    ligne["type"]       = first.get("type","")
    ligne["difficulté"] = first.get("difficulté","")
    ligne["reference"]  = first.get("reference","")
    for label, r in data.items():
        cle = label.lower().replace(" ","_").replace("(","").replace(")","").replace(".","")
        ligne[f"reponse_{cle}"]  = r.get("reponse_modele","")
        ligne[f"chars_{cle}"]    = len(r.get("reponse_modele",""))
        ligne[f"erreur_{cle}"]   = r.get("erreur", False)

    rapport.append(ligne)

rapport_path = RESULTS_DIR / "comparaison_modeles.jsonl"
with open(rapport_path, "w", encoding="utf-8") as f:
    for l in rapport:
        f.write(json.dumps(l, ensure_ascii=False) + "\n")

print(f"\n  ✓ Rapport comparatif → {rapport_path}")
print(f"{'═'*65}\n")