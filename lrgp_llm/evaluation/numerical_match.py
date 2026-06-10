# numerical_match.py
import json, re
from pathlib import Path

def extraire_nombres(texte: str) -> list[float]:
    """Extrait tous les nombres d'un texte."""
    patterns = [
        r'\d+[.,]\d+(?:[eE][+-]?\d+)?',  # décimaux + notation scientifique
        r'\d+(?:[eE][+-]?\d+)?',           # entiers
    ]
    nombres = []
    for p in patterns:
        for m in re.finditer(p, texte):
            try:
                n = float(m.group().replace(',', '.'))
                if 1e-15 < abs(n) < 1e15:  # filtrer les valeurs absurdes
                    nombres.append(n)
            except:
                pass
    return list(set(nombres))

def match_numerique(ref: str, pred: str, tolerance: float = 0.05) -> dict:
    """
    Compare les nombres de ref et pred.
    Retourne le taux de correspondance avec tolérance 5%.
    """
    nums_ref  = extraire_nombres(ref)
    nums_pred = extraire_nombres(pred)

    if not nums_ref:
        return {"score": None, "raison": "pas_de_nombres_ref"}

    matches = 0
    for n_ref in nums_ref:
        for n_pred in nums_pred:
            if abs(n_ref) > 1e-10:
                erreur_rel = abs(n_ref - n_pred) / abs(n_ref)
                if erreur_rel <= tolerance:
                    matches += 1
                    break

    score = matches / len(nums_ref)
    return {
        "score":      round(score, 3),
        "n_ref":      len(nums_ref),
        "n_pred":     len(nums_pred),
        "matches":    matches,
        "nums_ref":   nums_ref[:5],
        "nums_pred":  nums_pred[:5],
    }

RESULTS_DIR = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\evaluation\\results")
FICHIERS = {
    "Baseline RAG":  "baseline_responses.jsonl",
    "V4 sans RAG":   "v4_no_rag_responses.jsonl",
    "V4 avec RAG":   "v4_rag_responses.jsonl",
    "SRAR-GP":       "srar_gp_responses.jsonl",  
}

print(f"\n{'═'*65}")
print(f"  NUMERICAL MATCH — Questions CALCUL")
print(f"{'═'*65}")

for label, fichier in FICHIERS.items():
    path = RESULTS_DIR / fichier
    if not path.exists():
        continue
    with open(path, encoding="utf-8") as f:
        resultats = [json.loads(l) for l in f if l.strip()]

    calculs = [r for r in resultats if r.get("type") == "CALCUL"]
    scores  = []
    for r in calculs:
        m = match_numerique(r.get("reference",""), r.get("reponse_modele",""))
        if m["score"] is not None:
            scores.append(m["score"])

    moy = sum(scores)/len(scores) if scores else 0
    print(f"\n  {label}")
    print(f"  Questions CALCUL : {len(calculs)}")
    print(f"  Numerical Match  : {moy*100:.1f}%")
    print(f"  Détail scores    : {[round(s,2) for s in scores]}")