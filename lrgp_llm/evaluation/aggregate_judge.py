"""
aggregate_final.py
Agrégation finale : 4 systèmes évalués sur la grille enrichie de 6 critères.
Gère les deux scores (standard et enrichi) présents dans les JSON.
"""

import json
from pathlib import Path
from statistics import mean

RESULTS_DIR = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\evaluation\\results")

FICHIERS = {
    "Baseline RAG":  "judge_baseline.json",
    "V4 sans RAG":   "judge_v4_no_rag.json",
    "V4 avec RAG":   "judge_v4_rag.json",
    "SRAR-GP":       "judge_plus_srar_gp.json",
}


def charger(fichier: str) -> list:
    """Charge un fichier JSON d'évaluations, tolérant aux blocs markdown."""
    path = RESULTS_DIR / fichier
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        contenu = f.read().strip()
        # Tolérance pour les blocs markdown ```json ... ```
        if contenu.startswith("```"):
            parts = contenu.split("```")
            contenu = parts[1] if len(parts) > 1 else parts[0]
            if contenu.startswith("json"):
                contenu = contenu[4:]
        return json.loads(contenu)


def stats(evaluations: list) -> dict:
    """Calcule les statistiques pour un système."""
    if not evaluations:
        return None
    
    criteres = [
        "exactitude", "rigueur", "physique", 
        "clarte", "sources", "fiabilite_epistemique"
    ]
    par_critere = {c: [] for c in criteres}
    standard = []
    enrichi  = []
    par_type_standard = {"CALCUL": [], "FACTUEL": [], "COMPARAISON": []}
    par_type_enrichi  = {"CALCUL": [], "FACTUEL": [], "COMPARAISON": []}
    
    for e in evaluations:
        for c in criteres:
            v = e.get(c)
            if isinstance(v, (int, float)):
                par_critere[c].append(v)
        
        ss = e.get("score_global_standard") or e.get("score_global")
        se = e.get("score_global_enrichi")  or e.get("score_global")
        
        if isinstance(ss, (int, float)):
            standard.append(ss)
            t = e.get("type", "")
            if t in par_type_standard:
                par_type_standard[t].append(ss)
        
        if isinstance(se, (int, float)):
            enrichi.append(se)
            t = e.get("type", "")
            if t in par_type_enrichi:
                par_type_enrichi[t].append(se)
    
    return {
        "n":                len(evaluations),
        "score_standard":   round(mean(standard), 2) if standard else None,
        "score_enrichi":    round(mean(enrichi),  2) if enrichi  else None,
        "par_critere":      {c: round(mean(v), 2) if v else None for c, v in par_critere.items()},
        "par_type_std":     {t: (round(mean(v), 2), len(v)) if v else (None, 0) for t, v in par_type_standard.items()},
        "par_type_enr":     {t: (round(mean(v), 2), len(v)) if v else (None, 0) for t, v in par_type_enrichi.items()},
    }


print(f"\n[Aggregate Final] Chargement des évaluations...\n")

stats_tous = {}
for syst, fichier in FICHIERS.items():
    evals = charger(fichier)
    if evals:
        print(f"  ✓ {syst:<15} : {len(evals)} évaluations")
        stats_tous[syst] = stats(evals)
    else:
        print(f"  ✗ {syst:<15} : fichier introuvable ({fichier})")
        stats_tous[syst] = None


# ══════════════════════════════════════════════════════════════
# TABLEAU 1 — Comparaison STANDARD (grille V1-V4, 5 critères)
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*95}")
print(f"  TABLEAU 1 — SCORE STANDARD (5 critères, comparable historiquement)")
print(f"{'═'*95}\n")

print(f"  {'Système':<15} | {'Score':<6} | {'Exact':<6} | {'Rig.':<5} | {'Phys.':<6} | {'Clar.':<6} | {'Sour.':<6}")
print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

for syst, st in stats_tous.items():
    if st is None:
        print(f"  {syst:<15} | -- pas de données --")
        continue
    pc = st["par_critere"]
    print(f"  {syst:<15} | "
          f"{str(st['score_standard'] or '-'):<6} | "
          f"{str(pc['exactitude'] or '-'):<6} | "
          f"{str(pc['rigueur'] or '-'):<5} | "
          f"{str(pc['physique'] or '-'):<6} | "
          f"{str(pc['clarte'] or '-'):<6} | "
          f"{str(pc['sources'] or '-'):<6}")


# ══════════════════════════════════════════════════════════════
# TABLEAU 2 — Comparaison ENRICHIE (6 critères, fiabilité)
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*95}")
print(f"  TABLEAU 2 — SCORE ENRICHI (6 critères, + fiabilité épistémique)")
print(f"{'═'*95}\n")

print(f"  {'Système':<15} | {'Score':<6} | {'Standard':<9} | {'Δ (gain)':<9} | {'Fiab. Épist.':<14}")
print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*14}")

for syst, st in stats_tous.items():
    if st is None:
        continue
    ss = st["score_standard"]
    se = st["score_enrichi"]
    delta = round(se - ss, 2) if (ss is not None and se is not None) else None
    fe = st["par_critere"].get("fiabilite_epistemique")
    
    delta_str = f"+{delta}" if delta is not None and delta >= 0 else str(delta)
    
    print(f"  {syst:<15} | "
          f"{str(se or '-'):<6} | "
          f"{str(ss or '-'):<9} | "
          f"{delta_str:<9} | "
          f"{str(fe or '-'):<14}")


# ══════════════════════════════════════════════════════════════
# TABLEAU 3 — Fiabilité épistémique (le nouveau critère)
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*95}")
print(f"  TABLEAU 3 — FIABILITÉ ÉPISTÉMIQUE (visualisation)")
print(f"{'═'*95}\n")

print(f"  {'Système':<15} | Score | Visualisation")
print(f"  {'-'*15}-+-{'-'*5}-+-{'-'*40}")

for syst, st in stats_tous.items():
    if st is None:
        continue
    fe = st["par_critere"].get("fiabilite_epistemique")
    if fe is None:
        continue
    bar = "█" * int(fe * 5) + "░" * (25 - int(fe * 5))
    print(f"  {syst:<15} | {fe:<5} | {bar} ({fe}/5)")


# ══════════════════════════════════════════════════════════════
# TABLEAU 4 — Performance par type de question (score enrichi)
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*95}")
print(f"  TABLEAU 4 — SCORE ENRICHI PAR TYPE DE QUESTION")
print(f"{'═'*95}\n")

print(f"  {'Système':<15} | {'CALCUL':<14} | {'FACTUEL':<14} | {'COMPARAISON':<14}")
print(f"  {'-'*15}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")

for syst, st in stats_tous.items():
    if st is None:
        continue
    pt = st["par_type_enr"]
    
    calcul   = f"{pt['CALCUL'][0]} (n={pt['CALCUL'][1]})"        if pt['CALCUL'][0] else "-"
    factuel  = f"{pt['FACTUEL'][0]} (n={pt['FACTUEL'][1]})"      if pt['FACTUEL'][0] else "-"
    comp     = f"{pt['COMPARAISON'][0]} (n={pt['COMPARAISON'][1]})" if pt['COMPARAISON'][0] else "-"
    
    print(f"  {syst:<15} | {calcul:<14} | {factuel:<14} | {comp:<14}")


# ══════════════════════════════════════════════════════════════
# SYNTHÈSE FINALE POUR LE RAPPORT
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*95}")
print(f"  SYNTHÈSE POUR LE RAPPORT R&D")
print(f"{'═'*95}\n")

print(f"  Métrique                              | Score | Commentaire")
print(f"  --------------------------------------+-------+-----------------------------")

for syst, st in stats_tous.items():
    if st is None:
        continue
    ss = st["score_standard"]
    se = st["score_enrichi"]
    if ss is not None and se is not None:
        gain = round(se - ss, 2)
        print(f"  {syst:<14} (standard)             | {ss}/5  | grille V1-V4 historique")
        print(f"  {syst:<14} (enrichi)              | {se}/5  | + fiabilité épistémique (+{gain})")

print(f"\n  Rappel Numerical Match (CALCUL uniquement) :")
print(f"    Baseline RAG : 29.1%")
print(f"    V4 sans RAG  : 75.7%")
print(f"    V4 avec RAG  : 72.0%")
print(f"    SRAR-GP      : 67.9%  (avec biais méthodologique documenté)")

print(f"\n{'═'*95}\n")