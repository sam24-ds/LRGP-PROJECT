"""
benchmark_srar_gp.py
Génère srar_gp_responses.jsonl en interrogeant l'API SRAR-GP
sur les questions du benchmark_test.jsonl.

Format de sortie IDENTIQUE à baseline_responses.jsonl
pour réutiliser tous les scripts d'évaluation existants.

Usage : python evaluation/benchmark_srar_gp.py
"""

import json
import time
import re
import requests
import sys
from pathlib import Path

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════
API_URL = "http://localhost:8000/v1/chat/completions"

BENCHMARK_FILE = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\benchmark\\split\\benchmark_test.jsonl")
OUTPUT_FILE    = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\evaluation\\results\\srar_gp_responses.jsonl")

# Timeout par question (voie CALCUL peut prendre ~3 min)
TIMEOUT = 360  # 6 minutes max par question

# Pause entre questions (laisser Ollama respirer)
PAUSE_BETWEEN = 3  # secondes


# ════════════════════════════════════════════════════════════════
# APPEL API
# ════════════════════════════════════════════════════════════════
def call_srar_gp(question: str) -> dict:
    """Appelle l'API SRAR-GP en mode verbose et capture toutes les métadonnées."""
    payload = {
        "model": "srar-gp-verbose",
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "temperature": 0.1,
    }
    
    t0 = time.time()
    
    try:
        response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
        latency = time.time() - t0
        
        if response.status_code != 200:
            return _erreur(f"HTTP {response.status_code}", latency)
        
        data = response.json()
        contenu = data["choices"][0]["message"]["content"]
        
        # ── Extraire les métadonnées depuis le mode verbose ──
        voie = "UNKNOWN"
        parcours = []
        
        voie_match = re.search(r"\*\*Voie\*\*\s*:\s*(\w+)", contenu)
        if voie_match:
            voie = voie_match.group(1)
        
        parcours_match = re.search(r"\*\*Agents traversés\*\*\s*:\s*([^\n]+)", contenu)
        if parcours_match:
            parcours = [a.strip() for a in parcours_match.group(1).split("→")]
        
        # Détection des comportements spéciaux
        parcours_str = str(parcours).lower()
        renegociation   = "correction_code" in parcours_str or "correction_physique" in parcours_str
        web_search_used = "web_search" in parcours_str
        missing_data    = "missing_data_handler" in parcours_str
        calcul_failed   = "calculation_expert_failed" in parcours_str
        
        return {
            "response":        contenu,
            "latency":         round(latency, 2),
            "voie":            voie,
            "parcours":        parcours,
            "renegociation":   renegociation,
            "web_search_used": web_search_used,
            "missing_data":    missing_data,
            "calcul_failed":   calcul_failed,
            "status":          "ok",
        }
    
    except requests.Timeout:
        return _erreur(f"TIMEOUT > {TIMEOUT}s", time.time() - t0)
    except Exception as e:
        return _erreur(str(e)[:200], time.time() - t0)


def _erreur(msg: str, latency: float) -> dict:
    return {
        "response":        f"ERROR: {msg}",
        "latency":         round(latency, 2),
        "voie":            "ERROR",
        "parcours":        [],
        "renegociation":   False,
        "web_search_used": False,
        "missing_data":    False,
        "calcul_failed":   True,
        "status":          "error",
    }


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def run_benchmark(resume: bool = False, limit: int = None):
    """Exécute le benchmark complet.
    
    Args:
        resume: Si True, reprend là où on s'était arrêté
        limit: Nombre maximum de questions à traiter
    """
   

    # Charger les questions
    if not BENCHMARK_FILE.exists():
        print(f"✗ Fichier introuvable : {BENCHMARK_FILE}")
        sys.exit(1)
    
    questions = []
    with open(BENCHMARK_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    print(f"\n{'═'*70}")
    print(f"  BENCHMARK SRAR-GP — {len(questions)} questions")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"{'═'*70}\n")

    if limit:
        questions = questions[:limit]
        print(f"  ⚠ Mode pilote : limité aux {limit} premières questions\n")
    
    # Mode resume : skip les questions déjà traitées
    deja_traitees = set()
    if resume and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    deja_traitees.add(json.loads(line).get("id"))
        print(f"  ⚠ Mode reprise : {len(deja_traitees)} questions déjà traitées\n")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Vérifier que l'API tourne
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        print(f"  [API] Health check : {health.json()}\n")
    except Exception as e:
        print(f"  ✗ API LRGP non accessible sur http://localhost:8000")
        print(f"  ✗ Lance d'abord : python rag/api_server.py\n")
        sys.exit(1)
    
    # Boucle principale
    mode = "a" if resume else "w"
    
    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            
            if q["id"] in deja_traitees:
                print(f"[{i}/{len(questions)}] {q['id']} — DÉJÀ FAIT, skip")
                continue
            
            print(f"\n[{i}/{len(questions)}] {q['id']} ({q['type']})")
            print(f"  → {q['question'][:120]}")
            
            result = call_srar_gp(q["question"])
            
            # Affichage résumé
            statut = "✓" if result["status"] == "ok" else "✗"
            print(f"  {statut} Voie : {result['voie']}")
            print(f"  {statut} Latence : {result['latency']}s")
            
            badges = []
            if result["web_search_used"]: badges.append("WEB")
            if result["renegociation"]:   badges.append("RENEGO")
            if result["missing_data"]:    badges.append("MISSING_DATA")
            if result["calcul_failed"]:   badges.append("CALC_FAILED")
            if badges:
                print(f"  {statut} Spéciaux : {' | '.join(badges)}")
            
            # Construction de l'entrée
            entry = {
                "id":              q["id"],
                "type":            q["type"],
                "question":        q["question"],
                "reference":       q.get("answer", ""),
                "reponse_modele":  result["response"],
                "latence_sec":     result["latency"],
                "voie":            result["voie"],
                "parcours":        result["parcours"],
                "renegociation":   result["renegociation"],
                "web_search_used": result["web_search_used"],
                "missing_data":    result["missing_data"],
                "calcul_failed":   result["calcul_failed"],
                "domaine":         q.get("domaine", ""),
                "difficulté":      q.get("difficulté", ""),
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()  # écrire immédiatement (anti-crash)
            
            # Pause pour ne pas saturer Ollama
            if i < len(questions):
                time.sleep(PAUSE_BETWEEN)
    
    # ── Récapitulatif final ──
    print(f"\n{'═'*70}")
    print(f"  ✓ BENCHMARK TERMINÉ")
    print(f"{'═'*70}\n")
    
    # Charger tous les résultats pour stats
    resultats = []
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                resultats.append(json.loads(line))
    
    # Stats globales
    n_total = len(resultats)
    n_ok    = sum(1 for r in resultats if r.get("voie") not in ("ERROR", ""))
    
    par_voie = {}
    for r in resultats:
        v = r.get("voie", "?")
        par_voie[v] = par_voie.get(v, 0) + 1
    
    latence_moy = sum(r["latence_sec"] for r in resultats) / n_total
    
    n_web      = sum(1 for r in resultats if r["web_search_used"])
    n_renego   = sum(1 for r in resultats if r["renegociation"])
    n_missing  = sum(1 for r in resultats if r["missing_data"])
    n_failed   = sum(1 for r in resultats if r.get("calcul_failed"))
    
    print(f"  Total questions    : {n_total}")
    print(f"  Réussites          : {n_ok} / {n_total} ({100*n_ok/n_total:.1f}%)")
    print(f"  Latence moyenne    : {latence_moy:.1f}s")
    print(f"\n  Répartition par voie :")
    for v, n in par_voie.items():
        print(f"    {v:<15} : {n}")
    
    print(f"\n  Activations spéciales :")
    print(f"    Web Search       : {n_web} ({100*n_web/n_total:.0f}%)")
    print(f"    Re-négociation   : {n_renego} ({100*n_renego/n_total:.0f}%)")
    print(f"    Missing Data     : {n_missing} ({100*n_missing/n_total:.0f}%)")
    print(f"    Échecs calcul    : {n_failed} ({100*n_failed/n_total:.0f}%)")
    
    print(f"\n  Résultats : {OUTPUT_FILE}")
    print(f"  Prochaine étape : python evaluation/numerical_match.py")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    # Mode resume si argument --resume
    resume = "--resume" in sys.argv
     # Détecter --limit N
    limit = None
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i+1 < len(sys.argv):
            limit = int(sys.argv[i+1])
    run_benchmark(resume=resume, limit=limit)