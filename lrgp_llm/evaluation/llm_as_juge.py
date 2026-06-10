"""
llm_judge.py
Évaluation automatique des 3 modèles via Gemini 3.1 Pro Preview.
Utilise le Batch API — 50% moins cher, pas de 503.

Usage :
    python evaluation/llm_judge.py --prepare
    python evaluation/llm_judge.py --submit
    python evaluation/llm_judge.py --status
    python evaluation/llm_judge.py --collect
    python evaluation/llm_judge.py --test   (synchrone, 5 questions)
"""

import argparse
import json
import os
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

RESULTS_DIR  = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\evaluation\\results")
BATCH_DIR    = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\evaluation\\results\\batch_judge")
BATCH_DIR.mkdir(parents=True, exist_ok=True)

REQUESTS_FILE = BATCH_DIR / "judge_requests.jsonl"
SOURCES_FILE  = BATCH_DIR / "judge_sources.jsonl"
JOB_STATE     = BATCH_DIR / "judge_job_state.json"
OUTPUT_FILE   = RESULTS_DIR / "llm_judge_scores.jsonl"

MODEL = "gemini-3.1-pro-preview"

FICHIERS = {
    "baseline_rag": "baseline_responses.jsonl",
    "v4_no_rag":    "v4_no_rag_responses.jsonl",
    "v4_rag":       "v4_rag_responses.jsonl",
}

PROMPT_JUDGE = """Tu es un expert en génie des procédés et séparation membranaire au LRGP Nancy.
Évalue cette réponse à une question technique de génie des procédés.

Note chaque critère de 1 à 5 :
- exactitude  : les valeurs numériques et faits scientifiques sont-ils corrects ?
- rigueur     : la démarche de calcul est-elle correcte et complète ?
- physique    : les unités et ordres de grandeur sont-ils cohérents ?
- clarte      : la réponse est-elle claire, structurée, compréhensible ?
- sources     : les sources sont-elles citées correctement ?

Grille :
  5 = parfait   4 = bon   3 = acceptable   2 = insuffisant   1 = incorrect/absent

Question : {question}

Réponse de référence (corrigé expert) : {reference}

Réponse à évaluer : {reponse}

Réponds UNIQUEMENT avec un JSON valide sans texte avant ni après :
{{"exactitude": X, "rigueur": X, "physique": X, "clarte": X, "sources": X, "score_global": X.X, "commentaire": "..."}}

score_global = moyenne des 5 critères (1 décimale)."""


# ══════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════
def charger_donnees() -> tuple[dict, list]:
    """Charge les résultats des 3 modèles."""
    donnees = {}
    for label, fichier in FICHIERS.items():
        path = RESULTS_DIR / fichier
        if not path.exists():
            print(f"  ⚠ {fichier} introuvable")
            continue
        with open(path, encoding="utf-8") as f:
            resultats = [json.loads(l) for l in f if l.strip()]
        donnees[label] = {r["id"]: r for r in resultats}
        print(f"  {label:<20} : {len(resultats)} réponses")

    tous_ids = list(list(donnees.values())[0].keys())
    return donnees, tous_ids


def parser_scores(contenu: str) -> dict | None:
    """Parse la réponse JSON du juge."""
    try:
        contenu = contenu.strip()
        if "```" in contenu:
            for p in contenu.split("```"):
                p = p.strip()
                if p.startswith("json"):
                    p = p[4:]
                if p.strip().startswith("{"):
                    contenu = p.strip()
                    break
        scores = json.loads(contenu)
        criteres = ["exactitude","rigueur","physique","clarte","sources"]
        if "score_global" not in scores:
            vals = [scores.get(c,3) for c in criteres]
            scores["score_global"] = round(sum(vals)/len(vals),1)
        return scores
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — PRÉPARER
# ══════════════════════════════════════════════════════════════════
def preparer():
    print(f"\n{'═'*60}")
    print(f"  ÉTAPE 1 — Préparation requêtes LLM-judge")
    print(f"{'═'*60}")

    donnees, tous_ids = charger_donnees()

    requetes      = []
    sources_index = []  # liste ordonnée pour liaison par index
    idx = 0

    for qid in tous_ids:
        ref_data  = list(donnees.values())[0].get(qid, {})
        question  = ref_data.get("question","")
        reference = ref_data.get("reference","")
        qtype     = ref_data.get("type","?")

        for label, resultats_dict in donnees.items():
            r       = resultats_dict.get(qid, {})
            reponse = r.get("reponse_modele","")

            # Skip les réponses vides
            if len(reponse) < 30:
                continue

            prompt = PROMPT_JUDGE.format(
                question=question[:500],
                reference=reference[:500],
                reponse=reponse[:1500],
            )

            # ✓ Bonne pratique — pas de custom_id dans la requête
            requetes.append({
                "contents": [{"parts": [{"text": prompt}], "role": "user"}],
                "config": {
                    "temperature":     0.0,
                    "max_output_tokens": 300
                },
            })

            # ✓ Bonne pratique — sources ordonnées séparément
            sources_index.append({
                "index":  idx,
                "id":     qid,
                "modele": label,
                "type":   qtype,
            })
            idx += 1

    # Sauvegarder
    with open(REQUESTS_FILE, "w", encoding="utf-8") as f:
        for r in requetes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(SOURCES_FILE, "w", encoding="utf-8") as f:
        for s in sources_index:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    cout = len(requetes) * 500 * 1.0/1e6 + len(requetes) * 300 * 6.0/1e6
    print(f"\n  Requêtes préparées  : {len(requetes)}")
    print(f"  (41 questions × {len(donnees)} modèles = ~{41*len(donnees)} évaluations)")
    print(f"  Coût estimé batch   : ~${cout:.2f}")
    print(f"\n  Lance : --submit")


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — SOUMETTRE
# ══════════════════════════════════════════════════════════════════
def soumettre():
    print(f"\n{'═'*60}")
    print(f"  ÉTAPE 2 — Soumission batch LLM-judge")
    print(f"{'═'*60}")

    with open(REQUESTS_FILE, encoding="utf-8") as f:
        requetes = [json.loads(l) for l in f if l.strip()]

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        job = client.batches.create(
            model=MODEL,
            src=requetes,
            config={"display_name": f"lrgp-judge-{int(time.time())}"},
        )
        state = {
            "job_name":     job.name,
            "submitted_at": time.time(),
            "n_requests":   len(requetes),
        }
        with open(JOB_STATE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        print(f"  ✓ Batch soumis : {job.name}")
        print(f"  Statut         : {job.state}")
        print(f"  Lance --status pour suivre")

    except Exception as e:
        print(f"❌ {e}")


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — STATUT
# ══════════════════════════════════════════════════════════════════
def statut() -> str:
    if not JOB_STATE.exists():
        print("❌ Aucun job — lance --prepare puis --submit")
        return ""
    with open(JOB_STATE, encoding="utf-8") as f:
        state = json.load(f)
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        job     = client.batches.get(name=state["job_name"])
        elapsed = (time.time() - state["submitted_at"]) / 60
        print(f"\n  Job    : {state['job_name']}")
        print(f"  Statut : {job.state}")
        print(f"  Durée  : {elapsed:.0f} min")
        if hasattr(job,"request_counts") and job.request_counts:
            rc = job.request_counts
            print(f"  Requêtes : total={rc.total} ok={rc.succeeded} échec={rc.failed}")
        return str(job.state)
    except Exception as e:
        print(f"❌ {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — COLLECTER
# ══════════════════════════════════════════════════════════════════
def collecter():
    print(f"\n{'═'*60}")
    print(f"  ÉTAPE 4 — Collecte scores LLM-judge")
    print(f"{'═'*60}")

    st = statut()
    if "SUCCEEDED" not in st:
        print(f"  ⚠ Batch non terminé ({st})")
        return

    with open(JOB_STATE, encoding="utf-8") as f:
        state = json.load(f)

    # ✓ Bonne pratique — sources ordonnées
    sources_index = []
    with open(SOURCES_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sources_index.append(json.loads(line))

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    job    = client.batches.get(name=state["job_name"])

    tous_scores = []
    erreurs     = 0

    # ✓ Bonne pratique — liaison par index avec enumerate
    for i, response in enumerate(job.dest.inlined_responses or []):

        # ✓ Bonne pratique — vérifier si la requête a échoué
        if getattr(response, "response", None) is None:
            print(f"  ⚠ index {i} : requête échouée")
            erreurs += 1
            continue

        src = sources_index[i] if i < len(sources_index) else {}

        try:
            contenu = response.response.candidates[0].content.parts[0].text
            scores  = parser_scores(contenu)

            if scores:
                entree = {
                    "id":           src.get("id","?"),
                    "modele":       src.get("modele","?"),
                    "type":         src.get("type","?"),
                    "score_global": scores["score_global"],
                    "exactitude":   scores.get("exactitude",3),
                    "rigueur":      scores.get("rigueur",3),
                    "physique":     scores.get("physique",3),
                    "clarte":       scores.get("clarte",3),
                    "sources":      scores.get("sources",3),
                    "commentaire":  scores.get("commentaire",""),
                }
                tous_scores.append(entree)
            else:
                erreurs += 1

        except Exception as e:
            print(f"  ⚠ index {i} : {str(e)[:60]}")
            erreurs += 1

    # Sauvegarder
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in tous_scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Rapport
    afficher_rapport(tous_scores)
    print(f"\n  Erreurs  : {erreurs}")
    print(f"  ✓ Scores → {OUTPUT_FILE}")


# ══════════════════════════════════════════════════════════════════
# MODE TEST — synchrone
# ══════════════════════════════════════════════════════════════════
def mode_test():
    print(f"\n{'═'*60}")
    print(f"  MODE TEST — 5 questions (synchrone)")
    print(f"{'═'*60}")

    donnees, tous_ids = charger_donnees()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    for qid in tous_ids[:5]:
        ref_data  = list(donnees.values())[0].get(qid,{})
        question  = ref_data.get("question","")
        reference = ref_data.get("reference","")
        qtype     = ref_data.get("type","?")

        print(f"\n  [{qid}] [{qtype}] {question[:55]}...")

        for label, resultats_dict in donnees.items():
            reponse = resultats_dict.get(qid,{}).get("reponse_modele","")
            if len(reponse) < 30:
                print(f"    {label:<20} → vide")
                continue

            for tentative in range(3):
                try:
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=PROMPT_JUDGE.format(
                            question=question[:500],
                            reference=reference[:500],
                            reponse=reponse[:1500],
                        ),
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=300
                        ),
                    )
                    scores = parser_scores(response.text)
                    if scores:
                        print(f"    {label:<20} → score={scores['score_global']} "
                              f"(E:{scores['exactitude']} R:{scores['rigueur']} "
                              f"P:{scores['physique']} C:{scores['clarte']} "
                              f"S:{scores['sources']})")
                    break
                except Exception as e:
                    if "503" in str(e):
                        print(f"    503 — attente {30*(tentative+1)}s...")
                        time.sleep(30*(tentative+1))
                    else:
                        print(f"    ✗ {str(e)[:60]}")
                        break
            time.sleep(0.5)


# ══════════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════════
def afficher_rapport(tous_scores: list):
    from collections import defaultdict
    by_modele = defaultdict(list)
    for s in tous_scores:
        by_modele[s["modele"]].append(s)

    criteres = ["exactitude","rigueur","physique","clarte","sources","score_global"]

    print(f"\n{'═'*65}")
    print(f"  RAPPORT LLM-JUDGE")
    print(f"{'═'*65}")
    print(f"  {'Modèle':<22} {'Exact':>6} {'Rigeur':>7} {'Phys':>6} "
          f"{'Clarté':>7} {'Src':>5} {'GLOBAL':>7}")
    print(f"  {'─'*22} {'─'*6} {'─'*7} {'─'*6} {'─'*7} {'─'*5} {'─'*7}")

    for label, scores in by_modele.items():
        vals = {}
        for c in criteres:
            vals[c] = round(sum(s[c] for s in scores)/len(scores), 2)
        print(f"  {label:<22} {vals['exactitude']:>6.2f} {vals['rigueur']:>7.2f} "
              f"{vals['physique']:>6.2f} {vals['clarte']:>7.2f} "
              f"{vals['sources']:>5.2f} {vals['score_global']:>7.2f}")

    # Par type de question
    print(f"\n  PAR TYPE DE QUESTION")
    print(f"  {'─'*50}")
    for qtype in ["CALCUL","FACTUEL","COMPARAISON"]:
        print(f"\n  {qtype} :")
        for label, scores in by_modele.items():
            subset = [s for s in scores if s.get("type") == qtype]
            if subset:
                moy = sum(s["score_global"] for s in subset)/len(subset)
                print(f"    {label:<22} : {moy:.2f}/5  (n={len(subset)})")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--submit",  action="store_true")
    parser.add_argument("--status",  action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--test",    action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY manquante dans .env")
        return

    if args.test:       mode_test()
    elif args.prepare:  preparer()
    elif args.submit:   soumettre()
    elif args.status:   statut()
    elif args.collect:  collecter()
    else: parser.print_help()


if __name__ == "__main__":
    main()