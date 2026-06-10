"""
run_benchmark.py
Génère les réponses baseline via la chaîne RAG complète.
Reproduit à l'identique le comportement de chain.py en production.
Usage : python evaluation/run_benchmark.py
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import ollama
from prompts import SYSTEM_LRGP

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chain import LRGPChain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="qwen3.5:9b",
                    help="Modèle Ollama à utiliser")
parser.add_argument("--output", default="baseline_responses.jsonl",
                    help="Fichier de sortie dans evaluation/results/")
parser.add_argument("--no-rag", action="store_true",
                    help="Mode sans RAG — appel direct Ollama")
args = parser.parse_args()



# ── Chemins ───────────────────────────────────────────────────────
TEST_PATH   = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\benchmark\\split\\benchmark_test.jsonl")
RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_OUT = RESULTS_DIR / args.output
HUMAN_FORM   = RESULTS_DIR / "formulaire_notation_v4_with_rag.json"


def charger_questions() -> list[dict]:
    with open(TEST_PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def generer_formulaire(resultats: list[dict]) -> None:
    fiches = []
    for r in resultats:
        fiches.append({
            "id":             r["id"],
            "question":       r["question"],
            "type":           r.get("type", "?"),
            "domaine":        r.get("domaine", ""),
            "difficulté":     r.get("difficulté", "?"),
            "reponse_modele": r["reponse_modele"],
            "sources":        r.get("sources", []),
            "reference":      r.get("reference", ""),
            "notation_annotateur_1": {
                "exactitude_factuelle": None,
                "rigueur_demarche":     None,
                "pertinence_physique":  None,
                "clarte_pedagogique":   None,
                "citation_sources":     None,
                "commentaire":          "",
            },
            "notation_annotateur_2": {
                "exactitude_factuelle": None,
                "rigueur_demarche":     None,
                "pertinence_physique":  None,
                "clarte_pedagogique":   None,
                "citation_sources":     None,
                "commentaire":          "",
            },
        })
    with open(HUMAN_FORM, "w", encoding="utf-8") as f:
        json.dump(fiches, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Formulaire → {HUMAN_FORM} ({len(fiches)} fiches)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="qwen3.5:9b")
    parser.add_argument("--output", default="baseline_responses.jsonl")
    parser.add_argument("--no-rag", action="store_true")
    args = parser.parse_args()

    # ← BASELINE_OUT défini ici — avant tout if/else
    BASELINE_OUT = RESULTS_DIR / args.output

    print(f"\n{'═'*60}")
    print(f"  GÉNÉRATION RÉPONSES")
    print(f"  Modèle : {args.model}")
    print(f"  Mode   : {'sans RAG' if args.no_rag else 'avec RAG'}")
    print(f"  Output : {BASELINE_OUT}")
    print(f"{'═'*60}")

    questions = charger_questions()
    print(f"  Questions : {len(questions)}")
    resultats = []  # ← défini ici aussi

    if args.no_rag:
        for i, q in enumerate(questions, 1):
            print(f"  [{i:2d}/{len(questions)}] {q['question'][:60]}...")
            response = ollama.chat(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_LRGP},
                    {"role": "user",   "content": f"Question : {q['question']}"},
                ],
                options={"temperature": 0.1, "think": False, "num_predict": 2048},
            )
            resultats.append({
                "id":             q["id"],
                "question":       q["question"],
                "reference":      q["answer"],
                "type":           q["type"],
                "domaine":        q.get("domaine",""),
                "difficulté":     q["difficulté"],
                "reponse_modele": response.message.content,
                "modele":         args.model + "_no_rag",
                "timestamp":      datetime.now().isoformat(),
            })
            print(f"           ✓ {len(response.message.content)} chars")

    else:
        print(f"\n  Initialisation LRGPChain...", end=" ", flush=True)
        chain = LRGPChain(
            llm_backend    = "ollama",
            model_name     = args.model,
            ollama_url     = "http://localhost:11434",
            top_k_retrieve = 20,
            top_k_rerank   = 5,
            temperature    = 0.1,
            verbose        = True,
        )
        print("✓")

        t_debut = time.time()
        for i, q in enumerate(questions, 1):
            print(f"\n  [{i:2d}/{len(questions)}] {q['question'][:60]}...")
            print(f"           Type: {q['type']} | Niveau: {q['difficulté']}")
            t0 = time.time()
            try:
                response = chain.ask(q["question"])
                duree    = time.time() - t0
                resultats.append({
                    "id":                   q["id"],
                    "question":             q["question"],
                    "reference":            q["answer"],
                    "type":                 q["type"],
                    "domaine":              q.get("domaine",""),
                    "difficulté":           q["difficulté"],
                    "reponse_modele":       response.answer,
                    "question_type_classe": response.question_type,
                    "sources":              [s.source for s in response.sources],
                    "n_sources":            len(response.sources),
                    "context_chars":        response.context_chars,
                    "duree_s":              round(duree, 1),
                    "modele":               args.model + "_rag",
                    "timestamp":            datetime.now().isoformat(),
                })
                print(f"           ✓ {duree:.1f}s | {len(response.answer)} chars")
            except Exception as e:
                resultats.append({
                    "id":             q["id"],
                    "question":       q["question"],
                    "reference":      q["answer"],
                    "reponse_modele": f"ERREUR: {str(e)}",
                    "modele":         args.model + "_rag",
                    "erreur":         True,
                })
                print(f"           ✗ {str(e)[:60]}")

        duree_totale = time.time() - t_debut
        n_ok = sum(1 for r in resultats if not r.get("erreur"))
        print(f"\n  {n_ok}/{len(resultats)} réponses — {duree_totale/60:.1f} min")

    # ← Sauvegarde ICI — après les deux blocs
    with open(BASELINE_OUT, "w", encoding="utf-8") as f:
        for r in resultats:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    generer_formulaire(resultats)
    print(f"\n✓ {len(resultats)} réponses → {BASELINE_OUT}")
    print(f"✓ Formulaire → {HUMAN_FORM}")




if __name__ == "__main__":
    main()