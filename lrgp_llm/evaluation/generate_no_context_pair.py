"""
generate_no_context_pairs.py
Dérive des paires no-context depuis train_clean ou eval_clean.
Supprime le bloc contexte RAG dans input — output conservé identique.
Split Before Augment : train → train_v3 | eval → eval_v3

Usage :
    python evaluation/generate_no_context_pairs.py --source train
    python evaluation/generate_no_context_pairs.py --source eval
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

# ── Chemins ───────────────────────────────────────────────────────
TRAIN_CLEAN    = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\train_clean.jsonl")
EVAL_CLEAN     = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\eval_clean.jsonl")
OUT_TRAIN_NC   = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\no_context_train.jsonl")
OUT_EVAL_NC    = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\no_context_eval.jsonl")

SEED           = 42
SCORE_MIN      = 4
OUTPUT_MIN     = 150

SOURCES_REF    = {"examen_2023", "document_reference", "examen_2023_pairs"}

INSTRUCTION_NC = (
    "Tu es un expert en génie des procédés au LRGP Nancy. "
    "Réponds directement depuis tes connaissances scientifiques. "
    "Pour les calculs, détaille toutes les étapes avec les unités. "
    "Cite les références si tu les connais."
)

TRACES_CONTEXTE = [
    "d'après le contexte",
    "selon le contexte",
    "le contexte fourni",
    "dans le document fourni",
    "le texte mentionne",
    "d'après le texte",
    "comme indiqué dans le document",
    "selon le document",
    "le document indique",
    "dans le contexte",
    "le texte ci-dessus",
    "d'après les documents",
    "les documents fournis",
    "selon les documents",
]


# ══════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════
def extraire_question(input_text: str) -> str:
    """Supprime le bloc Contexte — garde uniquement la question."""
    if "Question :" in input_text:
        return input_text.split("Question :")[-1].strip()
    elif "Question:" in input_text:
        return input_text.split("Question:")[-1].strip()
    return input_text.strip()


def contient_trace(output: str) -> bool:
    """Retourne True si l'output contient des références au contexte."""
    o = output.lower()
    return any(trace in o for trace in TRACES_CONTEXTE)


def est_eligible(p: dict) -> bool:
    """Vérifie qu'une paire est éligible pour la dérivation no-context."""
    return (
        p.get("qualite_estimee", 0) >= SCORE_MIN
        and len(p.get("output", "")) >= OUTPUT_MIN
        and p.get("source", "") not in SOURCES_REF
        and "Contexte :" in p.get("input", "")
        and not contient_trace(p.get("output", ""))
    )


def deriver_paires(paires: list[dict]) -> list[dict]:
    """Dérive les paires no-context depuis les paires RAG."""
    paires_nc     = []
    rejetees_trace = 0
    rejetees_court = 0
    rejetees_ref   = 0
    rejetees_score = 0

    for p in paires:
        output = p.get("output", "")
        input_ = p.get("input", "")

        # Filtre source référence
        if p.get("source", "") in SOURCES_REF:
            rejetees_ref += 1
            continue

        # Filtre score
        if p.get("qualite_estimee", 0) < SCORE_MIN:
            rejetees_score += 1
            continue

        # Filtre longueur output
        if len(output) < OUTPUT_MIN:
            rejetees_court += 1
            continue

        # Filtre — doit avoir un contexte à supprimer
        if "Contexte :" not in input_:
            continue

        # Filtre traces contexte dans output
        if contient_trace(output):
            rejetees_trace += 1
            continue

        question = extraire_question(input_)
        if not question:
            continue

        paire_nc = {
            "instruction": INSTRUCTION_NC,
            "input":       f"Question : {question}",
            "output":      output,
            "type":        p.get("type", "FACTUEL"),
            "domaine":     p.get("domaine", ""),
            "qualite_estimee": p.get("qualite_estimee", SCORE_MIN),
            "source":      "no_context_derived",
        }
        paires_nc.append(paire_nc)

    return paires_nc, {
        "trace":  rejetees_trace,
        "court":  rejetees_court,
        "ref":    rejetees_ref,
        "score":  rejetees_score,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["train", "eval", "both"],
        default="both",
        help="Source : train_clean, eval_clean ou les deux"
    )
    args = parser.parse_args()

    sources = []
    if args.source in ("train", "both"):
        sources.append(("train", TRAIN_CLEAN, OUT_TRAIN_NC))
    if args.source in ("eval", "both"):
        sources.append(("eval", EVAL_CLEAN, OUT_EVAL_NC))

    for nom, chemin, sortie in sources:
        print(f"\n{'═'*58}")
        print(f"  SOURCE : {chemin.name}  →  {sortie.name}")
        print(f"{'═'*58}")

        # Charger
        with open(chemin, encoding="utf-8") as f:
            paires = [json.loads(l) for l in f if l.strip()]
        print(f"  Paires chargées       : {len(paires)}")

        # Dériver
        paires_nc, rejets = deriver_paires(paires)

        # Mélanger
        random.seed(SEED)
        random.shuffle(paires_nc)

        # Stats
        types   = Counter(p["type"] for p in paires_nc)
        n_think = sum(1 for p in paires_nc if "<think>" in p["output"])

        print(f"\n  Résultat :")
        print(f"    Paires dérivées       : {len(paires_nc)}")
        print(f"    Rejetées (trace ctx)  : {rejets['trace']}")
        print(f"    Rejetées (trop court) : {rejets['court']}")
        print(f"    Rejetées (référence)  : {rejets['ref']}")
        print(f"    Rejetées (score < {SCORE_MIN}) : {rejets['score']}")
        print(f"\n    Avec <think>  : {n_think}/{len(paires_nc)} ({n_think/max(len(paires_nc),1)*100:.0f}%)")
        print(f"    Types         : {dict(types)}")

        # Aperçu
        if paires_nc:
            ex = paires_nc[0]
            print(f"\n  Exemple :")
            print(f"    input  : {ex['input'][:80]}...")
            print(f"    output : {ex['output'][:80]}...")
            print(f"    <think>: {'oui' if '<think>' in ex['output'] else 'non'}")

        # Sauvegarder
        with open(sortie, "w", encoding="utf-8") as f:
            for p in paires_nc:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        print(f"\n  ✓ {sortie}")

    print(f"\n  Lance ensuite :")
    if args.source in ("train", "both"):
        print(f"    python evaluation/generate_dataset_v3.py --prepare")
    print(f"    python evaluation/fusion_v3.py")


if __name__ == "__main__":
    main()