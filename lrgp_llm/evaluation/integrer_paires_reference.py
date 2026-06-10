"""
integrer_paires_reference.py
Intègre les paires Q/R issues des documents de référence
dans le dataset de fine-tuning.

Usage : python evaluation/integrer_paires_reference.py
        --fichier "data/datasets/reference_pairs/examen_2023.json"
"""

import argparse
import json
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("data/datasets/reference_pairs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHAMPS_REQUIS = ["instruction", "input", "output", "type", "domaine"]

INSTRUCTION_DEFAULT = (
    "Tu es un assistant expert en génie des procédés. "
    "Résous le problème avec un raisonnement concis, "
    "puis donne une réponse finale claire."
)


def valider_et_normaliser(paire: dict, source: str) -> dict | None:
    """
    Vérifie et complète les champs manquants.
    Retourne None si la paire est invalide.
    """
    # Champs obligatoires
    if not paire.get("input") or not paire.get("output"):
        return None

    # Normaliser
    paire.setdefault("instruction", INSTRUCTION_DEFAULT)
    paire.setdefault("type", "FACTUEL")
    paire.setdefault("domaine", "autre")
    paire.setdefault("qualite_estimee", 5)  # corrigé vérifié → 5/5
    paire.setdefault("source", source)

    # Vérifier type valide
    if paire["type"] not in {"CALCUL", "FACTUEL", "COMPARAISON", "PROCEDURE"}:
        paire["type"] = "FACTUEL"

    return paire


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fichier", required=True,
                        help="Fichier JSON ou JSONL généré via Gemini")
    args = parser.parse_args()

    fichier = Path(args.fichier)
    source  = fichier.stem[:40]

    print(f"\n{'═'*55}")
    print(f"  INTÉGRATION PAIRES RÉFÉRENCE")
    print(f"  Fichier : {fichier.name}")
    print(f"{'═'*55}")

    # Lire le fichier — supporte JSON et JSONL
    with open(fichier, encoding="utf-8") as f:
        contenu = f.read().strip()

    try:
        # Essayer JSON d'abord
        data = json.loads(contenu)
        paires_brutes = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        # Sinon JSONL
        paires_brutes = [
            json.loads(l) for l in contenu.splitlines() if l.strip()
        ]

    print(f"  Paires lues : {len(paires_brutes)}")

    # Valider et normaliser
    paires_valides = []
    rejets = 0
    for p in paires_brutes:
        p_norm = valider_et_normaliser(p, source)
        if p_norm:
            paires_valides.append(p_norm)
        else:
            rejets += 1

    print(f"  Valides     : {len(paires_valides)}")
    print(f"  Rejetées    : {rejets}")

    if not paires_valides:
        print("  ⚠ Aucune paire valide")
        return

    # Stats
    types      = Counter(p["type"] for p in paires_valides)
    avec_think = sum(1 for p in paires_valides
                     if "<think>" in p.get("output", ""))

    print(f"\n  Par type :")
    for t, n in sorted(types.items()):
        print(f"    {t:<15} {n}")
    print(f"  Avec <think> : {avec_think}/{len(paires_valides)}")

    # Sauvegarder dans reference_pairs/
    output = OUTPUT_DIR / f"{source}_pairs.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for p in paires_valides:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  ✓ Sauvegardé → {output}")
    print(f"  Lance ensuite : python evaluation/fusion_datasets.py")


if __name__ == "__main__":
    main()