"""
benchmark_manager.py
Gestion complète du benchmark LRGP.
Usage :
    python evaluation/benchmark_manager.py --action stats
    python evaluation/benchmark_manager.py --action split
    python evaluation/benchmark_manager.py --action check
    python evaluation/benchmark_manager.py --action add
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

BENCHMARK_PATH = Path("questions.jsonl")
SPLIT_DIR      = Path("benchmark/split")

TYPES_VALIDES  = {"CALCUL", "FACTUEL", "COMPARAISON", "PROCEDURE"}
NIVEAUX_VALIDES = {"N1", "N2", "N3", "N4"}


# ══════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════
def charger_questions() -> list[dict]:
    if not BENCHMARK_PATH.exists():
        return []
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sauvegarder_questions(questions: list[dict]) -> None:
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def sauvegarder_split(nom: str, questions: list[dict]) -> None:
    # Créer le dossier split/ s'il n'existe pas
    split_dir = SPLIT_DIR / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    path = split_dir / f"benchmark_{nom}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  ✓ benchmark_{nom}.jsonl — {len(questions)} questions")

# ══════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════
def afficher_stats(questions: list[dict]) -> None:
    print(f"\n{'═'*58}")
    print(f"  BENCHMARK LRGP — {len(questions)} questions")
    print(f"{'═'*58}")

    if not questions:
        print("  Aucune question.")
        return

    types    = Counter(q.get("type",       "?") for q in questions)
    domaines = Counter(q.get("domaine",    "?") for q in questions)
    niveaux  = Counter(q.get("difficulté", "?") for q in questions)
    sources  = Counter(q.get("source",     "?") for q in questions)

    print(f"\n  Par type :")
    for t, n in sorted(types.items()):
        bar = "█" * n
        print(f"    {t:<15} {n:3d}  {bar}")

    print(f"\n  Par domaine :")
    for d, n in sorted(domaines.items(), key=lambda x: -x[1]):
        print(f"    {d:<35} {n:3d}")

    print(f"\n  Par niveau :")
    for nv, n in sorted(niveaux.items()):
        bar = "█" * n
        print(f"    {nv}  {n:3d}  {bar}")

    print(f"\n  Par source :")
    for s, n in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {s:<35} {n:3d}")

    # Vérifier les splits existants
    print(f"\n  Splits existants :")
    for nom in ["train", "val", "test"]:
        p = SPLIT_DIR / "split" / f"benchmark_{nom}.jsonl"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                n = sum(1 for l in f if l.strip())
            print(f"    benchmark_{nom}.jsonl  — {n} questions  ✓")
        else:
            print(f"    benchmark_{nom}.jsonl  — non créé")

    print(f"\n  Progression : {len(questions)}/80 minimum")
    print(f"{'═'*58}\n")


# ══════════════════════════════════════════════════════════════════
# SPLIT ANTI-CONTAMINATION 60/10/30
# ══════════════════════════════════════════════════════════════════
def split_anti_contamination(
    questions: list[dict],
    ratio_train: float = 0.60,
    ratio_val:   float = 0.10,
    ratio_test:  float = 0.30,
    seed:        int   = 42,
) -> tuple[list, list, list]:
    """
    Split stratifié par type pour garder la distribution.
    Reproductible via seed fixe.
    """
    random.seed(seed)

    # Grouper par type
    par_type = defaultdict(list)
    for q in questions:
        par_type[q.get("type", "FACTUEL")].append(q)

    train_set, val_set, test_set = [], [], []

    for qtype, qs in par_type.items():
        qs_melange = qs.copy()
        random.shuffle(qs_melange)
        n = len(qs_melange)

        n_train = math.ceil(n * ratio_train)
        n_val   = max(1, math.ceil(n * ratio_val))   # au moins 1 par type
        n_test  = n - n_train - n_val

        # Garantir au moins 1 question de test par type
        if n_test < 1 and n >= 3:
            n_train -= 1
            n_test   = 1

        train_set.extend(qs_melange[:n_train])
        val_set.extend(  qs_melange[n_train:n_train + n_val])
        test_set.extend( qs_melange[n_train + n_val:])

    # Mélanger chaque split (pour ne pas avoir les types groupés)
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set


def verifier_contamination_split(
    train: list, val: list, test: list
) -> None:
    """
    Vérifie qu'aucun ID n'apparaît dans deux partitions.
    """
    ids_train = {q["id"] for q in train}
    ids_val   = {q["id"] for q in val}
    ids_test  = {q["id"] for q in test}

    overlap_tv = ids_train & ids_val
    overlap_tt = ids_train & ids_test
    overlap_vt = ids_val   & ids_test

    ok = True
    if overlap_tv:
        print(f"  ⚠ Overlap train/val  : {overlap_tv}")
        ok = False
    if overlap_tt:
        print(f"  ⚠ Overlap train/test : {overlap_tt}")
        ok = False
    if overlap_vt:
        print(f"  ⚠ Overlap val/test   : {overlap_vt}")
        ok = False

    if ok:
        print(f"  ✓ Aucun overlap entre les partitions")


# ══════════════════════════════════════════════════════════════════
# CHECK — DOUBLONS ET QUALITÉ
# ══════════════════════════════════════════════════════════════════
def verifier_qualite(questions: list[dict]) -> None:
    print(f"\n  Vérification qualité — {len(questions)} questions")
    print(f"  {'─'*50}")

    erreurs = []

    for q in questions:
        qid = q.get("id", "???")

        # Champs obligatoires
        for champ in ["id", "question", "answer", "type", "domaine", "difficulté"]:
            if not q.get(champ):
                erreurs.append(f"  [{qid}] Champ manquant : {champ}")

        # Type valide
        if q.get("type") not in TYPES_VALIDES:
            erreurs.append(f"  [{qid}] Type invalide : {q.get('type')}")

        # Niveau valide
        if q.get("difficulté") not in NIVEAUX_VALIDES:
            erreurs.append(f"  [{qid}] Niveau invalide : {q.get('difficulté')}")

        # Réponse non vide
        if q.get("answer", "").strip() in ("", "À compléter"):
            erreurs.append(f"  [{qid}] Réponse vide ou placeholder")

        # Question trop courte
        if len(q.get("question", "")) < 20:
            erreurs.append(f"  [{qid}] Question trop courte (<20 chars)")

    # Doublons par ID
    ids = [q.get("id") for q in questions]
    doublons_id = [id_ for id_, n in Counter(ids).items() if n > 1]
    if doublons_id:
        erreurs.append(f"  IDs en double : {doublons_id}")

    # Doublons par texte (50 premiers chars)
    textes = defaultdict(list)
    for q in questions:
        key = q.get("question", "")[:60].lower().strip()
        textes[key].append(q.get("id"))
    doublons_texte = {k: v for k, v in textes.items() if len(v) > 1}
    if doublons_texte:
        for texte, ids_dup in doublons_texte.items():
            erreurs.append(f"  Questions similaires {ids_dup} : '{texte[:50]}...'")

    if erreurs:
        print(f"  ✗ {len(erreurs)} problème(s) détecté(s) :")
        for e in erreurs:
            print(e)
    else:
        print(f"  ✓ Qualité OK — aucun problème détecté")

    # Stats réponses "À compléter"
    a_completer = [q["id"] for q in questions
                   if "compléter" in q.get("answer", "").lower()]
    if a_completer:
        print(f"\n  ⚠ {len(a_completer)} réponses 'À compléter' :")
        for qid in a_completer[:5]:
            print(f"    {qid}")
        if len(a_completer) > 5:
            print(f"    ... et {len(a_completer)-5} autre(s)")


# ══════════════════════════════════════════════════════════════════
# ADD — AJOUT INTERACTIF
# ══════════════════════════════════════════════════════════════════
def ajouter_question_interactive(questions: list[dict]) -> dict | None:
    print("\n─── Nouvelle question ───")

    ids_existants = {q["id"] for q in questions}
    n = len(questions) + 1
    while f"Q{n:03d}" in ids_existants:
        n += 1
    qid = f"Q{n:03d}"

    question = input(f"Question [{qid}] : ").strip()
    if not question:
        return None

    answer = input("Réponse / Solution : ").strip()
    if not answer:
        answer = "À compléter"

    print(f"Type : {', '.join(sorted(TYPES_VALIDES))}")
    qtype = input("Type [FACTUEL] : ").strip().upper() or "FACTUEL"
    if qtype not in TYPES_VALIDES:
        qtype = "FACTUEL"

    domaine = input("Domaine : ").strip() or "autre"

    print(f"Niveau : N1 (simple) → N4 (calcul complexe)")
    niveau = input("Niveau [N2] : ").strip().upper() or "N2"
    if niveau not in NIVEAUX_VALIDES:
        niveau = "N2"

    source = input("Source [examen_prof] : ").strip() or "examen_prof"

    return {
        "id":         qid,
        "question":   question,
        "answer":     answer,
        "type":       qtype,
        "domaine":    domaine,
        "difficulté": niveau,
        "source":     source,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Gestion du benchmark LRGP"
    )
    parser.add_argument(
        "--action",
        choices=["stats", "split", "check", "add"],
        default="stats",
        help="Action à effectuer"
    )
    args = parser.parse_args()

    questions = charger_questions()

    if args.action == "stats":
        afficher_stats(questions)

    elif args.action == "check":
        afficher_stats(questions)
        verifier_qualite(questions)

    elif args.action == "split":
        if len(questions) < 10:
            print(f"⚠ Trop peu de questions ({len(questions)}) — minimum 10 requis")
            return

        print(f"\n  Split 60/10/30 sur {len(questions)} questions...")
        train, val, test = split_anti_contamination(questions)

        print(f"\n  Résultat :")
        print(f"    Train : {len(train)} questions ({len(train)/len(questions)*100:.0f}%)")
        print(f"    Val   : {len(val)}   questions ({len(val)/len(questions)*100:.0f}%)")
        print(f"    Test  : {len(test)}  questions ({len(test)/len(questions)*100:.0f}%)")

        # Vérifier la distribution par type dans chaque split
        print(f"\n  Distribution par type :")
        for nom, split in [("Train", train), ("Val", val), ("Test", test)]:
            types = Counter(q["type"] for q in split)
            dist  = " | ".join(f"{t}:{n}" for t,n in sorted(types.items()))
            print(f"    {nom:<6} : {dist}")

        # Vérifier l'absence d'overlap
        print(f"\n  Vérification overlaps :")
        verifier_contamination_split(train, val, test)

        # Sauvegarder
        print(f"\n  Sauvegarde :")
        sauvegarder_split("train", train)
        sauvegarder_split("val",   val)
        sauvegarder_split("test",  test)

        print(f"\n  ✓ Split terminé — dossier : {SPLIT_DIR}")

    elif args.action == "add":
        print(f"\nBenchmark actuel : {len(questions)} questions")
        print("Entrée vide pour terminer.\n")
        n_ajoutees = 0
        while True:
            q = ajouter_question_interactive(questions)
            if not q:
                break
            questions.append(q)
            sauvegarder_questions(questions)
            n_ajoutees += 1
            print(f"  ✓ {q['id']} ajoutée ({len(questions)} total)\n")

        if n_ajoutees > 0:
            print(f"\n  {n_ajoutees} question(s) ajoutée(s)")
            print(f"  Relance le split : python evaluation/benchmark_manager.py --action split")


if __name__ == "__main__":
    main()