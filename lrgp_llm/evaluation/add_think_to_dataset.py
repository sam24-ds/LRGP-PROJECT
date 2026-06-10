"""
add_think_to_dataset.py
Enrichit les paires CALCUL/COMPARAISON du dataset avec des blocs <think>.
Utilise le Gemini Batch API pour minimiser le coût.

Usage :
    python evaluation/add_think_to_dataset.py --prepare
    python evaluation/add_think_to_dataset.py --submit
    python evaluation/add_think_to_dataset.py --status
    python evaluation/add_think_to_dataset.py --collect
    python evaluation/add_think_to_dataset.py --all
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ── Chemins ───────────────────────────────────────────────────────
DATASET_RAW   = Path("data/datasets/dataset_raw.jsonl")
BATCH_DIR     = Path("data/datasets/batch_think")
BATCH_DIR.mkdir(parents=True, exist_ok=True)

REQUESTS_FILE = BATCH_DIR / "think_requests.jsonl"
JOB_STATE     = BATCH_DIR / "think_job_state.json"
ENRICHED_OUT  = BATCH_DIR / "think_enriched.jsonl"

MODEL = "gemini-3.1-pro-preview"

PROMPT_ADD_THINK = """Tu es un expert en génie des procédés et séparation membranaire.

Reformule ce champ "output" en ajoutant un bloc <think> synthétique
AVANT la réponse finale existante.

Format attendu OBLIGATOIRE :
<think>Étape 1 : [identifier l'équation ou le principe depuis le contexte]
Étape 2 : [lister les données numériques avec unités]
Calcul : [résolution étape par étape]
Vérification : [cohérence dimensionnelle ou logique]</think>

Réponse finale :[réponse originale conservée intégralement]

Règles strictes :
- Le bloc <think> doit être court — 3 à 6 étapes MAXIMUM
- Ne pas modifier la réponse finale — la conserver mot pour mot
- Le raisonnement doit être exact et vérifiable
- Si l'output contient déjà <think>, retourne-le tel quel sans modification

Output original à enrichir :
{output}

Réponds UNIQUEMENT avec le nouvel output enrichi, sans texte avant ni après."""


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — PRÉPARER LES REQUÊTES BATCH
# ══════════════════════════════════════════════════════════════════
def preparer() -> None:
    print(f"\n{'═'*58}")
    print(f"  ÉTAPE 1 — Préparation requêtes enrichissement <think>")
    print(f"{'═'*58}")

    # Charger le dataset
    with open(DATASET_RAW, encoding="utf-8") as f:
        toutes_paires = [json.loads(l) for l in f if l.strip()]

    # Filtrer les paires CALCUL/COMPARAISON sans <think>
    a_enrichir =[
        (i, p) for i, p in enumerate(toutes_paires)
        if p.get("type") in ("CALCUL", "COMPARAISON")
        and "<think>" not in p.get("output", "")
    ]

    deja_ok = sum(
        1 for p in toutes_paires
        if "<think>" in p.get("output", "")
    )

    print(f"  Paires totales          : {len(toutes_paires)}")
    print(f"  Déjà avec <think>       : {deja_ok}")
    print(f"  CALCUL/COMP sans think  : {len(a_enrichir)}")

    # Construire les requêtes batch
    requetes = []
    ordered_indexes =[]  # Liste des index originaux pour retrouver l'ordre

    for idx, (i_original, paire) in enumerate(a_enrichir):
        ordered_indexes.append(i_original)

        prompt = PROMPT_ADD_THINK.format(
            output=paire["output"][:2000]
        )

        requete = {
            "contents": [{"parts": [{"text": prompt}], "role": "user"}],
            "config": {  # Remplacé generation_config par config
                "temperature":     0.2,
                "max_output_tokens": 1500,
                "thinking_config": {"thinking_level": "LOW"},
            },
        }
        # custom_id a été retiré car interdit en inline
        requetes.append(requete)

    # Sauvegarder les requêtes
    with open(REQUESTS_FILE, "w", encoding="utf-8") as f:
        for r in requetes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Sauvegarder la map index (une simple liste dans le même ordre)
    map_file = BATCH_DIR / "index_map.json"
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(ordered_indexes, f)

    # Sauvegarder le dataset complet pour la reconstruction
    dataset_backup = BATCH_DIR / "dataset_backup.jsonl"
    with open(dataset_backup, "w", encoding="utf-8") as f:
        for p in toutes_paires:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Coût estimé
    cout = len(requetes) * 1000 * 1.0 / 1e6 + len(requetes) * 1500 * 6.0 / 1e6
    print(f"\n  ✓ {len(requetes)} requêtes → {REQUESTS_FILE}")
    print(f"  Coût estimé batch       : ~${cout:.2f}")
    print(f"\n  Lance maintenant        : --submit")


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — SOUMETTRE
# ══════════════════════════════════════════════════════════════════
def soumettre() -> None:
    print(f"\n{'═'*58}")
    print(f"  ÉTAPE 2 — Soumission batch")
    print(f"{'═'*58}")

    if not REQUESTS_FILE.exists():
        print("❌ Fichier requêtes introuvable — lance --prepare d'abord")
        return

    with open(REQUESTS_FILE, encoding="utf-8") as f:
        requetes =[json.loads(l) for l in f if l.strip()]

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    print(f"  {len(requetes)} requêtes à soumettre...")

    try:
        job = client.batches.create(
            model=MODEL,
            src=requetes,
            config={"display_name": f"lrgp-think-{int(time.time())}"},
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
        print(f"\n  Lance --status pour suivre")

    except Exception as e:
        print(f"❌ Erreur : {e}")


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — STATUT
# ══════════════════════════════════════════════════════════════════
def verifier_statut() -> str:
    if not JOB_STATE.exists():
        print("❌ Aucun job en cours")
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

        if hasattr(job, "request_counts") and job.request_counts:
            rc = job.request_counts
            print(f"  Requêtes : total={rc.total} | "
                  f"ok={rc.succeeded} | échec={rc.failed}")

        return str(job.state)

    except Exception as e:
        print(f"❌ {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — COLLECTER ET RECONSTRUIRE
# ══════════════════════════════════════════════════════════════════
def collecter() -> None:
    print(f"\n{'═'*58}")
    print(f"  ÉTAPE 4 — Collecte et reconstruction dataset")
    print(f"{'═'*58}")

    statut = verifier_statut()
    if "SUCCEEDED" not in statut:
        print(f"\n  ⚠ Batch non terminé ({statut})")
        return

    with open(JOB_STATE, encoding="utf-8") as f:
        state = json.load(f)

    # Charger la map ordonnée et le dataset original
    with open(BATCH_DIR / "index_map.json", encoding="utf-8") as f:
        ordered_indexes = json.load(f)

    with open(BATCH_DIR / "dataset_backup.jsonl", encoding="utf-8") as f:
        dataset =[json.loads(l) for l in f if l.strip()]

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    job    = client.batches.get(name=state["job_name"])

    # Appliquer les enrichissements
    enrichis  = 0
    erreurs   = 0

    # Remplacé inline_responses par dest.inlined_responses
    for i, response in enumerate(job.dest.inlined_responses or[]):
        
        # Récupération de l'index original via l'ordre
        i_original = ordered_indexes[i] if i < len(ordered_indexes) else None
        
        if i_original is None:
            continue

        # Sécurité si la requête a échoué individuellement
        if getattr(response, "response", None) is None:
            erreurs += 1
            continue

        try:
            nouvel_output = (
                response.response.candidates[0]
                .content.parts[0].text.strip()
            )

            # Vérifier que le <think> est présent
            if "<think>" in nouvel_output:
                dataset[i_original]["output"] = nouvel_output
                enrichis += 1
            else:
                # Pas de <think> généré — garder l'original
                erreurs += 1

        except Exception as e:
            erreurs += 1

    print(f"\n  Enrichissements appliqués : {enrichis}")
    print(f"  Échecs (gardé original)   : {erreurs}")

    # Stats finales
    avec_think = sum(
        1 for p in dataset if "<think>" in p.get("output", "")
    )
    print(f"  Avec <think> final        : "
          f"{avec_think}/{len(dataset)} "
          f"({avec_think/len(dataset)*100:.0f}%)")

    # Sauvegarder le dataset enrichi
    with open(ENRICHED_OUT, "w", encoding="utf-8") as f:
        for p in dataset:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  ✓ Dataset enrichi → {ENRICHED_OUT}")
    print(f"  Lance ensuite : python evaluation/fusion_datasets.py")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--submit",  action="store_true")
    parser.add_argument("--status",  action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--all",     action="store_true")
    args = parser.parse_args()

    if args.prepare or args.all:
        preparer()
        if args.all:
            soumettre()
    elif args.submit:
        soumettre()
    elif args.status:
        verifier_statut()
    elif args.collect:
        collecter()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()