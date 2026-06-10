"""
generate_dataset_batch.py
Génération du dataset LRGP via Gemini Batch API.
50% moins cher, pas de 503, traitement asynchrone.

Usage :
    python evaluation/generate_dataset_batch.py --prepare   # prépare le JSONL
    python evaluation/generate_dataset_batch.py --submit    # soumet le batch
    python evaluation/generate_dataset_batch.py --status    # vérifie l'état
    python evaluation/generate_dataset_batch.py --collect   # récupère les résultats
    python evaluation/generate_dataset_batch.py --all       # tout en un
    python evaluation/generate_dataset_batch.py --test      # test 20 paires
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import qdrant_client

load_dotenv()

# ── Chemins ───────────────────────────────────────────────────────
CHUNKS_DIR   = Path("ingestion/data/chunks")
SPLITS_DIR   = Path("data/datasets/benchmark/split")
OUTPUT_DIR   = Path("data/datasets")
BATCH_DIR    = Path("data/datasets/batch")
BATCH_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUESTS_FILE  = BATCH_DIR / "batch_requests.jsonl"   # requêtes à envoyer
JOB_STATE_FILE = BATCH_DIR / "batch_job_state.json"   # nom du job en cours
OUTPUT_RAW     = OUTPUT_DIR / "dataset_raw.jsonl"
OUTPUT_TRAIN   = OUTPUT_DIR / "train.jsonl"
OUTPUT_EVAL    = OUTPUT_DIR / "eval.jsonl"

# ── Config ────────────────────────────────────────────────────────
N_PAIRES_CIBLE   = 1500
RATIO_GENERATION = 1.35
BATCH_SIZE       = 5       # paires générées par requête
SCORE_MIN        = 4
RATIO_TRAIN      = 0.85
SEED             = 42

MODEL_GEN   = "gemini-3.1-pro-preview"  #gemini-3.1-pro-preview #gemini-3-flash-preview
MODEL_JUDGE = "gemini-3.1-pro-preview"

# ── Prompts ───────────────────────────────────────────────────────
PROMPT_GENERATION = """Tu es un expert en génie des procédés au LRGP Nancy.
À partir du contexte scientifique, génère {n} paires question/réponse.

Format JSON obligatoire pour chaque paire :
{{
  "instruction": "Tu es un assistant expert en génie des procédés. Résous le problème avec un raisonnement concis, puis donne une réponse finale claire.",
  "input": "Contexte :\\n[texte du contexte]\\n\\nQuestion : [question précise]",
  "output": "[voir règles ci-dessous]",
  "type": "CALCUL|FACTUEL|COMPARAISON",
  "domaine": "[domaine technique]",
  "qualite_estimee": [1-5]
}}

Règles pour le champ "output" :

Pour CALCUL et COMPARAISON (obligatoire) :
<think>Étape 1 : [identifier l'équation depuis le contexte]
Étape 2 : [lister les données avec unités]
Calcul : [résolution numérique étape par étape]
Vérification : [cohérence dimensionnelle]</think>

Réponse finale : [résultat avec unités]

[Source: nom_article]

Pour FACTUEL (réponse directe, pas de <think>) :
[réponse concise et précise]

[Source: nom_article]

Règles strictes :
- Baser UNIQUEMENT sur le contexte fourni
- Raisonnement court et EXACT — 3 à 6 étapes maximum
- Réponse finale concise après </think>
- Toujours citer [Source: nom_article]
- 70 à 80%% de paires CALCUL/COMPARAISON avec <think>

Réponds UNIQUEMENT avec un tableau JSON valide sans texte avant ni après.

Contexte scientifique :
{contexte}"""


# ══════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════
def charger_chunks() -> list[dict]:
    tous =[]
    for f in CHUNKS_DIR.glob("*.jsonl"):
        try:
            with open(f, encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        c = json.loads(line)
                        if len(c.get("text", "")) >= 200:
                            tous.append(c)
        except Exception:
            pass
    return tous


def filtrer_chunks(chunks: list[dict]) -> list[dict]:
    MOTS_CLES =["membrane", "perméabilité", "flux", "CO2", "CH4",
                 "transfert", "coefficient", "pression", "Barrer",
                 "permeability", "hollow fiber", "contactor", "K_OV"]
    return[
        c for c in chunks
        if any(kw.lower() in c.get("text", "").lower() for kw in MOTS_CLES)
        and not any(x in c.get("text", "")[:50].lower()
                    for x in ["references", "bibliography", "©"])
    ]


def recuperer_connexes(texte, model, qdrant, top_k=3) -> list[str]:
    vec = model.encode([texte], normalize_embeddings=True,
                       convert_to_numpy=True)[0].tolist()
    results = qdrant.query_points(
        collection_name="lrgp_corpus",
        query=vec, using="dense", limit=top_k+1,
    ).points
    connexes =[]
    for r in results:
        t = r.payload.get("text", "")
        if t != texte and t:
            connexes.append(
                f"[Source: {r.payload.get('source_file','?')}]\n{t}"
            )
        if len(connexes) >= top_k:
            break
    return connexes


def charger_questions_bench() -> list[str]:
    questions =[]
    for nom in ["train", "val", "test"]:
        p = SPLITS_DIR / f"benchmark_{nom}.jsonl"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line).get("question",""))
    return questions


def verifier_contamination(question, questions_bench, model, seuil=0.85):
    if not questions_bench:
        return False
    vecs = model.encode(
        [question] + questions_bench[:50],
        normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    )
    for vec_b in vecs[1:]:
        if float(vecs[0] @ vec_b) > seuil:
            return True
    return False


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — PRÉPARER LE FICHIER BATCH JSONL
# ══════════════════════════════════════════════════════════════════
def preparer_batch(n_cible: int) -> None:
    print(f"\n{'═'*60}")
    print(f"  ÉTAPE 1 — Préparation des requêtes batch")
    print(f"{'═'*60}")

    n_requetes = int((n_cible * RATIO_GENERATION) / BATCH_SIZE)
    print(f"  Requêtes à préparer : {n_requetes}")

    print(f"  Chargement BGE-M3...", end=" ", flush=True)
    embed_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    print("✓")

    print(f"  Connexion Qdrant...", end=" ", flush=True)
    qdrant = qdrant_client.QdrantClient("localhost", port=6333)
    print("✓")

    tous_chunks  = charger_chunks()
    chunks_utiles = filtrer_chunks(tous_chunks)
    questions_bench = charger_questions_bench()

    print(f"  Chunks utiles  : {len(chunks_utiles):,}")
    print(f"  Questions bench: {len(questions_bench)}")

    random.seed(SEED)
    random.shuffle(chunks_utiles)

    requetes =[]
    chunks_traites = 0

    for chunk in chunks_utiles[:n_requetes * 2]:  # marge
        if len(requetes) >= n_requetes:
            break

        try:
            connexes = recuperer_connexes(chunk["text"], embed_model, qdrant)
            source   = chunk.get("metadata", {}).get("source_file",
                       chunk.get("source", "?"))
            contexte = f"[Source: {source}]\n{chunk['text']}"
            for c in connexes:
                contexte += f"\n\n---\n\n{c}"
            contexte = contexte[:4000]

            prompt = PROMPT_GENERATION.format(
                n=BATCH_SIZE,
                contexte=contexte,
            )

            requete = {
                "_custom_id": f"gen_{chunks_traites:04d}", # Variable locale
                "contents": [{"parts": [{"text": prompt}], "role": "user"}],
                "config": { # Remplace "generation_config"
                    "temperature": 0.4,
                    "max_output_tokens": 3000,
                    "thinking_config": {"thinking_level": "LOW"},
                },
                "_contexte": contexte,  # métadonnée locale
            }
            requetes.append(requete)
            chunks_traites += 1

        except Exception as e:
            print(f"  ⚠ chunk {chunks_traites} : {e}")

        if chunks_traites % 50 == 0:
            print(f"  {chunks_traites}/{n_requetes} requêtes préparées...")

    # Sauvegarder le JSONL — format Gemini Batch (sans les clés commençant par _)
    with open(REQUESTS_FILE, "w", encoding="utf-8") as f:
        for r in requetes:
            r_clean = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(r_clean, ensure_ascii=False) + "\n")

    # Sauvegarder les contextes séparément pour la reconstruction
    contextes_file = BATCH_DIR / "contextes.jsonl"
    with open(contextes_file, "w", encoding="utf-8") as f:
        for r in requetes:
            f.write(json.dumps({
                "custom_id": r.get("_custom_id", ""),
                "contexte":  r.get("_contexte", ""),
            }, ensure_ascii=False) + "\n")

    print(f"\n  ✓ {len(requetes)} requêtes sauvegardées → {REQUESTS_FILE}")
    print(f"  ✓ Contextes sauvegardés → {contextes_file}")
    print(f"\n  Lance maintenant : --submit")


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — SOUMETTRE LE BATCH
# ══════════════════════════════════════════════════════════════════
def soumettre_batch() -> None:
    print(f"\n{'═'*60}")
    print(f"  ÉTAPE 2 — Soumission du batch")
    print(f"{'═'*60}")

    if not REQUESTS_FILE.exists():
        print("❌ Fichier de requêtes introuvable — lance --prepare d'abord")
        return

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Lire les requêtes inline
    requetes =[]
    with open(REQUESTS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                requetes.append(json.loads(line))

    print(f"  {len(requetes)} requêtes à soumettre...")
    print(f"  Modèle : {MODEL_GEN}")

    try:
        # Soumettre le batch
        batch_job = client.batches.create(
            model=MODEL_GEN,
            src=requetes,
            config={"display_name": f"lrgp-dataset-{int(time.time())}"},
        )

        # Sauvegarder l'état du job
        state = {
            "job_name":    batch_job.name,
            "submitted_at": time.time(),
            "n_requests":   len(requetes),
            "model":        MODEL_GEN,
        }
        with open(JOB_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        print(f"\n  ✓ Batch soumis : {batch_job.name}")
        print(f"  Statut initial : {batch_job.state}")
        print(f"\n  Lance --status pour suivre l'avancement")
        print(f"  Lance --collect quand le batch est terminé")

    except Exception as e:
        print(f"❌ Erreur soumission : {e}")


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — VÉRIFIER LE STATUT
# ══════════════════════════════════════════════════════════════════
def verifier_statut() -> str:
    if not JOB_STATE_FILE.exists():
        print("❌ Aucun job en cours — lance --submit d'abord")
        return ""

    with open(JOB_STATE_FILE, encoding="utf-8") as f:
        state = json.load(f)

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        job = client.batches.get(name=state["job_name"])
        elapsed = (time.time() - state["submitted_at"]) / 60

        print(f"\n  Job    : {state['job_name']}")
        print(f"  Statut : {job.state}")
        print(f"  Durée  : {elapsed:.0f} minutes")

        if hasattr(job, 'request_counts') and job.request_counts:
            rc = job.request_counts
            print(f"  Requêtes : total={rc.total} | "
                  f"ok={rc.succeeded} | "
                  f"échec={rc.failed} | "
                  f"en cours={rc.processing}")

        return str(job.state)

    except Exception as e:
        print(f"❌ Erreur statut : {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — COLLECTER LES RÉSULTATS
# ══════════════════════════════════════════════════════════════════
def collecter_resultats() -> None:
    print(f"\n{'═'*60}")
    print(f"  ÉTAPE 4 — Collecte des résultats")
    print(f"{'═'*60}")

    statut = verifier_statut()
    if "JOB_STATE_SUCCEEDED" not in statut and "SUCCEEDED" not in statut:
        print(f"\n  ⚠ Le batch n'est pas encore terminé (statut: {statut})")
        print(f"  Relance --status dans quelques minutes")
        return

    with open(JOB_STATE_FILE, encoding="utf-8") as f:
        state = json.load(f)

    # Charger les contextes EN LISTE ORDONNÉE (l'ordre est préservé par l'API)
    contextes_ordonnes =[]
    ctx_file = BATCH_DIR / "contextes.jsonl"
    if ctx_file.exists():
        with open(ctx_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    contextes_ordonnes.append(json.loads(line))

    client  = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    job     = client.batches.get(name=state["job_name"])

    # Récupérer les résultats avec enumerate pour faire la liaison avec contextes_ordonnes
    paires_brutes =[]
    for i, response in enumerate(job.dest.inlined_responses or[]):
        # Récupération du contexte via l'index
        ctx_data = contextes_ordonnes[i] if i < len(contextes_ordonnes) else {}
        custom_id = ctx_data.get("custom_id", f"gen_{i:04d}")
        contexte = ctx_data.get("contexte", "")

        try:
            # Sécurité si la requête a échoué individuellement (503, blocage de sécurité, etc.)
            if getattr(response, "response", None) is None:
                print(f"  ⚠ {custom_id} : La requête a échoué côté API.")
                continue

            contenu = response.response.candidates[0].content.parts[0].text
            contenu = contenu.strip()

            # Nettoyer markdown
            if "```" in contenu:
                for partie in contenu.split("```"):
                    partie = partie.strip()
                    if partie.startswith("json"):
                        partie = partie[4:]
                    if partie.strip().startswith("["):
                        contenu = partie.strip()
                        break

            paires = json.loads(contenu)

            for p in paires:
                if "input" in p and contexte:
                    # Injecter le vrai contexte
                    p["input"] = p["input"].replace(
                        "<contexte>", contexte
                    )
                p["custom_id"] = custom_id
                paires_brutes.append(p)

        except Exception as e:
            print(f"  ⚠ {custom_id} : {e}")

    print(f"  {len(paires_brutes)} paires extraites")

    # Charger BGE-M3 pour anti-contamination
    print(f"  Chargement BGE-M3 pour anti-contamination...", end=" ")
    embed_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    print("✓")

    questions_bench = charger_questions_bench()

    # Filtrer
    paires_validees =[]
    rejetees_score  = 0
    rejetees_contam = 0

    for p in paires_brutes:
        # Score
        score = p.get("qualite_estimee", 3)
        if score < SCORE_MIN:
            rejetees_score += 1
            continue

        # Anti-contamination
        input_text = p.get("input", "")
        question   = ""
        if "Question :" in input_text:
            question = input_text.split("Question :")[-1].strip()

        if question and verifier_contamination(
            question, questions_bench, embed_model
        ):
            rejetees_contam += 1
            continue

        paires_validees.append(p)

    print(f"\n  Acceptées (≥{SCORE_MIN}/5)  : {len(paires_validees)}")
    print(f"  Rejetées score       : {rejetees_score}")
    print(f"  Rejetées contam      : {rejetees_contam}")

    # Sauvegarder raw
    with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
        for p in paires_validees:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Split train/eval
    random.seed(SEED)
    random.shuffle(paires_validees)
    n_train = int(len(paires_validees) * RATIO_TRAIN)
    train   = paires_validees[:n_train]
    eval_   = paires_validees[n_train:]

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(OUTPUT_EVAL, "w", encoding="utf-8") as f:
        for p in eval_:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Coût estimé
    n_req    = state.get("n_requests", 0)
    cout_est = n_req * BATCH_SIZE * 5000 * 1.0 / 1e6  # $1/M batch
    cout_est += n_req * BATCH_SIZE * 3000 * 6.0 / 1e6  # $6/M batch output

    print(f"\n  ✓ train.jsonl : {len(train)} paires")
    print(f"  ✓ eval.jsonl  : {len(eval_)} paires")
    print(f"  Coût estimé   : ~${cout_est:.2f}")


# ══════════════════════════════════════════════════════════════════
# MODE TEST
# ══════════════════════════════════════════════════════════════════
def mode_test() -> None:
    """Test synchrone sur 20 paires — sans Batch API."""
    print(f"\n{'═'*60}")
    print(f"  MODE TEST — 20 paires (synchrone)")
    print(f"{'═'*60}")

    api_key = os.getenv("GEMINI_API_KEY")
    client  = genai.Client(api_key=api_key)

    print(f"  Chargement BGE-M3...", end=" ", flush=True)
    embed_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    print("✓")

    print(f"  Connexion Qdrant...", end=" ", flush=True)
    qdrant = qdrant_client.QdrantClient("localhost", port=6333)
    print("✓")

    chunks_utiles = filtrer_chunks(charger_chunks())
    random.seed(SEED)
    random.shuffle(chunks_utiles)

    paires_test = []
    for i, chunk in enumerate(chunks_utiles[:10]):
        print(f"\n  [{i+1}/10] {chunk.get('source','?')[:40]}...",
              end=" ", flush=True)

        connexes = recuperer_connexes(chunk["text"], embed_model, qdrant)
        source   = chunk.get("metadata", {}).get("source_file",
                   chunk.get("source", "?"))
        contexte = f"[Source: {source}]\n{chunk['text']}"
        for c in connexes:
            contexte += f"\n\n---\n\n{c}"
        contexte = contexte[:4000]

        prompt = PROMPT_GENERATION.format(n=2, contexte=contexte)

        for tentative in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_GEN,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.4,
                        max_output_tokens=3000,
                        thinking_config=types.ThinkingConfig(
                            thinking_level="LOW"
                        ),
                    ),
                )
                contenu = response.text.strip()
                if "```" in contenu:
                    for partie in contenu.split("```"):
                        partie = partie.strip()
                        if partie.startswith("json"):
                            partie = partie[4:]
                        if partie.strip().startswith("["):
                            contenu = partie.strip()
                            break
                paires = json.loads(contenu)
                paires_test.extend(paires)
                print(f"✓ +{len(paires)}")
                break
            except Exception as e:
                if "503" in str(e):
                    print(f"\n    503 — attente {30*(tentative+1)}s...",
                          end=" ")
                    time.sleep(30 * (tentative + 1))
                else:
                    print(f"✗ {str(e)[:60]}")
                    break

        time.sleep(1)

    print(f"\n  Total paires générées : {len(paires_test)}")

    # Aperçu
    for p in paires_test[:3]:
        print(f"\n  [{p.get('type','?')}] {p.get('input','')[:100]}...")
        print(f"  → {p.get('output','')[:100]}...")

    # Sauvegarder pour vérification
    test_out = BATCH_DIR / "test_paires.jsonl"
    with open(test_out, "w", encoding="utf-8") as f:
        for p in paires_test:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n  ✓ Résultats sauvegardés → {test_out}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare",  action="store_true")
    parser.add_argument("--submit",   action="store_true")
    parser.add_argument("--status",   action="store_true")
    parser.add_argument("--collect",  action="store_true")
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--test",     action="store_true")
    parser.add_argument("--n",        type=int, default=N_PAIRES_CIBLE)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not args.status:
        print("❌ GEMINI_API_KEY manquante dans .env")
        return

    if args.test:
        mode_test()
    elif args.prepare or args.all:
        preparer_batch(args.n)
        if args.all:
            soumettre_batch()
    elif args.submit:
        soumettre_batch()
    elif args.status:
        verifier_statut()
    elif args.collect:
        collecter_resultats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()