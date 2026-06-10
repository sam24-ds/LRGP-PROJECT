"""
generate_dataset_v3.py
Génère de nouvelles paires no-context depuis le corpus brut LRGP.
100% no-context — pas de contexte RAG dans les paires.
Bonnes pratiques API Gemini Batch appliquées (File API, Safety OFF, JSON mode).

Usage :
    python evaluation/generate_dataset_v3.py --prepare
    python evaluation/generate_dataset_v3.py --submit
    python evaluation/generate_dataset_v3.py --status
    python evaluation/generate_dataset_v3.py --collect
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ── Chemins ───────────────────────────────────────────────────────
CORPUS_DIR    = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\ingestion\\data\\chunks")
BATCH_DIR     = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\batch_v3")
BATCH_DIR.mkdir(parents=True, exist_ok=True)

REQUESTS_FILE  = BATCH_DIR / "v3_requests_2.jsonl"
SOURCES_FILE   = BATCH_DIR / "v3_sources_2.jsonl"   # dictionnaire de liaison
JOB_STATE      = BATCH_DIR / "v3_job_state_2.json"
OUTPUT_RAW     = BATCH_DIR / "v3_raw_2.jsonl"

MODEL      = "gemini-3.1-pro-preview"
N_ARTICLES = 300
BATCH_SIZE = 4  # Ajusté à 4 pour garantir assez de tokens avec le <think>

INSTRUCTION_NC = (
    "Tu es un expert en génie des procédés au LRGP Nancy. "
    "Réponds directement depuis tes connaissances scientifiques. "
    "Pour les calculs, détaille toutes les étapes avec les unités. "
    "Cite les références si tu les connais."
)

TRACES_CONTEXTE =[
    "d'après le contexte", "selon le contexte",
    "le contexte fourni", "dans le document fourni",
    "le texte mentionne", "d'après le texte",
    "comme indiqué dans le document", "le document indique",
    "dans le contexte", "selon le document",
    "le texte ci-dessus", "d'après les documents",
    "les documents fournis",
]

PROMPT_V3 = """Tu es un expert en génie des procédés et séparation membranaire au LRGP Nancy.

À partir du contenu scientifique ci-dessous, génère {n} paires question/réponse
que tu pourrais répondre DE MÉMOIRE — comme un expert qui connaît ce domaine
sans avoir le document sous les yeux.

Règles STRICTES :
- Les réponses doivent être AUTONOMES — ne pas mentionner "le document", "le contexte"
- Pour CALCUL : bloc <think> obligatoire avec étapes détaillées (<think>......</think>)
- Pour FACTUEL : réponse directe et précise
- Citer [Source: nom_article] si pertinent
- 60% CALCUL/COMPARAISON avec <think>, 40% FACTUEL sans
- Échapper correctement les sauts de ligne (\\n) pour ne pas casser le JSON.

Format JSON obligatoire (Renvoie UNIQUEMENT un tableau d'objets) :
[{{
  "input": "Question : [question précise]",
  "output": "[réponse autonome]",
  "type": "CALCUL|FACTUEL|COMPARAISON",
  "domaine": "[domaine technique]",
  "qualite_estimee": 5
}}]

Contenu scientifique :
{contenu}"""


# ══════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════
def charger_articles() -> dict[str, str]:
    """Charge les articles depuis les chunks — regroupe par source."""
    articles = {}
    MOTS_CLES =[
        "membrane", "CO2", "CH4", "perméabilité", "flux",
        "transfert", "K_OV", "contacteur", "hollow fiber",
        "adsorption", "absorption", "osmose", "Barrer",
    ]
    for f in CORPUS_DIR.glob("*.jsonl"):
        try:
            with open(f, encoding="utf-8") as fp:
                chunks = [json.loads(l) for l in fp if l.strip()]
            if not chunks:
                continue
            texte = "\n\n".join(
                c.get("text","") for c in chunks[:3]
                if len(c.get("text","")) >= 200
            )
            if (len(texte) >= 500 and
                any(kw.lower() in texte.lower() for kw in MOTS_CLES)):
                articles[f.stem[:50]] = texte[:5000]
        except Exception:
            pass
    return articles


def contient_trace(output: str) -> bool:
    o = output.lower()
    return any(t in o for t in TRACES_CONTEXTE)


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — PRÉPARER
# ══════════════════════════════════════════════════════════════════
def preparer():
    print(f"\n{'═'*60}")
    print(f"  ÉTAPE 1 — Préparation requêtes V3 (no-context)")
    print(f"{'═'*60}")

    articles = charger_articles()
    print(f"  Articles disponibles : {len(articles)}")

    random.seed(42)
    sources = list(articles.keys())
    random.shuffle(sources)
    sources = sources[:N_ARTICLES]

    requetes       = []
    sources_dict   =[]   # pour la liaison par custom_id

    # Désactivation des filtres de sécurité (évite les erreurs sur le vocabulaire scientifique)
    safety_settings =[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    for i, source in enumerate(sources):
        contenu = articles[source]
        prompt  = PROMPT_V3.format(n=BATCH_SIZE, contenu=contenu)
        
        custom_id = f"v3_{i:04d}"

        # STRUCTURE FILE API (Au lieu de la liste inline)
        requetes.append({
            "custom_id": custom_id,
            "request": {
                "contents": [{"parts":[{"text": prompt}], "role": "user"}],
                "generation_config": {
                    "temperature": 0.3,
                    "max_output_tokens": 6000, 
                    "response_mime_type": "application/json", # FORCE LE FORMAT JSON
                    "thinking_config": {"thinking_level": "LOW"},
                },
                "safety_settings": safety_settings
            }
        })

        # Sauvegarde de la source associée à l'ID
        sources_dict.append({
            "custom_id": custom_id,
            "source": source,
        })

    # Sauvegarder les requêtes
    with open(REQUESTS_FILE, "w", encoding="utf-8") as f:
        for r in requetes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Sauvegarder les sources
    with open(SOURCES_FILE, "w", encoding="utf-8") as f:
        for s in sources_dict:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    cout = len(requetes) * BATCH_SIZE * 6000 * 1.0/1e6
    print(f"  ✓ {len(requetes)} requêtes → {REQUESTS_FILE}")
    print(f"  Coût estimé batch   : ~${cout:.2f}")
    print(f"  Paires attendues    : ~{len(requetes)*BATCH_SIZE}")
    print(f"\n  Lance : python generate_dataset_v3.py --submit")


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — SOUMETTRE
# ══════════════════════════════════════════════════════════════════
def soumettre():
    print(f"\n{'═'*60}")
    print(f"  ÉTAPE 2 — Soumission batch V3 (File API)")
    print(f"{'═'*60}")

    if not REQUESTS_FILE.exists():
        print("❌ Fichier de requêtes introuvable. Lancez --prepare.")
        return

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        print("  1. Upload du fichier JSONL...", end=" ", flush=True)
        uploaded_file = client.files.upload(
            file=str(REQUESTS_FILE),
            config={'mime_type': 'application/jsonl'}
        )
        print("✓")

        print("  2. Création du Job Batch...", end=" ", flush=True)
        job = client.batches.create(
            model=MODEL,
            src=uploaded_file.name,
            config={"display_name": f"lrgp-v3-no-context-{int(time.time())}"},
        )
        print("✓")

        state = {
            "job_name":     job.name,
            "file_name":    uploaded_file.name,
            "submitted_at": time.time(),
        }
        with open(JOB_STATE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        print(f"\n  ✓ Batch soumis : {job.name}")
        print(f"  Statut : {job.state}")
        print(f"  Lance python generate_dataset_v3.py --status pour suivre")

    except Exception as e:
        print(f"\n❌ Erreur : {e}")


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
            print(f"  Requêtes : total={rc.total} | ok={rc.succeeded} | échec={rc.failed}")
        return str(job.state)
    except Exception as e:
        print(f"❌ {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — COLLECTER
# ══════════════════════════════════════════════════════════════════
def collecter():
    print(f"\n{'═'*60}")
    print(f"  ÉTAPE 4 — Collecte résultats V3")
    print(f"{'═'*60}")

    st = statut()
    if "SUCCEEDED" not in st:
        print(f"\n  ⚠ Batch non terminé ({st})")
        return

    with open(JOB_STATE, encoding="utf-8") as f:
        state = json.load(f)

    # 1. Charger la map des sources {custom_id: source}
    source_map = {}
    with open(SOURCES_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                source_map[data["custom_id"]] = data["source"]

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    job    = client.batches.get(name=state["job_name"])

    # 2. Identifier l'identifiant du fichier de sortie (Adapté au dernier SDK Google)
    file_name = None
    if getattr(job, 'dest', None) and getattr(job.dest, 'file_name', None):
        file_name = job.dest.file_name
    elif getattr(job, 'output_uri', None):
        file_name = "files/" + job.output_uri.split("/files/")[-1]

    if not file_name:
        print("❌ Identifiant de fichier introuvable. Dump pour analyse :")
        print(job.model_dump_json(indent=2))
        return

    print(f"\n  Fichier cible Google : {file_name}")
    content_bytes = b""

    # 3. Essai 1 : SDK Python 
    try:
        print("  Téléchargement via SDK...", end=" ", flush=True)
        # On essaie le SDK. Si Pydantic fait une erreur de keyword argument, on passera au REST
        content_bytes = client.files.download(file=file_name)
        print("✓ Succès !")
    except Exception as e:
        print(f"✗ Échec du SDK ({e})")
        
        # 4. Essai 2 : Fallback API REST Direct (100% fiable)
        print("  Récupération des métadonnées (API REST)...", end=" ", flush=True)
        import requests
        headers = {"x-goog-api-key": os.getenv("GEMINI_API_KEY")}
        meta_url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}"
        
        try:
            meta = requests.get(meta_url, headers=headers).json()
            if "error" in meta:
                print(f"❌ Erreur API REST : {meta['error']}")
            else:
                print("✓")
                # Google renvoie "uri" pour le lien de téléchargement direct
                dl_url = meta.get("uri") or meta.get("downloadUri")
                if dl_url:
                    print(f"  Téléchargement depuis le serveur Google...")
                    resp = requests.get(dl_url, headers=headers)
                    if resp.status_code == 200:
                        content_bytes = resp.content
                        print("  ✓ Succès du téléchargement REST !")
                    else:
                        print(f"  ❌ Erreur REST : {resp.status_code} - {resp.text}")
                else:
                    print(f"  ❌ L'API n'a pas fourni de lien de téléchargement :\n{meta}")
        except Exception as e2:
            print(f"  ❌ Erreur réseau : {e2}")

    if not content_bytes:
        print("\n❌ Impossible de récupérer les données. Annulation de l'extraction.")
        return

    # 5. Écriture du fichier brut récupéré
    results_file = BATCH_DIR / "v3_batch_results.jsonl"
    with open(results_file, "wb") as f:
        f.write(content_bytes)

    paires  =[]
    erreurs_json = 0
    erreurs_api = 0

    # 6. Extraction des données
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            res = json.loads(line)
            
            custom_id = res.get("custom_id")
            response = res.get("response")
            source = source_map.get(custom_id, "?")

            if not response or "candidates" not in response:
                print(f"  ⚠ {custom_id} : Erreur serveur ou blocage sécurité")
                erreurs_api += 1
                continue

            try:
                contenu = response["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                # Nettoyage Markdown de sécurité
                if "```" in contenu:
                    for p in contenu.split("```"):
                        p = p.strip()
                        if p.startswith("json"): p = p[4:]
                        if p.strip().startswith("["):
                            contenu = p.strip()
                            break
                            
                batch = json.loads(contenu)

                if isinstance(batch, dict): 
                    batch =[batch]

                for p in batch:
                    output = p.get("output","")

                    if contient_trace(output): continue
                    if len(output) < 100: continue

                    paires.append({
                        "instruction": INSTRUCTION_NC,
                        "input":       p.get("input",""),
                        "output":      output,
                        "type":        p.get("type","FACTUEL"),
                        "domaine":     p.get("domaine",""),
                        "qualite_estimee": p.get("qualite_estimee", 4),
                        "source":      f"v3_{source}",
                    })

            except Exception as e:
                erreurs_json += 1

    # 7. Affichage des Statistiques
    n_think = sum(1 for p in paires if "<think>" in p["output"])
    types_count = Counter(p["type"] for p in paires)

    print(f"\n  Bilan final :")
    print(f"  Paires extraites   : {len(paires)}")
    print(f"  Échecs sécurité/API: {erreurs_api}")
    print(f"  Erreurs JSON       : {erreurs_json}")
    print(f"  Avec <think>       : {n_think}/{max(1, len(paires))} ({n_think/max(len(paires),1)*100:.0f}%)")
    print(f"  Types              : {dict(types_count)}")

    with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
        for p in paires:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  ✓ Sauvegardé dans : {OUTPUT_RAW}")
    print(f"  Lance : python evaluation/fusion_v3.py")
# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--submit",  action="store_true")
    parser.add_argument("--status",  action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not args.status:
        print("❌ GEMINI_API_KEY manquante dans .env")
        return

    if args.prepare:   preparer()
    elif args.submit:  soumettre()
    elif args.status:  statut()
    elif args.collect: collecter()
    else: parser.print_help()


if __name__ == "__main__":
    main()