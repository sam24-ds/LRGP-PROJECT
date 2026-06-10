"""
test_retrieval_qdrant.py
Test de retrieval sur le corpus LRGP indexé dans Qdrant.
"""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import NamedVector, Query, NearestQuery

COLLECTION = "lrgp_corpus"

# ── Charger BGE-M3 ────────────────────────────────────────────────
print("Chargement BGE-M3...", end=" ", flush=True)
model = SentenceTransformer("BAAI/bge-m3", device="cuda")
print("✓")

# ── Connecter Qdrant ──────────────────────────────────────────────
client = qdrant_client.QdrantClient("localhost", port=6333)
info   = client.get_collection(COLLECTION)
print(f"Collection : {COLLECTION} — {info.points_count:,} points\n")

# ── Questions de test ─────────────────────────────────────────────
questions = [
    "Quelle est la perméabilité du CO2 pour une membrane PDMS ?",
    "What is the CO2 flux through a hollow fiber membrane contactor ?",
    "Coefficient de transfert global K_OV membrane",
]

for question in questions:
    print(f"{'─'*60}")
    print(f"Question : {question}\n")

    # Encoder la question
    q_vec = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()

    # ── Recherche dense (API v1.17+) ──────────────────────────────
    results = client.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        using="dense",
        limit=5,
    ).points

    print(f"Top 5 résultats (dense search) :\n")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score  : {r.score:.4f}")
        print(f"       Source : {r.payload.get('source_file', '?')[:55]}")
        print(f"       Page   : {r.payload.get('page', '?')}")
        print(f"       Texte  : {r.payload.get('text', '')[:180]}...")
        print()

print(f"{'─'*60}")
print("✓ Test retrieval terminé")