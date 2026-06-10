"""
relancer_gros_docs.py
Rechunke les documents qui ont échoué — batch réduit pour les gros fichiers.
"""
import os, sys, json
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from chunk_documents import (
    charger_bgem3, creer_collection_qdrant,
    chunker_document, sauvegarder_chunks_jsonl_safe,
    indexer_par_lots, COLLECTION_NAME
)
import qdrant_client

NOMS = [
    "32.2",
    "AIChE Boucif et al 2007_docx",
    "Baker Membrane Technology Wiley 2004",
    "CES 2012 revised manuscript_docx",
    "DesignofGasPermeationSystemsChapter4",
    "MEMBRANE HANDBOOK 1992",
    "Model of Vapor-Liquid Equilibria for Aqueous Acid Gas-Alkanolamine Systems. 2. Representation of H2S and C02 Solubility in Aqueous MDEA and C02 Solubility in Aqueous Mixtures of MDEA",
]

PARSED_DIR = Path("data/parsed")
CHUNKS_DIR = Path("data/chunks")
BATCH_SIZE_GROS = 8   # batch réduit pour les gros documents

def encoder_chunks_gros(chunks, model):
    """Encodage par petits batches pour les gros documents."""
    import numpy as np
    import re
    from collections import Counter

    def sparse(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        from collections import Counter
        tf = Counter(tokens)
        total = max(sum(tf.values()), 1)
        indices, values, seen = [], [], set()
        for token, freq in tf.items():
            idx = abs(hash(token)) % 50000
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
                values.append(round(freq / total, 6))
        return {"indices": indices, "values": values}

    # Traiter par batches de BATCH_SIZE_GROS
    for start in range(0, len(chunks), BATCH_SIZE_GROS):
        batch = chunks[start:start + BATCH_SIZE_GROS]
        textes = [c["text"] for c in batch]
        vecs = model.encode(
            textes,
            batch_size=BATCH_SIZE_GROS,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        for i, chunk in enumerate(batch):
            chunk["dense_vector"]  = vecs[i].tolist()
            chunk["sparse_vector"] = sparse(chunk["text"])

        done = min(start + BATCH_SIZE_GROS, len(chunks))
        print(f"    Encodé {done}/{len(chunks)} chunks...", end="\r", flush=True)

    print()
    return chunks

# ── Main ──────────────────────────────────────────────────────────
print("\nChargement BGE-M3...")
model = charger_bgem3()

client = qdrant_client.QdrantClient("localhost", port=6333)
info   = client.get_collection(COLLECTION_NAME)
offset = info.points_count

print(f"Points Qdrant actuels : {offset:,}\n")

for nom in NOMS:
    json_path  = PARSED_DIR / (nom + ".json")
    jsonl_path = CHUNKS_DIR / (nom + "_chunks.jsonl")

    if not json_path.exists():
        print(f"❌ JSON introuvable : {nom[:50]}")
        continue

    if jsonl_path.exists():
        print(f"⏭ Déjà chunké : {nom[:50]}")
        continue

    taille_mo = json_path.stat().st_size / (1024*1024)
    print(f"\n[{nom[:55]}]  ({taille_mo:.1f} Mo)")

    try:
        # Chunking
        print(f"  Chunking...", end=" ", flush=True)
        chunks = chunker_document(json_path)
        print(f"✓ {len(chunks)} chunks")

        if not chunks:
            print(f"  ⚠ Aucun chunk produit")
            continue

        # Encodage avec batch réduit
        print(f"  Encodage BGE-M3 (batch={BATCH_SIZE_GROS})...")
        chunks = encoder_chunks_gros(chunks, model)

        # Indexation Qdrant
        print(f"  Indexation Qdrant...", end=" ", flush=True)
        offset = indexer_par_lots(chunks, client, offset, lot=200)
        print(f"✓")

        # Sauvegarde JSONL
        sauvegarder_chunks_jsonl_safe(chunks, nom)
        print(f"  ✓ Terminé — {len(chunks)} chunks indexés")

    except Exception as e:
        print(f"  ✗ Erreur : {str(e)[:100]}")

# Bilan final
info = client.get_collection(COLLECTION_NAME)
print(f"\n{'═'*50}")
print(f"  Points Qdrant final : {info.points_count:,}")
print(f"{'═'*50}\n")