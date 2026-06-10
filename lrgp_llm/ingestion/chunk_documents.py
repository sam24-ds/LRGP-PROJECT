"""
chunk_documents.py
Chunking des DoclingDocuments avec HybridChunker
et indexation dans Qdrant via BGE-M3.

Usage : python ingestion/chunk_documents.py
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.WARNING)

# ── Docling ───────────────────────────────────────────────────────
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

# ── LlamaIndex + Docling chunker ──────────────────────────────────
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.core import Document as LIDocument
from llama_index.core.schema import TextNode

# ── Qdrant ────────────────────────────────────────────────────────
import qdrant_client
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, SparseVector,
    SparseVectorParams, SparseIndexParams,
    NamedVector, NamedSparseVector,
)

# ── BGE-M3 ────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import re
from collections import Counter

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
BASE_DIR    = Path(__file__).parent  # dossier ingestion/
PARSED_DIR  = BASE_DIR / "data" / "parsed"
CHUNKS_DIR  = BASE_DIR / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


QDRANT_HOST       = "localhost"
QDRANT_PORT       = 6333
COLLECTION_NAME   = "lrgp_corpus"
DENSE_DIM         = 1024     # dimension BGE-M3 dense
BATCH_SIZE        = 32       # chunks par batch pour les embeddings
CHUNK_MAX_TOKENS  = 512      # taille max chunk enfant
CHUNK_OVERLAP     = 0        # HybridChunker gère l'overlap via hiérarchie


def charger_bgem3() -> SentenceTransformer:
    """Charge BGE-M3 via sentence-transformers sur GPU."""
    print("  Chargement BGE-M3...", end=" ", flush=True)
    model = SentenceTransformer(
        "BAAI/bge-m3",
        device="cuda",
    )
    print("✓")
    return model


def _sparse_from_text(text: str) -> dict:
    """
    Vecteur sparse basé sur TF-IDF simplifié.
    Approximation du lexical weights natif de BGE-M3.
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    tf     = Counter(tokens)
    total  = max(sum(tf.values()), 1)

    indices, values = [], []
    seen = set()
    for token, freq in tf.items():
        idx = abs(hash(token)) % 50000
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
            values.append(round(freq / total, 6))

    return {"indices": indices, "values": values}


def creer_collection_qdrant(client: qdrant_client.QdrantClient) -> None:
    """Crée la collection Qdrant avec vecteurs dense + sparse."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"  Collection '{COLLECTION_NAME}' existe déjà — skip")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"  ✓ Collection '{COLLECTION_NAME}' créée")


def nettoyer_formule(orig: str) -> str:
    """
    Nettoie le texte brut d'une équation extraite depuis 'orig'.
    Supprime les espaces multiples et préfixe avec [Équation].
    """
    import re
    cleaned = re.sub(r'\s+', ' ', orig).strip()
    return f"[Équation] {cleaned}" if cleaned else ""



def extraire_texte_item(item) -> str:
    """
    Retourne le meilleur texte disponible pour un élément Docling.
    Priorité : text → orig (fallback pour équations vides)
    """
    text = ""
    orig = ""

    # Récupérer text
    if hasattr(item, 'text') and item.text:
        text = item.text.strip()

    # Récupérer orig
    if hasattr(item, 'orig') and item.orig:
        orig = item.orig.strip()

    # Cas normal : text disponible
    if text:
        return text

    # Fallback : text vide mais orig disponible
    # → typiquement une équation avec fractions complexes
    if orig and orig != text:
        label = ""
        if hasattr(item, 'label'):
            label = str(item.label.value) if hasattr(item.label, 'value') else str(item.label)

        if 'formula' in label.lower() or 'equation' in label.lower():
            return nettoyer_formule(orig)
        else:
            # Pour les autres types, retourner orig brut
            return orig

    return ""


def chunker_document(json_path: Path) -> list[dict]:
    """
    Charge un DoclingDocument JSON et le découpe en chunks.
    """
    # ── Lire le JSON brut ─────────────────────────────────────────
    with open(json_path, encoding="utf-8") as f:
        doc_data = json.load(f)

    doc = DoclingDocument.model_validate(doc_data)

    # ── Patcher les formules vides avec orig ──────────────────────
    n_formules_fallback = 0
    for item in doc.texts:
        label = str(item.label.value) if hasattr(item.label, 'value') else str(item.label)
        if not item.text.strip() and hasattr(item, 'orig') and item.orig.strip():
            texte_fallback = extraire_texte_item(item)
            if texte_fallback:
                item.text = texte_fallback
                if 'formula' in label.lower():
                    n_formules_fallback += 1

    # ── DoclingNodeParser attend le JSON sérialisé ────────────────
    # Pas le texte exporté — le JSON complet du DoclingDocument
    parser = DoclingNodeParser(max_tokens=CHUNK_MAX_TOKENS)

    # Créer le document LlamaIndex avec le JSON brut comme contenu
    json_str = doc.model_dump_json()

    li_doc = LIDocument(
        text=json_str,                    # ← JSON brut, pas export_to_text()
        metadata={
            "source":   json_path.stem,
            "doc_name": doc_data.get("name", json_path.stem),
        }
    )

    nodes  = parser.get_nodes_from_documents([li_doc])
    chunks = []

    for i, node in enumerate(nodes):
        text = node.get_content().strip()
        if not text or len(text) < 50:
            continue

        meta  = node.metadata or {}
        chunk = {
            "chunk_id":    f"{json_path.stem}__chunk_{i:04d}",
            "source":      json_path.stem + ".pdf",
            "text":        text,
            "n_chars":     len(text),
            "chunk_index": i,
            "metadata": {
                "source_file":       json_path.stem,
                "section":           meta.get("section", ""),
                "page":              meta.get("page_no", 0),
                "chunk_type":        meta.get("type", "text"),
                "formules_fallback": n_formules_fallback,
            }
        }
        chunks.append(chunk)

    return chunks


def encoder_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
) -> list[dict]:
    """
    Encode les chunks avec BGE-M3 dense + sparse approximé.
    Dense  : vrai BGE-M3 1024d via sentence-transformers
    Sparse : TF approximé (pas les lexical weights natifs)
    """
    textes = [c["text"] for c in chunks]

    dense_vecs = model.encode(
        textes,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    for i, chunk in enumerate(chunks):
        chunk["dense_vector"]  = dense_vecs[i].tolist()
        chunk["sparse_vector"] = _sparse_from_text(chunk["text"])

    return chunks


def indexer_dans_qdrant(
    chunks: list[dict],
    client: qdrant_client.QdrantClient,
    offset: int,
) -> int:
    """
    Indexe les chunks dans Qdrant.
    Retourne le nouvel offset (pour les IDs uniques).
    """
    points = []
    for i, chunk in enumerate(chunks):
        point = PointStruct(
            id=offset + i,
            vector={
                "dense":  chunk["dense_vector"],
                "sparse": SparseVector(
                    indices=chunk["sparse_vector"]["indices"],
                    values=chunk["sparse_vector"]["values"],
                ),
            },
            payload={
                "chunk_id":    chunk["chunk_id"],
                "source":      chunk["source"],
                "text":        chunk["text"],
                "n_chars":     chunk["n_chars"],
                "chunk_index": chunk["chunk_index"],
                **chunk["metadata"],
            }
        )
        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    return offset + len(chunks)

def indexer_par_lots(chunks, client, offset, lot=200):
    """
    Indexe les chunks dans Qdrant par lots de `lot` points.
    Évite le WinError 10053 sur les gros documents.
    """
    from qdrant_client.models import PointStruct, SparseVector

    total = len(chunks)
    for start in range(0, total, lot):
        batch = chunks[start:start + lot]
        points = []
        for i, chunk in enumerate(batch):
            point = PointStruct(
                id=offset + start + i,
                vector={
                    "dense": chunk["dense_vector"],
                    "sparse": SparseVector(
                        indices=chunk["sparse_vector"]["indices"],
                        values=chunk["sparse_vector"]["values"],
                    ),
                },
                payload={
                    "chunk_id":    chunk["chunk_id"],
                    "source":      chunk["source"],
                    "text":        chunk["text"],
                    "n_chars":     chunk["n_chars"],
                    "chunk_index": chunk["chunk_index"],
                    **chunk["metadata"],
                }
            )
            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        done = min(start + lot, total)
        print(f"    Indexé {done}/{total} points...", end="\r", flush=True)

    print()
    return offset + total

def sauvegarder_chunks_jsonl(chunks: list[dict], source: str) -> None:
    """Sauvegarde les chunks en JSONL dans data/chunks/."""
    out = CHUNKS_DIR / (source + "_chunks.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # Ne pas sauvegarder les vecteurs (trop volumineux)
            chunk_sans_vecteurs = {
                k: v for k, v in chunk.items()
                if k not in ("dense_vector", "sparse_vector")
            }
            f.write(json.dumps(chunk_sans_vecteurs, ensure_ascii=False) + "\n")

def sauvegarder_chunks_jsonl_safe(chunks, nom):
    """Sauvegarde JSONL avec nom de fichier tronqué si trop long."""
    # Tronquer le nom à 100 caractères max pour éviter les chemins trop longs
    nom_safe = nom[:100].strip()
    out = CHUNKS_DIR / (nom_safe + "_chunks.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for chunk in chunks:
            chunk_sans_vecteurs = {
                k: v for k, v in chunk.items()
                if k not in ("dense_vector", "sparse_vector")
            }
            f.write(json.dumps(chunk_sans_vecteurs, ensure_ascii=False) + "\n")
    print(f"  ✓ JSONL sauvegardé : {out.name}")


def main() -> None:
    # ── Collecter les JSONs à traiter ────────────────────────────
    jsons = sorted([
        p for p in PARSED_DIR.glob("*.json")
        if p.name != "parsing_report.json"
        and p.name != "bibliography.json"
    ])

    # Charger le rapport de parsing — ne traiter que les OK
    rapport_path = PARSED_DIR / "parsing_report.json"
    with open(rapport_path, encoding="utf-8") as f:
        rapport = json.load(f)

    ok_stems = {
        Path(k).stem
        for k, v in rapport.items()
        if isinstance(v, dict) and v.get("statut") == "ok"
    }

    jsons = [j for j in jsons if j.stem in ok_stems]

    print(f"\n{'═'*62}")
    print(f"  CHUNKING + INDEXATION QDRANT")
    print(f"{'═'*62}")
    print(f"  Documents à traiter  : {len(jsons)}")
    print(f"  Collection Qdrant    : {COLLECTION_NAME}")
    print(f"  Max tokens / chunk   : {CHUNK_MAX_TOKENS}")

    # ── Vérifier chunks déjà traités (reprise) ───────────────────
    chunks_existants = {p.stem.replace("_chunks", "") for p in CHUNKS_DIR.glob("*.jsonl")}
    jsons_a_traiter  = [j for j in jsons if j.stem not in chunks_existants]
    skips = len(jsons) - len(jsons_a_traiter)
    print(f"  Déjà traités (skip)  : {skips}")
    print(f"  À traiter            : {len(jsons_a_traiter)}")
    print(f"{'─'*62}")

    if not jsons_a_traiter:
        print("  ✓ Tout est déjà indexé.")
        return

    # ── Charger BGE-M3 ───────────────────────────────────────────
    model = charger_bgem3()

    # ── Connecter Qdrant ─────────────────────────────────────────
    print(f"  Connexion Qdrant {QDRANT_HOST}:{QDRANT_PORT}...", end=" ")
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("✓")
    creer_collection_qdrant(client)

    # Récupérer l'offset courant pour les IDs
    info   = client.get_collection(COLLECTION_NAME)
    offset = info.points_count

    # ── Traitement ───────────────────────────────────────────────
    stats = {"docs": 0, "chunks": 0, "erreurs": 0}
    t_debut = time.time()

    for i, json_path in enumerate(jsons_a_traiter, 1):
        print(f"  [{i:4d}/{len(jsons_a_traiter)}] {json_path.stem[:52]:<52}",
              end=" ", flush=True)
        t0 = time.time()

        try:
            # 1. Chunking
            chunks = chunker_document(json_path)
            if not chunks:
                print(f"⚠ 0 chunks")
                continue

            # 2. Encoding BGE-M3
            chunks = encoder_chunks(chunks, model)

            # 3. Indexation Qdrant
            offset = indexer_dans_qdrant(chunks, client, offset)

            # 4. Sauvegarde JSONL
            sauvegarder_chunks_jsonl(chunks, json_path.stem)

            duree = time.time() - t0
            stats["docs"]   += 1
            stats["chunks"] += len(chunks)
            print(f"✓ {len(chunks):3d} chunks  {duree:.1f}s")

        except Exception as e:
            stats["erreurs"] += 1
            print(f"✗ {str(e)[:60]}")

        # ETA toutes les 50 docs
        if i % 50 == 0:
            elapsed = time.time() - t_debut
            eta_min = (elapsed / i) * (len(jsons_a_traiter) - i) / 60
            print(f"\n  {'─'*58}")
            print(f"  {i}/{len(jsons_a_traiter)} docs | "
                  f"{stats['chunks']:,} chunks | ETA ~{eta_min:.0f} min")
            print(f"  {'─'*58}\n")

    # ── Rapport final ────────────────────────────────────────────
    elapsed = time.time() - t_debut
    info    = client.get_collection(COLLECTION_NAME)

    print(f"\n{'═'*62}")
    print(f"  RAPPORT FINAL CHUNKING")
    print(f"{'═'*62}")
    print(f"  Documents traités    : {stats['docs']}")
    print(f"  Chunks créés         : {stats['chunks']:,}")
    print(f"  Erreurs              : {stats['erreurs']}")
    print(f"  Points dans Qdrant   : {info.points_count:,}")
    print(f"  Durée totale         : {elapsed/3600:.1f}h")
    print(f"  Vitesse moyenne      : "
          f"{elapsed/max(1,stats['docs']):.1f}s/doc")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()