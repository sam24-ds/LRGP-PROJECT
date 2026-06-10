"""
chunk_documents.py
Chunking des DoclingDocuments avec HybridChunker
et indexation dans Qdrant via BGE-M3.

VERSION 2 — Adaptée aux livres scientifiques :
  - Chunking adaptatif (articles 512 / livres 1024 tokens)
  - Classification du contenu (theory/exercise/example/definition/equation)
  - Filtrage des contenus inutiles (index, biblio, copyright)
  - Metadata enrichi (doc_type, content_category, has_equation)

Usage : python ingestion/chunk_documents.py
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional
from collections import Counter

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

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
BASE_DIR    = Path(__file__).parent
PARSED_DIR  = BASE_DIR / "data" / "parsed"
CHUNKS_DIR  = BASE_DIR / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


QDRANT_HOST       = "localhost"
QDRANT_PORT       = 6333
COLLECTION_NAME   = "lrgp_corpus"
DENSE_DIM         = 1024
BATCH_SIZE        = 32

# ── NOUVEAU : Chunking adaptatif selon le type de document ────────
CHUNK_MAX_TOKENS_ARTICLE = 512    # articles (papers, thèses < 80 pages)
CHUNK_MAX_TOKENS_LIVRE   = 1024   # livres (manuels, handbooks)

# Seuil de détection livre
LIVRE_SEUIL_PAGES        = 80     # >80 pages = livre

# Mots-clés dans le nom de fichier indiquant un livre
INDICES_LIVRE = [
    "mulder", "baker", "strathmann", "geankoplis", "perry", "treybal",
    "handbook", "encyclopedia", "textbook", "fundamentals", "principles",
    "introduction_to", "membrane_technology", "transport_phenomena",
    "unit_operations", "chemical_engineering", "process_engineering",
]


# ══════════════════════════════════════════════════════════════════
# DÉTECTION DU TYPE DE DOCUMENT
# ══════════════════════════════════════════════════════════════════
def detecter_type_document(doc_data: dict, json_path: Path) -> str:
    """
    Détecte si le document est un livre ou un article.
    
    Critères (OR) :
      1. Nombre de pages >= LIVRE_SEUIL_PAGES
      2. Nom du fichier contient un indice de livre
    
    Returns:
        "livre" ou "article"
    """
    # Critère 1 : nombre de pages
    n_pages = len(doc_data.get("pages", {}))
    if n_pages >= LIVRE_SEUIL_PAGES:
        return "livre"
    
    # Critère 2 : nom du fichier
    nom = json_path.stem.lower()
    if any(idx in nom for idx in INDICES_LIVRE):
        return "livre"
    
    return "article"


# ══════════════════════════════════════════════════════════════════
# CLASSIFICATION DU CONTENU
# ══════════════════════════════════════════════════════════════════
def classifier_contenu(text: str) -> str:
    """
    Classifie le contenu d'un chunk pour aider le Reranker.
    
    Returns:
      - exercise        : énoncé d'exercice
      - worked_example  : exemple résolu avec solution
      - definition      : encadré de définition
      - equation_block  : bloc dominé par les équations
      - theory          : texte théorique (défaut)
    """
    text_lower = text.lower()
    
    # Exercices
    if re.search(r'exercice\s+\d+|problem\s+\d+|exercise\s+\d+|problème\s+\d+', text_lower):
        return "exercise"
    
    # Exemples résolus
    if re.search(r'exemple\s+\d+|example\s+\d+|solution\s*[:.]|résolution\s*[:.]|worked example', text_lower):
        return "worked_example"
    
    # Définitions
    if re.search(r'définition\s*[:.]|definition\s*[:.]|est défini[e]?\s+comme|is defined as', text_lower):
        return "definition"
    
    # Equations dominantes (>2 marqueurs)
    if text.count("[Équation]") >= 2:
        return "equation_block"
    
    return "theory"


# ══════════════════════════════════════════════════════════════════
# FILTRE PRÉ-INDEXATION
# ══════════════════════════════════════════════════════════════════
def doit_etre_indexe(text: str, meta: dict) -> bool:
    """
    Filtre les contenus à ne PAS indexer (réduit le bruit).
    
    Returns False si le contenu doit être ignoré.
    """
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    
    # Trop court
    if len(text_stripped) < 80:
        return False
    
    # Index alphabétique (succession de termes courts)
    lines = [l.strip() for l in text_stripped.split("\n") if l.strip()]
    if len(lines) > 10:
        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len < 30:  # lignes très courtes = probable index/glossaire
            return False
    
    # Bibliographie / index / glossaire
    section = (meta.get("section") or "").lower()
    sections_a_exclure = [
        "references", "bibliography", "bibliographie", "index",
        "glossary", "glossaire", "nomenclature", "list of figures",
        "list of tables", "table des matières", "table of contents",
        "remerciements", "acknowledgments", "preface", "préface",
    ]
    if any(kw in section for kw in sections_a_exclure):
        return False
    
    # Copyright, ISBN, préface
    if any(kw in text_lower[:300] for kw in [
        "copyright ©", "all rights reserved", "isbn", "© 20",
        "reproduction interdite", "tous droits réservés"
    ]):
        return False
    
    # Liste de références bibliographiques (heuristique simple)
    # ex : ligne commençant par [1], [2]... ou Author, Year
    n_refs = len(re.findall(r'^\s*\[\d+\]|\(\d{4}\)\.\s', text_stripped, re.M))
    if n_refs > 5 and len(text_stripped) < 2000:
        return False
    
    return True


# ══════════════════════════════════════════════════════════════════
# FONCTIONS EXISTANTES (inchangées)
# ══════════════════════════════════════════════════════════════════
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
    """Vecteur sparse basé sur TF approximé."""
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
    """Nettoie le texte brut d'une équation extraite depuis 'orig'."""
    cleaned = re.sub(r'\s+', ' ', orig).strip()
    return f"[Équation] {cleaned}" if cleaned else ""


def extraire_texte_item(item) -> str:
    """Retourne le meilleur texte disponible pour un élément Docling."""
    text = ""
    orig = ""

    if hasattr(item, 'text') and item.text:
        text = item.text.strip()

    if hasattr(item, 'orig') and item.orig:
        orig = item.orig.strip()

    if text:
        return text

    if orig and orig != text:
        label = ""
        if hasattr(item, 'label'):
            label = str(item.label.value) if hasattr(item.label, 'value') else str(item.label)

        if 'formula' in label.lower() or 'equation' in label.lower():
            return nettoyer_formule(orig)
        else:
            return orig

    return ""


# ══════════════════════════════════════════════════════════════════
# CHUNKING — MODIFIÉ POUR ADAPTATIF
# ══════════════════════════════════════════════════════════════════
def chunker_document(json_path: Path) -> tuple[list[dict], dict]:
    """
    Charge un DoclingDocument JSON et le découpe en chunks.
    
    VERSION 2 — Chunking adaptatif + classification.
    
    Returns:
        (chunks, stats) où stats contient les compteurs par catégorie
    """
    # ── Lire le JSON brut ─────────────────────────────────────────
    with open(json_path, encoding="utf-8") as f:
        doc_data = json.load(f)

    doc = DoclingDocument.model_validate(doc_data)

    # ── NOUVEAU : détecter le type de document ────────────────────
    doc_type = detecter_type_document(doc_data, json_path)
    max_tokens = (
        CHUNK_MAX_TOKENS_LIVRE if doc_type == "livre"
        else CHUNK_MAX_TOKENS_ARTICLE
    )

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

    # ── Parser avec taille adaptée au type ────────────────────────
    parser = DoclingNodeParser(max_tokens=max_tokens)

    json_str = doc.model_dump_json()
    li_doc = LIDocument(
        text=json_str,
        metadata={
            "source":   json_path.stem,
            "doc_name": doc_data.get("name", json_path.stem),
        }
    )

    nodes  = parser.get_nodes_from_documents([li_doc])
    chunks = []
    
    # Stats par catégorie pour le rapport
    stats = {
        "doc_type":        doc_type,
        "max_tokens":      max_tokens,
        "total_nodes":     len(nodes),
        "filtres":         0,
        "trop_courts":     0,
        "indexes":         0,
        "categories":      Counter(),
    }

    for i, node in enumerate(nodes):
        text = node.get_content().strip()
        meta = node.metadata or {}
        
        # Filtre 1 : trop court
        if not text or len(text) < 50:
            stats["trop_courts"] += 1
            continue
        
        # Filtre 2 : NOUVEAU — contenu inutile
        if not doit_etre_indexe(text, meta):
            stats["filtres"] += 1
            continue
        
        # NOUVEAU : classification du contenu
        content_category = classifier_contenu(text)
        stats["categories"][content_category] += 1
        
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
                # NOUVEAUX CHAMPS
                "doc_type":          doc_type,
                "content_category":  content_category,
                "has_equation":      "[Équation]" in text,
            }
        }
        chunks.append(chunk)
        stats["indexes"] += 1

    return chunks, stats


# ══════════════════════════════════════════════════════════════════
# ENCODAGE / INDEXATION (inchangés)
# ══════════════════════════════════════════════════════════════════
def encoder_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
) -> list[dict]:
    """Encode les chunks avec BGE-M3 dense + sparse approximé."""
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
    """Indexe les chunks dans Qdrant."""
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
    """Indexe les chunks par lots de `lot` points (anti-WinError 10053)."""
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
    # Tronquer nom si trop long (anti-WindowsError)
    nom_safe = source[:100].strip()
    out = CHUNKS_DIR / (nom_safe + "_chunks.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for chunk in chunks:
            chunk_sans_vecteurs = {
                k: v for k, v in chunk.items()
                if k not in ("dense_vector", "sparse_vector")
            }
            f.write(json.dumps(chunk_sans_vecteurs, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
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
    print(f"  CHUNKING + INDEXATION QDRANT (V2 — adapté livres)")
    print(f"{'═'*62}")
    print(f"  Documents à traiter   : {len(jsons)}")
    print(f"  Collection Qdrant     : {COLLECTION_NAME}")
    print(f"  Chunk articles        : {CHUNK_MAX_TOKENS_ARTICLE} tokens")
    print(f"  Chunk livres          : {CHUNK_MAX_TOKENS_LIVRE} tokens")
    print(f"  Seuil détection livre : {LIVRE_SEUIL_PAGES} pages")

    # ── Vérifier chunks déjà traités (reprise) ───────────────────
    chunks_existants = {p.stem.replace("_chunks", "") for p in CHUNKS_DIR.glob("*.jsonl")}
    jsons_a_traiter  = [j for j in jsons if j.stem not in chunks_existants]
    skips = len(jsons) - len(jsons_a_traiter)
    print(f"  Déjà traités (skip)   : {skips}")
    print(f"  À traiter             : {len(jsons_a_traiter)}")
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

    info   = client.get_collection(COLLECTION_NAME)
    offset = info.points_count

    # ── Traitement ───────────────────────────────────────────────
    stats_global = {
        "docs":          0,
        "chunks":        0,
        "erreurs":       0,
        "livres":        0,
        "articles":      0,
        "filtres_total": 0,
        "categories":    Counter(),
    }
    t_debut = time.time()

    for i, json_path in enumerate(jsons_a_traiter, 1):
        print(f"  [{i:4d}/{len(jsons_a_traiter)}] {json_path.stem[:52]:<52}",
              end=" ", flush=True)
        t0 = time.time()

        try:
            # 1. Chunking (NOUVEAU : retourne stats)
            chunks, stats = chunker_document(json_path)
            
            doc_type_short = "📚" if stats["doc_type"] == "livre" else "📄"
            
            if not chunks:
                print(f"⚠ {doc_type_short} 0 chunks")
                continue

            # 2. Encoding BGE-M3
            chunks = encoder_chunks(chunks, model)

            # 3. Indexation Qdrant (par lots si gros)
            if len(chunks) > 200:
                offset = indexer_par_lots(chunks, client, offset, lot=200)
            else:
                offset = indexer_dans_qdrant(chunks, client, offset)

            # 4. Sauvegarde JSONL
            sauvegarder_chunks_jsonl(chunks, json_path.stem)

            duree = time.time() - t0
            stats_global["docs"]   += 1
            stats_global["chunks"] += len(chunks)
            stats_global["filtres_total"] += stats["filtres"]
            
            if stats["doc_type"] == "livre":
                stats_global["livres"] += 1
            else:
                stats_global["articles"] += 1
            
            stats_global["categories"].update(stats["categories"])
            
            # Affichage détaillé
            cats = stats["categories"]
            cat_summary = ""
            if cats.get("worked_example", 0) > 0:
                cat_summary += f" 💡{cats['worked_example']}"
            if cats.get("exercise", 0) > 0:
                cat_summary += f" 📝{cats['exercise']}"
            if cats.get("definition", 0) > 0:
                cat_summary += f" 📖{cats['definition']}"
            
            print(f"✓ {doc_type_short} {len(chunks):3d} chunks"
                  f" (filtré:{stats['filtres']}){cat_summary}"
                  f"  {duree:.1f}s")

        except Exception as e:
            stats_global["erreurs"] += 1
            print(f"✗ {str(e)[:60]}")

        # ETA toutes les 50 docs
        if i % 50 == 0:
            elapsed = time.time() - t_debut
            eta_min = (elapsed / i) * (len(jsons_a_traiter) - i) / 60
            print(f"\n  {'─'*58}")
            print(f"  {i}/{len(jsons_a_traiter)} docs | "
                  f"{stats_global['chunks']:,} chunks | "
                  f"📚 {stats_global['livres']} | "
                  f"📄 {stats_global['articles']} | "
                  f"ETA ~{eta_min:.0f} min")
            print(f"  {'─'*58}\n")

    # ── Rapport final ────────────────────────────────────────────
    elapsed = time.time() - t_debut
    info    = client.get_collection(COLLECTION_NAME)

    print(f"\n{'═'*62}")
    print(f"  RAPPORT FINAL CHUNKING")
    print(f"{'═'*62}")
    print(f"  Documents traités     : {stats_global['docs']}")
    print(f"    📚 Livres           : {stats_global['livres']}")
    print(f"    📄 Articles         : {stats_global['articles']}")
    print(f"  Chunks créés          : {stats_global['chunks']:,}")
    print(f"  Chunks filtrés (bruit): {stats_global['filtres_total']}")
    print(f"  Erreurs               : {stats_global['erreurs']}")
    print(f"  Points dans Qdrant    : {info.points_count:,}")
    print(f"  Durée totale          : {elapsed/3600:.1f}h")
    print(f"  Vitesse moyenne       : "
          f"{elapsed/max(1,stats_global['docs']):.1f}s/doc")
    
    # NOUVEAU : Répartition par catégorie de contenu
    if stats_global["categories"]:
        print(f"\n  Répartition par catégorie de contenu :")
        for cat, n in stats_global["categories"].most_common():
            emoji = {
                "theory":         "📚",
                "worked_example": "💡",
                "exercise":       "📝",
                "definition":     "📖",
                "equation_block": "🔢",
            }.get(cat, "  ")
            pct = 100 * n / stats_global["chunks"]
            print(f"    {emoji} {cat:<18} : {n:5d} chunks ({pct:.1f}%)")
    
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()