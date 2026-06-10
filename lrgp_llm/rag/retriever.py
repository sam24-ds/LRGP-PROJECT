"""
retriever.py
Pipeline de retrieval hybride pour le corpus LRGP.

Flux :
  Question
     ↓
  BGE-M3 encode (dense + sparse)
     ↓
  Qdrant hybrid search (dense + sparse) → top-20
     ↓
  bge-reranker-v2-m3 → top-5
     ↓
  Chunks contextualisés → LLM
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import qdrant_client
from qdrant_client.models import (
    NamedVector, NamedSparseVector,
    SparseVector, Prefetch, FusionQuery, Fusion,
)
from sentence_transformers import SentenceTransformer, CrossEncoder

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
COLLECTION      = "lrgp_corpus"
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333
TOP_K_RETRIEVE  = 20   # candidats avant reranking
TOP_K_RERANK    = 5    # résultats finaux après reranking
DENSE_MODEL     = "BAAI/bge-m3"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"


# ══════════════════════════════════════════════════════════════════
# DATACLASS — Résultat de retrieval
# ══════════════════════════════════════════════════════════════════
@dataclass
class RetrievalResult:
    chunk_id:    str
    source:      str
    text:        str
    score:       float
    rerank_score: float
    page:        int
    section:     str
    chunk_type:  str

    def format_for_llm(self) -> str:
        """Formate le chunk pour l'injection dans le prompt LLM."""
        header = f"[Source: {self.source} | Page: {self.page}]"
        if self.section:
            header += f"\n[Section: {self.section}]"
        return f"{header}\n{self.text}"


# ══════════════════════════════════════════════════════════════════
# RETRIEVER
# ══════════════════════════════════════════════════════════════════
class LRGPRetriever:
    """
    Retriever hybride dense + sparse avec reranking pour le LRGP.
    """

    def __init__(
        self,
        top_k_retrieve: int = TOP_K_RETRIEVE,
        top_k_rerank:   int = TOP_K_RERANK,
        use_reranker:   bool = True,
        device:         str = "cuda",
    ):
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank   = top_k_rerank
        self.use_reranker   = use_reranker
        self.device         = device

        # Connexion Qdrant
        self.client = qdrant_client.QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT
        )

        # Modèles — chargés à la première utilisation
        self._embedder  = None
        self._reranker  = None

    # ── Chargement paresseux des modèles ─────────────────────────
    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            print("  Chargement BGE-M3...", end=" ", flush=True)
            self._embedder = SentenceTransformer(
                DENSE_MODEL, device=self.device
            )
            print("✓")
        return self._embedder

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            print("  Chargement reranker...", end=" ", flush=True)
            self._reranker = CrossEncoder(
                RERANKER_MODEL, device=self.device
            )
            print("✓")
        return self._reranker

    # ── Encodage ─────────────────────────────────────────────────
    def _encode_dense(self, text: str) -> list[float]:
        vec = self.embedder.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0].tolist()

    def _encode_sparse(self, text: str) -> SparseVector:
        tokens = re.findall(r'\b\w+\b', text.lower())
        tf     = Counter(tokens)
        total  = max(sum(tf.values()), 1)
        indices, values, seen = [], [], set()
        for token, freq in tf.items():
            idx = abs(hash(token)) % 50000
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
                values.append(round(freq / total, 6))
        return SparseVector(indices=indices, values=values)

    # ── Recherche hybride dans Qdrant ─────────────────────────────
    def _hybrid_search(self, question: str) -> list:
        dense_vec  = self._encode_dense(question)
        sparse_vec = self._encode_sparse(question)

        # Hybrid search via Prefetch + Fusion RRF
        # RRF = Reciprocal Rank Fusion — combine les deux rankings
        try:
            results = self.client.query_points(
                collection_name=COLLECTION,
                prefetch=[
                    Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=self.top_k_retrieve,
                    ),
                    Prefetch(
                        query=sparse_vec,
                        using="sparse",
                        limit=self.top_k_retrieve,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=self.top_k_retrieve,
            ).points

        except Exception:
            # Fallback : dense uniquement si hybrid échoue
            results = self.client.query_points(
                collection_name=COLLECTION,
                query=dense_vec,
                using="dense",
                limit=self.top_k_retrieve,
            ).points

        return results

    # ── Reranking ─────────────────────────────────────────────────
    def _rerank(
        self,
        question: str,
        results: list,
    ) -> list[tuple]:
        """
        Reranke les résultats avec bge-reranker-v2-m3.
        Retourne les top_k_rerank meilleurs.
        """
        if not results:
            return []

        # Paires (question, chunk_text) pour le CrossEncoder
        pairs = [
            (question, r.payload.get("text", ""))
            for r in results
        ]

        scores = self.reranker.predict(pairs)

        # Trier par score décroissant
        ranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:self.top_k_rerank]

    # ── Interface principale ──────────────────────────────────────
    def retrieve(
        self,
        question: str,
        filter_source: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieval complet : hybrid search + reranking.

        Args:
            question      : question en langage naturel
            filter_source : filtrer par nom de source (optionnel)

        Returns:
            Liste de RetrievalResult triés par pertinence
        """
        # 1. Hybrid search
        raw_results = self._hybrid_search(question)

        if not raw_results:
            return []

        # 2. Filtre optionnel par source
        if filter_source:
            raw_results = [
                r for r in raw_results
                if filter_source.lower() in
                   r.payload.get("source_file", "").lower()
            ]

        # 3. Reranking
        if self.use_reranker and len(raw_results) > 1:
            ranked = self._rerank(question, raw_results)
        else:
            ranked = [(r, r.score) for r in raw_results]

        # 4. Construire les RetrievalResult
        output = []
        for point, rerank_score in ranked:
            p = point.payload
            output.append(RetrievalResult(
                chunk_id     = p.get("chunk_id", ""),
                source       = p.get("source_file", p.get("source", "")),
                text         = p.get("text", ""),
                score        = point.score,
                rerank_score = float(rerank_score),
                page         = p.get("page", 0),
                section      = p.get("section", ""),
                chunk_type   = p.get("chunk_type", "text"),
            ))
        # Filtrer les chunks sous le seuil de pertinence
        RERANK_THRESHOLD = 0.50
        output = [r for r in output if r.rerank_score >= RERANK_THRESHOLD]

        return output

    def format_context(self, results: list[RetrievalResult]) -> str:
        """
        Formate les résultats en contexte pour le prompt LLM.
        """
        if not results:
            return "Aucun document pertinent trouvé dans le corpus."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"--- Document {i} ---\n{r.format_for_llm()}")

        return "\n\n".join(parts)