"""
test_retriever.py
Test du retriever hybride LRGP.
Usage : python rag/test_retriever.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import LRGPRetriever

retriever = LRGPRetriever(
    top_k_retrieve=20,
    top_k_rerank=5,
    use_reranker=True,
)

questions = [
    "Quelle est la perméabilité du CO2 pour une membrane PDMS ?",
    "Coefficient de transfert global K_OV membrane contacteur",
    "CO2 flux hollow fiber membrane contactor aqueous amine",
]

for question in questions:
    print(f"\n{'═'*65}")
    print(f"Question : {question}")
    print(f"{'─'*65}")

    results = retriever.retrieve(question)

    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Rerank: {r.rerank_score:.4f} | Dense: {r.score:.4f}")
        print(f"       Source  : {r.source[:55]}")
        print(f"       Section : {r.section[:60] if r.section else '—'}")
        print(f"       Texte   : {r.text[:180]}...")

    print(f"\n  Contexte formaté pour LLM ({len(retriever.format_context(results))} chars)")