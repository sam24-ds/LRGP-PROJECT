# test_finetuned_v2.py
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ollama
from rag.retriever import LRGPRetriever
from rag.prompts import PROMPT_RAG, PROMPT_CALCUL, SYSTEM_LRGP

retriever = LRGPRetriever(top_k_retrieve=20, top_k_rerank=5)

questions = [
    ("Qu'est-ce que le coefficient de transfert global K_OV ?", "FACTUEL"),
    ("Calcule le flux de CO2, membrane PDMS 100µm, Δp=10 kPa.", "CALCUL"),
    ("Quelle est la perméabilité du CO2 pour une membrane PDMS ?", "FACTUEL"),
]

for model_name in ["qwen3.5:9b", "lrgp-expert"]:
    print(f"\n{'═'*60}")
    print(f"  MODÈLE : {model_name}")
    print(f"{'═'*60}")

    for question, qtype in questions:
        print(f"\n  Q [{qtype}]: {question[:60]}")

        # Retrieval
        sources  = retriever.retrieve(question)
        contexte = retriever.format_context(sources)

        # Génération directe via ollama
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system",
                 "content": SYSTEM_LRGP},
                {"role": "user",
                 "content": f"Contexte :\n{contexte}\n\nQuestion : {question}"},
            ],
            options={"temperature": 0.1, "think": False},
        )
        answer = response.message.content
        print(f"  R ({len(answer)} chars): {answer[:400]}")
        print(f"  Sources: {[s.source[:35] for s in sources[:2]]}")