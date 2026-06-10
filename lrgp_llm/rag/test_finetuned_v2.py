import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ollama
from rag.retriever import LRGPRetriever
from rag.prompts import SYSTEM_LRGP

retriever = LRGPRetriever(top_k_retrieve=20, top_k_rerank=5)

question = """À 25°C, une membrane d'osmose inverse est utilisée pour une solution d'alimentation en NaCl contenant 2,5 g/L (2,5 kg/m³, ρ = 999 kg/m³). La constante de perméabilité à l'eau est Aw = 4,81 × 10⁻⁴ kg/(s·m²·atm) et la constante de perméabilité du soluté est As = 4,42 × 10⁻⁷ m/s. Calculer le flux d'eau (Nw), le flux de soluté (Ns), la rétention R et la concentration C2 du perméat. Donnée : ΔP = 27,20 atm."""

print(f"Question : {question[:80]}...")

# Retrieval
sources  = retriever.retrieve(question)
contexte = retriever.format_context(sources)
print(f"Sources récupérées : {[s.source[:35] for s in sources[:3]]}")

# Génération V2
response = ollama.chat(
    model="lrgp-knowledge_v2",
    messages=[
        {"role": "system", "content": SYSTEM_LRGP},
        {"role": "user",   "content": f"Contexte :\n{contexte}\n\nQuestion : {question}"},
    ],
    options={"temperature": 0.1, "think": False, "num_predict": 1500},
)
print(f"\nRéponse V2 + RAG :")
print(response.message.content)