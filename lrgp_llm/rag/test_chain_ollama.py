"""
test_chain.py
Test de la chaîne RAG complète LRGP avec Ollama.
Usage : python rag/test_chain.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chain import LRGPChain

# ── Initialiser la chaîne ─────────────────────────────────────────
print("Initialisation de la chaîne RAG LRGP...")
chain = LRGPChain(
    llm_backend    = "ollama",
    model_name     = "qwen3.5:9b",
    ollama_url     = "http://localhost:11434",
    top_k_retrieve = 20,
    top_k_rerank   = 5,
    temperature    = 0.3,
    verbose        = True,
)
print("✓ Chaîne prête\n")

# ── Questions de test ─────────────────────────────────────────────
questions = [
    # Factuel
    "Quelle est la perméabilité du CO2 pour une membrane PDMS ?",

    # Calcul
    "Calcule le flux de CO2 à travers une membrane PDMS "
    "d'épaisseur 100 µm avec une différence de pression partielle de 10 kPa.",

    # Comparaison
    "Quelle membrane est plus sélective pour CO2/CH4 : PDMS ou PEBA ?",
]

for question in questions:
    response = chain.ask(question)
    response.afficher()