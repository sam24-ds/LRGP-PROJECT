# rag/test_chain.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chain import LRGPChain

# ── Test avec OpenAI (si tu as une clé) ──────────────────────────
chain = LRGPChain(
    llm_backend = "openai",
    model_name  = "gpt-4o-mini",
    temperature = 0.1,
    verbose     = True,
)

questions = [
    "Quelle est la perméabilité du CO2 pour une membrane PDMS ?",
    "Calcule le flux de CO2 à travers une membrane PDMS d'épaisseur 100 µm avec une pression partielle de 10 kPa.",
]

for q in questions:
    response = chain.ask(q)
    response.afficher()