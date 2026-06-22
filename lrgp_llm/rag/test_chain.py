# rag/test_chain.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chain import LRGPChain

# ── Test avec OpenAI (si tu as une clé) ──────────────────────────
chain = LRGPChain(
    llm_backend = "ollama",
    model_name  = "qwen3.5:9b",
    temperature = 0.1,
    verbose     = True,
)

questions = [
    "Comment estimer l'importance des effets de couplage de flux dans le dimensionnement d'un procédé de séparation de gaz par membranes?",
]

for q in questions:
    response = chain.ask(q)
    response.afficher()