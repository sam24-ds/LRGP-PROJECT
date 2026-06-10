"""
document_grader.py
Évalue la pertinence des sources RAG (Boucle 1 — Self-Reflection).
"""
import json
import ollama
from srar_gp.state import SRARState

GRADER_MODEL = "qwen3.5:27b"


PROMPT_GRADER = """Tu évalues la pertinence de documents pour répondre à une question.

═══════════════════════════════════════════════════════════════
QUESTION : {question}

DOCUMENTS RAG TROUVÉS :
{contexte}
═══════════════════════════════════════════════════════════════

ÉVALUATION :
Les documents ci-dessus contiennent-ils les informations nécessaires
pour répondre PRÉCISÉMENT à la question ?

Critères :
- Pertinence thématique (sujet couvert)
- Précision des données (valeurs, formules, références)
- Couverture (essentiel de la question abordé)

Réponds STRICTEMENT en JSON :
{{
  "pertinent": true ou false,
  "raison": "explication courte (1 phrase)",
  "manque": "ce qui manque si non pertinent"
}}

JSON :"""


def grader_documents(state: SRARState) -> SRARState:
    """Évalue la pertinence des sources RAG via Qwen 27B."""
    print(f"\n  ┌─ [DOC_GRADER] Évaluation pertinence des documents...")
    
    sources = state.get("sources_rag", [])
    contexte = state.get("context_rag", "")
    
    if not sources or not contexte.strip():
        print(f"  │  → 0 source RAG → non pertinent (fallback Web)")
        return {
            "document_pertinent": False,
            "agents_actives": ["doc_grader_no_sources"],
        }
    
    try:
        response = ollama.chat(
            model=GRADER_MODEL,
            messages=[{
                "role": "user",
                "content": PROMPT_GRADER.format(
                    question=state["question"][:500],
                    contexte=contexte[:2500],
                )
            }],
            think=False,
            format="json",
            options={"temperature": 0.0, "num_predict": 400},
            keep_alive="5m",
        )
        
        data = json.loads(response.message.content)
        pertinent = bool(data.get("pertinent", False))
        raison = data.get("raison", "")
        manque = data.get("manque", "")
        
        print(f"  │  → Pertinent : {'OUI ✓' if pertinent else 'NON ✗'}")
        print(f"  │  → Raison : {raison[:120]}")
        if not pertinent and manque:
            print(f"  │  → Manque : {manque[:120]}")
        
        return {
            "document_pertinent": pertinent,
            "agents_actives": [f"doc_grader_{'ok' if pertinent else 'fail'}"],
        }
    
    except Exception as e:
        print(f"  │  ⚠ Erreur Grader : {str(e)[:100]} — considère pertinent par défaut")
        return {
            "document_pertinent": True,
            "agents_actives": ["doc_grader_error"],
        }