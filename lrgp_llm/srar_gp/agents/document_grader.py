"""
document_grader.py
Évalue la pertinence des sources RAG (Boucle 1 — Self-Reflection).
"""
import json
import re
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
    
    contenu = ""  # Initialisation pour pouvoir l'afficher dans le bloc except
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
        
        contenu = response.message.content.strip()
        
        # ─── NETTOYAGE DU TEXTE POUR ÉVITER LES ERREURS JSON ───
        # 1. Retirer les balises <think> (si le modèle "réfléchit" malgré think=False)
        contenu = re.sub(r"<think>.*?</think>", "", contenu, flags=re.DOTALL).strip()
        
        # 2. Retirer les balises Markdown (ex: ```json ... ```)
        if "```" in contenu:
            match = re.search(r"```(?:json)?(.*?)```", contenu, re.DOTALL)
            if match:
                contenu = match.group(1).strip()
                
        # 3. Sécurité contre les réponses vides (qui causent "Expecting value: char 0")
        if not contenu:
            raise ValueError("Le modèle a renvoyé une chaîne vide.")
        # ───────────────────────────────────────────────────────
        
        data = json.loads(contenu)
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
    
    except json.JSONDecodeError as e:
        print(f"  │  ⚠ Erreur Parsing JSON : {str(e)}")
        print(f"  │  → Texte reçu : {contenu[:100]}...")
        print(f"  │  → Considéré NON pertinent par sécurité (Web Search déclenchée)")
        return {
            "document_pertinent": False,
            "agents_actives": ["doc_grader_error"],
        }
        
    except Exception as e:
        print(f"  │  ⚠ Erreur Grader : {str(e)[:100]}")
        print(f"  │  → Considéré NON pertinent par sécurité (Web Search déclenchée)")
        return {
            "document_pertinent": False,
            "agents_actives": ["doc_grader_error"],
        }