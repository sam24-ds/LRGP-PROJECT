"""
process_engineer.py
Process_Engineer — Rédige le Blueprint mathématique au format JSON.
Modèle : Qwen 3.5 27B.
"""
import ollama
import json
import re
from srar_gp.state import SRARState
from srar_gp.prompts.engineer_prompts import PROMPT_BLUEPRINT

ENGINEER_MODEL = "qwen3.5:27b"  # Modèle LLM pour le Process_Engineer

def rediger_blueprint(state: SRARState) -> SRARState:
    print(f"\n  ┌─ [PROCESS_ENGINEER] Rédaction du Blueprint...")
    
    contexte = state.get("context_rag", "")
    if not contexte and state.get("sources_rag"):
        contexte = "\n\n".join(
            f"[Source: {s.get('source', '?')}]"
            for s in state["sources_rag"][:5]
        )
    if not contexte:
        contexte = "(Aucun document du corpus LRGP — utiliser les connaissances générales)"
    
    response = ollama.chat(
        model=ENGINEER_MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT_BLUEPRINT.format(
                contexte=contexte[:3000],
                question=state["question"],
            )
        }],
        think=True,     # <-- DÉSACTIVÉ pour éviter la troncature
        format="json",   # <-- FORCER LE FORMAT JSON
        options={
            "temperature": 0.1, 
            "num_predict": 10000, 
            "num_ctx": 16384  # <-- Protège contre la limite de contexte
        },
    )
    
    raw_content = response.message.content.strip()
    
    # Nettoyage de sécurité
    raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    
    print(f"  │  → Blueprint généré ({len(raw_content)} chars)")
    
    # ── DEBUG : afficher le Blueprint complet ──
    print(f"\n{'='*70}")
    print(f"BLUEPRINT JSON COMPLET")
    print(f"{'='*70}")
    print(raw_content)
    print(f"{'='*70}\n")
    
    # Extraction des MISSING_DATA
    missing = []
    try:
        data = json.loads(raw_content)
        manquantes = data.get("donnees_manquantes", [])
        missing = [d for d in manquantes if "MISSING_DATA" in d]
    except json.JSONDecodeError:
        print("  │  ⚠ Erreur de parsing JSON dans le Blueprint.")
        
    return {
        "blueprint": raw_content,
        "missing_data": missing,
        "agents_actives": ["process_engineer"],
    }