"""
process_engineer.py
Process_Engineer — Rédige le Blueprint mathématique.
Modèle : Qwen 3.5 27B.
"""
import ollama
from srar_gp.state import SRARState
from srar_gp.prompts.engineer_prompts import PROMPT_BLUEPRINT

ENGINEER_MODEL = "qwen3.5:27b"


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
        think=False,
        options={"temperature": 0.1, "num_predict": 3000},
    )
    
    blueprint = response.message.content.strip()
    print(f"  │  → Blueprint généré ({len(blueprint)} chars)")
    
    # ── DEBUG : afficher le Blueprint complet ──
    print(f"\n{'='*70}")
    print(f"BLUEPRINT COMPLET")
    print(f"{'='*70}")
    print(blueprint)
    print(f"{'='*70}\n")
    
    if not blueprint:
        print(f"  │  ⚠ Blueprint vide — fallback")
        blueprint = (
            f"## DONNÉES\n{state['question']}\n\n"
            f"## MÉTHODE\nCalcul direct des équations"
        )
    
    missing = [l.strip() for l in blueprint.split("\n") if "MISSING_DATA" in l.upper()]
    
    return {
        "blueprint": blueprint,
        "missing_data": missing,
        "agents_actives": ["process_engineer"],
    }