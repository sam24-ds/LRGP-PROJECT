"""
main.py
Point d'entrée SRAR-GP — test interactif.

Usage : python -m srar_gp.main
"""
from srar_gp.graph import get_graph
from srar_gp.formatter import formatter_reponse



def ask_srar(question: str) -> dict:
    """Pose une question au système SRAR-GP."""
    graph = get_graph()
    
    initial_state = {
        "question":         question,
        "voie":             "UNKNOWN",
        "type_question":    "FACTUEL",
        "sources_rag":      [],
        "context_rag":      "",
        "document_pertinent": False,
        "blueprint":        "",
        "missing_data":     [],
        "code_python":      "",
        "resultat_numerique": "",
        "execution_errors": [],
        "validation_ok":    False,
        "validation_message": "",
        "reponse_finale":   "",
        "agents_actives":   [],
        # ── Sprint 3 ──
        "tentatives_renegociation": 0,
        "type_erreur":              "aucune",
        "critique_validator":       "",
        "code_python_historique":   [],
        "sources_web": [],
    }
    
    final_state = graph.invoke(initial_state)
    
    # ── Appliquer le formatter ──
    final_state["reponse_finale"] = formatter_reponse(final_state)
    
    return final_state


if __name__ == "__main__":
    graph1 = get_graph()
    print(graph1.draw_mermaid())
    
    print(f"\n{'═'*60}")
    print(f"  SRAR-GP — Sprint 1 (Squelette routage)")
    print(f"  Tape 'q' pour quitter")
    print(f"{'═'*60}\n")
    
    while True:
        question = input("\n🔹 Question : ").strip()
        if question.lower() in ("q", "quit", "exit"):
            print("Au revoir !")
            break
        if not question:
            continue
        
        result = ask_srar(question)
        
        print(f"\n  └─ Parcours : {' → '.join(result['agents_actives'])}")
        print(f"\n  📋 Réponse :")
        print(f"  {'─'*55}")
        print(f"  {result['reponse_finale']}")
        print(f"  {'─'*55}\n")