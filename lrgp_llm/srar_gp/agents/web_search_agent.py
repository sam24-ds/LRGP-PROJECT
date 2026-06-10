"""
web_search_agent.py
Agent fallback Web — appelé quand le RAG LRGP est insuffisant.
"""
from srar_gp.tools.web_search import web_search
from srar_gp.state import SRARState


def chercher_web(state: SRARState) -> SRARState:
    """Effectue une recherche web et enrichit le contexte existant."""
    print(f"\n  ┌─ [WEB_SEARCH] Recherche complémentaire sur le web...")
    
    contexte_existant = state.get("context_rag", "")
    
    resultats_web = web_search(
        query=state["question"],
        max_results=3,
        scientifique=True,
    )
    
    if not resultats_web:
        print(f"  │  ✗ Aucun résultat web — continuer avec ce qu'on a")
        return {
            "sources_web": [],
            "document_pertinent": False,
            "agents_actives": ["web_search_failed"],
        }
    
    # Construction du contexte enrichi
    contexte_web = "\n\n=== SOURCES WEB COMPLÉMENTAIRES ===\n"
    for r in resultats_web:
        contexte_web += f"\n[Source Web: {r['source']}]\n"
        contexte_web += f"URL: {r['url']}\n"
        contexte_web += f"{r['text'][:800]}\n"
    
    nouveau_contexte = contexte_existant + contexte_web
    
    print(f"  │  ✓ {len(resultats_web)} sources web ajoutées")
    for r in resultats_web:
        print(f"  │    • {r['source'][:60]}")
    
    return {
        "sources_web": [
            {"source": r["source"], "url": r["url"], "score": r["score"]}
            for r in resultats_web
        ],
        "context_rag": nouveau_contexte,
        "document_pertinent": True,
        "agents_actives": ["web_search"],
    }