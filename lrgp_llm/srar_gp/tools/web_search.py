"""
web_search.py
Wrapper Tavily pour recherche web complémentaire au RAG.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Charger .env depuis la racine du projet
load_dotenv(Path(__file__).parent.parent.parent / ".env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

try:
    from tavily import TavilyClient
    if not TAVILY_API_KEY:
        print("  [WebSearch] ⚠ TAVILY_API_KEY manquante dans .env")
        _client = None
    else:
        _client = TavilyClient(api_key=TAVILY_API_KEY)
        print("  [WebSearch] ✓ Tavily client initialisé")
except ImportError:
    print("  [WebSearch] ⚠ tavily-python non installé (pip install tavily-python)")
    _client = None


DOMAINES_PRIORITAIRES = [
    "doi.org",
    "sciencedirect.com",
    "wiley.com",
    "rsc.org",
    "nature.com",
    "acs.org",
    "springer.com",
    "elsevier.com",
    "mdpi.com",
    "researchgate.net",
    "techniques-ingenieur.fr",
    "hal.science",
    "pubs.acs.org",
]


def web_search(query: str, max_results: int = 3, scientifique: bool = True) -> list[dict]:
    """Recherche web ciblée."""
    if _client is None:
        return []
    
    # ── NOUVEAU : Tronquer si trop long (Tavily limite à 400 chars) ──
    query_originale = query
    if len(query) > 380:
        # Garder seulement les 380 premiers caractères
        query = query[:380].rsplit(' ', 1)[0]  # couper sur le dernier espace
        print(f"  [WebSearch] ⚠ Query tronquée ({len(query_originale)} → {len(query)} chars)")
    
    try:
        params = {
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
        }
        if scientifique:
            params["include_domains"] = DOMAINES_PRIORITAIRES
        
        response = _client.search(**params)
        results = []
        for r in response.get("results", []):
            results.append({
                "source": r.get("title", "Web")[:80],
                "url":    r.get("url", ""),
                "text":   r.get("content", "")[:1500],
                "score":  r.get("score", 0.0),
            })
        return results
    except Exception as e:
        print(f"  [WebSearch] ✗ Erreur : {str(e)[:100]}")
        return []