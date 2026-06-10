"""
state.py
État partagé du graphe SRAR-GP — passé entre les agents.
"""
from typing import Annotated, Literal
from typing_extensions import TypedDict
from operator import add


class SRARState(TypedDict):
    """État du graphe — chaque agent peut lire/écrire ces champs."""

    # ── Entrée utilisateur ──
    question: str
    
    # ── Routage (Director) ──
    voie: Literal["GENERAL", "DOCUMENTAIRE", "CALCUL", "UNKNOWN"]
    type_question: Literal["FACTUEL", "CALCUL", "COMPARAISON", "GENERAL"]
    
    # ── Sortie Librarian ──
    sources_rag: list[dict]       # chunks récupérés
    context_rag: str              # contexte assemblé
    document_pertinent: bool      # résultat du Document Grader

    # ── Web Search (Boucle 1) ──
    sources_web: list[dict]   # résultats Tavily
    
    # ── Sortie Process_Engineer (sprint 2) ──
    blueprint: str                # plan mathématique
    missing_data: list[str]       # données manquantes détectées
    
    # ── Sortie Calculation_Expert (sprint 2) ──
    code_python: str              # script généré
    resultat_numerique: str       # résultat de l'exécution
    execution_errors: list[str]   # stack traces
    
    # ── Sprint 3 — Boucle de re-négociation ──
    tentatives_renegociation: int        # Compteur (max 2)
    type_erreur: str                     # "code" | "physique" | "ambigu"
    critique_validator: str              # Diagnostic précis pour correction
    code_python_historique: list         # Historique des codes générés

    # ── Sortie Validation (sprint 3) ──
    validation_ok: bool
    validation_message: str
    
    # ── Sortie finale ──
    reponse_finale: str
    agents_actives: Annotated[list[str], add]  # trace du parcours