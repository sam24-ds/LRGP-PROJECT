"""
graph.py
Architecture SRAR-GP complète avec les 3 boucles d'auto-correction.

Boucle 1 : Document Grader → Web Search (DOC + CALCUL)
Boucle 2 : REPL Python (intégrée dans calculation_expert)
Boucle 3 : Validator → Coder/Engineer (renégociation)
"""
from langgraph.graph import StateGraph, END
from srar_gp.state import SRARState
from srar_gp.agents.director import classifier_question, reponse_generale
from srar_gp.agents.librarian import (
    extraire_documentaire,
    generer_reponse_documentaire,
)
from srar_gp.agents.document_grader import grader_documents
from srar_gp.agents.web_search_agent import chercher_web
from srar_gp.agents.process_engineer import rediger_blueprint
from srar_gp.agents.calculation_expert import generer_et_executer_code
from srar_gp.agents.validator import valider_resultat


# ══════════════════════════════════════════════════════════════
# LIBRARIAN pour voie CALCUL (RAG sans génération)
# ══════════════════════════════════════════════════════════════
def librarian_pour_calcul(state: SRARState) -> SRARState:
    """Variante du Librarian pour voie CALCUL : RAG uniquement."""
    print(f"\n  ┌─ [LIBRARIAN-RAG-CALCUL] Extraction documentaire pour calcul...")

    try:
        from srar_gp.agents.librarian import _chain

        sources = []
        if hasattr(_chain, "retriever"):
            if hasattr(_chain.retriever, "retrieve"):
                sources = _chain.retriever.retrieve(state["question"])
            elif hasattr(_chain.retriever, "get_relevant_documents"):
                sources = _chain.retriever.get_relevant_documents(state["question"])

        if not sources:
            print(f"  │  → Fallback via .ask()")
            response = _chain.ask(state["question"])
            sources_data = response.sources if hasattr(response, 'sources') else []
            contexte = "\n\n".join(
                f"[Source: {s.source[:60]}]" for s in sources_data[:5]
            )
            return {
                "sources_rag": [{"source": s.source, "score": getattr(s, 'score', 0)} for s in sources_data],
                "context_rag": contexte,
                "document_pertinent": False,
                "agents_actives": ["librarian_rag_calcul"],
            }

        contexte = "\n\n".join(
            f"[Source: {getattr(s, 'source', '?')[:60]}]\n{getattr(s, 'text', '')[:500]}"
            for s in sources[:5]
        )
        print(f"  │  → {len(sources)} sources extraites")

        return {
            "sources_rag": [{"source": getattr(s, 'source', '?'), "score": getattr(s, 'score', 0)} for s in sources],
            "context_rag": contexte,
            "document_pertinent": False,
            "agents_actives": ["librarian_rag_calcul"],
        }
    except Exception as e:
        print(f"  │  ⚠ Erreur : {str(e)[:100]}")
        return {
            "sources_rag": [],
            "context_rag": "",
            "document_pertinent": False,
            "agents_actives": ["librarian_rag_calcul_failed"],
        }


# ══════════════════════════════════════════════════════════════
# MISSING_DATA_HANDLER
# ══════════════════════════════════════════════════════════════
def gerer_missing_data(state: SRARState) -> SRARState:
    """Court-circuit du calcul si données insuffisantes."""
    print(f"\n  ┌─ [MISSING_DATA_HANDLER] Données insuffisantes détectées")
    missing_list = state.get("missing_data", [])
    print(f"  │  → {len(missing_list)} donnée(s) manquante(s)")

    variables = []
    for line in missing_list:
        if "MISSING_DATA" in line.upper():
            apres = line.split("MISSING_DATA", 1)[1].lstrip(":* ").strip()
            apres = apres.replace("**", "").replace("`", "").replace("*", "")
            variables.append(f"  • {apres[:300]}")
    if not variables:
        variables = [f"  • {line[:200]}" for line in missing_list[:5]]

    reponse = (
        f"⚠ **Données insuffisantes pour résoudre ce problème**\n\n"
        f"Le Process_Engineer a identifié {len(variables)} paramètre(s) manquant(s) :\n\n"
        + "\n".join(variables) +
        f"\n\n**Action requise** : Merci de fournir ces valeurs.\n\n"
        f"*Le système refuse d'inventer des valeurs.*"
    )

    return {
        "reponse_finale": reponse,
        "agents_actives": ["missing_data_handler"],
    }


# ══════════════════════════════════════════════════════════════
# SPRINT 3 — Boucle de re-négociation
# ══════════════════════════════════════════════════════════════
def correction_code(state: SRARState) -> SRARState:
    """Renvoie au Coder avec le code précédent + diagnostic Validator + stack trace."""
    print(f"\n  ┌─ [RENÉGOCIATION] Correction du code Python (avec contexte)...")

    from srar_gp.agents.calculation_expert import extraire_code, CODER_MODEL
    from srar_gp.tools.python_repl import execute_python
    import ollama

    tentatives = state.get("tentatives_renegociation", 0) + 1

    blueprint = state.get("blueprint", "")
    donnees = ""
    if "2. DONNÉES" in blueprint:
        idx = blueprint.find("2. DONNÉES")
        next_section = blueprint.find("\n## 3.", idx)
        if next_section == -1:
            next_section = idx + 1000
        donnees = blueprint[idx:next_section].strip()

    equations = ""
    if "3. ÉQUATIONS" in blueprint:
        idx = blueprint.find("3. ÉQUATIONS")
        next_section = blueprint.find("\n## 4.", idx)
        if next_section == -1:
            next_section = idx + 1000
        equations = blueprint[idx:next_section].strip()

    ordre_attendu = ""
    if "Résultat attendu" in blueprint or "Ordre de grandeur" in blueprint:
        idx = max(blueprint.find("Résultat attendu"), blueprint.find("Ordre de grandeur"))
        ordre_attendu = blueprint[idx:idx + 500].strip()

    # ── Récupérer le code précédent à corriger ──
    code_precedent = state.get("code_python", "")
    code_precedent_tronque = code_precedent[:3000]

    # ── Récupérer la dernière erreur d'exécution (si le code avait planté) ──
    erreurs = state.get("execution_errors", [])
    derniere_erreur = erreurs[-1][:600] if erreurs else ""

    # Section erreur Python conditionnelle (uniquement si une trace existe)
    bloc_erreur_python = ""
    if derniere_erreur:
        bloc_erreur_python = f"""ERREUR PYTHON À L'EXÉCUTION (corrige-la en priorité) :
{derniere_erreur}
═══════════════════════════════════════════════════════════════
"""

    PROMPT_SIMPLE = f"""Tu es un développeur Python. Écris un script Python COMPLET et AUTONOME.

═══════════════════════════════════════════════════════════════
LE VALIDATOR A REJETÉ LE CODE PRÉCÉDENT.
RAISON PRÉCISE :
{state.get('validation_message', '')}

CORRECTION OBLIGATOIRE :
{state.get('critique_validator', '')}
═══════════════════════════════════════════════════════════════
{bloc_erreur_python}CODE PRÉCÉDENT À CORRIGER (ne le recopie PAS tel quel — corrige l'erreur identifiée ci-dessus) :
```python
{code_precedent_tronque}
```
═══════════════════════════════════════════════════════════════

DONNÉES À UTILISER : {donnees}

ÉQUATION À CODER : {equations}

RÉSULTAT ATTENDU : {ordre_attendu}

INSTRUCTIONS STRICTES :

1.  Le code DOIT être AUTONOME et EXÉCUTABLE TEL QUEL
2.  Définis TOUTES les variables avec leurs valeurs numériques
3.  Applique le ou les équation
4.  Implemente de maniere rigoureuse les corrections soulignées par le Validator
5.  Affiche le résultat avec print()
6.  PAS de conversion d'unités sauf si l'équation l'exige
7.  Format : code Python pur dans un bloc ```python ... ```

CODE PYTHON COMPLET :"""

    response = ollama.chat(
        model=CODER_MODEL,
        messages=[{"role": "user", "content": PROMPT_SIMPLE}],
        options={"temperature": 0.0, "num_predict": 4000, "num_ctx": 16384},
    )

    code = extraire_code(response.message.content)
    print(f"  │  → Code corrigé ({len(code)} chars)")

    result = execute_python(code)
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    historique = state.get("code_python_historique", [])
    historique.append(state.get("code_python", ""))

    if result.success and stdout.strip():
        print(f"  │  ✓ Code corrigé exécuté avec succès")
        print(f"  │  → stdout : {stdout[:150]}")
        return {
            "code_python": code,
            "resultat_numerique": stdout.strip(),
            "code_python_historique": historique,
            "tentatives_renegociation": tentatives,
            "agents_actives": [f"correction_code_t{tentatives}"],
        }

    print(f"  │  ✗ Correction échouée : {stderr[:100]}")
    return {
        "code_python": code,
        "resultat_numerique": state.get("resultat_numerique", ""),
        "execution_errors": state.get("execution_errors", []) + [stderr[:500]],
        "code_python_historique": historique,
        "tentatives_renegociation": tentatives,
        "agents_actives": [f"correction_code_failed_t{tentatives}"],
    }


def correction_physique(state: SRARState) -> SRARState:
    """Renvoie au Process_Engineer pour reformuler le Blueprint."""
    print(f"\n  ┌─ [RENÉGOCIATION] Reformulation du Blueprint...")

    from srar_gp.agents.process_engineer import ENGINEER_MODEL
    import ollama
    import re

    tentatives = state.get("tentatives_renegociation", 0) + 1

    PROMPT_REFORMULATION = f"""Le Blueprint précédent a conduit à un résultat
PHYSIQUEMENT INCORRECT. Voici la critique du Validator.

═══════════════════════════════════════════════════════════════
BLUEPRINT PRÉCÉDENT :
{state['blueprint']}

DIAGNOSTIC DU VALIDATOR :
{state.get('validation_message', '')}

CORRECTION SUGGÉRÉE :
{state.get('critique_validator', '')}

QUESTION ORIGINALE :
{state['question']}
═══════════════════════════════════════════════════════════════

Reformule un NOUVEAU Blueprint en tenant compte de la critique. ⚠️ TU DOIS
OBLIGATOIREMENT RÉPONDRE AVEC EXACTEMENT LE MÊME FORMAT JSON QUE PRÉCÉDEMMENT.
AUCUN TEXTE AVANT OU APRÈS LE JSON.

JSON :"""

    response = ollama.chat(
        model=ENGINEER_MODEL,
        messages=[{"role": "user", "content": PROMPT_REFORMULATION}],
        think=False,
        format="json",  # <-- OBLIGATOIRE POUR ÉVITER QU'IL NE BAVARDE
        options={"temperature": 0.1, "num_predict": 4000, "num_ctx": 16384},
    )

    blueprint = re.sub(r"<think>.*?</think>", "", response.message.content, flags=re.DOTALL).strip()
    print(f"  │  → Nouveau Blueprint ({len(blueprint)} chars)")

    return {
        "blueprint": blueprint,
        "tentatives_renegociation": tentatives,
        "agents_actives": [f"correction_physique_t{tentatives}"],
    }


# ══════════════════════════════════════════════════════════════
# ROUTEURS
# ══════════════════════════════════════════════════════════════
def router(state: SRARState) -> str:
    """Aiguillage selon la voie déterminée par le Director."""
    return state["voie"]


def router_doc_grader(state: SRARState) -> str:
    """Aiguillage après Document Grader.
    Aiguille selon la voie et la pertinence des documents.
    """
    voie = state.get("voie", "")
    pertinent = state.get("document_pertinent", False)

    derniers_agents = state.get("agents_actives", [])
    deja_web_search = "web_search" in derniers_agents or "web_search_failed" in derniers_agents

    if pertinent or deja_web_search:
        if voie == "DOCUMENTAIRE":
            return "continuer_doc"
        return "continuer_calcul"

    return "web_search"


def router_apres_web(state: SRARState) -> str:
    """Après Web Search, aiguiller selon la voie."""
    voie = state.get("voie", "")
    if voie == "DOCUMENTAIRE":
        return "continuer_doc"
    return "continuer_calcul"


def router_missing_data(state: SRARState) -> str:
    """Filtre les vrais MISSING_DATA des faux positifs."""
    missing = state.get("missing_data", [])
    if not missing:
        return "calculation_expert"

    NEGATIONS = [
        "aucune donnée manquante",
        "aucun paramètre manquant",
        "pas de donnée manquante",
        "no missing data",
    ]
    vrais_missing = []
    for line in missing:
        l = line.lower()
        if any(neg in l for neg in NEGATIONS):
            continue
        vrais_missing.append(line)

    state["missing_data"] = vrais_missing

    if vrais_missing:
        return "missing_data_handler"
    return "calculation_expert"


def router_renegociation(state: SRARState) -> str:
    """Aiguillage après le Validator."""
    if state.get("validation_ok"):
        return "end"

    tentatives = state.get("tentatives_renegociation", 0)
    if tentatives >= 3:
        print(f"  │  → Max tentatives atteint ({tentatives}) → fin")
        return "end"

    type_erreur = state.get("type_erreur", "aucune")
    print(f"  │  → Type d'erreur : {type_erreur} (tentative {tentatives+1}/2)")

    # ── Aiguillage dynamique basé EXCLUSIVEMENT sur le verdict du Validator ──
    if type_erreur == "code":
        return "correction_code"
    elif type_erreur == "physique":
        return "correction_physique"

    return "end"


# ══════════════════════════════════════════════════════════════
# CONSTRUCTION DU GRAPHE
# ══════════════════════════════════════════════════════════════
def build_graph():
    """Architecture SRAR-GP complète avec les 3 boucles."""
    workflow = StateGraph(SRARState)

    # ── Nœuds ──
    workflow.add_node("director_classifier",     classifier_question)
    workflow.add_node("director_general",        reponse_generale)

    # Voie DOCUMENTAIRE (2 étapes)
    workflow.add_node("librarian_doc_extract",   extraire_documentaire)
    workflow.add_node("librarian_doc_gen",       generer_reponse_documentaire)

    # Voie CALCUL
    workflow.add_node("librarian_rag",           librarian_pour_calcul)

    # Boucle 1 — communs aux deux voies
    workflow.add_node("doc_grader",              grader_documents)
    workflow.add_node("web_search",              chercher_web)

    # Reste voie CALCUL
    workflow.add_node("process_engineer",        rediger_blueprint)
    workflow.add_node("missing_data_handler",    gerer_missing_data)
    workflow.add_node("calculation_expert",      generer_et_executer_code)
    workflow.add_node("validator",               valider_resultat)
    workflow.add_node("correction_code",         correction_code)
    workflow.add_node("correction_physique",     correction_physique)

    # ── Point d'entrée ──
    workflow.set_entry_point("director_classifier")

    # ── Routage Director ──
    workflow.add_conditional_edges(
        "director_classifier",
        router,
        {
            "GENERAL":      "director_general",
            "DOCUMENTAIRE": "librarian_doc_extract",
            "CALCUL":       "librarian_rag",
        }
    )

    # ══ BOUCLE 1 — Document Grader + Web Search (DOC + CALCUL) ══
    workflow.add_edge("librarian_doc_extract", "doc_grader")
    workflow.add_edge("librarian_rag",         "doc_grader")

    workflow.add_conditional_edges(
        "doc_grader",
        router_doc_grader,
        {
            "continuer_doc":    "librarian_doc_gen",
            "continuer_calcul": "process_engineer",
            "web_search":       "web_search",
        }
    )

    workflow.add_conditional_edges(
        "web_search",
        router_apres_web,
        {
            "continuer_doc":    "librarian_doc_gen",
            "continuer_calcul": "process_engineer",
        }
    )

    # ══ Voie CALCUL — suite ══
    workflow.add_conditional_edges(
        "process_engineer",
        router_missing_data,
        {
            "missing_data_handler": "missing_data_handler",
            "calculation_expert":   "calculation_expert",
        }
    )

    workflow.add_edge("calculation_expert", "validator")

    # ══ BOUCLE 3 — Re-négociation ══
    workflow.add_conditional_edges(
        "validator",
        router_renegociation,
        {
            "correction_code":     "correction_code",
            "correction_physique": "correction_physique",
            "end":                 END,
        }
    )

    workflow.add_edge("correction_code",     "validator")
    workflow.add_edge("correction_physique", "calculation_expert")

    # ── Sorties (END) ──
    workflow.add_edge("director_general",     END)
    workflow.add_edge("librarian_doc_gen",    END)
    workflow.add_edge("missing_data_handler", END)

    return workflow.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph