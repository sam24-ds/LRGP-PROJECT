import re
import ollama
from srar_gp.state import SRARState
from srar_gp.prompts.director_prompts import PROMPT_ROUTER, PROMPT_GENERAL

DIRECTOR_MODEL = "qwen3.5:27b"


# ── Patterns heuristiques (regex) ──
PATTERNS_GENERAL = [
    r"\bqui (es|est)[\s\-]?tu\b",          # "qui es-tu", "qui est tu"
    r"\bque (es|est)[\s\-]?tu\b",
    r"\bt['e]?es qui\b",
    r"\bpresent[ée]?[\s\-]?toi\b",
    r"\bbonjour\b",
    r"\bsalut\b",
    r"\bmerci\b",
    r"\bau revoir\b",
    r"\bque peux[\s\-]?tu\b",
    r"\bcomment (tu|fonctionn)\b",
    r"\btu es (un|quel|quoi)\b",
    r"\btu fais quoi\b",
    r"\bquel mod[èe]le\b",
    r"\bcomment[\s\-]?[çc]a va\b",
]

PATTERNS_CALCUL = [
    r"\bcalcule[zr]?\b",
    r"\bd[ée]termine[zr]?\b",
    r"\br[ée]sou[ds]?\b",
    r"\bquelle? est la valeur\b",
    r"\btrouve[zr]? la valeur\b",
]

PATTERNS_COMPARAISON = [
    r"\bcompare[zr]?\b",
    r"\bdiff[ée]rence entre\b",
    r"\b vs \b",
    r"\bavantages? et inconv[ée]nients?\b",
]


def heuristique_classification(question: str) -> str:
    """Classification heuristique par regex — fiable et rapide."""
    q = question.lower().strip()
    
    if any(re.search(p, q) for p in PATTERNS_GENERAL):
        return "GENERAL"
    if any(re.search(p, q) for p in PATTERNS_CALCUL):
        return "CALCUL"
    if any(re.search(p, q) for p in PATTERNS_COMPARAISON):
        return "COMPARAISON"
    return None  # incertain — laisser le LLM décider


def classifier_question(state: SRARState) -> SRARState:
    """Classe la question dans une voie de traitement.
    
    Stratégie hybride :
    1. Heuristique regex en PREMIER (rapide, fiable pour les cas évidents)
    2. LLM router en SECOND (pour les cas ambigus)
    3. Fallback FACTUEL si tout échoue
    """
    print(f"\n  ┌─ [DIRECTOR] Classification de la question...")
    
    # ── Étape 1 — Heuristique regex (instantanée) ──
    type_h = heuristique_classification(state["question"])
    if type_h:
        print(f"  │  → Heuristique : {type_h}")
        type_question = type_h
    else:
        # ── Étape 2 — LLM router (cas ambigus) ──
        print(f"  │  → Ambigu — appel au LLM router...")
        response = ollama.chat(
            model=DIRECTOR_MODEL,
            messages=[{
                "role": "user",
                "content": PROMPT_ROUTER.format(question=state["question"])
            }],
            think= False,
            options={"temperature": 0.0, "num_predict": 100},
        )
        contenu = response.message.content.strip().upper()
        print(f"  │  → LLM brut : '{contenu[:50]}'")
        
        # Extraction prioritaire (GENERAL en premier !)
        type_question = None
        for cat in ["GENERAL", "CALCUL", "COMPARAISON", "FACTUEL"]:
            if cat in contenu:
                type_question = cat
                break
        
        if type_question is None:
            print(f"  │  ⚠ LLM échec total — défaut FACTUEL")
            type_question = "FACTUEL"
    
    # Mapping voie
    if type_question == "GENERAL":
        voie = "GENERAL"
    elif type_question == "CALCUL":
        voie = "CALCUL"
    else:
        voie = "DOCUMENTAIRE"
    
    print(f"  │  → Type : {type_question} | Voie : {voie}")
    
    return {
        "type_question": type_question,
        "voie": voie,
        "agents_actives": ["director"],
    }


def reponse_generale(state: SRARState) -> SRARState:
    """Voie GÉNÉRAL — réponse directe sans mobiliser les autres agents."""
    print(f"  ┌─ [DIRECTOR] Réponse générale directe...")
    
    response = ollama.chat(
        model=DIRECTOR_MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT_GENERAL.format(question=state["question"])
        }],
        think = False,
        options={
            "temperature": 0.3,
            "num_predict": 1000,
            
        },
    )
    
    contenu = response.message.content

    print(response)
    
    # ── DEBUG : afficher la réponse brute ──
    print(f"  │  → Brut ({len(contenu)} chars) : {contenu[:200]}")
    
    # ── Nettoyage : extraire après </think> si présent ──
    if "</think>" in contenu:
        contenu = contenu.split("</think>", 1)[1].strip()
        print(f"  │  → Après nettoyage think : {len(contenu)} chars")
    
    # ── Fallback si vide ──
    if not contenu.strip():
        print(f"  │  ⚠ Réponse vide — fallback manuel")
        contenu = (
            "Je suis l'assistant IA du LRGP Nancy, basé sur un modèle Qwen 3.5 "
            "fine-tuné, couplé à un système RAG sur le corpus du laboratoire. "
            "Je suis spécialisé dans la séparation membranaire des gaz et "
            "le génie des procédés. Je peux répondre à vos questions "
            "théoriques, faire des calculs de dimensionnement, et citer "
            "les sources documentaires du LRGP."
        )
    
    return {
        "reponse_finale": contenu,
        "agents_actives": ["director_general"],
    }