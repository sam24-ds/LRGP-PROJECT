"""
validator.py
Validation_Expert — Vérifie la cohérence physique avec CoT structuré.
Sprint 3 : classifie les erreurs (code / physique / ambigu) pour la boucle.
"""
import json
import ollama
from srar_gp.state import SRARState

VALIDATOR_MODEL = "qwen3.5:27b"


PROMPT_VALIDATION = """Tu es un ingénieur senior LRGP. Tu valides un calcul scientifique.

═══════════════════════════════════════════════════════════════
INFORMATIONS DISPONIBLES
═══════════════════════════════════════════════════════════════

QUESTION POSÉE :
{question}

BLUEPRINT MATHÉMATIQUE :
{blueprint}

CODE PYTHON EXÉCUTÉ :
{code}

RÉSULTAT NUMÉRIQUE OBTENU :
{resultat}

═══════════════════════════════════════════════════════════════
PROCÉDURE DE VALIDATION (3 étapes obligatoires)
═══════════════════════════════════════════════════════════════

ÉTAPE 1 — Analyse du CODE Python :
  - Le code respecte-t-il les équations du Blueprint ?
  - Les unités sont-elles cohérentes à chaque ligne ?
  - L'algorithme est-il logique ?

ÉTAPE 2 — Analyse du RÉSULTAT :
  - Ordre de grandeur cohérent avec celui attendu dans le Blueprint ?
  - Signes physiques valides (pas de t<0, fraction>1) ?

ÉTAPE 3 — CLASSIFICATION DE L'ERREUR (si invalide) :
  - "code"     : erreur de syntaxe, conversion d'unités, bug logique du code
  - "physique" : erreur dans le Blueprint (formule fausse, hypothèse erronée)
  - "ambigu"   : la question elle-même est ambiguë (conventions multiples)
  - "aucune"   : pas d'erreur (cas valide=true)

═══════════════════════════════════════════════════════════════
RÈGLES STRICTES
═══════════════════════════════════════════════════════════════

1. NE JAMAIS rejeter un calcul à cause d'une faute dans la question.
   Si le code a CORRIGÉ une faute de frappe, c'est un BON comportement.

2. Si le code a fonctionné ET le résultat est plausible → VALIDE.

3. NE JAMAIS halluciner d'erreurs absentes du code réel.
   Regarde uniquement le CODE PYTHON, pas la question.

═══════════════════════════════════════════════════════════════
FORMAT DE RÉPONSE (JSON strict)
═══════════════════════════════════════════════════════════════

Réponds STRICTEMENT en JSON valide avec ces champs :

{{
  "raisonnement": "Étape 1: ... Étape 2: ... Étape 3: ...",
  "valide": true,
  "diagnostic": "explication courte (1-2 phrases)",
  "type_erreur": "aucune",
  "correction_suggeree": "",
  "synthese": "réponse finale formatée pour l'utilisateur"
}}

JSON :"""


def valider_resultat(state: SRARState) -> SRARState:
    """Valide la cohérence du résultat avec CoT et classification de l'erreur."""
    print(f"\n  ┌─ [VALIDATOR] Vérification physique...")
    
    if not state.get("resultat_numerique"):
        return {
            "validation_ok": False,
            "validation_message": "Le calcul n'a pas abouti à un résultat.",
            "type_erreur": "aucune",
            "critique_validator": "",
            "reponse_finale": (
                f"⚠ Calcul échoué.\n\n"
                f"Erreurs Python :\n" +
                "\n".join(state.get("execution_errors", [])[:2])
            ),
            "agents_actives": ["validator_no_result"],
        }
    
    # ── Construction du prompt en passant TOUTES les variables ──
    try:
        prompt = PROMPT_VALIDATION.format(
            question=state.get("question", "")[:500],
            blueprint=state.get("blueprint", "")[:2000],
            code=state.get("code_python", "")[:1500],
            resultat=state.get("resultat_numerique", "")[:500],
        )
    except KeyError as e:
        print(f"  │  ⚠ Erreur format prompt : {e}")
        return {
            "validation_ok": False,
            "validation_message": f"Erreur prompt : {e}",
            "type_erreur": "aucune",
            "critique_validator": "",
            "reponse_finale": (
                f"📊 Résultat : {state.get('resultat_numerique', '')}\n\n"
                f"⚠ Validation impossible — à vérifier manuellement."
            ),
            "agents_actives": ["validator_prompt_failed"],
        }
    
    # ── Tentative avec retry ──
    for tentative in range(2):
        try:
            response = ollama.chat(
                model=VALIDATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                think=False,
                format="json",
                options={"temperature": 0.1, "num_predict": 1500},
                keep_alive="5m",
            )
            contenu = response.message.content.strip()
            
            if not contenu:
                print(f"  │  ⚠ Réponse vide (tentative {tentative+1})")
                continue
            
            data = json.loads(contenu)
            valide = data.get("valide", False)
            raisonnement = data.get("raisonnement", "")
            diagnostic = data.get("diagnostic", "")
            type_erreur = data.get("type_erreur", "aucune")
            correction = data.get("correction_suggeree", "")
            synthese = data.get("synthese", "")
            
            print(f"  │  → Validation : {'✓ OK' if valide else '✗ ÉCHEC'}")
            if raisonnement:
                print(f"  │  → Raisonnement : {raisonnement[:150]}")
            print(f"  │  → Diagnostic : {diagnostic[:150]}")
            if not valide:
                print(f"  │  → Type erreur : {type_erreur}")
                if correction:
                    print(f"  │  → Correction suggérée : {correction[:150]}")
            
            if valide:
                reponse = synthese + f"\n\n📊 Résultat : {state['resultat_numerique']}"
            else:
                reponse = (
                    f"⚠ Résultat physiquement incohérent : {diagnostic}\n\n"
                    f"Raisonnement : {raisonnement}\n\n"
                    f"Résultat brut : {state['resultat_numerique']}"
                )
            
            return {
                "validation_ok": valide,
                "validation_message": diagnostic,
                "type_erreur": type_erreur,
                "critique_validator": correction,
                "reponse_finale": reponse,
                "agents_actives": ["validator"],
            }
        
        except json.JSONDecodeError as e:
            print(f"  │  ⚠ Parse JSON échoué (tentative {tentative+1}) : {e}")
            if tentative == 0:
                import time
                time.sleep(2)
                continue
        
        except Exception as e:
            print(f"  │  ⚠ Erreur Validator (tentative {tentative+1}) : {str(e)[:100]}")
            if tentative == 0:
                import time
                time.sleep(2)
                continue
    
    # ── Fallback : résultat sans validation ──
    print(f"  │  ⚠ Validator inopérant — résultat retourné sans validation")
    return {
        "validation_ok": False,
        "validation_message": "Validator inopérant",
        "type_erreur": "aucune",
        "critique_validator": "",
        "reponse_finale": (
            f"📊 Résultat : {state['resultat_numerique']}\n\n"
            f"⚠ Validation automatique impossible — à vérifier manuellement."
        ),
        "agents_actives": ["validator_failed"],
    }