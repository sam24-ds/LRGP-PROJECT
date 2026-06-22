"""
validator.py
Validation_Expert — Vérifie la cohérence physique avec CoT structuré.
Sprint 3 : classifie les erreurs (code / physique / ambigu) pour la boucle.
"""
import json
import ollama
from srar_gp.state import SRARState

VALIDATOR_MODEL = "qwen3.5:27b"


PROMPT_VALIDATION = """Tu es un ingénieur senior LRGP et un expert en diagnostic d'erreurs mathématiques et informatiques. 
Tu valides un calcul scientifique issu d'un processus en deux étapes : 
1) Un "Process Engineer" a rédigé le Blueprint (équations et hypothèses).
2) Un "Codeur Python" a exécuté ce Blueprint.

═══════════════════════════════════════════════════════════════
INFORMATIONS DISPONIBLES
═══════════════════════════════════════════════════════════════

QUESTION POSÉE :
{question}

BLUEPRINT MATHÉMATIQUE (Rédigé par le Process Engineer) :
{blueprint}

CODE PYTHON EXÉCUTÉ (Rédigé par le Codeur) :
{code}

RÉSULTAT NUMÉRIQUE OBTENU (OU ERREUR D'EXÉCUTION) :
{resultat}

═══════════════════════════════════════════════════════════════
PROCÉDURE DE VALIDATION (3 étapes obligatoires)
═══════════════════════════════════════════════════════════════

ÉTAPE 1 — Analyse du CODE Python et des unités :
  - Le code respecte-t-il les équations du Blueprint ?
  - Les variables critiques (débits, pressions, surfaces, etc.) sont-elles bien définies et correspondent-elles EXACTEMENT aux valeurs données par l'utilisateur ?
  - Le code a-t-il dû "inventer" des formules absurdes parce qu'il manquait des données ?
  -VÉRIFICATION CRITIQUE DES CONVERSIONS : Des erreurs résident parfois dans les facteurs de conversion (ex: Joules vers kWh nécessite de diviser par 3,6 \times 10^6 et non 10^9, conversion de pression Pa/bar, etc.).Vérifie rigoureusement ces facteurs dans le code blueprint avant de remettre en cause la théorie physique.


ÉTAPE 2 — Analyse du RÉSULTAT :
  - Si c'est un plantage (Crash/Traceback) : Le plantage vient-il d'une erreur de syntaxe ou d'une équation mathématiquement insoluble dictée par le Blueprint (ex: fsolve qui diverge) ?
  - Si c'est un chiffre : Le calcul est-il mathématiquement exact par rapport aux données fournies ? 
    ATTENTION : Si les données initiales de l'utilisateur mènent à un résultat dont l'ordre de grandeur est surprenant pour le matériau concerné, ce résultat DOIT quand même être validé. Les seules raisons valables pour rejeter un résultat numérique sont les aberrations logiques absolues (ex: masse négative, température en Kelvin négative, fraction molaire > 1).

ÉTAPE 3 — CLASSIFICATION DE L'ERREUR (si résultat invalide ou plantage) :
  L'étape la plus importante : à qui la faute ?
  - "physique" : Le Blueprint utilise de mauvaises équations, omet une donnée intermédiaire indispensable, ou définit un système mathématiquement insoluble/surcontraint. Ne classe JAMAIS en "physique" un calcul qui est juste mais dont le résultat final semble inhabituel uniquement à cause des chiffres imposés par l'utilisateur.
  - "code"     : Le Blueprint est PARFAIT et soluble, mais le développeur Python a fait une erreur de syntaxe pure (ex: NameError), un bug logique, ou a mal traduit une équation valide.
  - "ambigu"   : La question initiale de l'utilisateur est incohérente au point de rendre la modélisation impossible.
  - "aucune"   : Le résultat est valide.

═══════════════════════════════════════════════════════════════
RÈGLES STRICTES DE DIAGNOSTIC
═══════════════════════════════════════════════════════════════

1. SANCTUARISATION DES DONNÉES UTILISATEUR (RÈGLE D'OR) : Il est STRICTEMENT INTERDIT d'exiger ou de suggérer la modification des nombres fournis dans la question de l'utilisateur pour forcer un résultat "plus réaliste". Si les valeurs de départ mènent mathématiquement à un résultat physiquement improbable, le calcul mathématique PRIME. Tu DOIS valider l'itération ("valide": true) et tu pourras simplement signaler que le résultat est atypique dans ta "synthese".
2. RÈGLE D'IMPUTATION : Si le code a planté ou a inventé des absurdités (ex: F = P / RT sans volume), VÉRIFIE LE BLUEPRINT. S'il manque des conditions aux limites, tu DOIS renvoyer "type_erreur": "physique" pour forcer la réécriture des équations.
3. NE JAMAIS rejeter un calcul pour une faute d'orthographe dans la question.
4. Si le code a fonctionné, que les données de l'utilisateur ont été respectées à la lettre, et qu'il n'y a pas d'aberration absolue (ex: énergie négative) → VALIDE ("valide": true).
5. ADAPTABILITÉ AU BLUEPRINT : Tu dois juger le résultat selon les hypothèses posées dans le Blueprint. Si le Blueprint a fait le choix assumé d'un modèle simplifié, tu NE DOIS PAS rejeter le résultat sous prétexte qu'un modèle d'ingénierie complexe donnerait une valeur différente. 

═══════════════════════════════════════════════════════════════
FORMAT DE RÉPONSE (JSON strict)
═══════════════════════════════════════════════════════════════

INTERDICTION STRICTE : N'utilise AUCUN caractère d'échappement ou antislash ("\") dans ta réponse. Si tu dois citer une variable, écris "gamma" au lieu de "\gamma", "alpha" au lieu de "\alpha".

Réponds STRICTEMENT en JSON valide avec ces champs :

{{
  "raisonnement": "Étape 1: ... Étape 2: ... Étape 3: ...",
  "valide": false,
  "diagnostic": "explication courte de l'erreur (1-2 phrases) OU 'Aucune erreur, calcul correct.'",
  "type_erreur": "physique", 
  "correction_suggeree": "Ce qu'il faut changer dans le code ou le Blueprint (mettre null si valide=true)",
  "synthese": "Réponse finale formatée pour l'utilisateur. Si le calcul est mathématiquement juste mais physiquement atypique, explique-le ici sans rejeter la validation."
}}

JSON :"""


def valider_resultat(state: SRARState) -> SRARState:
    """Valide la cohérence du résultat avec CoT et classification de l'erreur."""
    print(f"\n  ┌─ [VALIDATOR] Vérification physique...")
    
    # ── Remplacement du coupe-circuit par la capture du plantage ──
    resultat_a_evaluer = state.get("resultat_numerique", "").strip()
    
    if not resultat_a_evaluer:
        # On capture les dernières erreurs Python s'il n'y a pas de résultat
        erreurs = "\n".join(state.get("execution_errors", [])[-2:])
        if erreurs:
            resultat_a_evaluer = f"[CRASH DU CODE - AUCUN RÉSULTAT NUMÉRIQUE]\nTraceback:\n{erreurs}"
        else:
            resultat_a_evaluer = "[CRASH DU CODE - AUCUNE TRACE D'ERREUR]"
            
        print(f"  │  ⚠ Pas de résultat numérique, transmission de l'erreur au LLM...")

    # ── Construction du prompt en passant TOUTES les variables ──
    try:
        prompt = PROMPT_VALIDATION.format(
            question=state.get("question", "")[:500],
            blueprint=state.get("blueprint", "")[:4000],
            code=state.get("code_python", ""), # Laisser le code en entier !
            resultat=resultat_a_evaluer[:2000],
        )
    except KeyError as e:
        print(f"  │  ⚠ Erreur format prompt : {e}")
        return {
            "validation_ok": False,
            "validation_message": f"Erreur prompt : {e}",
            "type_erreur": "aucune",
            "critique_validator": "",
            "reponse_finale": (
                f"📊 Résultat : {resultat_a_evaluer}\n\n"
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
                reponse = synthese + f"\n\n📊 Résultat : {resultat_a_evaluer}"
            else:
                reponse = (
                    f"⚠ Résultat physiquement incohérent : {diagnostic}\n\n"
                    f"Raisonnement : {raisonnement}\n\n"
                    f"Résultat brut : {resultat_a_evaluer}"
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
            f"📊 Résultat : {resultat_a_evaluer}\n\n"
            f"⚠ Validation automatique impossible — à vérifier manuellement."
        ),
        "agents_actives": ["validator_failed"],
    }