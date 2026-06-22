"""
calculation_expert.py
Calculation_Expert — Traduit le Blueprint en code Python + boucle REPL.
Modèle : DeepSeek-Coder 6.7B.
"""
import re
import ollama
from srar_gp.state import SRARState
from srar_gp.prompts.coder_prompts import PROMPT_CODE, PROMPT_CODE_FIX
from srar_gp.tools.python_repl import execute_python

CODER_MODEL = "deepseek-coder-v2:16b" #deepseek-coder:6.7b"
MAX_FIX_ATTEMPTS = 3


import re

def extraire_code(texte: str) -> str:
    """Extrait le bloc Python d'une réponse LLM.
    
    Robuste aux variations de format Markdown :
    - ```python ... ```
    - ``` ... ```
    - ```py ... ```
    - Code sans fences
    - Code avec fences résiduelles en début/fin
    """
    if not texte:
        return ""
    
    # ── Stratégie 1 : match ```python ... ``` ──
    match = re.search(r"```(?:python|py)\s*\n(.*?)\n```", texte, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # ── Stratégie 2 : match ``` ... ``` (sans langage) ──
    match = re.search(r"```\s*\n(.*?)\n```", texte, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Retirer "python" en début si le LLM l'a mis là
        if code.startswith(("python\n", "py\n")):
            code = code.split("\n", 1)[1] if "\n" in code else ""
        return code.strip()
    
    # ── Stratégie 3 : nettoyage ligne par ligne ──
    lignes = texte.split("\n")
    lignes_nettoyees = []
    for ligne in lignes:
        stripped = ligne.strip()
        # Skip les lignes de fences markdown isolées
        if stripped in ("```python", "```py", "```", "'''python", "'''"):
            continue
        lignes_nettoyees.append(ligne)
    
    code = "\n".join(lignes_nettoyees).strip()
    
    # ── Stratégie 4 : nettoyage final des fences résiduelles ──
    # Retirer ``` en début (avec ou sans langage)
    while code.startswith("```"):
        first_newline = code.find("\n")
        if first_newline == -1:
            code = code[3:].strip()
            break
        code = code[first_newline+1:].lstrip()
        # Retirer "python" ou "py" sur sa propre ligne en début
        if code.startswith(("python\n", "py\n")):
            code = code.split("\n", 1)[1] if "\n" in code else ""
    
    # Retirer ``` en fin
    while code.rstrip().endswith("```"):
        last_idx = code.rstrip().rfind("```")
        code = code[:last_idx].rstrip()
    
    # ── Validation finale ──
    code = code.strip()
    
    # Si vraiment vide ou inutilisable, retourner tel quel
    if not code or code in ("```python", "```", "python"):
        return ""
    
    return code


def generer_et_executer_code(state: SRARState) -> SRARState:
    """Génère le code Python et l'exécute avec boucle d'auto-correction."""
    print(f"\n  ┌─ [CALCULATION_EXPERT] Génération de code Python...")
    
    response = ollama.chat(
        model=CODER_MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT_CODE.format(blueprint=state["blueprint"])
        }],

        
        options={"temperature": 0.0, "num_predict": 4000, "num_ctx": 16384},
    )

    #print(f"blueprint envoyer au model : {state['blueprint']}") #debogage
    
    code = extraire_code(response.message.content)
    print(f"  │  → Code généré ({len(code)} chars)")
    
    errors = []
    for tentative in range(MAX_FIX_ATTEMPTS):
        print(f"  │  → Exécution Python (tentative {tentative+1}/{MAX_FIX_ATTEMPTS})...")
        result = execute_python(code)
        
        # ── FIX : sécuriser stdout/stderr ──
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        
        if result.success and stdout.strip():
            print(f"  │  ✓ Code exécuté avec succès")
            print(f"  │  → stdout : {stdout[:500]}")
            return {
                "code_python": code,
                "resultat_numerique": stdout.strip(),
                "execution_errors": errors,
                "agents_actives": ["calculation_expert"],
            }
        
        # Erreur ou stdout vide
        err_msg = stderr[:500] if stderr else "Pas de stderr — stdout vide"
        errors.append(err_msg)
        print(f"  │  ✗ Erreur : {err_msg[:500]}")
        
        if tentative < MAX_FIX_ATTEMPTS - 1:
            print(f"  │  → Demande de correction...")
            response = ollama.chat(
                model=CODER_MODEL,
                messages=[{
                    "role": "user",
                    "content": PROMPT_CODE_FIX.format(
                        code=code,
                        erreur=err_msg,
                    )
                }],
                options={"temperature": 0.1, "num_predict": 10000},
                think=False,
            )
            code = extraire_code(response.message.content)
    
    print(f"  │  ⚠ Échec après {MAX_FIX_ATTEMPTS} tentatives")
    return {
        "code_python": code,
        "resultat_numerique": "",
        "execution_errors": errors,
        "agents_actives": ["calculation_expert_failed"],
    }