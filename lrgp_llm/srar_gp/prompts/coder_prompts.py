"""
coder_prompts.py
Prompts du Calculation_Expert — traduit le Blueprint en code Python.
"""

PROMPT_CODE = """Tu es un développeur Python spécialisé en calcul scientifique.

Ta mission : traduire ce Blueprint mathématique en script Python EXÉCUTABLE.

═══════════════════════════════════════════════════════════════
BLUEPRINT :
{blueprint}
═══════════════════════════════════════════════════════════════

RÈGLES STRICTES :

1. Écris UN SEUL bloc de code Python complet et autonome.
2. Utilise UNIQUEMENT : numpy, scipy.optimize, scipy.integrate, math
3. Le code doit :
   - Définir toutes les constantes en début de script
   - Implémenter les équations du Blueprint exactement
   - Afficher le résultat final via print() avec unités
   - Inclure des commentaires français pour chaque étape

4. Format de sortie :
```python
# code ici
```

5. INTERDIT :
   - input(), open(), import os/sys/subprocess
   - import matplotlib (pas de plots)
   - Inventer des valeurs non données

CODE PYTHON :"""


PROMPT_CODE_FIX = """Le code précédent a échoué à l'exécution. Voici l'erreur :

═══════════════════════════════════════════════════════════════
CODE EXÉCUTÉ :
{code}

ERREUR :
{erreur}
═══════════════════════════════════════════════════════════════

Corrige le code en gardant la même logique. Réécris le script ENTIER corrigé.

CODE CORRIGÉ :"""