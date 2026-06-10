from srar_gp.agents.calculation_expert import extraire_code

# Cas 1 — Format propre
test1 = """Voici le code :

````python
import numpy as np
x = 5
print(x)
````

Voilà."""

# Cas 2 — Sans langage
test2 = """```
import numpy as np
print("hello")
````"""

# Cas 3 — Le bug observé sur Test C2
test3 = """```python
# Définition des constantes
P_CO2 = 3500.0
L = 50.0
Q = P_CO2 / L
print(f"Q = {Q} GPU")
```"""

# Cas 4 — Sans fences
test4 = """import numpy as np
x = 42
print(x)"""

# Cas 5 — Fences mal fermées
test5 = """```python
import numpy as np
print("test")"""

for i, test in enumerate([test1, test2, test3, test4, test5], 1):
    print(f"\n=== Test {i} ===")
    code = extraire_code(test)
    print(f"Extrait ({len(code)} chars):")
    print(code)
    print(f"\nPremier caractère : '{code[:1]}'")
    print(f"Commence par ``` : {code.startswith('```')}")
