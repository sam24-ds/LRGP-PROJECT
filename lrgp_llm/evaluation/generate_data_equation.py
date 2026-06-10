import json
import random
import math

def generate_quadratic_permeation_pairs(num_samples=100):
    dataset =[]
    
    for _ in range(num_samples):
        # 1. Génération de variables réalistes aléatoires
        x_in = round(random.uniform(0.05, 0.40), 2)  # Concentration entrée 5% à 40%
        alpha = random.randint(10, 60)               # Sélectivité entre 10 et 60
        r = round(random.uniform(0.05, 0.20), 3)     # Rapport de pression
        P_up = random.choice([5, 10, 15, 20])        # Pression amont (Bar)
        P_down = round(P_up * r, 3)                  # Pression aval calculée depuis r
        
        # 2. Calcul des coefficients du polynôme (ay² + by + c = 0)
        # y / (1-y) = alpha * (x - r*y) / ((1-x) - r*(1-y))
        # En développant :
        a = (alpha - 1) * r
        b = - (alpha * x_in + (1 - x_in) + r * (alpha - 1))
        c = alpha * x_in
        
        # Inversion des signes pour avoir un format classique (a > 0)
        a_calc = -a
        b_calc = -b
        c_calc = -c
        
        # 3. Résolution de l'équation
        delta = b_calc**2 - 4 * a_calc * c_calc
        sqrt_delta = math.sqrt(delta)
        y_max = (-b_calc - sqrt_delta) / (2 * a_calc)  # La solution physiquement possible est la soustraction ici
        y_max_pct = round(y_max * 100, 2)
        
        # 4. Génération de la réflexion (Chain-of-Thought)
        think_text = f"""<think>Étape 1 : Le rapport de pression r = {P_down} / {P_up} = {r:.3f} n'est pas négligeable. On doit utiliser l'équilibre local.
Étape 2 : Poser l'équation d'égalité des flux pour une membrane dense :
 y / (1-y) = α* · (x - r·y) / ((1-x) - r·(1-y))
Étape 3 : Remplacer par les valeurs numériques (x = {x_in}, r = {r:.3f}, α* = {alpha}) :
 y / (1-y) = {alpha} · ({x_in} - {r:.3f}·y) / ({(1-x_in):.2f} - {r:.3f}·(1-y))
 y / (1-y) = ({alpha*x_in:.2f} - {alpha*r:.3f}·y) / ({(1-x_in-r):.3f} + {r:.3f}·y)
Étape 4 : Développer pour obtenir l'équation quadratique de la forme ay² + by + c = 0 :
 y · ({(1-x_in-r):.3f} + {r:.3f}·y) = (1-y) · ({alpha*x_in:.2f} - {alpha*r:.3f}·y)
 {(1-x_in-r):.3f}·y + {r:.3f}·y² = {alpha*x_in:.2f} - {alpha*r:.3f}·y - {alpha*x_in:.2f}·y + {alpha*r:.3f}·y²
 {(1-x_in-r):.3f}·y + {r:.3f}·y² = {alpha*x_in:.2f} - {(alpha*r + alpha*x_in):.3f}·y + {alpha*r:.3f}·y²
 {(-a_calc):.3f}·y² + {(-b_calc):.3f}·y + {(-c_calc):.2f} = 0
 {a_calc:.3f}·y² + {b_calc:.3f}·y + {c_calc:.2f} = 0
Étape 5 : Résoudre le discriminant :
 Δ = ({b_calc:.3f})² - 4 · ({a_calc:.3f}) · ({c_calc:.2f}) = {b_calc**2:.3f} - {4*a_calc*c_calc:.3f} = {delta:.3f}
 √Δ ≈ {sqrt_delta:.3f}
 y = ({-b_calc:.3f} - {sqrt_delta:.3f}) / (2 · {a_calc:.3f}) = {(-b_calc - sqrt_delta):.3f} / {2*a_calc:.3f} = {y_max:.4f} ({y_max_pct}%).</think>"""

        # 5. Création du bloc JSON
        item = {
            "instruction": "Tu es un expert en génie des procédés au LRGP Nancy.",
            "input": f"Dans un module de perméation gazeuse, le composé préférentiel est à x_in={int(x_in*100)}%. P_up={P_up} Bar, P_down={P_down} Bar. Sélectivité idéale α*={alpha}. Calculer y_max au perméat.",
            "output": f"{think_text}\n\nRéponse finale : La fraction molaire maximale au perméat est y_max = {y_max_pct}%. Cette situation correspond à un taux de coupe nul (θ → 0).",
            "type": "CALCUL",
            "domaine": "perméation_gazeuse",
            "qualite_estimee": 5
        }
        dataset.append(item)
        
    return dataset

# Générer et sauvegarder
data = generate_quadratic_permeation_pairs(100)
with open("dataset_permeation.jsonl", "w", encoding="utf-8") as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print("100 paires d'entraînement générées avec succès dans dataset_permeation.jsonl !")