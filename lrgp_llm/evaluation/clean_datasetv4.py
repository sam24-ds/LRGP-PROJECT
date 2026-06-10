"""
clean_dataset_v4.py
Nettoyage complet du dataset selon les règles expertes.
- Supprime [Source:] dans input ET output
- Nettoie "Contexte :" orphelin dans input
- Bannit vocabulaire documentaire
- Détecte références invisibles
- Détecte paires hors sujet (version musclée)
- Vérifie cohérence thématique input/output

Usage : python evaluation/clean_dataset_v4.py
"""
import json
import re
from pathlib import Path
from collections import Counter

INPUT_TRAIN     = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\train_v4_clean.jsonl")
OUTPUT_TRAIN  = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\train_v4_clean_cleaned.jsonl")
INPUT_EVAL  = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\eval_v4_clean.jsonl")
OUTPUT_EVAL = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\eval_v4_clean_cleaned.jsonl")




# ─── RÈGLE 1 : Supprimer les sources ─────────────────────────────────────────
def supprimer_sources(text: str) -> str:
    # Supprime [Source: ...] ou [Sources: ...]
    text = re.sub(r'\[Source(s)?\s*:[^\]]*\]', '', text, flags=re.IGNORECASE)
    return text.strip()


# ─── RÈGLE 2 : Nettoyer l'input (Contexte et fausses amorces) ────────────────
def nettoyer_input(text: str) -> str:
    text = supprimer_sources(text)
    # Supprime le bloc "Contexte :\n..." s'il ne sert à rien
    text = re.sub(r'Contexte\s*:\s*\n.*?\n\n', '', text, flags=re.DOTALL)
    text = re.sub(r'^Contexte\s*:\s*\n?', '', text, flags=re.MULTILINE)
    # Enlever "Question : " au début si présent pour rendre le prompt naturel
    text = re.sub(r'^Question\s*:\s*', '', text, flags=re.IGNORECASE)
    # Nettoyer les sauts de ligne excessifs
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─── RÈGLE 3 : Bannir le vocabulaire documentaire dans l'output ──────────────
VOCAB_DOC =[
    r"selon le (document|texte|contexte|tableau|graphique|figure)",
    r"d'après le (document|texte|contexte)",
    r"le (tableau|graphique|figure|schéma)\s+\d+[\.\d]*",
    r"dans le (document|texte|contexte) (fourni|ci-dessus|indiqué)",
    r"le texte (indique|mentionne|précise|dit)",
    r"le document (indique|mentionne|précise)",
    r"comme (mentionné|indiqué) dans (le document|le texte|la source)",
    r"d'après les? (données? )?(du|de la) (document|source|texte)",
    r"cf\.\s*(tableau|figure|équation)\s*\d+",
    r"voir (tableau|figure)\s+\d+",
    r"dans (le|ce) (contexte fourni|texte fourni)"
]
regex_vocab = re.compile('|'.join(VOCAB_DOC), re.IGNORECASE)

def contient_vocab_doc(text: str) -> bool:
    return bool(regex_vocab.search(text))


# ─── RÈGLE 4 : Références invisibles dans l'input ────────────────────────────
REFS_INVISIBLES =[
    r"selon (la )?référence\s*\[?\d+\]?",
    r"d'après (la )?référence\s*\[?\d+\]?",
    r"les références\s*\[[\d,\s\-]+\]",
    r"d'après l'équation\s+\d+[\.\d]*",
    r"(tableau|figure|équation)\s+\d+\.\d+",
    r"cité(s)? (en|dans)\s*\[?\d+\]?",
    r"dans la figure \d+",
    r"dans le tableau \d+",
    r"les figures \d+ et \d+"
]
regex_refs = re.compile('|'.join(REFS_INVISIBLES), re.IGNORECASE)

def contient_refs_invisibles(text: str) -> bool:
    return bool(regex_refs.search(text))


# ─── RÈGLE 5 : Paires hors-sujet / Hallucinations croisées ───────────────────
def est_hors_sujet(paire: dict) -> bool:
    inp = paire.get("input", "").lower()
    out = paire.get("output", "").lower()

    # (Mots dans Output, Mots dans Input) -> Si les deux matchent, c'est hors-sujet
    incompatibles =[
        # Bactériologie vs Membranes
        (["bactériolog", "microbiolog", "npp", "e. coli"],["pervaporation", "membrane", "co2", "ch4"]),
        # Solvants spécifiques vs Polymères spécifiques (Le bug Davis/Zhao)
        (["zhao", "liao", "pvam", "mof", "membrane polymère"],["solvant", "davis", "rochelle", "freeman"]),
        # Le bug Graphène vs Colonne d'absorption
        (["flux convectif", "n_conv", "colonne"],["graphène", "dispersion", "mg/ml"]),
        # Le bug Électrodialyse mécanique vs Phénol
        (["phénol", "diffusivité", "émulsion", "deff"],["électrodialyse", "conception mécanique", "ro", "pression"]),
        # Le bug Perméabilité vs Module tubulaire géométrique
        (["nombre de tubes", "surface totale", "cylindre", "a_totale"],["sélectivité idéale", "perméabilités", "figures"]),
        # Distillation vs Adsorption
        (["distillation fraction", "rectification", "plateau théorique"], ["adsorption", "isotherme", "langmuir"])
    ]

    for mots_output, mots_input in incompatibles:
        if (any(m in out for m in mots_output) and any(m in inp for m in mots_input)):
            return True
    return False


# ─── MOTEUR DE NETTOYAGE ─────────────────────────────────────────────────────
def nettoyer_paire(paire: dict) -> tuple:
    output = paire.get("output", "")
    input_ = paire.get("input", "")

    if est_hors_sujet(paire):
        return None, "hors_sujet (Mismatch détecté)"

    if contient_refs_invisibles(input_):
        return None, "refs_invisibles_dans_input"

    if contient_vocab_doc(output):
        return None, "vocabulaire_documentaire_dans_output"

    output_clean = supprimer_sources(output)
    input_clean = nettoyer_input(input_)

    if len(output_clean) < 40:
        return None, "output_trop_court_apres_nettoyage"
    if len(input_clean) < 15:
        return None, "input_vide_apres_nettoyage"

    paire_clean = {
        "instruction": paire.get("instruction", "Tu es un expert en génie des procédés au LRGP Nancy."),
        "input": input_clean,
        "output": output_clean,
        "type": paire.get("type", "FACTUEL"),
        "domaine": paire.get("domaine", "Génie des procédés"),
        "qualite_estimee": paire.get("qualite_estimee", 5)
    }
    return paire_clean, "ok"


# ─── TRAITEMENT DU FICHIER ───────────────────────────────────────────────────
def traiter_fichier(input_path: Path, output_path: Path, nom: str) -> int:
    if not input_path.exists():
        print(f"Le fichier {input_path} est introuvable. Ignoré.")
        return 0

    with open(input_path, encoding="utf-8") as f:
        paires =[json.loads(l) for l in f if l.strip()]

    print(f"\n{'═'*65}")
    print(f"  NETTOYAGE : {nom} ({len(paires)} paires trouvées)")
    print(f"{'═'*65}")

    paires_clean =[]
    rejets = Counter()

    for p in paires:
        p_clean, raison = nettoyer_paire(p)
        if p_clean:
            paires_clean.append(p_clean)
        else:
            rejets[raison] += 1

    # Statistiques
    print(f"✅ Conservées : {len(paires_clean)}")
    print(f"❌ Rejetées   : {sum(rejets.values())}")
    for raison, n in sorted(rejets.items(), key=lambda x: -x[1]):
        print(f"    - {raison:<40} : {n}")

    # Sauvegarde
    with open(output_path, "w", encoding="utf-8") as f:
        for p in paires_clean:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n📁 Fichier propre sauvegardé sous : {output_path}")
    return len(paires_clean)

# ─── LANCEMENT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Démarrage du nettoyage du Dataset V4...")
    traiter_fichier(INPUT_TRAIN, OUTPUT_TRAIN, "DATASET D'ENTRAÎNEMENT")
    
    # Décommentez ces lignes si vous avez aussi un fichier d'évaluation
    traiter_fichier(INPUT_EVAL, OUTPUT_EVAL, "DATASET D'ÉVALUATION")
    
    print("\nTerminé ! Votre dataset est maintenant purgé des hallucinations et prêt pour le fine-tuning.")