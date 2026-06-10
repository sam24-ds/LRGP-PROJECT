"""
register_external_json.py
Enregistre des JSONs déjà produits (CSV, BIB, etc.)
dans parsing_report.json pour qu'ils soient pris en compte
par chunk_documents.py.

Usage : python ingestion/register_external_json.py
"""
import json
from pathlib import Path

PARSED_DIR   = Path("data/parsed")
RAPPORT_PATH = PARSED_DIR / "parsing_report.json"

# ── Charger le rapport existant ───────────────────────────────────
with open(RAPPORT_PATH, encoding="utf-8") as f:
    rapport = json.load(f)

# ── JSONs à enregistrer ───────────────────────────────────────────
# Lister tous les JSONs présents dans data/parsed/
# qui ne sont pas encore dans le rapport
exclus = {"parsing_report.json", "bibliography.json"}

nouveaux = []
for json_path in PARSED_DIR.glob("*.json"):
    if json_path.name in exclus:
        continue
    # Vérifier si déjà dans le rapport
    nom_fichier = json_path.stem + ".json"
    if nom_fichier not in rapport and json_path.stem not in rapport:
        nouveaux.append(json_path)

if not nouveaux:
    print("Aucun nouveau JSON à enregistrer.")
else:
    print(f"{len(nouveaux)} nouveau(x) JSON(s) trouvé(s) :\n")
    for json_path in nouveaux:
        # Calculer les stats de base
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Extraire le texte pour compter les chars
        texte = ""
        if "texts" in data:
            texte = " ".join(
                t.get("text", "") for t in data["texts"]
                if isinstance(t, dict)
            )

        n_chars  = len(texte)
        n_tables = len(data.get("tables", []))
        source_type = data.get("source_type", "external")

        # Entrée dans le rapport
        entree = {
            "statut":      "ok",
            "fichier":     json_path.name,
            "taille_mo":   round(json_path.stat().st_size / (1024*1024), 3),
            "duree_s":     0,
            "chars":       n_chars,
            "tables":      n_tables,
            "source_type": source_type,
        }

        rapport[json_path.name] = entree
        print(f"  ✓ {json_path.name:<50} {n_chars:,} chars  {n_tables} tableaux")

    # Sauvegarder le rapport mis à jour
    with open(RAPPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Rapport mis à jour — {len(nouveaux)} entrée(s) ajoutée(s)")
    print(f"  Relancer : python chunk_documents.py")