"""
corpus_stats.py — version multi-format
Analyse PDFs, DOCX et XLSX dans le corpus LRGP.
Usage : python ingestion/corpus_stats.py --corpus "C:/chemin/vers/corpus"
"""

import argparse
from pathlib import Path
from collections import defaultdict

# Formats supportés par Docling nativement
FORMATS = {
    ".pdf":  "PDF",
    ".docx": "Word",
    ".doc":  "Word (ancien)",
    ".xlsx": "Excel",
    ".xls":  "Excel (ancien)",
    ".pptx": "PowerPoint",
}

def analyser_corpus(corpus_path: str) -> None:
    root = Path(corpus_path)
    if not root.exists():
        print(f"❌ Dossier introuvable : {corpus_path}")
        return

    # Collecte récursive — tous formats
    tous_fichiers_bruts = []
    chemins_trop_longs  = []

    for p in root.rglob("*"):
        if p.suffix.lower() in FORMATS:
            try:
                _ = p.stat().st_size   # test d'accès
                tous_fichiers_bruts.append(p)
            except (FileNotFoundError, OSError):
                chemins_trop_longs.append(str(p))

    tous_fichiers = tous_fichiers_bruts

    if not tous_fichiers:
        print("❌ Aucun fichier reconnu dans ce dossier.")
        return

    # Stats globales
    taille_totale = sum(p.stat().st_size for p in tous_fichiers)
    taille_mo     = taille_totale / (1024 * 1024)

    # Répartition par type de fichier
    par_type = defaultdict(list)
    for f in tous_fichiers:
        par_type[f.suffix.lower()].append(f)

    # Répartition par sous-dossier (niveau 1)
    par_dossier = defaultdict(list)
    for f in tous_fichiers:
        try:
            sous_dossier = f.relative_to(root).parts[0]
        except IndexError:
            sous_dossier = "(racine)"
        par_dossier[sous_dossier].append(f)

    # Tailles pour stats
    tailles    = [f.stat().st_size / (1024 * 1024) for f in tous_fichiers]
    taille_moy = sum(tailles) / len(tailles)
    taille_max = max(tailles)

    # ── Affichage ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RAPPORT CORPUS LRGP — multi-format")
    print("═" * 60)
    print(f"  📁 Racine         : {root}")
    print(f"  📄 Total fichiers : {len(tous_fichiers)}")
    print(f"  💾 Taille totale  : {taille_mo:.1f} Mo  ({taille_mo/1024:.2f} Go)")
    print(f"  📊 Taille moyenne : {taille_moy:.1f} Mo / fichier")
    print(f"  📊 Plus gros      : {taille_max:.1f} Mo")
    print(f"  📂 Sous-dossiers  : {len(par_dossier)}")

    # Par type
    print("\n─ Répartition par format " + "─" * 34)
    print(f"  {'Format':<20} {'Fichiers':>9}  {'Taille':>10}  {'Traitement'}")
    print("─" * 60)
    for ext, label in FORMATS.items():
        fichiers = par_type.get(ext, [])
        if not fichiers:
            continue
        t = sum(f.stat().st_size for f in fichiers) / (1024 * 1024)
        print(f"  {label:<20} {len(fichiers):>9}  {t:>8.1f} Mo  Docling direct")

    # Par sous-dossier
    print("\n─ Répartition par sous-dossier " + "─" * 27)
    print(f"  {'Sous-dossier':<28} {'Fichiers':>9}  {'Taille':>10}")
    print("─" * 60)
    for dossier, fichiers in sorted(par_dossier.items()):
        t = sum(f.stat().st_size for f in fichiers) / (1024 * 1024)
        print(f"  {str(dossier):<28} {len(fichiers):>9}  {t:>8.1f} Mo")

    # Estimations
    n_pdf  = len(par_type.get(".pdf",  []))
    n_docx = len(par_type.get(".docx", []) + par_type.get(".doc", []))
    n_xlsx = len(par_type.get(".xlsx", []) + par_type.get(".xls", []))

    pages_estimees = int(taille_mo * 10)
    cout_ocr_max   = pages_estimees / 1000 * 1.0

    print("\n─ Estimations " + "─" * 44)
    print(f"  Pages estimées        : ~{pages_estimees:,}")
    print(f"  Temps parsing Docling : ~{max(1, pages_estimees//600)}–{max(2, pages_estimees//300)} heures (A6000)")
    print(f"  Coût OCR (10% fallback): ~{cout_ocr_max*0.1:.2f} $")
    print(f"\n  Docling supporte nativement :")
    print(f"    PDF  : {n_pdf} fichiers")
    print(f"    DOCX : {n_docx} fichiers")
    print(f"    XLSX : {n_xlsx} fichiers")
    print("═" * 60)

    # Chemins trop longs
    if chemins_trop_longs:
        print(f"\n  ⚠ {len(chemins_trop_longs)} fichier(s) ignorés — chemin trop long (>260 chars)")
        print(f"  Solution : activer les chemins longs Windows (voir README)")
        for p in chemins_trop_longs[:5]:
            print(f"    → ...{p[-60:]}")
        if len(chemins_trop_longs) > 5:
            print(f"    ... et {len(chemins_trop_longs)-5} autre(s)")

    # Extensions non reconnues
    tous_bruts = []
    for p in root.rglob("*"):
        try:
            if p.is_file():
                tous_bruts.append(p)
        except OSError:
            pass
    non_reconnus = [p for p in tous_bruts if p.suffix.lower() not in FORMATS]
    if non_reconnus:
        ext_inconnues = defaultdict(int)
        for f in non_reconnus:
            ext_inconnues[f.suffix.lower() or "(sans ext)"] += 1
        print(f"\n  ⚠ Extensions ignorées :")
        for ext, n in sorted(ext_inconnues.items(), key=lambda x: -x[1])[:8]:
            print(f"    {ext:<20} {n}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        required=True,
        help='Ex: "C:\\Users\\toi\\Documents\\corpus_lrgp"'
    )
    args = parser.parse_args()
    analyser_corpus(args.corpus)