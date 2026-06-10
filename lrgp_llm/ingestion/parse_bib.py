"""
parse_bib.py
Extrait les références BibTeX et crée un fichier de métadonnées global.
Utilisé pour enrichir les métadonnées des chunks à l'étape suivante.
"""
import argparse
import json
from pathlib import Path
import bibtexparser

def parse_bib(corpus_path: str) -> None:
    root = Path(corpus_path)
    bibs = list(root.rglob("*.bib"))
    print(f"\n{len(bibs)} fichiers .bib trouvés")

    toutes_refs = {}
    for bib_path in bibs:
        with open(bib_path, encoding="utf-8", errors="ignore") as f:
            db = bibtexparser.load(f)

        for entry in db.entries:
            toutes_refs[entry.get("ID", "unknown")] = {
                "titre":   entry.get("title",   ""),
                "auteurs": entry.get("author",  ""),
                "annee":   entry.get("year",    ""),
                "journal": entry.get("journal", entry.get("booktitle", "")),
                "doi":     entry.get("doi",     ""),
                "abstract":entry.get("abstract",""),
            }

    print(f"  {len(toutes_refs)} références extraites")

    out = Path("data/parsed/bibliography.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(toutes_refs, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Sauvegardé : {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()
    parse_bib(args.corpus)