"""
parse_csv.py
Convertit les fichiers CSV en chunks Markdown pour indexation dans Qdrant.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

def parse_csv(corpus_path: str) -> None:
    root       = Path(corpus_path)
    output_dir = Path("data/parsed")
    output_dir.mkdir(parents=True, exist_ok=True)

    csvs = list(root.rglob("*.csv"))
    print(f"\n{len(csvs)} fichiers CSV trouvés\n")

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", sep=None,
                             engine="python")  # détecte , ou ; automatiquement
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin-1", sep=None,
                             engine="python")

        print(f"  {csv_path.name} — {df.shape[0]} lignes × {df.shape[1]} colonnes")
        print(f"  Colonnes : {list(df.columns)}\n")

        # Convertir en Markdown pour indexation
        md_text = f"# Données tabulaires — {csv_path.stem}\n\n"
        md_text += df.to_markdown(index=False)

        # Sauvegarder comme JSON compatible avec le reste du pipeline
        doc_json = {
            "schema_name": "DoclingDocument",
            "name": csv_path.stem,
            "source_type": "csv",
            "texts": [
                {
                    "label": "title",
                    "text": f"Données tabulaires — {csv_path.stem}"
                },
                {
                    "label": "paragraph",
                    "text": md_text
                }
            ],
            "tables": [
                {
                    "label": "table",
                    "caption": {"text": csv_path.stem},
                    "data": {
                        "grid": [list(df.columns)] +
                                df.astype(str).values.tolist()
                    }
                }
            ]
        }

        out = output_dir / (csv_path.stem + ".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc_json, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Sauvegardé : {out.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()
    parse_csv(args.corpus)