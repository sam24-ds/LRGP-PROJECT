# supprimer_et_reindexer.py
import qdrant_client
from pathlib import Path

SOURCE = "nom_article_sans_extension"  # ex: "Baker_2004_Membrane"

client = qdrant_client.QdrantClient("localhost", port=6333)

# Supprimer tous les points de cette source dans Qdrant
client.delete(
    collection_name="lrgp_corpus",
    points_selector=qdrant_client.models.FilterSelector(
        filter=qdrant_client.models.Filter(
            must=[qdrant_client.models.FieldCondition(
                key="source_file",
                match=qdrant_client.models.MatchValue(value=SOURCE)
            )]
        )
    )
)
print(f"Points supprimés pour : {SOURCE}")

# Supprimer le JSON parsé et le JSONL chunks
Path(f"data/parsed/{SOURCE}.json").unlink(missing_ok=True)
Path(f"data/chunks/{SOURCE}_chunks.jsonl").unlink(missing_ok=True)

# Supprimer du rapport
import json
rapport_path = Path("data/parsed/parsing_report.json")
with open(rapport_path, encoding="utf-8") as f:
    rapport = json.load(f)
rapport.pop(SOURCE + ".pdf", None)
with open(rapport_path, "w", encoding="utf-8") as f:
    json.dump(rapport, f, ensure_ascii=False, indent=2)

print(f"Fichiers nettoyés — relancer parse_pdfs.py et chunk_documents.py")