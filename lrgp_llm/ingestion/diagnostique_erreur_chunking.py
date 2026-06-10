# diagnostic_erreurs.py
import json
from pathlib import Path

chunks_dir = Path("data/chunks")
parsed_dir = Path("data/parsed")

# Lister les JSONs parsés
parsed  = {p.stem for p in parsed_dir.glob("*.json") if p.name != "parsing_report.json"}
chunks  = {p.stem.replace("_chunks", "") for p in chunks_dir.glob("*.jsonl")}

manquants = parsed - chunks
print(f"\n{len(manquants)} documents parsés mais non chunkés :\n")
for m in sorted(manquants):
    print(f"  {m}")