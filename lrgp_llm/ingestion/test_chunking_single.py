# test_chunking_single.py
from pathlib import Path
from chunk_documents import chunker_document

json_path = Path("ingestion\data\parsed\Ahn_2013_IJGGC.json")
chunks = chunker_document(json_path)
print(f"\n{len(chunks)} chunks produits\n")
for c in chunks[:5]:
    print(f"  [{c['chunk_index']:03d}] {c['text'][:100]}...")
    if "[Équation]" in c["text"]:
        print(f"       ↑ Équation récupérée via fallback orig")
print()

# Chercher les chunks avec équations fallback
eq_chunks = [c for c in chunks if "[Équation]" in c["text"]]
print(f"Équations récupérées via fallback : {len(eq_chunks)}")
for c in eq_chunks:
    print(f"  → {c['text'][:120]}")



# Ajouter à test_chunking_single.py

# ── Chercher traces de tableaux dans TOUS les chunks ─────────────
print("\n── Recherche tableaux dans tous les chunks ──")
for c in chunks:
    text = c["text"]
    # Différents formats possibles selon DoclingNodeParser
    if any(kw in text.lower() for kw in ["table", "tableau", "tab.", "|", "<!-- table"]):
        print(f"\n  [{c['chunk_index']:03d}] {c['n_chars']} chars — CONTIENT TABLE")
        print(f"  Début : {text[:200]}")
        print(f"  ...")

# ── Inspecter directement le DoclingDocument pour les tableaux ───
import json
from docling_core.types.doc import DoclingDocument

json_path_check = Path("data/parsed") / (json_path.stem + ".json")
with open(json_path_check, encoding="utf-8") as f:
    doc_data = json.load(f)

doc = DoclingDocument.model_validate(doc_data)
tables = list(doc.tables)
print(f"\n── Tableaux dans le DoclingDocument : {len(tables)} ──")
for i, t in enumerate(tables[:3]):
    print(f"\n  Table {i+1} :")
    if hasattr(t, 'caption') and t.caption:
        cap = t.caption.text if hasattr(t.caption, 'text') else str(t.caption)
        print(f"  Caption : {cap[:80]}")
    try:
        df = t.export_to_dataframe()
        print(f"  Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        print(f"  Aperçu :\n{df.head(3).to_string()}")
    except Exception as e:
        print(f"  Export DataFrame : {e}")
    try:
        md = t.export_to_markdown()
        print(f"  Markdown ({len(md)} chars) :\n{md[:300]}")
    except Exception as e:
        print(f"  Export Markdown : {e}")
