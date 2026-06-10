import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from parse_pdfs import build_converter, parse_one, sauvegarder_json

fichiers = [
    Path(r"C:\Users\Samir\Desktop\3-Samir\corpus_lrgp\Membranes\Egle_2_BIBLIO\CES 2012 revised manuscript_docx.docx"),
    Path(r"C:\Users\Samir\Desktop\3-Samir\corpus_lrgp\Membranes\Egle_2_BIBLIO\AIChE Boucif et al 2007_docx.docx"),
]


output_dir = Path("data/parsed")
converter  = build_converter(use_gpu=True)

for f in fichiers:
    if not f.exists():
        print(f"❌ Introuvable : {f.name}")
        continue
    print(f"Parsing : {f.name}...", end=" ", flush=True)
    res = parse_one(f, converter)
    if res["statut"] == "ok":
        sauvegarder_json(f, converter, output_dir)
        print(f"✓ {res['chars']:,} chars")
    else:
        print(f"✗ {res.get('message','')}")