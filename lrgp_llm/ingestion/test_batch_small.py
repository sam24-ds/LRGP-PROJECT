"""
test_batch_small.py
Test de parsing sur un échantillon de 10 PDFs pour estimer la vitesse réelle.
Usage : python ingestion/test_batch_small.py --corpus "C:/chemin/corpus"
"""

import argparse
import time
import random
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

def tester_vitesse(corpus_path: str, n: int = 10) -> None:
    root = Path(corpus_path)

    # Collecter tous les PDFs accessibles
    pdfs = []
    for p in root.rglob("*.pdf"):
        try:
            _ = p.stat().st_size
            pdfs.append(p)
        except OSError:
            pass

    # Échantillon aléatoire de n PDFs
    echantillon = random.sample(pdfs, min(n, len(pdfs)))
    print(f"\n{'═'*60}")
    print(f"  Test vitesse Docling — {len(echantillon)} PDFs")
    print(f"{'═'*60}\n")

    # Config Docling optimisée batch
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr            = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    resultats = []
    for i, pdf in enumerate(echantillon, 1):
        taille_mo = pdf.stat().st_size / (1024*1024)
        print(f"  [{i:2d}/{len(echantillon)}] {pdf.name[:50]:<50} ({taille_mo:.1f} Mo)")

        t0 = time.time()
        try:
            result   = converter.convert(str(pdf))
            doc      = result.document
            texte    = doc.export_to_text()
            duree    = time.time() - t0
            n_chars  = len(texte)
            n_tables = len(list(doc.tables))

            # Évaluer la qualité
            if n_chars < 100:
                qualite = "⚠ VIDE"
                statut  = "fallback"
            elif n_chars < 500:
                qualite = "⚠ PAUVRE"
                statut  = "fallback"
            else:
                qualite = "✓ OK"
                statut  = "ok"

            print(f"          {qualite} — {n_chars:,} chars, {n_tables} tableaux, {duree:.1f}s")
            resultats.append({
                "nom": pdf.name, "taille_mo": taille_mo,
                "duree": duree, "chars": n_chars,
                "statut": statut, "tables": n_tables
            })

        except Exception as e:
            duree = time.time() - t0
            print(f"          ✗ ERREUR — {str(e)[:60]}")
            resultats.append({
                "nom": pdf.name, "taille_mo": taille_mo,
                "duree": duree, "chars": 0,
                "statut": "erreur", "tables": 0
            })

    # ── Résumé ───────────────────────────────────────────────────────
    ok       = [r for r in resultats if r["statut"] == "ok"]
    fallback = [r for r in resultats if r["statut"] == "fallback"]
    erreurs  = [r for r in resultats if r["statut"] == "erreur"]

    duree_moy  = sum(r["duree"] for r in resultats) / len(resultats)
    duree_total_estime = duree_moy * len(pdfs) / 3600  # heures

    print(f"\n{'─'*60}")
    print(f"  Résumé sur {len(echantillon)} PDFs")
    print(f"{'─'*60}")
    print(f"  ✓ OK (texte extrait)   : {len(ok)}")
    print(f"  ⚠ Fallback nécessaire  : {len(fallback)}")
    print(f"  ✗ Erreurs              : {len(erreurs)}")
    print(f"  Durée moyenne/fichier  : {duree_moy:.1f}s")
    print(f"  Durée totale estimée   : ~{duree_total_estime:.1f}h pour {len(pdfs)} PDFs")
    print(f"  Taux fallback estimé   : ~{len(fallback)/len(resultats)*100:.0f}%")

    if ok:
        chars_moy = sum(r["chars"] for r in ok) / len(ok)
        print(f"  Chars moyens (OK)      : {chars_moy:,.0f}")

    print(f"{'═'*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--n", type=int, default=10,
                        help="Nombre de PDFs à tester (défaut: 10)")
    args = parser.parse_args()
    tester_vitesse(args.corpus, args.n)