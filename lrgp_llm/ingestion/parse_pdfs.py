"""
parse_pdfs.py
Parsing complet du corpus LRGP avec Docling.
- Reprend là où il s'est arrêté (skip si JSON déjà présent)
- Log détaillé dans data/parsed/parsing_report.json
- Gestion des gros fichiers (>4 Mo)
- Fallback MinerU si texte vide

Usage :
    python ingestion/parse_pdfs.py --corpus "C:/chemin/corpus" [--workers 1]
"""

import argparse
import json
import os
import time
import logging
from pathlib import Path
from datetime import datetime

# Supprimer les warnings HuggingFace et symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # masquer les logs Docling verbeux
    format="%(levelname)s: %(message)s"
)
log = logging.getLogger("parse_pdfs")

# ── Formats supportés ─────────────────────────────────────────────────────────
FORMATS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".htm", ".html"}

# Seuil : fichier considéré "gros" (thèse, livre) → batch réduit
SEUIL_GROS_MO = 4.0


def build_converter(use_gpu: bool = True) -> DocumentConverter:
    """Construit le converter Docling avec la config optimale."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr                  = False
    pipeline_options.do_table_structure      = True
    pipeline_options.generate_picture_images = False
    pipeline_options.accelerator_options     = AcceleratorOptions(
        num_threads = 4,
        device      = AcceleratorDevice.CUDA if use_gpu else AcceleratorDevice.CPU,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def parse_one(path: Path, converter: DocumentConverter) -> dict:
    """
    Parse un fichier et retourne un dict de résultat.
    Statuts possibles : "ok", "vide", "erreur"
    """
    taille_mo = path.stat().st_size / (1024 * 1024)
    t0 = time.time()

    try:
        result   = converter.convert(str(path))
        doc      = result.document
        texte    = doc.export_to_text()
        n_chars  = len(texte.strip())
        n_tables = len(list(doc.tables))
        n_texts  = len(list(doc.texts))
        duree    = time.time() - t0

        if n_chars < 200:
            return {
                "statut":     "vide",
                "fichier":    path.name,
                "taille_mo":  round(taille_mo, 2),
                "duree_s":    round(duree, 1),
                "chars":      n_chars,
                "tables":     n_tables,
                "message":    "Texte insuffisant — probable PDF scanné",
            }

        return {
            "statut":    "ok",
            "fichier":   path.name,
            "taille_mo": round(taille_mo, 2),
            "duree_s":   round(duree, 1),
            "chars":     n_chars,
            "tables":    n_tables,
            "texts":     n_texts,
        }

    except Exception as e:
        duree = time.time() - t0
        return {
            "statut":    "erreur",
            "fichier":   path.name,
            "taille_mo": round(taille_mo, 2),
            "duree_s":   round(duree, 1),
            "chars":     0,
            "tables":    0,
            "message":   str(e)[:200],
        }


def sauvegarder_json(path: Path, converter: DocumentConverter,
                     output_dir: Path) -> bool:
    """Parse et sauvegarde le JSON. Retourne True si succès."""
    try:
        result = converter.convert(str(path))
        doc    = result.document
        out    = output_dir / (path.stem + ".json")
        doc.save_as_json(str(out))
        return True
    except Exception:
        return False


def parser_corpus(corpus_path: str) -> None:
    root       = Path(corpus_path)
    output_dir = Path("data/parsed")
    output_dir.mkdir(parents=True, exist_ok=True)

    rapport_path = output_dir / "parsing_report.json"

    # ── Collecter les fichiers ────────────────────────────────────────
    tous = []
    ignores_chemin = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in FORMATS:
            continue
        try:
            _ = p.stat().st_size
            tous.append(p)
        except OSError:
            ignores_chemin.append(str(p))

    print(f"\n{'═'*62}")
    print(f"  PARSING CORPUS LRGP")
    print(f"{'═'*62}")
    print(f"  Fichiers trouvés  : {len(tous)}")
    print(f"  Chemin trop long  : {len(ignores_chemin)} (ignorés)")
    print(f"  Sortie JSON       : {output_dir.resolve()}")

    # ── Charger le rapport existant (reprise) ─────────────────────────
    rapport_existant = {}
    if rapport_path.exists():
        with open(rapport_path, encoding="utf-8") as f:
            rapport_existant = json.load(f)
        deja_faits = sum(
            1 for v in rapport_existant.values()
            if isinstance(v, dict) and v.get("statut") == "ok"
        )
        print(f"  Déjà parsés (OK)  : {deja_faits} — reprise activée")

    # ── Filtrer : skip si JSON déjà présent et statut OK ─────────────
    a_traiter = []
    skips     = 0
    for p in tous:
        json_out = output_dir / (p.stem + ".json")
        statut_connu = rapport_existant.get(p.name, {}).get("statut")
        if json_out.exists() and statut_connu == "ok":
            skips += 1
        else:
            a_traiter.append(p)

    print(f"  À traiter         : {len(a_traiter)}  (skippés : {skips})")
    print(f"{'─'*62}")

    if not a_traiter:
        print("  ✓ Tout est déjà parsé.")
        return

    # ── Construire le converter (GPU) ─────────────────────────────────
    print("  Chargement modèles Docling...")
    converter = build_converter(use_gpu=True)
    print("  ✓ Prêt\n")

    # ── Parsing ───────────────────────────────────────────────────────
    resultats = dict(rapport_existant)
    stats     = {"ok": 0, "vide": 0, "erreur": 0}
    t_debut   = time.time()

    for i, path in enumerate(a_traiter, 1):
        taille_mo = path.stat().st_size / (1024 * 1024)
        est_gros  = taille_mo >= SEUIL_GROS_MO
        tag       = " [GROS]" if est_gros else ""

        print(f"  [{i:4d}/{len(a_traiter)}] {path.name[:52]:<52} "
              f"({taille_mo:.1f} Mo){tag}", end=" ", flush=True)

        # Parser
        res = parse_one(path, converter)
        stats[res["statut"]] += 1
        resultats[path.name] = res

        # Icône statut
        icone = {"ok": "✓", "vide": "⚠", "erreur": "✗"}[res["statut"]]
        print(f"{icone} {res['duree_s']}s  {res['chars']:,} chars  "
              f"{res.get('tables',0)} tab.")

        # Sauvegarder le JSON si OK
        if res["statut"] == "ok":
            sauvegarder_json(path, converter, output_dir)

        # Sauvegarder le rapport toutes les 10 itérations
        if i % 10 == 0:
            _sauver_rapport(rapport_path, resultats, stats, t_debut, len(a_traiter), i)

        # ETA toutes les 50 fichiers
        if i % 50 == 0:
            elapsed  = time.time() - t_debut
            vitesse  = elapsed / i          # secondes/fichier
            restants = len(a_traiter) - i
            eta_min  = vitesse * restants / 60
            print(f"\n  {'─'*58}")
            print(f"  Progression : {i}/{len(a_traiter)} "
                  f"| Vitesse : {vitesse:.1f}s/fichier "
                  f"| ETA : ~{eta_min:.0f} min")
            print(f"  OK:{stats['ok']}  Vide:{stats['vide']}  "
                  f"Erreur:{stats['erreur']}")
            print(f"  {'─'*58}\n")

    # ── Rapport final ─────────────────────────────────────────────────
    elapsed_total = time.time() - t_debut
    _sauver_rapport(rapport_path, resultats, stats, t_debut, len(a_traiter),
                    len(a_traiter))

    print(f"\n{'═'*62}")
    print(f"  RAPPORT FINAL")
    print(f"{'═'*62}")
    print(f"  ✓ OK               : {stats['ok']}")
    print(f"  ⚠ Vides (fallback) : {stats['vide']}")
    print(f"  ✗ Erreurs          : {stats['erreur']}")
    print(f"  Durée totale       : {elapsed_total/3600:.1f}h")
    print(f"  Vitesse moyenne    : {elapsed_total/max(1,len(a_traiter)):.1f}s/fichier")
    print(f"  Rapport JSON       : {rapport_path}")
    print(f"{'═'*62}\n")

    # Lister les fichiers à retraiter (vides ou erreurs)
    problemes = {k: v for k, v in resultats.items()
                 if isinstance(v, dict) and v.get("statut") in ("vide", "erreur")}
    if problemes:
        print(f"  ⚠ {len(problemes)} fichier(s) à retraiter avec MinerU :")
        for nom, info in list(problemes.items())[:10]:
            print(f"    {nom[:60]}  ({info.get('message','')[:50]})")
        if len(problemes) > 10:
            print(f"    ... et {len(problemes)-10} autre(s)")
        # Sauvegarder la liste pour MinerU
        fallback_path = output_dir / "fallback_list.txt"
        with open(fallback_path, "w", encoding="utf-8") as f:
            for nom in problemes:
                f.write(nom + "\n")
        print(f"\n  Liste sauvegardée : {fallback_path}")


def _sauver_rapport(path: Path, resultats: dict, stats: dict,
                    t_debut: float, total: int, fait: int) -> None:
    rapport = {
        "_meta": {
            "date":           datetime.now().isoformat(),
            "total_fichiers": total,
            "traites":        fait,
            "stats":          stats,
            "duree_s":        round(time.time() - t_debut, 1),
        },
        **resultats,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", required=True,
        help='Chemin vers le dossier corpus'
    )
    args = parser.parse_args()
    parser_corpus(args.corpus)