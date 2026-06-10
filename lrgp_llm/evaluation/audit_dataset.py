"""
audit_dataset.py
Audit et nettoyage du dataset de fine-tuning LRGP.
Charge train_final + eval_final ensemble, audite tout,
puis refait un split propre 85/15.

Usage : python evaluation/audit_dataset.py
"""
import json
import re
import random
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────
TRAIN_PATH       = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\train_final.jsonl")
EVAL_PATH        = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\eval_final.jsonl")
TRAIN_CLEAN_PATH = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\train_clean.jsonl")
EVAL_CLEAN_PATH  = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\eval_clean.jsonl")
RATIO_EVAL       = 0.15
SEED             = 42

# ── Regex précise — évite les faux positifs ───────────────────────
regex_qcm = re.compile(r'\n[A-D]\)|\b[A-D]\)\s')

# ── Chargement ────────────────────────────────────────────────────
with open(TRAIN_PATH, encoding="utf-8") as f:
    train_paires = [json.loads(l) for l in f if l.strip()]

with open(EVAL_PATH, encoding="utf-8") as f:
    eval_paires = [json.loads(l) for l in f if l.strip()]

print(f"\n{'═'*55}")
print(f"  AUDIT DATASET LRGP — COMPLET")
print(f"{'═'*55}")
print(f"  train_final.jsonl : {len(train_paires)} paires")
print(f"  eval_final.jsonl  : {len(eval_paires)} paires")
print(f"  Total             : {len(train_paires) + len(eval_paires)} paires")

# ── Séparer paires référence (examens corrigés) ───────────────────
# Les paires référence ont source="examen_2023" ou "document_reference"
# Elles vont toujours dans train — jamais dans eval
SOURCES_REFERENCE = {"examen_2023", "document_reference", "examen_2023_pairs"}

ref_paires = [
    p for p in train_paires
    if p.get("source", "") in SOURCES_REFERENCE
]
corpus_train = [
    p for p in train_paires
    if p.get("source", "") not in SOURCES_REFERENCE
]

# Corpus complet = corpus train + eval (à re-splitter proprement)
corpus_total = corpus_train + eval_paires

print(f"\n  Dont paires référence : {len(ref_paires)} (→ train uniquement)")
print(f"  Dont corpus total     : {len(corpus_total)} (→ à auditer)")

# ── Détection des problèmes ───────────────────────────────────────
def est_problematique(p: dict) -> bool:
    output = p.get("output", "")
    input_ = p.get("input", "")
    return (
        bool(regex_qcm.search(output)) or
        bool(regex_qcm.search(input_)) or
        len(output) < 100 or
        "à compléter" in output.lower() or
        "..." in output or
        any(x in output.lower() for x in [
            "transfert de chaleur", "dtlm",
            "échange thermique", "thermique"
        ])
    )

qcm         = [p for p in corpus_total
               if regex_qcm.search(p["output"]) or regex_qcm.search(p["input"])]
trop_court  = [p for p in corpus_total if len(p["output"]) < 100]
a_completer = [p for p in corpus_total
               if "à compléter" in p["output"].lower() or "..." in p["output"]]
chaleur     = [p for p in corpus_total
               if any(x in p["output"].lower() for x in [
                   "transfert de chaleur", "dtlm",
                   "échange thermique", "thermique"
               ])]

# Stats par fichier source
qcm_train = [p for p in qcm if p in corpus_train]
qcm_eval  = [p for p in qcm if p in eval_paires]
ch_train  = [p for p in chaleur if p in corpus_train]
ch_eval   = [p for p in chaleur if p in eval_paires]

print(f"\n{'─'*55}")
print(f"  {'Problème':<30} {'Train':>6} {'Eval':>6} {'Total':>7}")
print(f"{'─'*55}")
print(f"  {'QCM':<30} {len(qcm_train):>6} {len(qcm_eval):>6} {len(qcm):>7}")
print(f"  {'Trop courtes (<100 chars)':<30} {'-':>6} {'-':>6} {len(trop_court):>7}")
print(f"  {'À compléter / ...':<30} {'-':>6} {'-':>6} {len(a_completer):>7}")
print(f"  {'Confusion chaleur/matière':<30} {len(ch_train):>6} {len(ch_eval):>6} {len(chaleur):>7}")

paires_pb = {id(p) for p in qcm + trop_court + a_completer + chaleur}
n_pb = len(paires_pb)
print(f"{'─'*55}")
print(f"  {'TOTAL À SUPPRIMER':<30} {'':>6} {'':>6} {n_pb:>7} ({n_pb/len(corpus_total)*100:.1f}%)")

if qcm:
    print(f"\n  Exemple QCM :")
    print(f"  {qcm[0]['output'][:250]}")
if chaleur:
    print(f"\n  Exemple confusion chaleur :")
    print(f"  {chaleur[0]['output'][:250]}")

# ── Nettoyage ─────────────────────────────────────────────────────
corpus_propres  = [p for p in corpus_total if not est_problematique(p)]
corpus_rejetes  = [p for p in corpus_total if est_problematique(p)]

print(f"\n{'═'*55}")
print(f"  NETTOYAGE")
print(f"{'═'*55}")
print(f"  Corpus propres    : {len(corpus_propres)}")
print(f"  Corpus rejetés    : {len(corpus_rejetes)}")
print(f"  Référence (train) : {len(ref_paires)}")

# Re-split 85/15 sur corpus propres
random.seed(SEED)
random.shuffle(corpus_propres)
n_eval_  = int(len(corpus_propres) * RATIO_EVAL)
eval_clean  = corpus_propres[:n_eval_]
train_clean = corpus_propres[n_eval_:] + ref_paires  # référence → train

# Stats <think>
n_think = sum(1 for p in train_clean if "<think>" in p.get("output", ""))
print(f"\n  Split final :")
print(f"    train_clean.jsonl : {len(train_clean)} paires")
print(f"    eval_clean.jsonl  : {len(eval_clean)} paires")
print(f"    Avec <think>      : {n_think}/{len(train_clean)} ({n_think/len(train_clean)*100:.0f}%)")

# ── Sauvegarde ────────────────────────────────────────────────────
with open(TRAIN_CLEAN_PATH, "w", encoding="utf-8") as f:
    for p in train_clean:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

with open(EVAL_CLEAN_PATH, "w", encoding="utf-8") as f:
    for p in eval_clean:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"\n  ✓ {TRAIN_CLEAN_PATH}")
print(f"  ✓ {EVAL_CLEAN_PATH}")
print(f"{'═'*55}\n")