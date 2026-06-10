"""
fusion_v2.py
Fusionne train_clean + no_context_pairs + référence → dataset V2.
Usage : python evaluation/fusion_v2.py
"""
import json
import random
from pathlib import Path
from collections import Counter

SEED       = 42
RATIO_EVAL = 0.15

TRAIN_CLEAN = Path("data/datasets/train_clean.jsonl")
NO_CTX      = Path("data/datasets/no_context_pairs/no_context_pairs.jsonl")
REF_DIR     = Path("data/datasets/reference_pairs")
OUT_TRAIN   = Path("data/datasets/train_v2.jsonl")
OUT_EVAL    = Path("data/datasets/eval_v2.jsonl")

def charger(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# Charger
train_clean = charger(TRAIN_CLEAN)
no_ctx      = charger(NO_CTX) if NO_CTX.exists() else []
ref_paires  = []
for f in REF_DIR.glob("*.jsonl"):
    ref_paires.extend(charger(f))

print(f"\n{'═'*55}")
print(f"  FUSION DATASET V2")
print(f"{'═'*55}")
print(f"  train_clean (RAG)      : {len(train_clean)}")
print(f"  no_context (knowledge) : {len(no_ctx)}")
print(f"  référence (examens)    : {len(ref_paires)}")

# Séparer référence du train_clean (déjà dedans)
SOURCES_REF = {"examen_2023", "document_reference", "examen_2023_pairs"}
corpus_clean = [p for p in train_clean
                if p.get("source","") not in SOURCES_REF]

# Split corpus_clean 85/15
random.seed(SEED)
all_corpus = corpus_clean + no_ctx
random.shuffle(all_corpus)
n_eval  = int(len(all_corpus) * RATIO_EVAL)
eval_v2 = all_corpus[:n_eval]
train_v2 = all_corpus[n_eval:] + ref_paires

# Stats
types  = Counter(p.get("type","?") for p in train_v2)
n_think = sum(1 for p in train_v2 if "<think>" in p.get("output",""))
nc_count = sum(1 for p in train_v2 if p.get("source","") == "no_context_derived")
rag_count = sum(1 for p in train_v2 if p.get("source","") not in
               ("no_context_derived", "examen_2023", "document_reference"))

print(f"\n  Dataset V2 final :")
print(f"    Paires RAG           : {rag_count}")
print(f"    Paires no-context    : {nc_count}")
print(f"    Paires référence     : {len(ref_paires)}")
print(f"    ─────────────────────")
print(f"    train_v2.jsonl       : {len(train_v2)}")
print(f"    eval_v2.jsonl        : {len(eval_v2)}")
print(f"    Avec <think>         : {n_think}/{len(train_v2)} ({n_think/len(train_v2)*100:.0f}%)")
print(f"    Types : {dict(types)}")

# Sauvegarder
with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    for p in train_v2:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

with open(OUT_EVAL, "w", encoding="utf-8") as f:
    for p in eval_v2:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"\n  ✓ {OUT_TRAIN}")
print(f"  ✓ {OUT_EVAL}")
print(f"{'═'*55}\n")