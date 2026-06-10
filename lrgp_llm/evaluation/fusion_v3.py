"""
fusion_v3.py
Fusionne toutes les paires no-context → dataset V3 100% autonome.
Usage : python evaluation/fusion_v3.py
"""
import json, random
from pathlib import Path
from collections import Counter

SEED = 42

# Sources
NO_CTX_EXISTANT = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\no_context_pairs.jsonl")  # 378
NO_CTX_V3       = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\batch_v3\\v3_raw.jsonl")   # ~600
REF_DIR         = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\reference_pairs")          # 21

OUT_TRAIN = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\train_v3.jsonl")
OUT_EVAL  = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\eval_v3.jsonl")

def charger(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# Charger toutes les sources
nc_existant = charger(NO_CTX_EXISTANT) if NO_CTX_EXISTANT.exists() else []
nc_v3       = charger(NO_CTX_V3)       if NO_CTX_V3.exists()       else []
ref_paires  = []
for f in REF_DIR.glob("*.jsonl"):
    ref_paires.extend(charger(f))

print(f"\n{'═'*55}")
print(f"  FUSION DATASET V3 — 100% No-Context")
print(f"{'═'*55}")
print(f"  No-context dérivés  : {len(nc_existant)}")
print(f"  No-context nouveaux : {len(nc_v3)}")
print(f"  Référence examens   : {len(ref_paires)}")

# Fusionner corpus no-context (sans référence)
corpus = nc_existant + nc_v3
print(f"  Corpus total        : {len(corpus)}")

# Split Before Augment — split corpus 85/15
random.seed(SEED)
random.shuffle(corpus)
n_eval  = int(len(corpus) * 0.15)
eval_nc = corpus[:n_eval]
train_nc = corpus[n_eval:]

# Référence → train uniquement
train_v3 = train_nc + ref_paires
eval_v3  = eval_nc

random.shuffle(train_v3)
random.shuffle(eval_v3)

# Stats
n_think_train = sum(1 for p in train_v3 if "<think>" in p.get("output",""))
n_think_eval  = sum(1 for p in eval_v3  if "<think>" in p.get("output",""))
types_train   = Counter(p.get("type","?") for p in train_v3)

print(f"\n  Dataset V3 final :")
print(f"    train_v3.jsonl : {len(train_v3)} paires")
print(f"    eval_v3.jsonl  : {len(eval_v3)} paires")
print(f"    Avec <think>   : {n_think_train}/{len(train_v3)} ({n_think_train/len(train_v3)*100:.0f}%)")
print(f"    Types          : {dict(types_train)}")

# Sauvegarder
with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    for p in train_v3:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
with open(OUT_EVAL, "w", encoding="utf-8") as f:
    for p in eval_v3:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"\n  ✓ {OUT_TRAIN}")
print(f"  ✓ {OUT_EVAL}")
print(f"\n  Dans finetune_lora.py :")
print(f"    TRAIN_PATH = Path('data/datasets/train_v3.jsonl')")
print(f"    EVAL_PATH  = Path('data/datasets/eval_v3.jsonl')")
print(f"{'═'*55}\n")