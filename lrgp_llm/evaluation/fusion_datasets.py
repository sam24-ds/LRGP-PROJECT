# fusion_datasets.py complet
import json
import random
from pathlib import Path
from collections import Counter

SEED       = 42
RATIO_EVAL = 0.15   # 15% eval, 85% train

OUTPUT_TRAIN = Path("data/datasets/train_final.jsonl")
OUTPUT_EVAL  = Path("data/datasets/eval_final.jsonl")

# ── Charger paires corpus (Gemini batch) ──────────────────────────
corpus_pairs = []
# Charger uniquement le fichier brut du batch
# Remplacer la source corpus par le dataset enrichi
enriched_path = Path("data/datasets/batch_think/think_enriched.jsonl")
if enriched_path.exists():
    with open(enriched_path, encoding="utf-8") as f:
        corpus_pairs = [json.loads(l) for l in f if l.strip()]
else:
    # Fallback sur dataset_raw
    with open(Path("data/datasets/dataset_raw.jsonl"), encoding="utf-8") as f:
        corpus_pairs = [json.loads(l) for l in f if l.strip()]
# ── Charger paires référence (examens corrigés) ───────────────────
ref_pairs = []
for f in Path("data/datasets/reference_pairs").glob("*.jsonl"):
    with open(f, encoding="utf-8") as fp:
        ref_pairs.extend([json.loads(l) for l in fp if l.strip()])

print(f"\n{'═'*55}")
print(f"  FUSION DATASETS")
print(f"{'═'*55}")
print(f"  Corpus (Gemini batch) : {len(corpus_pairs)}")
print(f"  Référence (examens)   : {len(ref_pairs)}")
print(f"  Total brut            : {len(corpus_pairs) + len(ref_pairs)}")

# ── Fusionner ─────────────────────────────────────────────────────
toutes = corpus_pairs + ref_pairs

# ── Stats avant split ─────────────────────────────────────────────
types      = Counter(p.get("type","?") for p in toutes)
avec_think = sum(1 for p in toutes if "<think>" in p.get("output",""))
print(f"\n  Par type :")
for t, n in sorted(types.items()):
    print(f"    {t:<15} {n}")
print(f"  Avec <think> : {avec_think}/{len(toutes)} ({avec_think/len(toutes)*100:.0f}%)")

# ── Split train/eval ──────────────────────────────────────────────
# Les paires référence vont TOUJOURS dans train (trop précieuses)
# Les paires corpus sont splittées 85/15
random.seed(SEED)
random.shuffle(corpus_pairs)

n_eval  = int(len(corpus_pairs) * RATIO_EVAL)
eval_   = corpus_pairs[:n_eval]
train   = corpus_pairs[n_eval:] + ref_pairs  # ref → train uniquement

random.shuffle(train)

# ── Sauvegarder ───────────────────────────────────────────────────
with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
    for p in train:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

with open(OUTPUT_EVAL, "w", encoding="utf-8") as f:
    for p in eval_:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"\n  ✓ train_final.jsonl : {len(train)} paires")
print(f"  ✓ eval_final.jsonl  : {len(eval_)} paires")
print(f"{'═'*55}\n")