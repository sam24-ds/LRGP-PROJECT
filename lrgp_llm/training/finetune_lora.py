"""
finetune_lora.py
Fine-tuning LoRA BF16 de Qwen3.5-9B sur le dataset LRGP.
Stratégie : 3 epochs max + early stopping sur eval_loss.
"""

import os, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]           = "false"

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
import torch
import mlflow

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
MODEL_NAME     = "unsloth/Qwen3.5-9B"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT   = False
DTYPE          = None

# LoRA
LORA_RANK      = 16
LORA_ALPHA     = 16
LORA_DROPOUT   = 0
USE_RSLORA     = False

# Entraînement
LEARNING_RATE  = 2e-4
NUM_EPOCHS     = 3        # max — early stopping peut arrêter avant
BATCH_SIZE     = 2
GRAD_ACCUM     = 4        # batch effectif = 8
WARMUP_STEPS   = 5
WEIGHT_DECAY   = 0.01
LR_SCHEDULER   = "linear"
SEED           = 3407

# Point 3 — eval et save IDENTIQUES
EVAL_STEPS     = 50       # évaluer toutes les 50 steps
SAVE_STEPS     = 50       # sauvegarder toutes les 50 steps (= eval)

# Point 3 — early stopping
EARLY_STOPPING_PATIENCE = 3   # arrêter si pas d'amélioration pendant 3 évals

# Chemins
TRAIN_PATH   = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\test.jsonl")
EVAL_PATH    = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\eval_v4_clean_cleaned.jsonl")
OUTPUT_DIR   = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\training\\training\\checkpoints\\qwen35-9b-lrgp-v5")
LORA_DIR     = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\training\\training\\exports\\qwen35-9b-lrgp-lora-v5")
GGUF_DIR     = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\training\\training\\exports\\qwen35-9b-lrgp-gguf-v5")


# ══════════════════════════════════════════════════════════════════
# CHARGEMENT MODÈLE
# ══════════════════════════════════════════════════════════════════
def charger_modele():
    print(f"\n{'═'*60}")
    print(f"  Chargement {MODEL_NAME}")
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} Go")
    print(f"{'═'*60}\n")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = LOAD_IN_4BIT,
        dtype          = DTYPE,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_RANK,
        target_modules             = "all-linear",
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = LORA_DROPOUT,
        bias                       = "none",
        random_state               = SEED,
        use_rslora                 = USE_RSLORA,
        use_gradient_checkpointing = "unsloth",
    )
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════
# CHARGEMENT DATASET
# ══════════════════════════════════════════════════════════════════
def charger_dataset(tokenizer):
    def lire_jsonl(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    # Point 1 — jeux train et eval séparés
    # eval_final.jsonl = 171 paires déjà séparées dans fusion_datasets.py
    # Le modèle NE s'évalue PAS sur ses données d'entraînement
    train_data = lire_jsonl(TRAIN_PATH)
    eval_data  = lire_jsonl(EVAL_PATH)

    print(f"\n  Train : {len(train_data)} paires")
    print(f"  Eval  : {len(eval_data)} paires  ← séparé, jamais vu pendant l'entraînement")
    n_think = sum(1 for p in train_data if "<think>" in p.get("output",""))
    print(f"  <think> : {n_think}/{len(train_data)} ({n_think/len(train_data)*100:.0f}%)")

    def formater(exemple):
        messages = [
            {"role": "system",    "content": exemple["instruction"]},
            {"role": "user",      "content": exemple["input"]},
            {"role": "assistant", "content": exemple["output"]},
        ]
        texte = tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = False,
        )
        return {"text": texte}

    train_ds = Dataset.from_list(train_data).map(formater)
    eval_ds  = Dataset.from_list(eval_data).map(formater)

    # Aperçu format
    print(f"\n  Aperçu premier exemple :")
    print(f"  {train_ds[0]['text'][:300]}...")
    print(f"  <think> présent : {'<think>' in train_ds[0]['text']}")
    print(f"  im_start présent: {'<|im_start|>' in train_ds[0]['text']}")

    return train_ds, eval_ds


# ══════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════
def entrainer(model, tokenizer, train_ds, eval_ds):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        # Durée
        num_train_epochs            = NUM_EPOCHS,
        max_steps                   = -1,

        # Batch
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,

        # Learning rate
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = LR_SCHEDULER,
        warmup_steps                = WARMUP_STEPS,
        weight_decay                = WEIGHT_DECAY,
        optim                       = "adamw_8bit",

        # Précision
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),

        # Point 2 — eval_strategy = save_strategy OBLIGATOIREMENT
        eval_strategy               = "steps",
        eval_steps                  = EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,   # identique à eval_steps

        # Point 1 — load_best_model_at_end nécessite eval séparé
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,        # on minimise la loss
        save_total_limit            = 3,

        # Logging
        logging_steps               = 1,
        output_dir                  = str(OUTPUT_DIR),
        report_to                   = "none",

        # Dataset
        max_seq_length              = MAX_SEQ_LENGTH,
        dataset_text_field          = "text",
        packing                     = False,
        seed                        = SEED,
        run_name                    = "qwen35-9b-lrgp-v1",
    )

    # Point 3 — EarlyStoppingCallback pour vraiment arrêter si régression
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience  = EARLY_STOPPING_PATIENCE,
        early_stopping_threshold = 0.001,  # amélioration minimale requise
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,   # Point 1 — eval dataset séparé
        args          = training_args,
        callbacks     = [early_stopping],  # Point 3
    )

    # Infos avant lancement
    n_steps_par_epoch = len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)
    print(f"\n{'═'*60}")
    print(f"  PLAN D'ENTRAÎNEMENT")
    print(f"{'═'*60}")
    print(f"  Steps par epoch  : ~{n_steps_par_epoch}")
    print(f"  Steps totaux max : ~{n_steps_par_epoch * NUM_EPOCHS}")
    print(f"  Éval toutes les  : {EVAL_STEPS} steps")
    print(f"  Early stopping   : patience={EARLY_STOPPING_PATIENCE} évals")
    print(f"  Arrêt automatique si val_loss ne s'améliore pas")
    print(f"  Meilleur checkpoint rechargé automatiquement")
    print(f"{'═'*60}\n")

    vram_avant = torch.cuda.memory_reserved() / 1024**3
    print(f"  VRAM réservée avant : {vram_avant:.1f} Go\n")

    # Lancer avec MLflow
    with mlflow.start_run(run_name="qwen35-9b-lrgp-v5"):
        mlflow.log_params({
            "model":          MODEL_NAME,
            "lora_rank":      LORA_RANK,
            "learning_rate":  LEARNING_RATE,
            "num_epochs_max": NUM_EPOCHS,
            "batch_effectif": BATCH_SIZE * GRAD_ACCUM,
            "early_stopping": EARLY_STOPPING_PATIENCE,
            "n_train":        len(train_ds),
            "n_eval":         len(eval_ds),
        })

        trainer_stats = trainer.train()

        mlflow.log_metrics({
            "train_loss_final":  trainer_stats.training_loss,
            "train_runtime_min": trainer_stats.metrics["train_runtime"] / 60,
        })

    # Stats finales
    vram_max = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\n{'═'*60}")
    print(f"  ENTRAÎNEMENT TERMINÉ")
    print(f"  Loss finale       : {trainer_stats.training_loss:.4f}")
    print(f"  Durée réelle      : {trainer_stats.metrics['train_runtime']/60:.1f} min")
    print(f"  VRAM max utilisée : {vram_max:.1f} Go")
    print(f"  Meilleur modèle   : rechargé automatiquement")
    print(f"{'═'*60}\n")

    return trainer


# ══════════════════════════════════════════════════════════════════
# SAUVEGARDE
# ══════════════════════════════════════════════════════════════════
def sauvegarder(model, tokenizer):
    # Adaptateurs LoRA
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(LORA_DIR))
    tokenizer.save_pretrained(str(LORA_DIR))
    print(f"✓ Adaptateurs LoRA → {LORA_DIR}")

    # Export GGUF Q4_K_M pour Ollama
    print(f"\n  Export GGUF Q4_K_M...", end=" ", flush=True)
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(GGUF_DIR),
        tokenizer,
        quantization_method = "q4_k_m",
    )
    print(f"✓")
    print(f"✓ GGUF → {GGUF_DIR}")
    print(f"\n  Commande Ollama :")
    print(f"  ollama create lrgp-expert -f {GGUF_DIR}/Modelfile")
    print(f"  ollama run lrgp-expert")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'═'*60}")
    print(f"  FINE-TUNING LRGP — Qwen3.5-9B LoRA BF16")
    print(f"{'═'*60}")
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} Go")
    print(f"  BF16   : {torch.cuda.is_bf16_supported()}")

    if not TRAIN_PATH.exists():
        print(f"\n❌ Dataset introuvable : {TRAIN_PATH}")
        return
    if not EVAL_PATH.exists():
        print(f"\n❌ Eval dataset introuvable : {EVAL_PATH}")
        return

    model, tokenizer = charger_modele()
    train_ds, eval_ds = charger_dataset(tokenizer)
    entrainer(model, tokenizer, train_ds, eval_ds)
    sauvegarder(model, tokenizer)


if __name__ == "__main__":
    main()