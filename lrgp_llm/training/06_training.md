# Documentation — Module Training (fine-tuning LoRA)

> Projet **PRISME / SRAR-GP** — Assistant IA pour la recherche en génie des procédés (LRGP)
> Module : `training/`
> Ce module réalise l'**adaptation du modèle de base au domaine** par fine-tuning LoRA. C'est ici que sont produites les quatre versions V1→V4 du rapport (section 4.3), puis exportées au format GGUF pour Ollama.

> Il consomme les datasets produits par `evaluation/` (`04_data.md`) et produit le modèle `lrgp-expert` / `lrgp-knowledge_vN` utilisé par `rag/` (`05_rag.md`) et par l'agent *Librarian* de PRISME.

---

## 1. Vue d'ensemble

Le fine-tuning adapte le **comportement** du modèle (vocabulaire, ton, structure des réponses, discipline de citation) sans réentraîner depuis zéro. Conformément au rapport, c'est un fine-tuning **LoRA** (on fige les poids du modèle de base et on n'entraîne que de petites matrices d'adaptation) via **Unsloth** (intégration simple, efficace en mémoire).

Le module fait deux choses :

1. **Entraîner** (`finetune_lora.py`) — charge Qwen3.5-9B, applique LoRA, entraîne sur le dataset LRGP avec évaluation et early stopping, puis exporte les adaptateurs LoRA et un GGUF quantizé Q4_K_M.
2. **Déployer** (`create_model_file.py`) — génère le `Modelfile` Ollama qui rend le GGUF utilisable comme modèle local.

```
dataset (train_vN / eval_vN)
        ↓
Qwen3.5-9B + LoRA  ──entraînement Unsloth──►  meilleur checkpoint
        ↓
export adaptateurs LoRA  +  export GGUF Q4_K_M
        ↓
Modelfile Ollama  →  ollama create lrgp-expert
```

---

## 2. Arborescence du module

```
training/
├── finetune_lora.py        # ★ Script d'entraînement LoRA complet
├── create_model_file.py    # Génère le Modelfile Ollama pour le GGUF
├── mlflow.db               # Base MLflow (suivi des runs : params, métriques)
├── configs/                # Configurations d'entraînement
└── training/
    ├── checkpoints/        # Checkpoints d'entraînement, un dossier par version
    │   ├── qwen35-9b-lrgp        (V1)
    │   ├── qwen35-9b-lrgp-v2     (V2)
    │   ├── qwen35-9b-lrgp-v3     (V3)
    │   ├── qwen35-9b-lrgp-v4     (V4)
    │   └── qwen35-9b-lrgp-v5     (V5, en cours)
    └── exports/            # Modèles exportés (LoRA + GGUF), par version
        ├── qwen35-9b-lrgp-lora[-vN]    # adaptateurs LoRA légers
        └── qwen35-9b-lrgp-gguf[-vN]    # GGUF quantizé pour Ollama
```

L'arborescence reflète fidèlement la **trajectoire R&D** du rapport : cinq jeux de checkpoints (V1→V5) et, pour chaque version, un export LoRA *et* un export GGUF. C'est la matérialisation des itérations de fine-tuning.

---

## 3. Le script d'entraînement — `finetune_lora.py`

### 3.1 Le modèle de base et LoRA

Le script charge **Qwen3.5-9B** via Unsloth (`FastLanguageModel.from_pretrained`) puis y greffe les adaptateurs LoRA (`get_peft_model`) :

| Paramètre LoRA | Valeur | Rôle |
|----------------|--------|------|
| `LORA_RANK` (r) | 16 | Dimension des matrices d'adaptation |
| `LORA_ALPHA` | 16 | Facteur d'échelle |
| `LORA_DROPOUT` | 0 | Pas de dropout |
| `target_modules` | `all-linear` | Toutes les couches linéaires adaptées |
| `use_gradient_checkpointing` | `"unsloth"` | Économie mémoire (mode Unsloth optimisé) |

> **Note de précision** : Le script charge le modèle en **BF16** (`LOAD_IN_4BIT = False`), pas en 4-bit. C'est plus gourmand en VRAM mais de meilleure qualité d'entraînement. La quantization 4-bit n'intervient qu'à l'export GGUF (pour l'inférence Ollama), pas pendant l'entraînement. Cette distinction est importante : on entraîne en pleine précision, on déploie en quantizé.

### 3.2 Le formatage du dataset

`charger_dataset()` lit les fichiers `train`/`eval` (JSONL) et formate chaque paire au **format chat** du modèle via `tokenizer.apply_chat_template`, avec trois rôles :

```
system    → exemple["instruction"]
user      → exemple["input"]
assistant → exemple["output"]   (contient le bloc <think> pour les calculs)
```

Le script affiche des contrôles utiles avant l'entraînement : nombre de paires, pourcentage de paires avec `<think>`, et un aperçu vérifiant que les balises de chat (`<|im_start|>`) et le `<think>` sont bien présents dans le texte formaté.

> **Séparation train/eval stricte** : le dataset d'évaluation est séparé et **jamais vu pendant l'entraînement**. C'est ce qui rend `eval_loss` significative et permet l'early stopping (voir ci-dessous). Cette séparation a été construite en amont par les scripts de fusion (`04_data.md`).

### 3.3 La stratégie d'entraînement

| Hyperparamètre | Valeur | Commentaire |
|----------------|--------|-------------|
| `LEARNING_RATE` | 2 × 10⁻⁴ | Constant entre les versions (démarche d'isolation de variable) |
| `NUM_EPOCHS` | 3 (max) | Early stopping peut arrêter avant |
| `BATCH_SIZE` | 2 | Par device |
| `GRAD_ACCUM` | 4 | → **batch effectif = 8** |
| `WARMUP_STEPS` | 5 | |
| `WEIGHT_DECAY` | 0.01 | |
| `LR_SCHEDULER` | linéaire | |
| `optim` | `adamw_8bit` | Optimiseur 8-bit (économie mémoire) |
| `SEED` | 3407 | Reproductibilité |
| précision | BF16 si supporté, sinon FP16 | Détection automatique |

La logique d'entraînement repose sur trois mécanismes liés, soigneusement commentés dans le code (« Point 1/2/3 ») :

1. **Évaluation périodique** — toutes les 50 steps (`EVAL_STEPS`), la perte est mesurée sur le jeu d'eval séparé.
2. **Sauvegarde alignée** — `save_steps = eval_steps` (impératif technique pour `load_best_model_at_end`).
3. **Early stopping + meilleur modèle** — `EarlyStoppingCallback(patience=3, threshold=0.001)` arrête l'entraînement si `eval_loss` ne s'améliore pas pendant 3 évaluations consécutives, et `load_best_model_at_end=True` recharge automatiquement le **meilleur** checkpoint (pas le dernier). On minimise `eval_loss` (`greater_is_better=False`), en gardant au plus 3 checkpoints (`save_total_limit`).

Cette mécanique est ce qui protège contre le **surapprentissage** : plutôt que de figer 3 epochs aveuglément, on s'arrête dès que le modèle commence à régresser sur des données qu'il n'a jamais vues.

### 3.4 Suivi MLflow

L'entraînement est encapsulé dans un run **MLflow** (`mlflow.db` à la racine du module). Sont loggés : les paramètres (modèle, rang LoRA, learning rate, batch effectif, patience, tailles train/eval) et les métriques finales (loss d'entraînement, durée). C'est ce qui permet de comparer rigoureusement les runs V1→V5 — le support quantitatif des « loss finales » du tableau du rapport (0,857 → 0,764 → 0,763 → 0,580).

### Usage

Dans le dossier training lancer : 

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000

```

### 3.5 La sauvegarde et l'export

`sauvegarder()` produit deux artefacts :

1. **Adaptateurs LoRA** (`save_pretrained`) — fichier léger (quelques centaines de Mo) qui se charge par-dessus le modèle de base. C'est le format de référence pour conserver/recharger l'adaptation.
2. **GGUF Q4_K_M** (`save_pretrained_gguf` avec `quantization_method="q4_k_m"`) — le modèle fusionné et quantizé en 4 bits, prêt pour Ollama. C'est là que la quantization intervient, pour l'inférence locale.

Le script affiche directement la commande Ollama de création (`ollama create lrgp-expert -f .../Modelfile`).

### Usage

```bash
python training/finetune_lora.py
```

Le script vérifie l'existence des datasets, affiche le plan d'entraînement (steps/epoch, steps max, fréquence d'éval), entraîne, puis exporte. Il refuse de démarrer si un dataset est manquant.

---

## 4. La création du Modelfile — `create_model_file.py`

Petit utilitaire qui écrit le `Modelfile` Ollama du GGUF exporté. Le `Modelfile` indique à Ollama :

- le chemin du fichier GGUF (`FROM …Qwen3.5-9B.Q4_K_M.gguf`) ;
- les **tokens d'arrêt** (`<|endoftext|>`, `<|im_end|>`, `Human:`) — indispensables pour que le modèle s'arrête proprement en fin de réponse et ne « déborde » pas.

Une fois ce fichier écrit, `ollama create lrgp-expert -f <Modelfile>` enregistre le modèle dans Ollama, et il devient appelable par `rag/` et par PRISME.

```bash
python training/create_model_file.py
ollama create lrgp-expert -f training/.../Modelfile
```

> Ce script est un **correctif de chemin** : `finetune_lora.py` génère déjà un Modelfile à l'export GGUF, mais celui-ci peut pointer vers un chemin relatif incorrect. `create_model_file.py` le réécrit avec le chemin absolu correct et les bons tokens d'arrêt. C'est un rustine pratique, pas une étape conceptuellement nécessaire.

---

## 5. Divergences entre le code et le rapport

⚠️ Plusieurs écarts entre ce script (état actuel, orienté **V5**) et les paramètres documentés dans le rapport pour V1→V4. À réconcilier dans la doc finale, car ils peuvent prêter à confusion.

| Élément | Rapport (V1–V4) | Code actuel (`finetune_lora.py`) |
|---------|-----------------|----------------------------------|
| `LORA_ALPHA` | 32 | **16** |
| Batch effectif | 4 (batch size 4) | **8** (2 × 4) |
| Quantization entraînement | 4 bits | **BF16** (4-bit seulement à l'export) |
| `MAX_SEQ_LENGTH` | non précisé | 4096 |
| Early stopping | non mentionné | **présent** (patience 3) |
| `SEED` | non précisé | 3407 |

Ces écarts s'expliquent vraisemblablement par l'évolution vers **V5** (le script porte partout le suffixe `v5` : `OUTPUT_DIR`, `LORA_DIR`, `GGUF_DIR`, `run_name` MLflow). Le tableau du rapport (Annexe IV) décrivait la config figée des quatre premières itérations ; le code montre la configuration **en cours** pour V5. **Recommandation** : préciser dans la doc finale quelle config a produit quel modèle, idéalement en archivant un fichier de config par version dans `training/configs/`.

> Incohérence mineure supplémentaire : `run_name` du `SFTConfig` est `"qwen35-9b-lrgp-v1"` alors que le run MLflow est `"qwen35-9b-lrgp-v5"`. Vestige de copier-coller à corriger.

---

## 6. Le chemin TRAIN_PATH pointe vers `test.jsonl`

⚠️ Point à vérifier : dans le script fourni, `TRAIN_PATH` pointe vers `data/datasets/test.jsonl` et non vers un `train_v*.jsonl`. Or `test.jsonl` est décrit ailleurs (`04_data.md`) comme un **fichier de travail temporaire**.

Deux lectures possibles : soit c'est un état de test laissé en place (le vrai entraînement pointait vers `train_v4_clean_cleaned.jsonl`), soit `test.jsonl` contenait effectivement le dataset d'entraînement V5 au moment du run. L'`EVAL_PATH`, lui, pointe correctement vers `eval_v4_clean_cleaned.jsonl`. **À clarifier avant tout réentraînement** : remettre `TRAIN_PATH` sur le bon fichier de train versionné.

---

## 7. Place du module dans l'ensemble

```
evaluation/  ──produit──►  train_vN.jsonl / eval_vN.jsonl   (04_data.md)
                                  │
                          training/finetune_lora.py
                                  │
              ┌───────────────────┴───────────────────┐
       adaptateurs LoRA                          GGUF Q4_K_M
    (exports/...-lora-vN)                    (exports/...-gguf-vN)
                                                   │
                                          create_model_file.py
                                                   │
                                          ollama create lrgp-expert
                                                   │
                          ┌────────────────────────┴───────────────┐
                   rag/ (V4+RAG, 05_rag.md)              srar_gp/ (agent Librarian)
```

Le modèle fine-tuné produit ici est utilisé à deux endroits : comme générateur de la chaîne « V4+RAG » (module `rag/`), et — c'est son rôle dans l'architecture finale — comme **agent *Librarian*** de PRISME, le documentaliste spécialisé qui maîtrise le jargon du domaine. Conformément au rapport, le fine-tuning n'a pas suffi seul ; sa vraie valeur se révèle intégré dans l'architecture multi-agents, prochain module à documenter : **`srar_gp/`**.

---

## 8. Points d'attention transversaux

1. **`TRAIN_PATH` → `test.jsonl`** — à corriger pour pointer vers le dataset de train versionné (voir §6).
2. **Config code = V5, rapport = V1–V4** — documenter la correspondance version ↔ hyperparamètres ; archiver une config par version dans `configs/` (voir §5).
3. **`run_name` incohérent** (`v1` dans SFTConfig vs `v5` dans MLflow) — vestige à nettoyer.
4. **Chemins absolus Windows** — comme partout, à externaliser pour la portabilité.
5. **Modelfile à deux sources** — `finetune_lora.py` (export) et `create_model_file.py` (correctif) ; unifier pour éviter la confusion sur le chemin GGUF effectif.
6. **Nommage des modèles Ollama** — `lrgp-expert` (cette doc), `lrgp-knowledge_v5` / `lrgp-knowledge_v2` (module rag) : même famille, noms divergents. À consolider (rappel de `05_rag.md`).