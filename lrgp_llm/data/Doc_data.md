# Documentation — Le dossier `data/`

> Projet **PRISME / SRAR-GP** — Assistant IA pour la recherche en génie des procédés (LRGP)
> Ce document couvre **le dossier `data/`** : d'une part les **scripts de gestion du benchmark** qui y vivent (création, split, contrôle qualité), d'autre part une **cartographie complète des artefacts** (tous les fichiers de données produits par les pipelines d'ingestion et d'évaluation).

> Il complète les documents précédents : `01_ingestion.md`, `02_evaluation_partie1.md`, `03_evaluation_partie2.md`. Les artefacts listés ici sont les *sorties* des scripts documentés là-bas.

---

## 1. Vue d'ensemble

Le dossier `data/` est le **dépôt central de toutes les données** du projet. Il ne contient (presque) pas de logique métier : ce sont les entrées et sorties des pipelines. Deux grandes branches :

```
data/
├── parsed/        # Sortie du parsing Docling (un JSON par document) — voir 01_ingestion.md
│   ├── <document>.json
│   ├── bibliography.json
│   └── parsing_report.json
│
└── datasets/      # Tout ce qui concerne fine-tuning ET benchmark
    ├── benchmark/ # Le benchmark d'évaluation (questions + scripts de gestion)
    ├── batch*/    # Dossiers de travail des jobs Gemini Batch
    ├── reference_pairs/
    └── *.jsonl    # Les datasets de fine-tuning, par version
```

> Note : les chunks (`data/chunks/`) sont produits par l'ingestion et vivent sous `ingestion/data/chunks/` (voir `01_ingestion.md`). Les scripts d'évaluation les lisent à cet emplacement.

Particularité d'organisation : le sous-dossier `data/datasets/benchmark/` contient **du code** (`.py`) en plus des données. Ces scripts gèrent le benchmark et appartiennent logiquement à la chaîne d'évaluation ; ils sont documentés ci-dessous.

---

## 2. Les scripts de gestion du benchmark

Ces quatre scripts vivent dans `data/datasets/benchmark/` et forment la **chaîne amont du benchmark** : sa création, sa consolidation, son contrôle qualité et son découpage. Ils interviennent *avant* l'exécution du benchmark (documentée dans `02_evaluation_partie1.md` via `benchmark_srar_gp.py`).

### 2.1 `fusion_benchmark.py` — le socle du benchmark

C'est le **point de départ historique** du benchmark. Contrairement aux autres générateurs (qui passent par le RAG et un LLM), ce script contient les questions **écrites en dur**, directement dans le code, dans une grande liste `QUESTIONS`.

Ces questions proviennent de trois sources humaines, documentées par leur champ `source` :

| Source | Nature |
|--------|--------|
| `examen_corrige_2023` | Examen corrigé du laboratoire (questions + solutions vérifiées) |
| `polycopie` | Exercices issus du polycopié de cours (adsorption, osmose inverse) |
| `qcm_membranes` | Questions conceptuelles sur les procédés membranaires |

Le script assigne automatiquement les identifiants (`Q001`, `Q002`, …), sauvegarde le tout dans `questions.jsonl`, et affiche les statistiques (répartition par type, domaine, niveau, source). Il signale aussi les **domaines manquants** à compléter (transfert_matière, contacteurs_fibres, CH4_CO2) — précisément ceux que les scripts `extract_context_for_benchmark.py` viendront générer ensuite via le RAG.

C'est donc le **noyau initial** du benchmark, ensuite enrichi par les questions générées depuis le corpus.

```bash
python evaluation/fusion_benchmark.py   # (re)génère questions.jsonl depuis la liste en dur
```

> Les réponses marquées « À compléter » dans la liste sont des emplacements en attente du corrigé du polycopié — repérables ensuite par le contrôle qualité (§2.3).

### 2.2 `integrer_corrige.py` — ajout des questions générées

Une fois des questions générées via le RAG (script `extract_context_for_benchmark.py` → copier-coller dans le LLM → JSON), ce script les **intègre** au benchmark existant. Il lit `reponse_llm_calcul.json` (les nouvelles questions), leur assigne des IDs à la suite des existantes, complète le champ `source` (défaut `corpus_lrgp`), puis réécrit `questions.jsonl` et affiche les statistiques mises à jour.

C'est le pont entre la génération RAG (semi-manuelle) et le benchmark consolidé.

```bash
python evaluation/integrer_corrige.py
```

### 2.3 `benchmark_manager.py` — le couteau suisse du benchmark

Script central de gestion, piloté par `--action`. Il fait quatre choses :

| Action | Rôle |
|--------|------|
| `stats` | Affiche la composition du benchmark (type, domaine, niveau, source) et l'état des splits |
| `check` | `stats` + **contrôle qualité** complet |
| `split` | Découpe le benchmark en train/val/test **anti-contamination** |
| `add` | Ajout **interactif** d'une question en ligne de commande |

**Le contrôle qualité (`check`)** vérifie : champs obligatoires présents, type et niveau valides (parmi `CALCUL/FACTUEL/COMPARAISON/PROCEDURE` et `N1`–`N4`), réponses non vides ou « À compléter », questions assez longues, et surtout **les doublons** — par ID *et* par similarité de texte (60 premiers caractères). C'est ce qui garantit l'intégrité du benchmark avant évaluation.

**Le split anti-contamination (`split`)** est la fonction la plus importante. Il découpe en 60 % train / 10 % val / 30 % test, mais de façon **stratifiée par type** (pour préserver la distribution CALCUL/FACTUEL/COMPARAISON dans chaque partition) et **reproductible** (seed 42). Il garantit au moins une question de chaque type dans le test, puis vérifie explicitement qu'**aucun ID n'apparaît dans deux partitions** (`verifier_contamination_split`). C'est le garde-fou contre la fuite de données entre entraînement et évaluation.

```bash
python evaluation/benchmark_manager.py --action stats
python evaluation/benchmark_manager.py --action check
python evaluation/benchmark_manager.py --action split
python evaluation/benchmark_manager.py --action add
```

> **À noter** : la cible affichée est « 80 questions minimum », alors que le benchmark final en compte 41 (après `fusionner_test_val.py` qui a replié val dans test). L'écart s'explique par l'évolution du projet — la cible initiale n'a pas été atteinte, ce qui est cohérent avec la limite « benchmark de taille modeste » documentée dans le rapport (section 4.6.4).
>
> **Incohérence de chemins** : `BENCHMARK_PATH = "questions.jsonl"` (relatif) et la construction du dossier split (`SPLIT_DIR / "split"`) supposent un répertoire courant précis. À lancer depuis `data/datasets/benchmark/`, ou à fiabiliser avec des chemins absolus.

### 2.4 `debug_json.py` — réparation des sorties LLM

Le LLM générateur renvoie parfois un JSON cassé (tronqué, mal échappé). Ce script de **diagnostic** lit `reponse_llm.json`, tente un parsing direct, et en cas d'échec :

1. affiche le **contexte autour de l'erreur** (position exacte avec un marqueur `^^^`), ce qui aide à comprendre la corruption ;
2. tente une **extraction objet par objet** par expression régulière, ne gardant que les objets JSON valides contenant une `question` ;
3. sauvegarde ce qui a pu être récupéré dans `reponse_llm_corrigee.json`.

C'est l'équivalent, pour les questions de benchmark, de ce que `formatage.py` (partie 1) fait pour les paires de dataset. Outil de rattrapage ponctuel.

```bash
python evaluation/debug_json.py
```

### Chaîne complète de construction du benchmark

```bash
# 1. Socle écrit à la main (examens, polycopié, QCM)
python evaluation/fusion_benchmark.py                 # → questions.jsonl

# 2. Génération RAG des domaines manquants (semi-manuel)
python evaluation/extract_context_for_benchmark.py    # → prompts .txt
#    → copier-coller dans le LLM → reponse_llm_calcul.json
python evaluation/debug_json.py                       # (si JSON cassé)
python evaluation/integrer_corrige.py                 # → questions.jsonl enrichi

# 3. Contrôle et découpage
python evaluation/benchmark_manager.py --action check # qualité + doublons
python evaluation/benchmark_manager.py --action split # train/val/test anti-contamination

# 4. Consolidation finale (val replié dans test)
python evaluation/fusionner_test_val.py               # → benchmark_test.jsonl (41 questions)
```

---

## 3. Cartographie des artefacts — `data/datasets/`

Cette section recense **tous les fichiers de données** visibles dans l'arborescence, avec leur producteur et leur rôle. C'est la table de référence pour s'y retrouver.

### 3.1 Dossiers de travail des jobs Gemini Batch

Chaque pipeline qui appelle Gemini en mode batch a son propre dossier de travail. Tous suivent la même structure (requêtes + état du job + sources/contextes ordonnés + résultats bruts).

#### `batch/` — génération du dataset avec contexte (V1/V2)
Produit par `generate_dataset_finetuning.py`.

| Fichier | Rôle |
|---------|------|
| `batch_requests.jsonl` | Les requêtes envoyées à Gemini |
| `batch_job_state.json` | Nom du job + horodatage (pour le suivi) |
| `contextes.jsonl` | Les contextes RAG, **ordonnés** pour la ré-association par index |
| `test_paires.jsonl` | Sortie du mode `--test` (20 paires de contrôle) |

#### `batch_think/` — enrichissement `<think>`
Produit par `add_think_to_dataset.py`.

| Fichier | Rôle |
|---------|------|
| `think_requests.jsonl` | Requêtes d'enrichissement |
| `think_job_state.json` | État du job |
| `index_map.json` | Map ordonnée des index originaux (ré-association) |
| `dataset_backup.jsonl` | Copie de sauvegarde du dataset complet avant enrichissement |
| `think_enriched.jsonl` | **Sortie** : dataset avec blocs `<think>` ajoutés |

#### `batch_v3/` — génération sans contexte (V3)
Produit par `generate_dataset_fituning_v3.py`. On y voit des fichiers **dupliqués avec suffixe `_2`** (`v3_requests_2.jsonl`, `v3_raw_2.jsonl`, `v3_job_state_2.json`, `v3_sources_2.jsonl`) : ce sont les artefacts d'un **second passage** de génération (le script pointe explicitement vers les variantes `_2`). Les fichiers sans suffixe sont ceux du premier passage.

| Fichier | Rôle |
|---------|------|
| `v3_requests.jsonl` / `_2` | Requêtes (passage 1 / passage 2) |
| `v3_sources.jsonl` / `_2` | Sources ordonnées pour ré-association |
| `v3_job_state.json` / `_2` | État du job |
| `v3_raw.jsonl` / `_2` | Paires brutes générées |
| `v3_batch_results.jsonl` | Résultats consolidés du batch |

### 3.2 Le benchmark — `data/datasets/benchmark/`

| Fichier / dossier | Producteur | Rôle |
|-------------------|------------|------|
| `questions.jsonl` | `fusion_benchmark.py` + `integrer_corrige.py` | Toutes les questions du benchmark consolidées |
| `reponse_llm.json` | LLM (copier-coller) | Réponses brutes du LLM générateur |
| `reponse_llm_calcul.json` | LLM (copier-coller) | Idem, focalisé CALCUL |
| `split/benchmark_train.jsonl` | `benchmark_manager.py --action split` | Partition train (60 %) |
| `split/benchmark_val.jsonl` | idem, puis vidé | Partition val (10 %) — **vidée** par `fusionner_test_val.py` |
| `split/benchmark_test.jsonl` | idem, puis enrichi | Partition test (30 %) → **41 questions finales** |
| `prompts/*.txt` | `extract_context_for_benchmark.py` | 6 prompts de génération (à coller dans le LLM) |
| `prompts_calcul/*.txt` | variante `_for_calcul` | 3 prompts focalisés CALCUL |
| `no_context_pairs.jsonl` | `generate_no_context_pair.py` | Paires no-context dérivées |

### 3.3 Les paires de référence — `reference_pairs/`

Produit par `integrer_paires_reference.py`.

| Fichier | Rôle |
|---------|------|
| `examen_2023.json` | Examen corrigé brut (entrée) |
| `examen_2023_pairs.jsonl` | Paires normalisées (sortie) — vont **toujours dans train** |

### 3.4 Les datasets de fine-tuning (racine `data/datasets/`)

C'est ici que vivent les quatre versions du dataset, à différents stades de traitement. Le tableau les classe par version pour refléter les itérations du rapport (section 4.3).

| Fichier | Version | Stade | Producteur |
|---------|---------|-------|------------|
| `dataset_raw.jsonl` | — | Paires brutes générées | `generate_dataset_finetuning.py` |
| `train.jsonl` / `eval.jsonl` | brut | Premier split 85/15 | idem |
| `train_final.jsonl` / `eval_final.jsonl` | **V1/final** | Fusion corpus + référence | `fusion_datasets.py` |
| `train_clean.jsonl` / `eval_clean.jsonl` | nettoyé | Après audit + re-split | `audit_dataset.py` |
| `no_context_train.jsonl` / `no_context_eval.jsonl` | — | Paires no-context (Split Before Augment) | `generate_no_context_pair.py` |
| `no_context_pairs.jsonl` | — | No-context agrégées | (source de fusion V3) |
| `train_v3.jsonl` / `eval_v3.jsonl` | **V3** | 100 % no-context | `fusion_v3.py` |
| `train_v4_clean.jsonl` / `eval_v4_clean.jsonl` | **V4** | Curé manuellement | (curation) |
| `train_v4_clean_cleaned.jsonl` / `eval_v4_clean_cleaned.jsonl` | **V4** | Après nettoyage expert | `clean_datasetv4.py` |
| `output_v4.jsonl` | V4 | Réparé/normalisé | `formatage.py` |
| `test.jsonl` | — | Fichier de travail temporaire | — |

> La multiplicité des fichiers (`train`, `train_final`, `train_clean`, `train_v3`, `train_v4_clean`, `train_v4_clean_cleaned`) reflète fidèlement la **trajectoire R&D** : chaque version est conservée pour la traçabilité. C'est utile pour la reproductibilité, mais demande une convention de nommage claire (voir §5).

### 3.5 Le dossier `parsed/`

Produit par l'ingestion (voir `01_ingestion.md`). On y trouve un JSON par document parsé (`Alrrane.json`, `Capture.json`, …), plus `bibliography.json` (références BibTeX agrégées) et `parsing_report.json` (le journal de parsing qui pilote la reprise et le chunking).

---

## 4. Schéma de flux des données

```
                  ┌─────────────────── BENCHMARK ───────────────────┐
fusion_benchmark.py ──► questions.jsonl ──► benchmark_manager split ──► split/{train,val,test}
        ▲                     ▲                                              │
   (questions en dur)   integrer_corrige.py                          fusionner_test_val.py
                              ▲                                              │
                   extract_context (RAG) ──► prompts/*.txt              benchmark_test.jsonl (41 Q)
                                                                            │
                                                              benchmark_srar_gp.py ──► *_responses.jsonl
                                                                                              │
                                                            llm_as_juge.py / numerical_match.py
                                                                                              │
                                                            judge_*.json ──► aggregate_judge.py


                  ┌──────────────── FINE-TUNING ────────────────────┐
chunks ──► generate_dataset_finetuning ──► batch/ ──► dataset_raw ──► fusion_* ──► train_vN / eval_vN
       ──► generate_dataset_v3 ──────────► batch_v3/                     ▲
       ──► generate_data_equation ───────► dataset_permeation.jsonl      │
                                                          audit/clean ───┘
                              add_think (batch_think/) ──► think_enriched
```

---

## 5. Points d'attention transversaux

1. **Nommage des datasets** — la chaîne `train` → `train_final` → `train_clean` → `train_v4_clean` → `train_v4_clean_cleaned` est difficile à suivre. Une convention explicite (`train_v{N}_{stade}.jsonl`) et un fichier `DATASETS.md` recensant quelle version a servi à quel fine-tuning faciliteraient grandement la reproductibilité.
2. **Fichiers dupliqués `_2`** dans `batch_v3/` — vestiges d'un second passage de génération. À documenter ou nettoyer pour éviter la confusion sur la source réelle de V3.
3. **Code dans `data/`** — les scripts de benchmark vivent dans `data/datasets/benchmark/` alors qu'ils sont du code d'évaluation. Les déplacer sous `evaluation/` clarifierait la séparation code/données.
4. **Chemins relatifs fragiles** — `benchmark_manager.py`, `integrer_corrige.py` et `debug_json.py` utilisent des chemins relatifs (`questions.jsonl`) qui supposent un répertoire courant précis. À fiabiliser.
5. **Cible « 80 questions »** — affichée par plusieurs scripts mais jamais atteinte (41 questions finales). À aligner sur la réalité, ou à documenter comme objectif initial non tenu (cohérent avec le rapport).
6. **`test.jsonl`** — fichier de travail temporaire à la racine des datasets ; à supprimer s'il n'a plus d'usage.