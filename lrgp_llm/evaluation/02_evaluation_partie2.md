# Documentation — Évaluation (partie 2/2)

> Projet **PRISME / SRAR-GP** — Assistant IA pour la recherche en génie des procédés (LRGP)
> Module : `evaluation/`
> Seconde partie : **génération automatisée des paires de fine-tuning** (les itérations V1→V4), **fusion versionnée des datasets**, **LLM-Judge** (le moteur de notation), **métrique Numerical Match**, et les **utilitaires de notation / mesure**.

> Cette partie complète la partie 1/2. Elle documente notamment le script qui *produit* les fichiers `judge_*.json` consommés par `analyser_juge.py` et `aggregate_judge.py`, ainsi que la métrique `numerical_match` annoncée précédemment.

---

## 1. Où l'on en est

La partie 1 couvrait : la génération *semi-manuelle* des questions de benchmark (via prompts copiés-collés), le nettoyage des datasets, l'enrichissement `<think>`, l'exécution du benchmark et la lecture des notes.

Cette partie 2 couvre la chaîne **automatisée** qui a réellement produit les quatre versions du dataset de fine-tuning du rapport (section 4.3), plus le cœur de l'évaluation (le juge et les métriques). La logique d'ensemble :

```
chunks corpus  →  génération Gemini Batch  →  paires brutes  →  fusion + split  →  train/eval Vn
                                                                                        │
réponses des 4 systèmes  →  LLM-Judge (Gemini Batch)  →  judge_*.json  →  agrégation (partie 1)
                         →  Numerical Match (local)    →  taux CALCUL
```

---

## 2. Arborescence des fichiers (partie 2)

```
evaluation/
│
├── ─── Génération des paires de fine-tuning ───
│   ├── generate_dataset_finetuning.py      # Génération AVEC contexte RAG (V1/V2) — Gemini Batch
│   ├── generate_dataset_fituning_v3.py     # Génération SANS contexte (V3) — Gemini Batch
│   ├── generate_data_equation.py           # Génération SYNTHÉTIQUE de calculs (déterministe, sans LLM)
│   ├── generate_no_context_pair.py         # Dérive des paires no-context depuis les paires RAG
│   └── integrer_paires_reference.py        # Intègre les examens corrigés (paires de référence)
│
├── ─── Fusion & split versionnés ───
│   ├── fusion_datasets.py                  # Fusion V1/final (corpus + référence)
│   ├── fusion_2.py                         # Fusion V2 (+ no-context)
│   ├── fusion_v3.py                        # Fusion V3 (100% no-context)
│   └── fusionner_test_val.py               # Fusionne benchmark_val dans benchmark_test
│
├── ─── LLM-Judge & métriques ───
│   ├── llm_as_juge.py                      # ★ Le juge : note les réponses via Gemini (grille 5 critères)
│   ├── numerical_match.py                  # Métrique de justesse numérique (questions CALCUL)
│   ├── prepare_for_gemini.py               # Prépare le JSON à coller dans Gemini (jugement manuel SRAR-GP)
│   └── generer_formulaire_excel.py         # Convertit le formulaire de notation en Excel (notation humaine)
│
└── ─── Utilitaires ponctuels ───
    ├── temps_responses.py                  # Stats de temps de réponse par système
    └── relance_q052.py                     # Relance d'une question unique qui avait crashé (jetable)
```

---

## 3. Génération des paires de fine-tuning

Ces scripts sont ceux qui ont concrètement produit les datasets V1 à V4. Tous reposent sur le **Gemini Batch API** (asynchrone, ~50 % moins cher, pas d'erreurs 503), sauf le générateur synthétique d'équations qui est purement local.

### 3.1 Génération avec contexte RAG — `generate_dataset_finetuning.py`

C'est le générateur qui a produit les datasets **V1 et V2** (paires « avec contexte »). Le modèle apprend à répondre dans le style RAG, avec le contexte présent dans le prompt.

**Pipeline interne :**

1. **Chargement et filtrage des chunks** — lit tous les `*_chunks.jsonl`, ne garde que les chunks ≥ 200 caractères contenant un mot-clé du domaine (membrane, perméabilité, flux, CO2, CH4, K_OV…) et écarte les débuts de bibliographie/copyright.
2. **Enrichissement par chunks connexes** — pour chaque chunk, une recherche Qdrant remonte 3 passages voisins, concaténés au chunk d'origine pour former un contexte plus riche (plafonné à 4000 caractères).
3. **Génération** — chaque contexte est inséré dans `PROMPT_GENERATION`, qui demande à Gemini de produire 5 paires (`BATCH_SIZE`) au format JSON, en respectant les règles : bloc `<think>` obligatoire pour les CALCUL/COMPARAISON, citation `[Source: ...]`, 70-80 % de paires de calcul.
4. **Anti-contamination** — chaque paire générée est comparée par similarité d'embedding (BGE-M3) aux questions du benchmark ; au-delà d'un seuil de 0,85, elle est rejetée pour éviter toute fuite entre données d'entraînement et benchmark.
5. **Filtre qualité** — rejet des paires dont `qualite_estimee` < 4.
6. **Split** 85/15 → `train.jsonl` / `eval.jsonl`, plus `dataset_raw.jsonl`.

**Workflow batch en 4 commandes** (comme tous les scripts Gemini Batch du projet) :

```bash
python evaluation/generate_dataset_finetuning.py --prepare   # construit les requêtes + contextes.jsonl
python evaluation/generate_dataset_finetuning.py --submit    # soumet le batch
python evaluation/generate_dataset_finetuning.py --status     # suit l'avancement
python evaluation/generate_dataset_finetuning.py --collect    # collecte, anti-contamine, split
python evaluation/generate_dataset_finetuning.py --all        # prepare + submit
python evaluation/generate_dataset_finetuning.py --test       # test synchrone (20 paires, sans batch)
```

**Liaison par index** : comme pour l'enrichissement `<think>` (partie 1), les réponses du batch sont ré-associées à leur contexte par **ordre de soumission** (`contextes.jsonl` lu en liste ordonnée). Les clés locales préfixées `_` (`_custom_id`, `_contexte`) sont retirées avant l'envoi car interdites dans la requête Gemini.

> Cible : `N_PAIRES_CIBLE = 1500`, avec un `RATIO_GENERATION = 1.35` (on génère ~35 % de plus pour compenser les rejets). C'est ce qui donne les volumes V1 (994) et V2 (908) après nettoyage.

### 3.2 Génération sans contexte — `generate_dataset_fituning_v3.py`

C'est le générateur du dataset **V3**, celui de l'hypothèse « sans contexte » du rapport : faire répondre le modèle *de mémoire* plutôt qu'en reformulant un contexte fourni.

Différences clés avec le générateur V1/V2 :

- Les chunks sont **regroupés par article** (source), pas traités isolément ; on prend les 3 premiers chunks de chaque source comme « contenu scientifique » (jusqu'à 5000 caractères).
- Le prompt `PROMPT_V3` demande des paires **autonomes** : les réponses ne doivent jamais mentionner « le document » ou « le contexte » (liste `TRACES_CONTEXTE` pour le contrôle).
- L'`input` ne contient que `Question : ...`, sans bloc contexte.
- Répartition visée : 60 % CALCUL/COMPARAISON, 40 % FACTUEL.

Même workflow batch (`--prepare / --submit / --status / --collect`). C'est ce dataset qui a révélé le défaut décrit dans le rapport : privé de contexte, le modèle se mettait à halluciner du vocabulaire médical (« membrane » → dialyse rénale).

### 3.3 Génération synthétique d'équations — `generate_data_equation.py`

Script à part : **aucun appel LLM**, génération 100 % déterministe et locale. Il produit des paires de calcul de perméation gazeuse en tirant des variables aléatoires réalistes (concentration d'entrée, sélectivité, rapport de pression), puis en **résolvant l'équation quadratique** d'égalité des flux à travers une membrane dense. Chaque paire contient un bloc `<think>` détaillant le développement (équation, valeurs, discriminant, racine).

C'est ce script qui a produit `dataset_permeation.jsonl` (103 paires CALCUL, toutes avec `<think>`). Son intérêt : fournir des exemples de calcul **mathématiquement exacts par construction**, sans risque d'hallucination du générateur — précisément le type de données qui manquait pour les questions de raisonnement multi-étapes (limite V4 du rapport).

```bash
python evaluation/generate_data_equation.py   # → dataset_permeation.jsonl (100 paires)
```

> Réserve : les valeurs générées sont parfois physiquement extrêmes (ex. `y_max = 120 %`, signalé comme « taux de coupe nul θ → 0 »). Le réalisme physique de certaines paires synthétiques mériterait un contrôle.

### 3.4 Dérivation no-context — `generate_no_context_pair.py`

Plutôt que de regénérer, ce script **transforme** des paires existantes « avec contexte » en paires « sans contexte » : il retire le bloc `Contexte :` de l'`input` et conserve l'`output` à l'identique. C'est une alternative économique à la génération V3 (pas d'appel LLM).

Filtres d'éligibilité : qualité ≥ 4, output ≥ 150 caractères, source non-référence, présence d'un bloc `Contexte :` à retirer, et **absence de traces contextuelles** dans l'output (une réponse qui dit « d'après le document » devient incohérente une fois le contexte retiré — elle est donc rejetée).

Principe **Split Before Augment** : la dérivation se fait séparément sur `train_clean` et `eval_clean`, pour ne jamais mélanger les deux ensembles.

```bash
python evaluation/generate_no_context_pair.py --source train   # ou eval, ou both
```

### 3.5 Intégration des paires de référence — `integrer_paires_reference.py`

Les **examens corrigés** du laboratoire constituent des paires de très haute qualité (vérifiées humainement, `qualite_estimee = 5`). Ce script les valide, les normalise (champs requis, type par défaut, source) et les range dans `data/datasets/reference_pairs/<source>_pairs.jsonl`. Il accepte du JSON ou du JSONL en entrée.

Ces paires ont un statut spécial dans tout le projet : elles vont **toujours dans train, jamais dans eval** (voir fusions ci-dessous), car trop précieuses pour être « gaspillées » en évaluation et pour éviter toute fuite.

```bash
python evaluation/integrer_paires_reference.py --fichier "data/datasets/reference_pairs/examen_2023.json"
```

---

## 4. Fusion et split versionnés

Chaque version du dataset a sa propre fusion. Ce ne sont pas des doublons : chacune correspond à une itération expérimentale du rapport. Toutes partagent deux invariants — **seed 42**, et **les paires de référence vont uniquement dans train**.

| Script | Version | Sources fusionnées | Sortie |
|--------|---------|--------------------|--------|
| `fusion_datasets.py` | **V1 / final** | dataset enrichi `<think>` (ou `dataset_raw`) + référence | `train_final.jsonl` / `eval_final.jsonl` |
| `fusion_2.py` | **V2** | `train_clean` (RAG) + no-context + référence | `train_v2.jsonl` / `eval_v2.jsonl` |
| `fusion_v3.py` | **V3** | no-context existant + no-context nouveaux + référence | `train_v3.jsonl` / `eval_v3.jsonl` |

Chaque fusion affiche les mêmes statistiques de contrôle : répartition par type, pourcentage de paires avec `<think>`, comptes par origine.

```bash
python evaluation/fusion_datasets.py   # V1/final
python evaluation/fusion_2.py          # V2
python evaluation/fusion_v3.py         # V3
```

> **Redondance signalée** : `fusion_datasets.py` et `fusion_2.py` se recouvrent largement (même logique, sources légèrement différentes). Dans une consolidation du repo, une fonction de fusion paramétrée par version remplacerait avantageusement les trois fichiers.

### `fusionner_test_val.py` — consolidation du benchmark

Petit utilitaire qui fusionne le split de validation `benchmark_val.jsonl` dans `benchmark_test.jsonl` puis vide le premier. C'est ce qui a porté le benchmark final à ses **41 questions** (le split val n'étant plus utilisé). Affiche la répartition CALCUL / FACTUEL / COMPARAISON résultante.

```bash
python evaluation/fusionner_test_val.py
```

---

## 5. Le LLM-Judge — `llm_as_juge.py`

C'est le **moteur de notation** du projet : le script qui demande à Gemini de noter chaque réponse. Il produit les fichiers de scores que les agrégateurs de la partie 1 (`analyser_juge.py`, `aggregate_judge.py`) consomment ensuite.

### 5.1 Principe

Pour chaque question du benchmark et chaque système évalué, on envoie à **Gemini 3.1 Pro** un prompt (`PROMPT_JUDGE`) contenant la question, la **réponse de référence** (corrigé expert) et la réponse à noter. Le juge renvoie un JSON de notes sur 5 critères, chacun de 1 à 5 :

| Critère | Question posée au juge |
|---------|------------------------|
| `exactitude` | Les valeurs numériques et faits sont-ils corrects ? |
| `rigueur` | La démarche de calcul est-elle correcte et complète ? |
| `physique` | Unités et ordres de grandeur cohérents ? |
| `clarte` | Réponse claire, structurée, compréhensible ? |
| `sources` | Sources citées correctement ? |

Plus un `score_global` (moyenne) et un `commentaire` justificatif. La présence de la réponse de référence dans le prompt et l'obligation de commenter chaque note servent à **limiter les biais** du juge (préférence pour les réponses longues, jugements automatiques).

> **Important** : ce script implémente la grille **standard à 5 critères**. Le 6ᵉ critère du rapport — la **fiabilité épistémique** — n'est pas noté ici ; il est ajouté dans la passe d'évaluation enrichie (grille à 6 critères, agrégée par `aggregate_judge.py`). Cette doc reflète l'état du code fourni.

### 5.2 Paramètres de fiabilité

- **Température 0.0** — pour rendre la notation aussi déterministe que possible (la variance résiduelle du juge est documentée dans le rapport, ±0,2/5).
- **Troncatures** — question à 500 caractères, référence à 500, réponse à 1500, pour tenir dans le budget de contexte du juge.
- **Skip des réponses vides** (< 30 caractères).
- **Liaison par index** — `judge_sources.jsonl` conserve, dans l'ordre, l'`id`, le `modele` et le `type` de chaque requête, pour ré-associer les notes après le batch.

### 5.3 Workflow

```bash
python evaluation/llm_as_juge.py --prepare   # construit requêtes + sources
python evaluation/llm_as_juge.py --submit    # soumet le batch Gemini
python evaluation/llm_as_juge.py --status     # suit l'avancement
python evaluation/llm_as_juge.py --collect    # collecte, parse, écrit llm_judge_scores.jsonl + rapport
python evaluation/llm_as_juge.py --test       # mode synchrone sur 5 questions (debug, avec retry 503)
```

Le mode `--collect` affiche directement un rapport (moyennes par critère et par type de question). Le parseur tolère les réponses entourées de balises markdown ` ```json `.

> Le script fourni cible les trois systèmes Baseline / V4 sans RAG / V4 avec RAG (`FICHIERS`). Le système SRAR-GP est jugé via une passe distincte (voir `prepare_for_gemini.py` ci-dessous), ce qui explique pourquoi son fichier de notes (`judge_plus_srar_gp.json`) est produit séparément.

---

## 6. La métrique Numerical Match — `numerical_match.py`

Le LLM-Judge mesure la qualité globale, mais pas spécifiquement la **justesse numérique** d'un calcul. C'est le rôle de cette métrique, dédiée aux questions de type CALCUL.

### Fonctionnement

1. **Extraction** — `extraire_nombres()` extrait tous les nombres d'un texte par expressions régulières (décimaux, entiers, notation scientifique), en filtrant les valeurs absurdes (hors de l'intervalle 10⁻¹⁵ – 10¹⁵).
2. **Comparaison** — pour chaque nombre de la référence, on cherche un nombre de la réponse dont l'**erreur relative est ≤ 5 %** (tolérance configurable). Le score est le ratio de nombres de référence retrouvés.
3. **Agrégation** — le taux moyen est calculé par système, uniquement sur les questions CALCUL.

```bash
python evaluation/numerical_match.py
```

### Limite structurelle (documentée dans le rapport)

Cette métrique est **naïve par conception**. Sa logique d'extraction binaire pénalise les systèmes qui signalent leurs incertitudes ou présentent plusieurs scénarios, et favorise les systèmes catégoriques. C'est exactement ce qui explique le paradoxe des résultats : SRAR-GP obtient un Numerical Match plus bas (67,9 %) que V4 sans RAG (75,7 %) **non parce qu'il calcule moins bien, mais parce qu'il est plus prudent** dans sa formulation. Le rapport (section 4.6.4) en fait une limite explicite, à lire en regard de la fiabilité épistémique.

> Note : ce score isolé doit toujours être présenté avec son biais. Un bon Numerical Match n'est pas, à lui seul, un signe de supériorité.

---

## 7. Jugement / notation manuels

Deux scripts servent aux passes d'évaluation où l'humain ou une interface de chat intervient.

### `prepare_for_gemini.py`

Prépare les réponses de SRAR-GP au format JSON (id, type, question, référence, réponse) à **copier-coller dans l'interface Gemini** après le prompt LLM-Judge. C'est la passe de jugement « manuelle » du système multi-agents, par opposition au batch automatisé des autres systèmes. La sortie attendue (`srar_gp_evaluations.json`) est ensuite réintégrée dans la chaîne d'agrégation.

```bash
python evaluation/prepare_for_gemini.py
```

### `generer_formulaire_excel.py`

Convertit un formulaire de notation JSON en **classeur Excel mis en forme** (openpyxl, couleurs, bordures) destiné à la **notation humaine**. Sert quand un évaluateur humain note les réponses à la main, en complément ou en contrôle du juge LLM.

```bash
python evaluation/generer_formulaire_excel.py
```

---

## 8. Utilitaires ponctuels

### `temps_responses.py`

Calcule les statistiques de **temps de réponse** par système (moyenne, médiane, P90, min, max, total) et par type de question. Produit le tableau comparatif de latence.

```bash
python evaluation/temps_responses.py
```

> ⚠️ **Incohérence de champ à corriger** : ce script lit le champ `duree_s`, alors que `benchmark_srar_gp.py` (partie 1) écrit la latence sous le nom `latence_sec`. En l'état, il n'affichera donc « pas de données de temps » pour les fichiers produits par le benchmark SRAR-GP. À harmoniser (renommer l'un des deux champs).

### `relance_q052.py`

Script **jetable** : relance la seule question Q052, qui avait crashé pendant le benchmark, et propose (après confirmation interactive) de réécrire son entrée dans `srar_gp_responses.jsonl`. Utile une fois, sans valeur durable — typiquement à supprimer après usage, ou à généraliser en option `--id Q0XX` de `benchmark_srar_gp.py`.

---

## 9. Verdict sur l'utilité des fichiers

Comme demandé, le tri :

| Fichier | Statut | Remarque |
|---------|--------|----------|
| `generate_dataset.py` | ❌ **inutile** | Fichier **vide** (0 octet), placeholder jamais rempli → à supprimer |
| `relance_q052.py` | ⚠️ **jetable** | Correctif ponctuel d'une question ; à supprimer ou généraliser |
| `fusion_2.py` | ⚠️ **redondant** | Recouvre `fusion_datasets.py` ; candidat à fusion en script paramétré |
| `temps_responses.py` | ⚠️ **à corriger** | Bug de nom de champ (`duree_s` vs `latence_sec`) |
| `extract_context_for_benchmark__for_calcul.py` | ✅ utile | Variante CALCUL (vue en partie 1) ; doublon partiel assumé |
| Tous les autres | ✅ **utiles** | Chacun a un rôle distinct dans la chaîne |

Aucun autre fichier n'est à jeter : les trois générateurs (`finetuning`, `v3`, `equation`) correspondent à des stratégies différentes (avec contexte / sans contexte / synthétique), et les trois fusions à des versions différentes du dataset.

---

## 10. Enchaînement complet (partie 2)

```bash
# ── Générer les paires de fine-tuning ──
python evaluation/generate_dataset_finetuning.py --all      # V1/V2 : avec contexte
python evaluation/generate_dataset_fituning_v3.py --all     # V3 : sans contexte
python evaluation/generate_data_equation.py                 # calculs synthétiques
python evaluation/generate_no_context_pair.py --source both # dérivation no-context
python evaluation/integrer_paires_reference.py --fichier ...# examens corrigés

# ── Fusionner selon la version visée ──
python evaluation/fusion_v3.py        # (ex. pour V3)

# ── Noter les réponses ──
python evaluation/llm_as_juge.py --all       # juge automatique (3 systèmes)
python evaluation/prepare_for_gemini.py      # passe manuelle SRAR-GP
python evaluation/numerical_match.py         # justesse numérique (CALCUL)

# ── Mesures complémentaires ──
python evaluation/temps_responses.py         # latence (après correction du champ)
```

---

## 11. Points d'attention transversaux (rappel + nouveaux)

1. **Chemins absolus en dur** — encore présents partout (`C:\Users\Samir\...`) ; à externaliser.
2. **Trois noms de modèle Gemini** circulent — le code de cette salve utilise systématiquement `gemini-3.1-pro-preview` (génération, enrichissement, juge). À harmoniser sur la version réellement utilisée pour l'évaluation finale.
3. **Liaison par ordre d'index** — mécanisme répété dans tous les scripts batch (génération, enrichissement, juge). Robuste tant que l'API préserve l'ordre, mais fragile si une requête est réordonnée. Un `custom_id` natif serait plus sûr si l'API le permettait en mode inline.
4. **Grille 5 vs 6 critères** — `llm_as_juge.py` note 5 critères ; la fiabilité épistémique (6ᵉ) est ajoutée dans la passe enrichie agrégée par `aggregate_judge.py`. Cohérence à vérifier entre les deux passes.
5. **Champ de latence** — `duree_s` (attendu par `temps_responses.py`) vs `latence_sec` (écrit par `benchmark_srar_gp.py`) : bug à corriger.
6. **Numerical Match biaisé** — toujours interpréter ce taux avec sa limite (favorise les systèmes catégoriques), conformément au rapport.