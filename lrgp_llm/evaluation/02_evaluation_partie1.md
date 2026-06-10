# Documentation — Évaluation (partie 1/2)

> Projet **PRISME** — Assistant IA pour la recherche en génie des procédés (LRGP)
> Module : `evaluation/`
> Cette première partie couvre la **construction et le nettoyage des jeux de données** (fine-tuning et benchmark) puis l'**exécution et l'analyse du benchmark comparatif** des quatre systèmes.

> ⚠️ Documentation partielle. Le script qui appelle effectivement le LLM-Judge (Gemini) pour produire les fichiers `judge_*.json`, ainsi que la métrique `numerical_match`, seront documentés dans la partie 2/2. Ici sont documentés les scripts qui *préparent* les données et ceux qui *consomment* les sorties du juge.

---

## 1. Vue d'ensemble

Le module `evaluation/` répond à une difficulté structurelle du projet : il n'existe **aucun benchmark standard** pour le génie des procédés membranaires, et les métriques classiques de génération de texte (BLEU, ROUGE) ne capturent pas la justesse scientifique. Tout a donc dû être construit sur mesure.

Ce module remplit en réalité deux fonctions distinctes mais voisines, qui partagent les mêmes outils :

1. **Préparer les jeux de données** — celui qui sert au fine-tuning (paires question/réponse) et celui qui sert au benchmark d'évaluation (questions + réponses de référence). Cela inclut leur génération, leur nettoyage et leur enrichissement.
2. **Évaluer et comparer les systèmes** — exécuter le benchmark sur les quatre configurations comparées, puis agréger les notes du LLM-Judge en tableaux de résultats.

Les quatre systèmes comparés tout au long de l'évaluation sont :

| Système | Description |
|---------|-------------|
| **Baseline RAG** | Qwen 3.5 9B de base (non fine-tuné) + pipeline RAG |
| **V4 sans RAG** | Qwen 3.5 9B fine-tuné V4, seul |
| **V4 avec RAG** | Qwen 3.5 9B fine-tuné V4 + pipeline RAG |
| **SRAR-GP / PRISME** | Architecture multi-agents finale |

> *Note de nommage* : dans le code, l'architecture finale est appelée **SRAR-GP** (Self-Reflective Agentic RAG for Génie des Procédés). C'est le nom technique de ce qui est présenté sous le nom **PRISME** dans le rapport. Les deux désignent le même système.

---

## 2. Arborescence des fichiers (partie 1)

```
evaluation/
│
├── ─── Construction & préparation des datasets ───
│   ├── extract_context_for_benchmark.py        # Génère les prompts de création de questions (RAG → contexte)
│   ├── extract_context_for_benchmark_for_calcul.py  # Variante focalisée CALCUL
│   ├── formatage.py                            # Répare/normalise un JSONL mal formé
│   ├── audit_dataset.py                        # Audit + nettoyage + re-split 85/15
│   ├── clean_datasetv4.py                      # Nettoyage expert (sources, hors-sujet, vocab doc)
│   └── add_think_to_dataset.py                 # Enrichit les paires CALCUL avec un bloc <think> (Gemini Batch)
│
└── ─── Exécution & analyse du benchmark ───
    ├── benchmark_srar_gp.py                    # Interroge l'API SRAR-GP sur le benchmark
    ├── compare_models.py                       # Compare les réponses brutes des 3 systèmes (longueurs, types)
    ├── analyser_juge.py                        # Lit les judge_*.json → moyennes par critère (grille 5)
    └── aggregate_judge.py                      # Agrégation finale 4 systèmes (grilles standard + enrichie)
```

Exemple de jeu de données produit : `dataset_permeation.jsonl` (103 paires CALCUL, domaine perméation gazeuse, toutes enrichies d'un bloc `<think>`).

---

## 3. Construction des jeux de données

### 3.1 Génération des questions — `extract_context_for_benchmark.py`

C'est le point de départ de la création des questions, qu'il s'agisse du benchmark ou du dataset de fine-tuning. Le principe : on ne demande pas au LLM générateur d'inventer des questions « dans le vide », mais à partir de **vrais extraits du corpus** récupérés par le RAG. Cela ancre les questions dans le domaine réel et garantit que les réponses sont vérifiables.

**Fonctionnement :**

1. On définit une liste de **thèmes** (`THEMES`), chacun avec un domaine, une requête de recherche, un nombre de questions visé et les types souhaités (CALCUL / FACTUEL / COMPARAISON). Exemples de thèmes : transfert de matière, contacteurs à fibres creuses, séparation CH₄/CO₂, perméabilité des matériaux, modélisation membranaire, absorption CO₂ aux amines.
2. Pour chaque thème, la requête est encodée par **BGE-M3** et envoyée à Qdrant (recherche dense, `top_k=12`). Les passages remontés sont concaténés en un contexte.
3. Ce contexte est inséré dans un `PROMPT_TEMPLATE` qui demande au LLM de générer exactement N questions de qualité scientifique, **basées uniquement sur le contexte fourni**, au format JSON strict.
4. Chaque prompt complet est sauvegardé dans un fichier `.txt` (`data/datasets/benchmark/prompts/prompt_<domaine>.txt`).

L'opérateur copie ensuite chaque prompt dans l'interface de chat du LLM générateur, et colle le JSON retourné dans `questions.jsonl`. C'est une étape **semi-manuelle assumée** : elle garde l'humain dans la boucle pour la validation.

**Usage :**
```bash
python evaluation/extract_context_for_benchmark.py
```

**Variante `_for_calcul`** : version restreinte aux questions de type CALCUL, avec 3 thèmes seulement et `top_k=8`. Elle sert à densifier le benchmark en questions de calcul, qui sont les plus discriminantes entre systèmes.

> Point d'attention : les deux scripts utilisent le champ `source_file` du payload Qdrant. La requête de récupération de contexte utilise `top_k=12` au moment de l'appel même si le thème déclare un autre `top_k` — le `top_k` du thème n'est pas relu, c'est le `12` passé en argument qui prime.

### 3.2 Réparation de format — `formatage.py`

Le LLM générateur produit parfois un JSONL syntaxiquement cassé (objets concaténés, sauts de ligne non échappés). Ce script récupère ce qui peut l'être :

- il découpe le fichier sur le motif de début d'objet (`{"instruction"`),
- tente de parser chaque bloc,
- en cas d'échec, tente une réparation simple (échappement des `\n`),
- ne conserve que les objets valides et les réécrit proprement.

C'est un utilitaire de rattrapage, à lancer après une génération qui a mal tourné. Les chemins d'entrée/sortie sont codés en dur dans le fichier.

```bash
python evaluation/formatage.py
```

### 3.3 Le format des paires

Une paire du dataset (illustrée par `dataset_permeation.jsonl`) suit ce schéma :

```json
{
  "instruction": "Tu es un expert en génie des procédés au LRGP Nancy. ...",
  "input": "Dans un module de perméation gazeuse, ... Calculer y_max au perméat.",
  "output": "<think>Étape 1 : ...\nÉtape 2 : ...</think>\n\nRéponse finale : ...",
  "type": "CALCUL",
  "domaine": "perméation_gazeuse",
  "qualite_estimee": 5
}
```

Trois éléments structurent l'`output` des paires de calcul : un bloc de raisonnement `<think>...</think>` (les étapes), suivi d'une `Réponse finale :` qui contient la valeur numérique conclusive. C'est ce format qui est enrichi par le script suivant.

---

## 4. Nettoyage des datasets

Les itérations de fine-tuning du rapport (V1 → V4) ont montré que **la qualité prime sur la quantité** (principe LIMA). Deux scripts de nettoyage incarnent cet apprentissage, en supprimant les paires qui dégradent le modèle.

### 4.1 Audit et re-split — `audit_dataset.py`

Ce script fusionne `train_final` et `eval_final`, audite l'ensemble, supprime les paires problématiques, puis refait un split propre 85/15. Il cible précisément les défauts identifiés lors des itérations V1-V2 du rapport :

| Problème détecté | Critère |
|------------------|---------|
| **QCM mal formés** | regex `\n[A-D]\)` (réponses A/B/C/D) |
| **Réponses trop courtes** | `output` < 100 caractères |
| **Paires incomplètes** | présence de « à compléter » ou « ... » |
| **Confusion chaleur/matière** | présence de « transfert de chaleur », « DTLM », « thermique » |

Le QCM et la confusion chaleur/matière sont exactement les deux défauts marquants décrits pour V1 dans le rapport. Le script les élimine automatiquement.

**Logique de split importante** : les paires issues de sources de référence (`examen_2023`, `document_reference`) sont **toujours envoyées dans train, jamais dans eval**, pour éviter toute fuite des examens corrigés vers l'évaluation. Le re-split 85/15 (seed fixé à 42 pour la reproductibilité) ne s'applique qu'au reste du corpus. Le script affiche aussi le pourcentage de paires conservant un bloc `<think>`.

**Sorties :** `train_clean.jsonl` et `eval_clean.jsonl`.

```bash
python evaluation/audit_dataset.py
```

### 4.2 Nettoyage expert — `clean_datasetv4.py`

Ce nettoyage va plus loin que l'audit : il cible les défauts plus subtils observés notamment sur V3 (hallucinations hors-domaine) et la tension fine-tuning/RAG. Cinq règles sont appliquées :

1. **Suppression des sources** — retire les balises `[Source: ...]` de l'input et de l'output (le modèle ne doit pas apprendre à les recopier mécaniquement).
2. **Nettoyage de l'input** — supprime les blocs « Contexte : » orphelins et les amorces « Question : » pour rendre le prompt naturel.
3. **Bannissement du vocabulaire documentaire dans l'output** — rejette toute réponse contenant « selon le document », « d'après le texte », « voir tableau 3 », etc. Une réponse fine-tunée doit *intérioriser* la connaissance, pas renvoyer à un document absent.
4. **Détection des références invisibles dans l'input** — rejette les questions qui renvoient à « la référence [4] », « la figure 2.1 », « l'équation 5 » — références qui n'ont aucun sens hors de leur document d'origine.
5. **Détection des paires hors-sujet** — détecte les **hallucinations croisées** du générateur via des paires de mots-clés incompatibles. Par exemple : une question sur les membranes dont la réponse parle de bactériologie, ou le « bug Davis/Zhao » (mélange de solvants et de polymères incompatibles). Le code liste explicitement ces couples de bugs observés.

Les paires qui passent les cinq règles sont re-normalisées (instruction par défaut, type, domaine, qualité estimée) et conservées. Le script affiche un décompte détaillé des rejets par raison.

**Sorties :** `train_v4_clean_cleaned.jsonl` et `eval_v4_clean_cleaned.jsonl`.

```bash
python evaluation/clean_datasetv4.py
```

> Ordre recommandé : `audit_dataset.py` (audit grossier + re-split) puis `clean_datasetv4.py` (nettoyage fin). Les deux sont complémentaires : le premier traite la structure (QCM, longueur, split), le second le contenu sémantique (hors-sujet, vocabulaire).

---

## 5. Enrichissement par raisonnement — `add_think_to_dataset.py`

Une limite identifiée à la fin du fine-tuning : le modèle produisait des réponses de calcul plausibles dans la forme mais fausses sur le fond, faute de paires montrant un **raisonnement explicite étape par étape**. Ce script comble ce manque en ajoutant un bloc `<think>` aux paires de calcul.

### Principe

Pour chaque paire de type CALCUL ou COMPARAISON qui n'a pas encore de `<think>`, on demande à **Gemini 3.1 Pro** de reformuler l'`output` en y insérant, *avant* la réponse finale existante, un bloc de raisonnement court (3 à 6 étapes : identification de l'équation, données numériques, calcul, vérification dimensionnelle). La consigne est stricte : **ne pas modifier la réponse finale**, la conserver mot pour mot.

### Pourquoi le Batch API

L'enrichissement porte sur des centaines de paires. Pour minimiser le coût, le script utilise le **Gemini Batch API** (traitement asynchrone, moins cher que les appels synchrones). Le coût est estimé dans le code (de l'ordre du dollar pour l'ensemble).

### Workflow en quatre étapes

Le script est piloté par des arguments qui correspondent au cycle de vie d'un job batch :

```bash
python evaluation/add_think_to_dataset.py --prepare   # 1. construit les requêtes + sauvegarde la map d'index
python evaluation/add_think_to_dataset.py --submit    # 2. soumet le batch à Gemini
python evaluation/add_think_to_dataset.py --status     # 3. suit l'avancement (polling)
python evaluation/add_think_to_dataset.py --collect    # 4. collecte et reconstruit le dataset enrichi
# ou tout enchaîner :
python evaluation/add_think_to_dataset.py --all
```

| Étape | Ce qu'elle fait |
|-------|-----------------|
| `--prepare` | Filtre les paires CALCUL/COMP sans `<think>`, construit les requêtes, sauvegarde `index_map.json` (ordre des index originaux) et une copie de sauvegarde du dataset complet |
| `--submit` | Crée le job batch, enregistre `think_job_state.json` (nom du job, horodatage) |
| `--status` | Interroge l'état du job et le décompte des requêtes (réussies/échouées) |
| `--collect` | Vérifie que le job est `SUCCEEDED`, applique les enrichissements en se repérant via `index_map.json`, ne garde l'enrichissement que si `<think>` est bien présent (sinon garde l'original), écrit `think_enriched.jsonl` |

La reconstruction par **ordre d'index** (`index_map.json`) est le mécanisme clé : comme l'API batch inline ne permet pas de `custom_id`, on s'appuie sur le fait que les réponses reviennent dans l'ordre de soumission pour les ré-associer aux paires d'origine.

> Sécurité intégrée : si une requête individuelle a échoué ou si le `<think>` n'a pas été généré, la paire d'origine est conservée intacte. Aucune paire n'est perdue.

---

## 6. Exécution du benchmark sur SRAR-GP — `benchmark_srar_gp.py`

Une fois le benchmark de questions construit, ce script génère les réponses du système multi-agents en interrogeant son **API locale**.

### Fonctionnement

- Il lit `benchmark_test.jsonl` (les questions du benchmark).
- Pour chaque question, il appelle l'API SRAR-GP (`http://localhost:8000/v1/chat/completions`) en **mode verbose** (`model: "srar-gp-verbose"`), température 0.1.
- Du mode verbose, il extrait des **métadonnées de parcours** par expressions régulières : la voie empruntée (Générale / Documentaire / Calcul), la liste des agents traversés, et des indicateurs de comportement spécial : recherche web déclenchée, re-négociation (boucle de correction), donnée manquante signalée, échec de calcul.
- Le format de sortie est **identique** à `baseline_responses.jsonl`, ce qui permet de réutiliser tous les scripts d'analyse en aval.

### Robustesse

- **Timeout généreux** (360 s) car la voie CALCUL, avec ses boucles d'auto-correction, peut prendre plusieurs minutes.
- **Pause de 3 s** entre questions pour ne pas saturer le serveur Ollama.
- **Mode reprise** (`--resume`) : saute les questions déjà présentes dans le fichier de sortie.
- **Mode pilote** (`--limit N`) : ne traite que les N premières questions.
- **Health check** avant de démarrer : vérifie que l'API tourne, sinon arrête proprement avec le rappel de lancer `rag/api_server.py`.
- **Flush immédiat** après chaque écriture (anti-crash : on ne perd jamais une réponse déjà obtenue).

### Récapitulatif final

À la fin, le script affiche des statistiques agrégées : taux de réussite, latence moyenne, répartition par voie, et fréquence des activations spéciales (web search, re-négociation, missing data, échecs de calcul). Ce sont ces chiffres qui documentent le **comportement réel** de l'architecture, au-delà de la qualité des réponses.

```bash
python evaluation/benchmark_srar_gp.py [--resume] [--limit N]
```

---

## 7. Analyse des réponses brutes — `compare_models.py`

Avant même de noter la qualité, ce script compare les **caractéristiques de surface** des réponses des trois systèmes (Baseline RAG, V4 sans RAG, V4 avec RAG), à partir de leurs fichiers `*_responses.jsonl`.

Il produit :

- des **statistiques globales** par système : nombre de réponses, erreurs, réponses vides (< 50 caractères), longueur moyenne ;
- la **longueur moyenne par type** de question (CALCUL / FACTUEL / COMPARAISON) ;
- une **comparaison question par question** sur un échantillon de questions CALCUL (aperçu côte à côte des réponses) ;
- un rapport consolidé `comparaison_modeles.jsonl` avec, pour chaque question, les réponses et longueurs de chaque système.

Cet outil sert surtout au diagnostic : il révèle par exemple qu'un système répond systématiquement très court (signe de superficialité) ou produit des réponses vides. Il ne juge pas la justesse — c'est le rôle du LLM-Judge.

```bash
python evaluation/compare_models.py
```

---

## 8. Lecture des notes du LLM-Judge

Deux scripts consomment les fichiers `judge_*.json` produits par le LLM-Judge (Gemini). Ils ne notent rien eux-mêmes : ils agrègent et présentent.

> Le script qui *produit* ces `judge_*.json` (appel effectif à Gemini avec la grille de notation) sera documenté en partie 2/2.

### 8.1 Analyse grille standard — `analyser_juge.py`

Lit les notes des trois systèmes sur la **grille standard à 5 critères** (exactitude, rigueur, physique, clarté, sources) et affiche :

- les moyennes par critère et le score global de chaque système ;
- le score global **par type de question** ;
- le **top 3 et pire 3** de chaque système (avec id, type, score et commentaire du juge), utile pour repérer les cas extrêmes à inspecter.

```bash
python evaluation/analyser_juge.py
```

### 8.2 Agrégation finale — `aggregate_judge.py`

C'est le script qui produit les **tableaux de résultats du rapport** (section 4.7). Il agrège les **quatre** systèmes sur la grille **enrichie à 6 critères** (les 5 standard + la **fiabilité épistémique**, critère original du projet). Il gère le fait que les fichiers contiennent à la fois un score standard et un score enrichi.

Il produit cinq tableaux :

1. **Score standard** (5 critères) — comparable aux versions historiques.
2. **Score enrichi** (6 critères) — avec le **delta** (gain/perte) entre standard et enrichi, et la note de fiabilité épistémique. C'est ici qu'apparaît le résultat clé du rapport : SRAR-GP/PRISME est le seul système qui **gagne** au passage à la grille enrichie.
3. **Fiabilité épistémique** — visualisation en barres.
4. **Score enrichi par type de question** (CALCUL / FACTUEL / COMPARAISON), avec la taille d'échantillon n.
5. **Synthèse pour le rapport** — récapitulatif texte, complété par un rappel des taux de **Numerical Match** (la métrique de calcul, documentée en partie 2).

Tolérance utile : le chargeur sait extraire le JSON même s'il est entouré de balises markdown ` ```json `, fréquentes quand les fichiers ont été copiés depuis une interface de chat.

```bash
python evaluation/aggregate_judge.py
```

---

## 9. Enchaînement complet (partie 1)

```bash
# ── A. Construire le benchmark / dataset ──
python evaluation/extract_context_for_benchmark.py    # génère les prompts
#   → copier-coller dans le LLM, récupérer questions.jsonl
python evaluation/formatage.py                        # (si JSONL cassé)

# ── B. Nettoyer le dataset de fine-tuning ──
python evaluation/audit_dataset.py                    # audit + re-split 85/15
python evaluation/clean_datasetv4.py                  # nettoyage expert

# ── C. Enrichir avec le raisonnement <think> ──
python evaluation/add_think_to_dataset.py --all       # Gemini Batch

# ── D. Exécuter le benchmark ──
python evaluation/benchmark_srar_gp.py                # réponses SRAR-GP
#   (+ scripts équivalents pour Baseline / V4, hors de cette salve)

# ── E. Analyser ──
python evaluation/compare_models.py                   # caractéristiques de surface
python evaluation/analyser_juge.py                    # grille standard (5 critères)
python evaluation/aggregate_judge.py                  # grille enrichie (6 critères) → tableaux finaux
```

---

## 10. Points d'attention transversaux

1. **Chemins absolus en dur** — la plupart de ces scripts contiennent des chemins Windows absolus (`C:\Users\Samir\...`). C'est à externaliser dans une config ou des variables d'environnement pour la portabilité.
2. **SRAR-GP = PRISME** — même système, deux noms ; à uniformiser dans la doc finale selon le choix retenu.
3. **Étape semi-manuelle de génération** — la création des questions passe par un copier-coller dans une interface de chat. C'est volontaire (validation humaine) mais ce n'est pas automatisable en l'état.
4. **Dépendance API Gemini** — `add_think_to_dataset.py` et le LLM-Judge nécessitent `GEMINI_API_KEY` dans l'environnement. C'est l'une des deux exceptions documentées à la règle d'exécution locale (rôle de juge / générateur externe).
5. **Modèle de juge** — le code mentionne Gemini 3.1 Pro pour l'enrichissement et la synthèse ; le rapport mentionne Gemini 3.5 Pro / 2.5 Pro selon les sections comme juge. À harmoniser sur la version effectivement utilisée pour l'évaluation finale.
6. **Numerical Match** — les taux affichés en fin d'`aggregate_judge.py` (Baseline 29.1 %, V4 sans RAG 75.7 %, V4 avec RAG 72.0 %, SRAR-GP 67.9 %) proviennent d'un autre script (partie 2). Le biais méthodologique du Numerical Match envers les systèmes catégoriques — qui explique le score plus bas de SRAR-GP — est documenté dans le rapport (section 4.6.4).