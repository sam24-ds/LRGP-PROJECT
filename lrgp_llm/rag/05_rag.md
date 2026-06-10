# Documentation — Module RAG (voie documentaire)

> Projet **PRISME / SRAR-GP** — Assistant IA pour la recherche en génie des procédés (LRGP)
> Module : `rag/`
> Ce module implémente le **pipeline RAG en production** : récupération hybride, prompts spécialisés, chaîne LangChain, et le **serveur API** qui expose à la fois le RAG simple (V4) et l'architecture multi-agents SRAR-GP à l'interface Open WebUI.

> Il s'appuie sur la base vectorielle construite par l'ingestion (`01_ingestion.md`) et alimente l'évaluation (`02`/`03_evaluation`) ainsi que la voie documentaire de PRISME.

---

## 1. Vue d'ensemble

Là où le module `ingestion/` *remplit* la base vectorielle, le module `rag/` l'*interroge*. C'est la phase de **récupération à la volée** : à chaque question, on cherche les passages pertinents, on les raffine, on les injecte dans un prompt adapté, et on génère une réponse ancrée.

Le module remplit trois rôles :

1. **Le retriever** (`retriever.py`) — recherche hybride dense+sparse dans Qdrant, suivie d'un reranking. C'est la brique réutilisée partout (chaîne RAG, génération de dataset, voie documentaire de PRISME).
2. **La chaîne RAG** (`chain.py` + `prompts.py`) — orchestre routage → retrieval → prompt → LLM. C'est la configuration « V4 + RAG » du rapport, l'approche monolithique de référence.
3. **Le serveur API** (`api_server.py`) — expose le tout en HTTP compatible OpenAI, pour qu'Open WebUI (et les scripts d'évaluation) puissent dialoguer avec le système.

```
Question
   ↓
Router (classifie : CALCUL / FACTUEL / …)      ← prompts.py
   ↓
Retriever (hybrid search + reranking)          ← retriever.py
   ↓
Prompt adapté au type                          ← prompts.py
   ↓
LLM (Ollama local ou API cloud)                ← chain.py
   ↓
Réponse + sources                              ← chain.py
```

---

## 2. Arborescence du module

```
rag/
├── retriever.py            # ★ Recherche hybride dense+sparse + reranking
├── prompts.py              # Templates de prompts (système, RAG, calcul, router)
├── chain.py                # Chaîne LangChain : router → retrieval → prompt → LLM
├── api_server.py           # ★ Serveur FastAPI OpenAI-compatible (RAG + SRAR-GP)
├── run_benchmark.py        # Génère les réponses baseline/V4 sur le benchmark
│
├── test_retriever.py       # Test du retriever seul
├── test_chain.py           # Test de la chaîne (backend OpenAI)
├── test_chain_ollama.py    # Test de la chaîne (backend Ollama)
├── test_direct.py          # Appel Ollama brut (sans RAG)
├── test_finetuned.py       # Compare base vs fine-tuné, avec RAG
├── test_finetuned_v2.py    # Test ciblé d'une version fine-tunée
│
└── evaluation/results/     # Formulaires de notation produits par run_benchmark.py
```

---

## 3. Le retriever — `retriever.py`

C'est la brique fondamentale du système. Elle transforme une question en une liste de passages pertinents, prêts à être injectés dans un prompt.

### 3.1 Le flux complet

```
Question
   ↓ BGE-M3 encode (dense 1024d + sparse TF)
   ↓ Qdrant hybrid search (RRF fusion) → top-20
   ↓ bge-reranker-v2-m3 → top-5
   ↓ filtre seuil de pertinence (≥ 0.50)
Chunks contextualisés
```

### 3.2 L'encodage de la question

La question est encodée de deux façons, exactement comme les chunks le furent à l'ingestion :

- **Dense** — BGE-M3 produit un vecteur 1024 dimensions normalisé.
- **Sparse** — calcul TF maison (`_encode_sparse`) : tokenisation, comptage de fréquence, projection par hachage `% 50000`.

> ⚠️ **Cohérence critique avec l'ingestion** : le calcul sparse de la question doit être *identique* à celui des chunks. C'est bien le cas — le code est le même (`abs(hash(token)) % 50000`, fréquence relative). Mais cela hérite de la même limite : `hash()` de Python n'est pas déterministe entre exécutions sans `PYTHONHASHSEED` fixé. Si l'indexation et la requête tournent dans des processus aux seeds différents, les index sparse ne correspondront pas, dégradant la branche sparse de la recherche. **À fiabiliser** (fixer `PYTHONHASHSEED`, ou passer aux lexical weights natifs de BGE-M3).

### 3.3 La recherche hybride (RRF)

`_hybrid_search` combine les deux recherches via **Reciprocal Rank Fusion** (Fusion.RRF de Qdrant) : deux `Prefetch` (un dense, un sparse) remontent chacun top-20, puis Qdrant fusionne les deux classements. Si la recherche hybride échoue (collection sans vecteur sparse, par ex.), un **fallback dense-only** prend le relais — le système ne plante jamais sur ce point.

### 3.4 Le reranking

Les ~20 candidats sont réordonnés par le **CrossEncoder bge-reranker-v2-m3**, qui évalue finement la pertinence question↔passage (plus précis mais plus lent que la recherche vectorielle). On garde les top-5.

Deux raffinements importants :

- **Seuil de pertinence** — après reranking, les chunks dont le score est < 0.50 (`RERANK_THRESHOLD`) sont écartés. C'est ce qui permet à la chaîne de basculer en mode « sans contexte » quand rien n'est assez pertinent — un élément clé de la fiabilité épistémique (ne pas répondre sur du bruit).
- **Filtre optionnel par source** — `filter_source` permet de restreindre la recherche à un document précis.

### 3.5 Chargement paresseux

Les modèles (BGE-M3, reranker) sont chargés **à la première utilisation** (`@property` avec cache), pas à l'instanciation. Cela évite de payer le coût de chargement GPU si le retriever n'est finalement pas utilisé.

### 3.6 Sortie

Chaque résultat est un `RetrievalResult` : `chunk_id`, `source`, `text`, `score` (dense/RRF), `rerank_score`, `page`, `section`, `chunk_type`. La méthode `format_context()` les assemble en un bloc texte prêt pour le prompt, chaque passage préfixé par `[Source: … | Page: …]`.

---

## 4. Les prompts — `prompts.py`

Les prompts encodent les apprentissages du rapport sur les défauts des petits modèles (superficialité, hallucinations numériques). Tous sont des `ChatPromptTemplate` LangChain.

### 4.1 Le system prompt — `SYSTEM_LRGP`

Le prompt système pose cinq **règles absolues** qui structurent tout le comportement :

1. **Ancrage RAG** — les faits et données numériques doivent venir des documents fournis ; une donnée manquante doit être signalée.
2. **Savoir fondamental autorisé** — les lois de base (gaz parfaits, Dalton, bilan matière) sont permises pour relier les informations.
3. **Interdiction numérique** — pas de calcul complexe ni d'intégrale « de tête », jamais de résultat numérique deviné. C'est la traduction directe du constat du rapport : un 9B ne calcule pas de façon fiable.
4. **Citation** — toujours `[Source: X]` quand la source est présente.
5. **Formatage** — Markdown + LaTeX inline pour les maths.

### 4.2 Les prompts spécialisés par type

| Prompt | Type de question | Stratégie |
|--------|------------------|-----------|
| `PROMPT_RAG` | FACTUEL (défaut) | Force la **profondeur** : « agis comme un professeur d'université », réponse exhaustive, mécanismes physico-chimiques expliqués. Contre la superficialité des petits modèles. |
| `PROMPT_CALCUL` | CALCUL | **Interdit la résolution mentale**. Impose un protocole : spécifications → formules littérales → **script Python (scipy/numpy)**, puis arrêt immédiat. Le modèle produit le *code*, pas le résultat. |
| `PROMPT_NO_CONTEXT` | (aucune source pertinente) | **Fallback honnête** : annonce dès la 1ʳᵉ ligne que le corpus ne contient pas la réponse, répond prudemment depuis les connaissances générales, recommande de vérifier la littérature. |
| `PROMPT_ROUTER` | (classification) | Classe la question en CALCUL / FACTUEL / COMPARAISON / PROCEDURE / GENERAL, en un seul mot-clé. |

> Décision de conception notable : pour les calculs, le RAG simple **délègue à l'humain l'exécution du code** (« Veuillez l'exécuter dans votre environnement Python »). C'est précisément la limite que l'architecture multi-agents PRISME lèvera, en exécutant le code automatiquement via le REPL et la boucle de validation. Le `PROMPT_CALCUL` est donc le « chaînon manquant » entre l'approche monolithique et l'approche agentique.

### 4.3 Helpers

- `choisir_prompt(type)` — renvoie `PROMPT_CALCUL` pour les calculs, `PROMPT_RAG` sinon.
- `formater_historique(messages)` — convertit un historique de conversation en messages LangChain (`HumanMessage`/`AIMessage`).

---

## 5. La chaîne RAG — `chain.py`

`LRGPChain` orchestre les briques en un pipeline LangChain (LCEL). C'est la configuration **« V4 + RAG »** du rapport — l'aboutissement de l'approche monolithique.

### 5.1 Les étapes de `ask()`

1. **Classification** — un LLM « router » (température 0.0) classe la question via `PROMPT_ROUTER`.
2. **Retrieval** — le retriever remonte les passages pertinents (avec filtre source optionnel).
3. **Choix du prompt** — si aucune source ne passe le seuil de pertinence → `PROMPT_NO_CONTEXT` ; sinon le prompt adapté au type.
4. **Génération** — `prompt | llm | StrOutputParser()`, avec l'historique éventuel.
5. **Retour** — un `RAGResponse` (réponse, sources, type, modèle, taille de contexte).

### 5.2 Backends LLM interchangeables

La chaîne supporte trois backends, sélectionnables à l'instanciation :

- **`ollama`** — exécution locale (le cas nominal, conforme à la contrainte de confidentialité). Modèles Ollama quantizés.
- **`openai`** / **`anthropic`** — API cloud, pour comparaison ou test. Non utilisés en production (contrainte d'exécution locale).

Le `num_predict` (longueur max de génération, défaut 2048) est exposé pour contrôler la verbosité.

### 5.3 Streaming

`ask_stream()` génère la réponse token par token (`chain.stream`), pour une expérience interactive dans l'interface.

> Le `router_llm` est une seconde instance du même modèle à température 0.0. Sur une machine à carte unique, c'est le même modèle Ollama servi deux fois — pas de coût mémoire supplémentaire (appels stateless), mais une latence de classification à chaque question. Pour les questions simples, on pourrait court-circuiter le routeur.

---

## 6. Le serveur API — `api_server.py`

C'est le point d'entrée qui rend tout le système utilisable depuis **Open WebUI** (l'interface des chercheurs). Un serveur **FastAPI** expose une API **compatible OpenAI**, ce qui évite d'écrire un connecteur spécifique.

### 6.1 Trois modèles exposés

L'API présente trois « modèles » sélectionnables dans l'interface :

| Modèle exposé | Route interne | Description |
|---------------|---------------|-------------|
| `lrgp-rag` | `LRGPChain` (V4 + RAG) | RAG simple, rapide (~12 s) |
| `srar-gp` | `ask_srar()` (multi-agents) | Architecture PRISME complète (5–180 s selon la voie) |
| `srar-gp-verbose` | idem + footer | SRAR-GP avec le **parcours des agents** affiché |

L'aiguillage se fait sur le nom du modèle : si « srar » ou « agentic » apparaît → voie multi-agents ; sinon → RAG simple.

### 6.2 Initialisation

Au démarrage, le serveur (1) instancie la chaîne RAG sur le modèle `lrgp-knowledge_v5`, et (2) **pré-charge le graphe SRAR-GP** pour éviter une latence au premier appel. C'est un point important : sans ce préchauffage, la première question paierait le coût de construction du graphe d'agents.

### 6.3 Le mode verbose

Pour `srar-gp-verbose`, le serveur ajoute en pied de réponse un bloc structuré : la **voie** empruntée, les **agents traversés**, le nombre de **re-négociations** et de **sources web** utilisées. C'est exactement ce que `benchmark_srar_gp.py` (évaluation) parse par regex pour produire les statistiques de comportement. Les deux scripts sont donc couplés par ce format.

### 6.4 Le filtre des requêtes système Open WebUI

Détail pratique mais crucial : Open WebUI envoie automatiquement, après chaque réponse, des requêtes parasites (génération de titre de conversation, de tags, de follow-ups). Les router vers SRAR-GP gaspillerait des minutes de calcul multi-agents pour générer un titre.

`_est_requete_systeme_openwebui()` détecte ces requêtes par leur signature (`### Task:`, « generate a concise title », « suggest follow-up »…) et les **court-circuite** vers un appel direct à Qwen 27B (rapide, sans les agents). Sans ce filtre, l'interface serait inutilisable.

### 6.5 Format de réponse

`_format_response()` produit le format OpenAI, en mode streamé (chunks SSE mot par mot) ou non. Le serveur tourne avec `timeout_keep_alive=300` — **critique** : la voie CALCUL peut prendre jusqu'à 3 minutes, et un timeout standard couperait la connexion.

### Usage

```bash
python rag/api_server.py
# Open WebUI → Settings → Connections → OpenAI API
#   URL : http://localhost:8000/v1
#   Key : n'importe quoi
```

Un endpoint `/health` permet de vérifier que l'API tourne (utilisé par `benchmark_srar_gp.py` avant de lancer le benchmark).

---

## 7. Génération des réponses baseline — `run_benchmark.py`

Ce script génère les réponses de la **Baseline RAG** et des variantes V4 sur le benchmark, en reproduisant fidèlement le comportement de `chain.py`. C'est lui qui produit les fichiers `*_responses.jsonl` consommés par tous les analyseurs d'évaluation.

Deux modes :

- **avec RAG** (défaut) — instancie `LRGPChain` et appelle `ask()` pour chaque question ; enregistre réponse, sources, type classé, durée.
- **`--no-rag`** — appel Ollama direct avec le seul `SYSTEM_LRGP`, sans retrieval. C'est ce qui produit la configuration « V4 sans RAG ».

Il génère aussi un **formulaire de notation humaine** (`formulaire_notation_*.json`) : une fiche par question avec des emplacements vides pour deux annotateurs (exactitude, rigueur, pertinence physique, clarté, citations, commentaire). C'est l'entrée de la notation humaine, en complément du LLM-Judge.

```bash
python evaluation/run_benchmark.py --model qwen3.5:9b                    # Baseline RAG
python evaluation/run_benchmark.py --model lrgp-knowledge_v5 --output v4_rag_responses.jsonl
python evaluation/run_benchmark.py --model lrgp-knowledge_v5 --no-rag --output v4_no_rag_responses.jsonl
```

> Le champ de durée écrit ici est `duree_s` — cohérent avec `temps_responses.py`. Rappel : `benchmark_srar_gp.py` écrit `latence_sec` (incohérence signalée dans `03_evaluation_partie2.md`).

---

## 8. Scripts de test

Une batterie de tests à différents niveaux, utile pour le diagnostic :

| Script | Ce qu'il teste |
|--------|----------------|
| `test_retriever.py` | Le retriever seul : affiche les passages remontés (scores dense + rerank, source, section) pour 3 questions types |
| `test_chain.py` | La chaîne complète avec le backend **OpenAI** (gpt-4o-mini) — pour comparaison cloud |
| `test_chain_ollama.py` | La chaîne complète avec **Ollama** (qwen3.5:9b) — le cas nominal |
| `test_direct.py` | Appel Ollama **brut**, sans RAG ni prompt — sanity check minimal |
| `test_finetuned.py` | Compare le modèle **de base vs fine-tuné** (`lrgp-expert`), avec RAG, sur 3 questions |
| `test_finetuned_v2.py` | Test ciblé d'une version (`lrgp-knowledge_v2`) sur un problème d'osmose inverse |

Ces tests illustrent aussi l'historique des noms de modèles fine-tunés (voir §9).

---

## 9. Points d'attention transversaux

1. **Noms de modèles Ollama multiples** — le code référence `lrgp-knowledge_v5` (api_server), `lrgp-knowledge_v2`, `lrgp-expert`, et `qwen3.5:9b` selon les fichiers. Ce sont les itérations successives du fine-tuning, mais c'est une source de confusion. À consolider : un seul nom de modèle de production, documenté.
2. **Sparse non déterministe** — le `hash() % 50000` partagé entre ingestion et requête doit être rendu déterministe (`PYTHONHASHSEED`) ou remplacé par les lexical weights natifs de BGE-M3 (voir `01_ingestion.md`, même limite).
3. **Chemins absolus** — `run_benchmark.py` contient un chemin Windows en dur vers le benchmark. À externaliser.
4. **Double parsing d'arguments** — `run_benchmark.py` parse ses arguments deux fois (au niveau module *et* dans `main()`) ; le premier bloc est mort. À nettoyer.
5. **Seuil de rerank en dur** — `RERANK_THRESHOLD = 0.50` est codé dans le retriever ; le remonter en paramètre faciliterait le réglage selon le besoin de rappel vs précision.
6. **Cohérence verbose ↔ benchmark** — le format du footer verbose d'`api_server.py` et le parsing regex de `benchmark_srar_gp.py` sont couplés. Toute modification du footer casse le parsing des métadonnées de parcours. À garder synchronisés.

---

## 10. Place du module dans l'ensemble

```
ingestion/  ──remplit──►  Qdrant (lrgp_corpus)
                              ▲
                              │ interroge
                    rag/retriever.py  ──────────────┐
                              │                       │ réutilisé par
                    rag/chain.py (V4+RAG)             ├──► evaluation (run_benchmark)
                              │                       ├──► génération dataset (extract_context)
                    rag/api_server.py                 └──► srar_gp/ (voie documentaire de PRISME)
                         │        │
                   "lrgp-rag"  "srar-gp"
                         │        │
                    Open WebUI (chercheurs)
```

Le retriever est la brique la plus réutilisée du projet : il sert la chaîne RAG, mais aussi la génération de questions de benchmark, la création du dataset de fine-tuning, et la voie documentaire de l'architecture multi-agents. C'est le prochain module à documenter : **`srar_gp/`**, où ce même retriever devient l'outil de l'agent *Librarian*.