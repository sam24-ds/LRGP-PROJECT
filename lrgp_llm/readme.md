# PRISME / SRAR-GP — Assistant IA fiable pour la recherche en génie des procédés

> **PRISME** (*Self-Reflective Agentic RAG for Génie des Procédés*, nom technique **SRAR-GP**) est un assistant IA scientifique **exécuté en local**, conçu au Laboratoire Réactions et Génie des Procédés (LRGP, CNRS UMR 7274 / Université de Lorraine). Il répond aux questions techniques des chercheurs en s'appuyant sur la littérature du laboratoire, en citant ses sources, et en signalant ses propres limites plutôt qu'en inventant.

---

## 1. De quoi parle ce projet ?

Les grands modèles de langage généralistes produisent souvent des réponses bien structurées mais **numériquement fausses**, citent des sources inexistantes, et ne signalent pas leurs incertitudes. Dans un laboratoire de recherche, c'est rédhibitoire — et les données scientifiques ne peuvent pas toujours être confiées à un cloud externe.

PRISME répond à trois exigences difficilement compatibles avec un seul modèle :

- **Précision numérique** — les calculs sont exécutés par du vrai code Python, pas devinés par un LLM.
- **Traçabilité documentaire** — chaque affirmation est ancrée dans le corpus du laboratoire, avec citation.
- **Fiabilité épistémique** — le système dit « je ne sais pas » quand l'information manque, plutôt qu'il n'hallucine.

La réponse architecturale est l'**asymétrie cognitive** : plutôt qu'un gros modèle généraliste pour tout, plusieurs modèles spécialisés collaborent dans une architecture multi-agents, avec trois boucles d'auto-correction. C'est un **Proof of Concept** R&D mené sur ~8 semaines.

---

## 2. Architecture en bref

```
Question
   │
   ▼
Director (Qwen 27B)  ── classifie ──┐
   │                                 │
   ├─ VOIE GÉNÉRALE   → réponse directe
   ├─ VOIE DOCUMENTAIRE → Librarian (Qwen 9B fine-tuné) + RAG hybride + Grader + Web
   └─ VOIE CALCUL    → Process Engineer → Calculation Expert (code Python) → Validator
                        avec 3 boucles d'auto-correction
```

Les composants techniques : **Qdrant** (base vectorielle, recherche hybride), **BGE-M3** + **bge-reranker-v2-m3** (embeddings + reranking), **Docling** (parsing documentaire), **Unsloth/LoRA** (fine-tuning), **LangGraph** (orchestration des agents), **Ollama** (inférence locale), **Open WebUI** (interface).

---

## 3. Arborescence du projet

```
lrgp_llm/
│
├── ingestion/                  # 📥 Construction de la base vectorielle  → voir 01_ingestion.md
│   ├── corpus_stats.py             # inventaire du corpus
│   ├── parse_pdfs.py               # parsing Docling (PDF/DOCX/XLSX)
│   ├── parse_csv.py / parse_bib.py # parseurs spécialisés
│   ├── chunk_documents.py          # chunking + indexation Qdrant (v1)
│   ├── chunk_documents_livre.py    # chunking adaptatif + classification (v2)
│   └── ...                         # utilitaires (reprise, diagnostic, suppression)
│
├── rag/                        # 🔍 Pipeline RAG en production        → voir 05_rag.md
│   ├── retriever.py                # recherche hybride dense+sparse + reranking
│   ├── prompts.py                  # prompts spécialisés (RAG, calcul, router)
│   ├── chain.py                    # chaîne LangChain (V4+RAG)
│   ├── api_server.py               # serveur FastAPI OpenAI-compatible
│   └── run_benchmark.py            # génération des réponses baseline/V4
│
├── training/                   # 🎓 Fine-tuning LoRA                   → voir 06_training.md
│   ├── finetune_lora.py            # entraînement Unsloth + export GGUF
│   ├── create_model_file.py        # Modelfile Ollama
│   └── training/                   # checkpoints + exports (V1→V5)
│
├── srar_gp/                    # 🧠 Architecture multi-agents PRISME   → voir 07_srar_gp.md
│   ├── graph.py                    # câblage LangGraph + 3 boucles
│   ├── state.py                    # état partagé
│   ├── main.py                     # point d'entrée ask_srar()
│   ├── formatter.py                # mise en forme par voie
│   ├── agents/                     # director, librarian, grader, engineer, coder, validator
│   ├── prompts/                    # prompts par agent
│   └── tools/                      # python_repl (sandbox), web_search (Tavily)
│
├── evaluation/                 # 📊 Benchmark + datasets             → voir 02/03/04
│   ├── benchmark_srar_gp.py        # exécution du benchmark
│   ├── llm_as_juge.py              # LLM-Judge (Gemini)
│   ├── numerical_match.py          # métrique de justesse numérique
│   ├── generate_dataset_*.py       # génération des paires de fine-tuning
│   ├── fusion_*.py / audit_*.py    # fusion et nettoyage des datasets
│   └── ...
│
├── data/                       # 💾 Données (entrées/sorties)         → voir 04_data.md
│   ├── parsed/                     # JSON Docling + rapport de parsing
│   └── datasets/                   # datasets fine-tuning + benchmark + batchs Gemini
│
├── docs/                       # 📖 Documentation (ces fichiers .md)
├── environment.yaml            # environnement conda
├── .env                        # clés API (NON commité)
└── .env.example                # modèle de clés API
```

---

## 4. Prérequis

- **OS** : Windows 11 (le projet a été développé et testé sous Windows ; les chemins sont à adapter sous Linux).
- **GPU NVIDIA** avec CUDA — indispensable. Le projet a été dimensionné pour une carte de 48 Go de VRAM (voir §7).
- **Miniforge / conda** — gestionnaire d'environnements.
- **Ollama** — serveur d'inférence local : <https://ollama.com>
- **Docker** — pour exécuter Qdrant.
- Deux **clés API** : Gemini (génération de datasets + juge) et Tavily (recherche web).

---

## 5. Installation

### 5.1 Environnement Python principal

```bash
# 1. Cloner / récupérer le projet
cd lrgp_llm

# 2. Créer l'environnement conda
conda env create -f environment.yaml
conda activate lrgp_llm

# 3. Installer PyTorch avec la bonne build CUDA (vérifier la version avec nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> Pour mettre à jour l'environnement après modification du yaml :
> ```bash
> conda env update -f environment.yaml --prune
> ```

### 5.2 Second environnement : le sandbox de calcul

Le sandbox d'exécution du code (`srar_gp/tools/python_repl.py`) tourne dans un environnement **séparé et isolé** pour la sécurité. Le créer :

```bash
conda create -n srar-repl python=3.11 numpy scipy sympy
```

> Les autres librairies scientifiques (fluids, thermo, CoolProp, cantera…) sont **auto-installées** par le REPL au besoin (whitelist dans `python_repl.py`). Adapter le chemin `PYTHON_EXE` dans ce fichier vers le python de cet env.

### 5.3 Clés API

Créer un fichier `.env` à la **racine du projet** (copier `.env.example`) :

```dotenv
GEMINI_API_KEY=votre_cle_gemini_ici
TAVILY_API_KEY=votre_cle_tavily_ici
```

> ⚠️ Ajouter `.env` au `.gitignore` — ne jamais committer les clés réelles.

### 5.4 Services externes

```bash
# Qdrant (base vectorielle) dans un conteneur Docker
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Ollama : télécharger les modèles nécessaires
ollama pull qwen3.5:27b
ollama pull qwen3.5:9b
ollama pull deepseek-coder:6.7b
# + créer le modèle fine-tuné (après training)
ollama create lrgp-knowledge_v5 -f <chemin>/Modelfile
```

---

## 6. Démarrage rapide

```bash
# 1. (Une fois) Construire la base vectorielle depuis le corpus
python ingestion/parse_pdfs.py --corpus "C:/.../corpus_lrgp"
python ingestion/chunk_documents_livre.py

# 2. Lancer le serveur API (expose RAG + SRAR-GP)
python rag/api_server.py
#    → http://localhost:8000/v1

# 3. Connecter Open WebUI
#    Settings → Connections → OpenAI API
#    URL : http://localhost:8000/v1   |   Key : n'importe quoi
#    Modèles : lrgp-rag, srar-gp, srar-gp-verbose

# (Optionnel) Lancer le benchmark d'évaluation
python evaluation/benchmark_srar_gp.py
```

L'ordre des prérequis au démarrage : **Docker/Qdrant** et **Ollama** doivent tourner *avant* `api_server.py`.

---

## 7. Machine de référence

Le projet a été développé et exécuté sur la station de travail dédiée du LRGP :

| Composant | Spécification |
|-----------|---------------|
| Système | Windows 11 Pro (64 bits) |
| Processeur | Intel Xeon (x86-64) |
| Mémoire système | 256 Go RAM DDR4 |
| **Carte graphique** | **NVIDIA RTX A6000** |
| **VRAM** | **48 Go GDDR6 ECC** (architecture Ampere) |
| Stockage | NVMe SSD 2 To |
| Connectivité | Réseau local + accès Internet contrôlé |

**La VRAM de 48 Go est le facteur dimensionnant du projet.** Elle permet de charger plusieurs modèles moyens simultanément (7 à 30 B en quantization 4 bits), mais impose un **chargement dynamique** via Ollama dès que la combinaison dépasse la limite. Les modèles inactifs sont déchargés pour libérer la mémoire de la voie active, au prix de quelques secondes de latence.

Empreinte VRAM des modèles (quantization Q4_K_M sauf embeddings) :

| Rôle | Modèle | VRAM |
|------|--------|------|
| Director / Engineer / Validator / Grader | Qwen 3.5 27B (instance partagée) | ~17 Go |
| Librarian | Qwen 3.5 9B + LoRA | ~6,6 Go |
| Calculation Expert | DeepSeek-Coder 6.7B | ~3,8 Go |
| Embeddings | BGE-M3 | ~1,2 Go |
| Reranking | bge-reranker-v2-m3 | ~0,6 Go |
| **Total (voie complète)** | | **~29,6 Go** |

### Environnement logiciel

| Composant | Version / détail |
|-----------|------------------|
| Python | 3.11 |
| Distribution | Miniforge3 (conda-forge) |
| Env principal | `lrgp_llm` |
| Env sandbox calcul | `srar-repl` (isolé) |
| Serveur inférence | Ollama |
| Base vectorielle | Qdrant (Docker) |
| Interface | Open WebUI |
| Orchestration | LangGraph / LangChain |
| Suivi entraînement | MLflow |

### Accès externes (exceptions à l'exécution locale)

Conformément à la contrainte de confidentialité, **toute l'inférence des modèles reste locale**. Seules deux exceptions, sans données confidentielles :

- **API Tavily** — recherche web (Boucle 1 de SRAR-GP).
- **API Gemini** — génération de datasets et rôle de LLM-Judge (évaluation).

---

## 8. Documentation détaillée

| Document | Module couvert |
|----------|----------------|
| `01_ingestion.md` | Pipeline d'ingestion : parsing, chunking, indexation Qdrant |
| `02_evaluation_partie1.md` | Construction/nettoyage des datasets, exécution du benchmark |
| `03_evaluation_partie2.md` | Génération des datasets, LLM-Judge, métriques |
| `04_data.md` | Gestion du benchmark + cartographie des artefacts `data/` |
| `05_rag.md` | Pipeline RAG : retriever, prompts, chaîne, serveur API |
| `06_training.md` | Fine-tuning LoRA et export des modèles |
| `07_srar_gp.md` | Architecture multi-agents PRISME et ses 3 boucles |

**Ordre de lecture conseillé** : suivre le flux des données — `01` (ingestion) → `05` (RAG) → `06` (training) → `07` (PRISME) pour le système ; `04` → `02` → `03` pour l'évaluation.

---

## 9. Résultats clés (Proof of Concept)

Évaluation sur un benchmark de 41 questions, grille à 6 critères (juge Gemini) :

- **PRISME est le seul système qui gagne** au passage à la grille enrichie (intégrant la fiabilité épistémique), où il arrive **en tête** (4,27/5 sur ce critère).
- Sur les questions de **calcul**, il devance l'approche monolithique fine-tunée.
- Sur le **score global standard**, une baseline RAG bien conçue reste légèrement devant — écart imputable au modèle de code (DeepSeek-Coder 6.7B), dont le remplacement par Qwen3-Coder 33B est en cours.

La valeur du POC n'est pas dans une performance absolue, mais dans la démonstration qu'un assistant scientifique **local, traçable et honnête sur ses limites** est techniquement réalisable — et dans la base technique réutilisable qu'il laisse au laboratoire.

---

## 10. Limites connues et perspectives

- **Dataset de fine-tuning modeste** (~600 paires pour V4) — mise à l'échelle prévue.
- **Modèle de code insuffisant** sur le raisonnement multi-étapes — remplacement par Qwen3-Coder 33B en cours.
- **Benchmark limité** (41 questions) — élargissement prévu.
- **Portabilité** — nombreux chemins absolus Windows en dur, à externaliser ; deux noms de modèle Ollama coexistent (à consolider).
- **Sparse non déterministe** — le vecteur sparse repose sur `hash()` Python (à fixer via `PYTHONHASHSEED` ou les lexical weights natifs de BGE-M3).