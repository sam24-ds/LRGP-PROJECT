# Documentation — Pipeline d'ingestion documentaire

> Projet **PRISME** — Assistant IA pour la recherche en génie des procédés (LRGP)
> Module : `ingestion/`
> Cette partie couvre la transformation du corpus brut (PDF, DOCX, XLSX, CSV, BibTeX) en une base vectorielle Qdrant interrogeable.

---

## 1. Vue d'ensemble

Un système de RAG ne vaut que par les documents qu'il indexe : sans corpus de qualité, aucune sophistication algorithmique ne produit de réponses fiables. Le module `ingestion/` est donc la fondation du projet. Il prend en entrée le corpus documentaire du laboratoire et produit en sortie une collection Qdrant de plus de 155 000 segments vectorisés, chacun accompagné de sa métadonnée.

Le pipeline suit une architecture classique en deux temps, séparant nettement la **préparation des documents** (coûteuse, exécutée une fois) de la **récupération à la volée** (rapide, exécutée à chaque requête). Le module `ingestion/` ne couvre que le premier temps.

```
corpus brut          parsing            chunking           indexation
(PDF/DOCX/XLSX)  →   Docling      →   HybridChunker   →   Qdrant
                     (JSON)            (JSONL)            (vecteurs + payload)
```

Concrètement, le flux complet enchaîne les étapes suivantes :

1. **Analyse du corpus** — inventaire et estimation (`corpus_stats.py`)
2. **Parsing** — extraction structurée vers JSON (`parse_pdfs.py`, plus les parseurs spécialisés `parse_csv.py`, `parse_bib.py`)
3. **Enregistrement des JSON externes** — intégration des sources non-PDF dans le rapport (`register_external_json.py`)
4. **Chunking + vectorisation + indexation** — découpage, encodage BGE-M3, écriture dans Qdrant (`chunk_documents.py`, puis sa version 2 `chunk_documents_livre.py`)

Des utilitaires de maintenance (reprise, diagnostic, suppression) complètent l'ensemble.

---

## 2. Arborescence du module

```
ingestion/
├── corpus_stats.py              # Analyse / inventaire du corpus (multi-format)
├── parse_pdfs.py                # Parsing principal Docling (PDF/DOCX/XLSX/PPTX/HTML)
├── parse_csv.py                 # Parsing des fichiers CSV → JSON pipeline
├── parse_bib.py                 # Extraction des références BibTeX → bibliography.json
├── register_external_json.py    # Enregistre les JSON externes dans le rapport de parsing
├── chunk_documents.py           # Chunking + indexation Qdrant (v1, articles)
├── chunk_documents_livre.py     # Chunking adaptatif + classification (v2, articles & livres)
│
├── test_batch_small.py          # Test de vitesse parsing (CPU)
├── test_batch_small_gpu.py      # Test de vitesse parsing (GPU)
├── test_chunking_single.py      # Test du chunking sur un document unique
├── test_retrieval_qdrant.py     # Test de retrieval sur la base indexée
│
├── relancer_doc.py              # Reparse des documents ciblés
├── relancer_gros_docs.py        # Rechunk des gros fichiers (batch réduit)
├── diagnostique_erreur_chunking.py  # Liste les documents parsés mais non chunkés
├── delete_book_chunks.py        # Supprime les chunks d'une source dans Qdrant
└── supprimer_et_rindexer.py     # Supprime + nettoie une source (Qdrant + fichiers)
```

Les données produites sont organisées ainsi :

```
ingestion/data/
├── parsed/                      # Sortie du parsing
│   ├── <document>.json          # un DoclingDocument par fichier source
│   ├── parsing_report.json      # journal de parsing (statut, durée, stats)
│   └── bibliography.json        # références BibTeX agrégées
└── chunks/                      # Sortie du chunking
    └── <document>_chunks.jsonl  # un fichier JSONL de chunks par document
```

---

## 3. Étape 1 — Analyse du corpus (`corpus_stats.py`)

Avant tout traitement lourd, ce script dresse un état des lieux du corpus. Il sert à dimensionner le travail (volume, temps estimé) et à repérer les anomalies (chemins trop longs, formats non reconnus).

### Ce qu'il fait

Il parcourt récursivement le dossier corpus et reconnaît les formats supportés nativement par Docling : PDF, DOCX/DOC, XLSX/XLS et PPTX. Pour chaque format, il produit :

- le nombre de fichiers et la taille totale par type,
- la répartition par sous-dossier de premier niveau,
- des estimations dérivées (nombre de pages approximatif, durée de parsing prévue sur la station A6000, coût OCR éventuel).

Il signale aussi deux catégories de problèmes : les fichiers dont le chemin dépasse la limite Windows de 260 caractères (ignorés à l'accès), et les extensions non reconnues (comptées et affichées).

### Usage

```bash
python ingestion/corpus_stats.py --corpus "C:/chemin/vers/corpus_lrgp"
```

### Point d'attention

L'estimation de pages repose sur une heuristique grossière (`pages ≈ taille_Mo × 10`). Elle donne un ordre de grandeur, pas un chiffre exact. La sortie est purement informative : ce script ne modifie rien et ne produit aucun fichier.

---

## 4. Étape 2 — Parsing (`parse_pdfs.py`)

C'est le cœur de la phase de préparation. Il transforme chaque document source en un `DoclingDocument` sérialisé en JSON.

### Pourquoi Docling

Docling a été préféré aux extracteurs classiques (PyPDF, pdfminer) car il préserve la structure des documents scientifiques complexes — équations, tableaux, hiérarchie des sections — au lieu de produire un flot de texte plat. Cette structure est exploitée plus loin par le chunker.

### Configuration du convertisseur

La fonction `build_converter()` construit un `DocumentConverter` Docling avec une configuration optimisée pour le corpus :

- `do_ocr = False` — l'OCR est désactivé par défaut (les PDF natifs dominent le corpus) ;
- `do_table_structure = True` — la reconnaissance de structure des tableaux est activée ;
- `generate_picture_images = False` — les images ne sont pas extraites (inutiles pour le RAG textuel) ;
- accélérateur **CUDA** avec 4 threads, exploitant la RTX A6000.

### Mécanismes de robustesse

Le script est conçu pour traiter un corpus volumineux sur plusieurs heures, ce qui impose de la résilience :

- **Reprise sur interruption** — un `parsing_report.json` est tenu à jour ; au redémarrage, tout document déjà marqué `"ok"` avec son JSON présent est sauté. On peut donc relancer le script sans tout recommencer.
- **Sauvegarde incrémentale** — le rapport est réécrit toutes les 10 itérations, et un ETA est affiché toutes les 50.
- **Détection des documents vides** — un fichier produisant moins de 200 caractères est marqué `"vide"` (probable PDF scanné) plutôt que `"ok"`. Ces fichiers sont listés dans `fallback_list.txt` pour un retraitement ultérieur (par exemple via un outil OCR/MinerU).
- **Isolation des erreurs** — une exception sur un fichier le marque `"erreur"` sans interrompre le reste du lot.

### Statuts possibles

Chaque document reçoit l'un des trois statuts suivants dans le rapport :

| Statut | Signification | Suite |
|--------|---------------|-------|
| `ok` | Texte extrait (≥ 200 caractères) | Sera chunké |
| `vide` | Texte insuffisant (probable scan) | À retraiter (fallback) |
| `erreur` | Exception pendant le parsing | À diagnostiquer |

### Usage

```bash
python ingestion/parse_pdfs.py --corpus "C:/chemin/corpus" [--workers 1]
```

Le rapport final récapitule le nombre d'OK / vides / erreurs, la durée totale et la vitesse moyenne, et écrit `fallback_list.txt` si des documents nécessitent un retraitement.

---

## 5. Étape 2 bis — Parseurs spécialisés

Tous les documents ne passent pas par Docling. Deux formats ont leur propre parseur, dont la sortie est rendue **compatible avec le reste du pipeline** (même schéma JSON que les `DoclingDocument`).

### `parse_csv.py` — données tabulaires

Convertit chaque fichier CSV en JSON indexable. Le contenu est rendu sous forme Markdown (`df.to_markdown()`) pour rester lisible par le modèle d'embedding, et la structure tabulaire d'origine est conservée dans le champ `tables`. La détection du séparateur (`,` ou `;`) et de l'encodage (UTF-8 puis Latin-1 en repli) est automatique.

```bash
python ingestion/parse_csv.py --corpus "C:/chemin/corpus"
```

### `parse_bib.py` — références bibliographiques

Agrège toutes les entrées BibTeX du corpus dans un unique `data/parsed/bibliography.json` (titre, auteurs, année, journal, DOI, abstract). Ce fichier sert à enrichir les métadonnées des chunks à une étape ultérieure ; il n'est **pas** indexé tel quel.

```bash
python ingestion/parse_bib.py --corpus "C:/chemin/corpus"
```

### `register_external_json.py` — réintégration

Les JSON produits hors Docling (CSV, etc.) ne figurent pas dans `parsing_report.json`, donc le chunker les ignorerait. Ce script les détecte, calcule leurs statistiques de base (nombre de caractères, de tableaux) et les ajoute au rapport avec le statut `"ok"`, afin qu'ils soient pris en compte au chunking.

```bash
python ingestion/register_external_json.py
```

> **Ordre important** : exécuter `parse_csv.py` (ou tout autre parseur externe) **puis** `register_external_json.py` **avant** le chunking. Le chunker ne traite que les entrées marquées `"ok"` dans le rapport.

---

## 6. Étape 3 — Chunking et indexation

C'est ici que les documents structurés deviennent des vecteurs interrogeables. Deux versions coexistent : une version initiale (`chunk_documents.py`) et une version enrichie (`chunk_documents_livre.py`) qui est celle effectivement retenue pour le corpus final.

### 6.1 Principe commun

Pour chaque document parsé marqué `"ok"`, le traitement enchaîne quatre opérations :

1. **Chunking** — découpage du `DoclingDocument` en segments avec le `DoclingNodeParser` de LlamaIndex (chunking *structure-aware*, qui respecte la hiérarchie du document).
2. **Encodage** — chaque chunk est vectorisé par **BGE-M3** sur GPU, qui produit une représentation dense (1024 dimensions) ; un vecteur sparse est calculé en parallèle.
3. **Indexation** — les chunks sont écrits dans la collection Qdrant `lrgp_corpus` (vecteurs + métadonnées en payload).
4. **Sauvegarde JSONL** — les chunks (sans les vecteurs, trop volumineux) sont archivés dans `data/chunks/<doc>_chunks.jsonl`, ce qui sert de trace et de garde-fou de reprise.

### 6.2 Le détail du chunking

Le `DoclingNodeParser` ne reçoit **pas** du texte brut, mais le JSON sérialisé du `DoclingDocument` (`doc.model_dump_json()`). C'est volontaire : c'est ce qui lui permet d'exploiter la structure du document (sections, tableaux, formules) plutôt que de découper à l'aveugle.

Un point subtil concerne les **équations**. Docling laisse parfois le champ `text` d'une formule vide tout en conservant sa version brute dans le champ `orig`. Avant le chunking, le code « patche » ces formules : il récupère `orig`, le nettoie et le préfixe par `[Équation]`. Cela évite de perdre les équations — qui sont précisément le contenu le plus précieux pour les questions de calcul.

Les chunks trop courts (moins de 50 caractères) sont écartés.

### 6.3 Le vecteur sparse : une approximation assumée

BGE-M3 sait nativement produire des *lexical weights* (poids sparse). Or, dans ce pipeline, le vecteur sparse n'est **pas** issu de BGE-M3 mais d'un calcul TF (fréquence de termes) maison dans `_sparse_from_text()` :

```python
tokens = re.findall(r'\b\w+\b', text.lower())
tf     = Counter(tokens)
# chaque token est projeté sur un index par hachage modulo 50000
idx = abs(hash(token)) % 50000
value = freq / total   # fréquence relative
```

C'est une **approximation** explicitement documentée dans le code (« approximation du lexical weights natif de BGE-M3 »). Elle a l'avantage d'être simple et rapide, mais deux limites méritent d'être connues :

- le hachage modulo 50 000 peut produire des **collisions** (deux tokens distincts pointant sur le même index) ;
- `hash()` de Python n'est **pas déterministe entre exécutions** par défaut (sauf `PYTHONHASHSEED` fixé), ce qui peut faire varier les index sparse d'un lancement à l'autre. C'est sans conséquence si tout le corpus est indexé d'un seul tenant, mais à garder en tête lors d'ajouts incrémentaux.

C'est une piste d'amélioration identifiée : remplacer cette approximation par les vrais lexical weights de BGE-M3 renforcerait la cohérence de la recherche hybride.

### 6.4 La collection Qdrant

`creer_collection_qdrant()` crée (si absente) une collection `lrgp_corpus` configurée pour la recherche hybride :

- vecteur **dense** nommé `"dense"` — 1024 dimensions, distance cosinus ;
- vecteur **sparse** nommé `"sparse"` — index en mémoire.

Chaque point porte un payload riche : `chunk_id`, `source`, `text`, `n_chars`, `chunk_index`, plus les champs de métadonnée (`source_file`, `section`, `page`, `chunk_type`, etc.).

### 6.5 Gestion des gros documents

L'indexation se fait par lots de 200 points (`indexer_par_lots`). Ce découpage évite une erreur réseau Windows (`WinError 10053`) observée quand un upsert unique transporte trop de points sur un gros document. La version 2 bascule automatiquement en mode « par lots » dès qu'un document dépasse 200 chunks.

### 6.6 Reprise

Comme le parsing, le chunking est reprenable : un document dont le `_chunks.jsonl` existe déjà est sauté. L'`offset` des identifiants Qdrant est initialisé à partir du `points_count` courant de la collection, ce qui garantit des IDs uniques même en plusieurs passes.

### Usage

```bash
# Prérequis : Qdrant doit tourner (conteneur Docker sur localhost:6333)
python ingestion/chunk_documents_livre.py
```

---

## 7. La version 2 : chunking adaptatif et classification (`chunk_documents_livre.py`)

La version initiale traitait tous les documents avec une taille de chunk fixe de 512 tokens. Cette uniformité posait problème pour les livres et thèses, dont les démonstrations s'étendent sur plusieurs paragraphes : un découpage trop fin séparait l'énoncé du résultat. La version 2 introduit trois améliorations.

### 7.1 Chunking adaptatif (512 / 1024 tokens)

La taille de chunk dépend désormais du **type de document** :

| Type | Taille de chunk | Justification |
|------|-----------------|---------------|
| Article (< 80 pages) | **512 tokens** | Taille standard des pipelines RAG académiques |
| Livre / thèse (≥ 80 pages) | **1024 tokens** | Préserve la cohérence des démonstrations longues |

La détection (`detecter_type_document`) repose sur deux critères en OU logique :

1. le nombre de pages du `DoclingDocument` est ≥ 80 ;
2. le nom du fichier contient un indice de livre (`handbook`, `baker`, `mulder`, `fundamentals`, `transport_phenomena`, etc. — liste `INDICES_LIVRE`).

L'effet est mesurable : les questions portant sur des démonstrations mathématiques bénéficient des chunks longs, qui contiennent à la fois l'énoncé, les étapes et le résultat.

### 7.2 Classification automatique du contenu

Chaque chunk est étiqueté par `classifier_contenu()` dans l'une de cinq catégories, par règles heuristiques (motifs lexicaux et typographiques) :

| Catégorie | Détection | Usage au retrieval |
|-----------|-----------|--------------------|
| `exercise` | « exercice 3 », « problem 5 » | — |
| `worked_example` | « solution : », « example 2 », « worked example » | Privilégié pour les questions de **calcul** |
| `definition` | « est défini comme », « is defined as » | Privilégié pour les questions **conceptuelles** |
| `equation_block` | ≥ 2 marqueurs `[Équation]` | Privilégié pour le **calcul** |
| `theory` | (catégorie par défaut) | Privilégié pour les questions **conceptuelles** |

Cette étiquette est stockée dans la métadonnée (`content_category`), ce qui permet au moteur de recherche de filtrer ou pondérer les résultats selon le type de question. Deux autres champs sont ajoutés : `doc_type` (livre/article) et `has_equation` (booléen).

### 7.3 Filtre de qualité pré-indexation

`doit_etre_indexe()` écarte le « bruit » documentaire avant indexation, pour éviter de polluer les résultats de recherche :

- segments trop courts (< 80 caractères significatifs) ;
- index alphabétiques et glossaires (détectés par des lignes nombreuses et très courtes, longueur moyenne < 30 caractères) ;
- sections de bibliographie, nomenclature, table des matières, remerciements (détectées sur la métadonnée `section`) ;
- mentions de copyright, ISBN, « all rights reserved » ;
- listes de références bibliographiques (heuristique sur les motifs `[1]`, `(2020). `).

Ce filtrage a un coût (quelques pour cent de segments écartés) mais améliore nettement la pertinence globale.

### 7.4 Rapport enrichi

Le rapport final de la version 2 distingue livres et articles, compte les chunks filtrés comme bruit, et affiche la répartition des chunks par catégorie de contenu — utile pour vérifier que le corpus contient bien des exemples résolus et des définitions, et pas uniquement de la théorie.

---

## 8. Utilitaires de maintenance

Ces scripts ne font pas partie du flux nominal mais sont essentiels en pratique, pour corriger ou rejouer une partie du traitement.

### Reprise et retraitement

- **`relancer_doc.py`** — reparse une liste de documents nommément désignés (chemins en dur dans le script), en réutilisant les fonctions de `parse_pdfs.py`. Pratique quand un document précis a échoué.
- **`relancer_gros_docs.py`** — rechunk une liste de gros documents avec un **batch d'encodage réduit** (`BATCH_SIZE_GROS = 8`) pour limiter la pression mémoire GPU, puis indexe par lots. À utiliser pour les manuels et handbooks volumineux.

### Diagnostic

- **`diagnostique_erreur_chunking.py`** — compare l'ensemble des documents parsés (`data/parsed/*.json`) à l'ensemble des documents chunkés (`data/chunks/*.jsonl`) et liste ceux qui ont été parsés mais **non chunkés**. Premier réflexe quand le compte de chunks paraît bas.

### Suppression / réindexation

- **`delete_book_chunks.py`** — supprime de Qdrant tous les chunks d'une source donnée (par filtre sur le champ `source`), avec confirmation interactive et affichage du delta de points. Sert à retirer un document mal indexé.
- **`supprimer_et_rindexer.py`** — va plus loin : supprime les points Qdrant **et** les fichiers associés (`parsed/<src>.json`, `chunks/<src>_chunks.jsonl`) **et** l'entrée du `parsing_report.json`, pour repartir d'une ardoise propre sur cette source avant de relancer parsing + chunking.

> ⚠️ Ces deux scripts utilisent des filtres différents (`source` vs `source_file`) selon la version du payload — vérifier le champ effectivement présent dans la collection avant suppression.

---

## 9. Scripts de test

- **`test_batch_small.py`** / **`test_batch_small_gpu.py`** — parsent un échantillon aléatoire de N PDF (10 par défaut) pour mesurer la vitesse réelle de Docling et **extrapoler la durée totale** du parsing complet. La version `_gpu` active l'accélérateur CUDA ; comparer les deux donne le gain GPU. Elles affichent aussi le taux de documents nécessitant un fallback (texte vide/pauvre).
- **`test_chunking_single.py`** — chunke un document unique et inspecte le résultat : aperçu des premiers chunks, repérage des équations récupérées via le fallback `orig`, recherche de traces de tableaux, et inspection directe des tableaux du `DoclingDocument` (export DataFrame / Markdown). Outil de débogage du chunking.
- **`test_retrieval_qdrant.py`** — valide le résultat final en interrogeant Qdrant : charge BGE-M3, encode quelques questions types (perméabilité CO₂ d'une membrane PDMS, flux à travers un contacteur, coefficient de transfert global) et affiche les 5 meilleurs passages en recherche dense, avec score, source et page. C'est le test de bout en bout du module.

---

## 10. Récapitulatif du flux complet

```bash
# 0. (Une fois) Lancer Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 1. Inventorier le corpus
python ingestion/corpus_stats.py --corpus "C:/.../corpus_lrgp"

# 2. (Optionnel) Estimer la vitesse sur un échantillon
python ingestion/test_batch_small_gpu.py --corpus "C:/.../corpus_lrgp" --n 10

# 3. Parser tout le corpus (reprenable)
python ingestion/parse_pdfs.py --corpus "C:/.../corpus_lrgp"

# 4. Parser les formats spécialisés
python ingestion/parse_csv.py --corpus "C:/.../corpus_lrgp"
python ingestion/parse_bib.py --corpus "C:/.../corpus_lrgp"

# 5. Réintégrer les JSON externes dans le rapport
python ingestion/register_external_json.py

# 6. Chunker + indexer (version retenue)
python ingestion/chunk_documents_livre.py

# 7. Valider le retrieval
python ingestion/test_retrieval_qdrant.py
```

### État final attendu

À l'issue du pipeline, la base vectorielle contient **155 274 segments** (chiffre du corpus final), chacun avec :

- un embedding dense (1024 dim, BGE-M3) et un vecteur sparse (TF approximé) ;
- un payload : source, page d'origine, type de document (livre/article), catégorie de contenu (theory / worked_example / etc.), présence d'équation.

C'est sur cette base que reposent toutes les opérations de recherche du système PRISME (voie documentaire et alimentation du contexte des agents).

---

## 11. Limites connues et pistes d'amélioration

Cohérentes avec les limites documentées dans le rapport de stage :

1. **Vecteur sparse approximé** — remplacer le TF maison par les *lexical weights* natifs de BGE-M3 (déterminisme, pas de collisions de hachage).
2. **Parsing des pages très denses** — Docling échoue ponctuellement sur des pages à tableaux/figures/équations imbriqués (quelques pour cent de pages perdues sur les livres). Stratégie de reprise sur erreur en place ; un fallback OCR dédié reste à industrialiser.
3. **Classification heuristique** — la catégorisation du contenu repose sur des règles lexicales ; un classifieur appris serait plus robuste mais plus lourd.
4. **Détection de type binaire** — le seuil de 80 pages est un compromis ; certains longs articles ou courts chapitres peuvent être mal classés.