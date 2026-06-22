# Documentation — Module SRAR-GP / PRISME (architecture multi-agents)

> Projet **PRISME / SRAR-GP** — Assistant IA pour la recherche en génie des procédés (LRGP)
> Module : `srar_gp/`
> C'est le **cœur du projet** : l'architecture multi-agents qui orchestre plusieurs modèles spécialisés et trois boucles d'auto-correction. C'est ce que le rapport présente sous le nom **PRISME**.

> Il intègre tout ce qui précède : le retriever de `rag/` (`05_rag.md`), le modèle fine-tuné de `training/` (`06_training.md`), et la base vectorielle de l'ingestion (`01_ingestion.md`). C'est le dernier maillon, celui qui transforme des briques séparées en un système cohérent.

---

## 1. Vue d'ensemble

L'idée structurante est l'**asymétrie cognitive** : plutôt qu'un seul gros modèle généraliste pour tout, on confie chaque tâche au modèle le mieux adapté. Le système se comporte comme une équipe aux compétences complémentaires.

PRISME décompose chaque question en facettes (documentaire, calcul, validation) traitées par des agents distincts, puis recompose le résultat — d'où la métaphore du prisme optique. Concrètement, c'est un **graphe d'états LangGraph** : les agents sont les nœuds, les transitions conditionnelles sont les arêtes, et un objet d'état unique circule entre eux en accumulant les données produites.

```
                          Question
                             │
                   ┌─────────▼─────────┐
                   │ Director (Qwen 27B)│  classifie la question
                   └─────────┬─────────┘
            ┌────────────────┼────────────────┐
            ▼                ▼                 ▼
       VOIE GÉNÉRALE   VOIE DOCUMENTAIRE   VOIE CALCUL
       (réponse        (Librarian +        (Engineer → Coder →
        directe)        Grader + Web)       Validator + 3 boucles)
```

Trois voies, trois niveaux de complexité croissante. La force du système est dans la troisième, avec ses boucles d'auto-correction.

---

## 2. Arborescence du module

```
srar_gp/
├── main.py                      # Point d'entrée : ask_srar() + REPL interactif
├── graph.py                     # ★ Câblage LangGraph : nœuds, arêtes, 3 boucles
├── state.py                     # SRARState : l'état partagé entre tous les agents
├── formatter.py                 # Mise en forme finale selon la voie
│
├── agents/                      # Les agents (= nœuds du graphe)
│   ├── director.py              # Orchestrateur : classifie + voie générale
│   ├── librarian.py             # Documentaliste (Qwen 9B fine-tuné)
│   ├── document_grader.py       # Évalue la pertinence des sources (Boucle 1)
│   ├── web_search_agent.py      # Recherche web complémentaire (Boucle 1)
│   ├── process_engineer.py      # Rédige le Blueprint mathématique
│   ├── calculation_expert.py    # Génère + exécute le code Python (Boucle 2)
│   └── validator.py             # Valide la cohérence physique (Boucle 3)
│
├── prompts/                     # Prompts par agent
│   ├── director_prompts.py      # Router + réponse générale
│   ├── engineer_prompts.py      # Blueprint (règles MISSING_DATA, anti-rumination)
│   └── coder_prompts.py         # Génération + correction de code
│
└── tools/                       # Outils déterministes
    ├── python_repl.py           # Sandbox d'exécution Python (Boucle 2)
    └── web_search.py            # Wrapper Tavily (Boucle 1)
```

---

## 3. L'état partagé — `state.py`

Tout le système repose sur un unique objet `SRARState` (un `TypedDict`) qui circule de nœud en nœud. Chaque agent en lit certains champs et en écrit d'autres. C'est ce qui rend le système traçable et débogable : à tout moment, l'état contient l'historique de ce qui a été produit.

Les champs principaux, par étape :

| Groupe | Champs | Écrit par |
|--------|--------|-----------|
| Entrée | `question` | (initial) |
| Routage | `voie`, `type_question` | Director |
| Documentaire | `sources_rag`, `context_rag`, `document_pertinent` | Librarian, Grader |
| Web | `sources_web` | Web Search |
| Calcul (plan) | `blueprint`, `missing_data` | Process Engineer |
| Calcul (exécution) | `code_python`, `resultat_numerique`, `execution_errors` | Calculation Expert |
| Re-négociation | `tentatives_renegociation`, `type_erreur`, `critique_validator`, `code_python_historique` | Validator, boucles |
| Validation | `validation_ok`, `validation_message` | Validator |
| Sortie | `reponse_finale`, `agents_actives` | tous |

Le champ `agents_actives` mérite une attention particulière : il est typé `Annotated[list[str], add]`. L'annotation `add` indique à LangGraph d'**accumuler** (concaténer) les contributions de chaque nœud au lieu de les écraser. C'est ce qui construit automatiquement la **trace du parcours** — la liste ordonnée des agents traversés, exploitée par le mode verbose et l'évaluation.

---

## 4. Les agents

### 4.1 Director — `director.py` (Qwen 27B)

L'orchestrateur. Il remplit deux rôles :

**Classification** (`classifier_question`) — détermine la voie via une stratégie **hybride** en deux temps :
1. **Heuristique regex d'abord** — des motifs (`PATTERNS_GENERAL`, `PATTERNS_CALCUL`, `PATTERNS_COMPARAISON`) couvrent les cas évidents instantanément, sans appel LLM. « bonjour » → GENERAL, « calcule » → CALCUL, etc.
2. **LLM router ensuite** — seulement pour les cas ambigus que la regex ne tranche pas, via `PROMPT_ROUTER` (température 0.0).
3. **Fallback FACTUEL** si tout échoue.

Le type est ensuite mappé sur une voie : GENERAL → générale, CALCUL → calcul, le reste (FACTUEL/COMPARAISON) → documentaire.

**Réponse générale** (`reponse_generale`) — pour les questions conversationnelles/méta (« qui es-tu ? »), le Director répond directement sans mobiliser les autres agents, avec un fallback en dur décrivant le système si la réponse est vide.

> Choix de conception malin : faire la classification d'abord par regex évite un appel LLM (et sa latence) sur la majorité des questions évidentes. Le LLM n'est sollicité que sur le doute.

### 4.2 Librarian — `librarian.py` (Qwen 9B fine-tuné V4/V5)

Le documentaliste spécialisé — c'est ici qu'intervient le **modèle fine-tuné** produit par `training/`. Il utilise le `LRGPChain` du module `rag/` (donc le retriever hybride + reranker).

Architecture en **deux phases** pour la voie documentaire :
1. `extraire_documentaire` — recherche RAG **seule**, sans génération. Remplit `sources_rag` et `context_rag`. (Séparer l'extraction de la génération permet d'insérer le Document Grader entre les deux.)
2. `generer_reponse_documentaire` — génération finale, avec un **prompt « expert senior »** très exigeant : vocabulaire technique précis, esprit critique systématique (limites et défis obligatoires), structure complète, citations. C'est la traduction directe de l'apprentissage du rapport : forcer la profondeur pour contrer la superficialité du 9B.

Une variante `librarian_pour_calcul` (définie dans `graph.py`) fait la même extraction RAG mais pour alimenter la voie calcul, sans générer de réponse.

> Le fichier conserve une fonction `rechercher_et_repondre` marquée **deprecated** — vestige de l'architecture mono-phase avant l'insertion du Grader. À nettoyer.

### 4.3 Document Grader — `document_grader.py` (Qwen 27B) — Boucle 1

L'évaluateur de pertinence. Il reçoit les sources RAG et juge, en JSON strict, si elles suffisent à répondre précisément (`pertinent: true/false` + raison + ce qui manque). C'est lui qui **déclenche la Boucle 1** : si les documents sont insuffisants, le graphe bascule vers la recherche web.

Robustesse : zéro source → non pertinent direct (fallback web) ; erreur de parsing → considéré pertinent par défaut (on ne bloque pas sur une erreur du juge).

### 4.4 Web Search Agent — `web_search_agent.py` — Boucle 1

Le filet de sécurité documentaire. Quand le Grader juge le corpus local insuffisant, cet agent interroge le web via Tavily (`tools/web_search.py`), concatène les résultats au contexte existant, et marque `document_pertinent: True` pour laisser le flux continuer. C'est la **première boucle d'auto-correction** : le système ne se résigne pas à un corpus lacunaire.

### 4.5 Process Engineer — `process_engineer.py` (Qwen 27B)

L'ingénieur procédés. Pour la voie calcul, il transforme la question en un **Blueprint** — un cahier des charges mathématique structuré en 5 sections (hypothèses, données numériques, équations littérales numérotées, méthode numérique, résultat attendu).

Le prompt (`engineer_prompts.py`) est remarquablement défensif, encodant des leçons apprises :
- **Règle MISSING_DATA** — un format strict pour signaler une donnée absente, avec interdiction explicite de l'utiliser pour dire « rien ne manque » (source de faux positifs).
- **Gestion des valeurs anormales** — si une valeur est physiquement impossible, calculer quand même (sans corriger) et laisser le Validator juger.
- **Anti-rumination** — interdiction de lister plusieurs hypothèses concurrentes ; choisir, justifier en une phrase, avancer. « Si le Blueprint dépasse 3500 caractères, c'est que tu rumines. »
- **Anti-redondance** — ne jamais définir deux variables pour la même quantité (ex. `L_um` et `L_m`), source classique de bugs d'unités.

### 4.6 Calculation Expert — `calculation_expert.py` (DeepSeek-Coder 6.7B) — Boucle 2

Le développeur. Il traduit le Blueprint en code Python (`PROMPT_CODE` : numpy/scipy uniquement, autonome, résultat via `print()`) et l'exécute dans le sandbox. **La Boucle 2 est interne à cet agent** : en cas d'échec d'exécution, la trace d'erreur est renvoyée au modèle avec `PROMPT_CODE_FIX`, jusqu'à `MAX_FIX_ATTEMPTS = 3` tentatives.

Une fonction `extraire_code` particulièrement robuste gère les multiples formats de blocs Markdown que DeepSeek peut produire (4 stratégies en cascade, plus un nettoyage des « tokens corrompus » mentionnés dans le rapport).

### 4.7 Validator — `validator.py` (Qwen 27B) — Boucle 3

Le contrôleur final. Il confronte question + Blueprint + code + résultat, et valide en JSON la **cohérence physique** via une procédure en 3 étapes (analyse du code, analyse du résultat, classification de l'erreur). Le point clé est la **classification du type d'erreur**, qui pilote la Boucle 3 :

| `type_erreur` | Signification | Renvoi vers |
|---------------|---------------|-------------|
| `code` | Bug de syntaxe/unité/logique du code | Correction code |
| `physique` | Formule fausse / hypothèse erronée dans le Blueprint | Reformulation Blueprint |
| `ambigu` | La question elle-même est ambiguë | Fin |
| `aucune` | Pas d'erreur (valide) | Fin |

Le prompt insiste sur une règle subtile : ne jamais rejeter un calcul à cause d'une faute dans la *question* — si le code a corrigé une faute de frappe, c'est un bon comportement. Et ne pas halluciner d'erreurs absentes du code réel.

---

## 5. Les trois boucles d'auto-correction

C'est ce qui justifie le « Self-Reflective » de SRAR-GP. Chacune adresse un type d'erreur différent.

```
BOUCLE 1 (information)    Doc Grader juge insuffisant → Web Search → réinjecte
BOUCLE 2 (exécution)      Code échoue → trace renvoyée au Coder → max 3 fois
BOUCLE 3 (physique)       Validator détecte incohérence → classifie l'erreur :
                            ├─ "code"     → correction_code     → re-valide
                            └─ "physique" → correction_physique → re-code → re-valide
                          max 2 tentatives, puis fin honnête
```

**Boucle 1 — Information augmentée.** Dans `graph.py`, le Document Grader est commun aux voies documentaire ET calcul. S'il juge les chunks insuffisants → nœud `web_search` → réinjection dans le flux. Garde-fou anti-bouclage : si une recherche web a déjà eu lieu (`deja_web_search`), on ne la relance pas.

**Boucle 2 — Exécution déterministe.** Interne au Calculation Expert (§4.6). Jusqu'à 3 corrections sur la base de la stack trace.

**Boucle 3 — Re-négociation physique.** La plus sophistiquée. Après le Validator, `router_renegociation` décide : si valide → fin ; si max atteint (2) ou échec technique récent → fin honnête ; sinon, selon `type_erreur`, renvoi vers `correction_code` (qui re-code et re-valide) ou `correction_physique` (qui reformule le Blueprint puis repasse par le Coder). Le compteur `tentatives_renegociation` empêche les boucles infinies.

> Philosophie commune aux trois boucles, fidèle au rapport : **l'honnêteté de l'échec est préférée à la plausibilité d'une réponse inventée.** Au-delà des limites de tentatives, le système signale l'échec plutôt que d'halluciner.

---

## 6. Le câblage du graphe — `graph.py`

`build_graph()` assemble tout en un `StateGraph` LangGraph. Le flux complet :

```
director_classifier ─┬─ GENERAL ──────► director_general ──────────────────► END
                     ├─ DOCUMENTAIRE ─► librarian_doc_extract ─┐
                     └─ CALCUL ───────► librarian_rag ─────────┤
                                                                ▼
                                                          doc_grader
                                          ┌──────────────────┼──────────────┐
                                    (pertinent/déjà web)  (insuffisant)   │
                                          │                  ▼              │
                                          │             web_search ─────────┤
                                          ▼                                 ▼
                            DOC: librarian_doc_gen → END    CALCUL: process_engineer
                                                                        │
                                                          router_missing_data
                                                          ┌─────────────┴──────────────┐
                                                    (manque)                      (complet)
                                                          ▼                            ▼
                                              missing_data_handler → END      calculation_expert
                                                                                       │ (Boucle 2 interne)
                                                                                       ▼
                                                                                   validator
                                                                    router_renegociation (Boucle 3)
                                                          ┌──────────────┬──────────────┐
                                                    correction_code  correction_physique  end → END
                                                          │                  │
                                                          ▼                  ▼
                                                      validator      calculation_expert
```

Les nœuds non-agents ajoutés dans `graph.py` :
- **`missing_data_handler`** — court-circuit honnête : si le Process Engineer signale de vraies données manquantes, le système refuse de calculer et liste ce qui manque (« le système refuse d'inventer des valeurs »). Le routeur `router_missing_data` filtre au passage les faux positifs (négations type « aucune donnée manquante »).
- **`correction_code`** / **`correction_physique`** — les nœuds de la Boucle 3 (re-code avec prompt simplifié / reformulation du Blueprint).

Le graphe est compilé une fois et mis en cache (`get_graph()` avec singleton `_graph`), ce qui explique le pré-chargement par `api_server.py` (`05_rag.md`).

---

## 7. Les outils déterministes — `tools/`

### 7.1 Sandbox Python — `python_repl.py`

L'outil le plus critique de la voie calcul. Il exécute le code généré dans un **environnement conda isolé dédié** (`srar-repl`), pas dans le processus principal. Caractéristiques :

- **Sécurité** — une liste de motifs interdits (`os.system`, `subprocess`, `eval`, `exec`, `open`, `socket`, `requests`…) bloque le code dangereux avant exécution.
- **Whitelist de librairies** — seules les librairies scientifiques autorisées (numpy, scipy, sympy, fluids, thermo, CoolProp, cantera…) peuvent être **auto-installées** si un import manque, jusqu'à 2 fois.
- **Timeout** — 60 s par défaut, pour éviter les boucles infinies dans le code généré.
- **UTF-8 forcé** — corrections spécifiques Windows (`PYTHONUTF8`, décodage manuel en mode binaire) pour gérer les caractères spéciaux.

C'est le « moteur d'exécution déterministe » du rapport : c'est lui qui garantit que le résultat numérique est *calculé*, pas *halluciné* par un LLM.

### 7.2 Recherche web — `web_search.py`

Wrapper autour de **Tavily**, restreint à des **domaines scientifiques prioritaires** (doi.org, sciencedirect, rsc.org, nature, acs, hal.science, techniques-ingénieur…). Tronque les requêtes trop longues (limite Tavily à ~400 caractères). Dégrade proprement si la clé API manque (retourne une liste vide plutôt que de planter).

---

## 8. Formatage et point d'entrée

### 8.1 `formatter.py`

Met en forme la réponse finale **selon la voie** empruntée :
- **GÉNÉRALE** — réponse brute.
- **DOCUMENTAIRE** — réponse + footer sources LRGP.
- **CALCUL** — format structuré riche : résultat en avant, puis démarche (hypothèses / équations / méthode extraites du Blueprint), validation physique, sources, et un bloc repliable `<details>` avec le Blueprint complet, le code Python et la trace du parcours. En cas de validation échouée ou de données manquantes, le format s'adapte (avertissement + détails techniques).

C'est ce formatage qui produit la présentation soignée vue dans les traces verbose du rapport (Annexe III).

### 8.2 `main.py`

Le point d'entrée. `ask_srar(question)` initialise l'état complet, invoque le graphe, applique le formatter, et retourne l'état final. C'est la fonction appelée par `api_server.py` (`05_rag.md`) et par `benchmark_srar_gp.py` (`02_evaluation`). En exécution directe (`python -m srar_gp.main`), il offre un REPL interactif et affiche le diagramme Mermaid du graphe.

---

## 9. Correspondance avec le rapport (les 6 agents)

| Rapport (nom PRISME) | Code (fichier) | Modèle |
|----------------------|----------------|--------|
| Director | `director.py` | Qwen 27B |
| Librarian | `librarian.py` | Qwen 9B fine-tuné |
| Document Grader | `document_grader.py` | Qwen 27B |
| Process Engineer | `process_engineer.py` | Qwen 27B |
| Calculation Expert | `calculation_expert.py` | DeepSeek-Coder 6.7B |
| Validator | `validator.py` | Qwen 27B |

Quatre rôles (Director, Engineer, Grader, Validator) partagent la **même instance Qwen 27B** (appels stateless), conformément à l'optimisation VRAM décrite dans le rapport (Annexe I). Le Web Search n'est pas un « agent LLM » mais un outil, d'où le compte de 6 agents + outils.

---

## 10. Points d'attention transversaux

1. **`librarian.py` : fonction dépréciée** — `rechercher_et_repondre` est un vestige de l'architecture mono-phase ; à supprimer.
2. **Modèle Librarian** — `lrgp-knowledge_v5` en dur dans `librarian.py` ; cohérence à maintenir avec le nom de modèle déployé (rappel des `05`/`06`).
3. **Chemin du REPL en dur** — `PYTHON_EXE` pointe vers `C:\Users\Samir\miniforge3\envs\srar-repl\python.exe` ; à externaliser, c'est un point de rupture majeur pour la portabilité.
4. **Dépendances DEBUG** — plusieurs agents (`director.py`, `process_engineer.py`) impriment des blocs de debug verbeux (`print(response)`, « BLUEPRINT COMPLET ») ; à passer derrière un flag verbose.
5. **Limites de boucles codées en dur** — `MAX_FIX_ATTEMPTS=3` (Boucle 2) et le seuil de 2 tentatives (Boucle 3) sont dispersés dans le code ; les centraliser dans une config faciliterait le réglage (le rapport mentionne ces valeurs comme configurables).
6. **`tentatives_renegociation` partagé** — le compteur est commun aux deux types de correction (code et physique) ; une longue séquence mixte pourrait atteindre la limite globale plus vite qu'attendu. Comportement à vérifier.
7. **Web Search dans la voie calcul** — le Grader et le Web Search sont câblés aussi pour CALCUL, mais leur intérêt y est moindre (un calcul a surtout besoin de données, pas de prose web). À évaluer.

---

## 11. Conclusion : la place de ce module

`srar_gp/` est le point de convergence de tout le projet. Il réutilise le retriever (`rag/`), le modèle fine-tuné (`training/`), la base vectorielle (`ingestion/`), et il est évalué par le harnais (`evaluation/`). C'est l'incarnation du principe directeur du rapport — l'asymétrie cognitive — et de son résultat principal : une architecture qui n'est pas la plus brillante sur chaque question isolée, mais la plus **fiable épistémiquement**, capable de signaler ses limites (données manquantes, validation échouée, échec après N tentatives) plutôt que d'inventer.

C'est aussi le module le plus directement « vivant » : c'est lui qui tourne derrière le modèle `srar-gp` exposé à Open WebUI, et c'est son comportement (voies empruntées, boucles déclenchées) que le benchmark mesure.



## Correctifs de robustesse des agents revision 22/06/2026

Trois correctifs ont été apportés aux prompts de validation et de modélisation,
en réponse à des biais classiques des LLM observés en conditions réelles. Ils
ne changent pas l'architecture, mais fiabilisent fortement les agents Validator
et Process Engineer.

### 1. Préservation des données utilisateur (Validator)

**Symptôme.** Sur un calcul de perméabilité d'une membrane PDMS, le système
calculait correctement mais obtenait un résultat éloigné de l'intuition
physique du LLM. Le Validator rejetait alors systématiquement le résultat ;
sous la pression de produire une valeur « acceptable », le Process Engineer
finissait par modifier discrètement les données d'entrée de l'utilisateur (par
exemple en changeant le flux) pour bidouiller le résultat final.

**Correctif.** Une règle d'or impose désormais la primauté des mathématiques
sur les croyances physiques du Validator : si le calcul est juste au regard des
données fournies, il DOIT valider, quitte à émettre des réserves dans sa
synthèse textuelle. Les données d'entrée de l'utilisateur ne sont jamais
modifiées.

**Résultat.** Le système valide le calcul sans altérer les données d'entrée.

### 2. Vérification prioritaire des conversions d'unités (Validator)

**Symptôme.** Sur un calcul d'énergie de captage du CO₂, une simple erreur de
conversion (division par 3,6×10⁹ au lieu de 3,6×10⁶ pour obtenir des kWh)
donnait 0,26 kWh/tonne au lieu de ~263. Le Validator détectait l'incohérence
physique mais accusait le modèle théorique plutôt que la conversion,
déclenchant une boucle infinie où les agents échangeaient des théories de plus
en plus complexes tout en conservant l'erreur de conversion triviale.

**Correctif.** Une directive ajoutée à l'Étape 1 du prompt du Validator
l'oblige à vérifier rigoureusement les facteurs de conversion d'unités avant de
remettre en cause la physique du Blueprint.

### 3. Approche macroscopique et bridage JSON (Process Engineer / Validator)

**Symptôme.** Le Process Engineer produisait des Blueprints verbeux et exigeait
du Calculation Expert la résolution de systèmes différentiels complexes (ODEs,
matrices NumPy) pour de simples calculs d'énergie. Cela provoquait des
troncatures côté serveur et des plantages de code, que le Validator imputait à
tort au codeur ou au modèle physique — relançant des boucles de correction qui
saturaient la mémoire.

**Correctif.** Passage en JSON strict pour brider la verbosité (éliminant la
troncature) et règle d'approche macroscopique imposant des modèles algébriques
simples et découplés pour les énergies minimales.

**Résultat.** Temps de réponse divisés par ~3, code Python généré sans bug,
résultat thermodynamique fiable.

### Fil conducteur

Ces trois correctifs adressent deux biais récurrents des LLM :

1. **Modifier les données pour plaire à l'évaluateur** — corrigé par la
   préservation des données utilisateur.
2. **Chercher des erreurs intellectuellement complexes plutôt que vérifier les
   détails triviaux** (les divisions par 1000) — corrigé par la directive sur
   les conversions d'unités.

### Perspective : Validator connecté au RAG / recherche web (V2)

Doter le Validator de ses propres outils de fact-checking, suivant les
architectures *Actor-Critic* de l'état de l'art. Aujourd'hui, le Validator juge
la plausibilité physique d'un résultat à partir de ses poids internes, qui
peuvent le tromper sur un procédé novateur ou spécifique. Avec un accès au RAG
LRGP (et au web), il pourrait vérifier un résultat contre la littérature et
valider avec une certitude documentée, citations à l'appui.

**Garde-fou impératif.** La préservation des données utilisateur doit être
maintenue : si l'utilisateur impose une sélectivité α = 40 et que la
littérature trouvée indique 20, le Validator ne doit pas rejeter le calcul. Il
valide le calcul mathématique en ajoutant une note signalant l'écart
(« la valeur imposée de 40 dépasse l'état de l'art trouvé, qui est de 20 »).