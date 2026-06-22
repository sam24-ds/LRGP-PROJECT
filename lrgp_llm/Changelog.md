# Changelog — PRISME / SRAR-GP

Toutes les évolutions notables du projet sont consignées ici.

---

## 22/06/2026 Correctifs de robustesse des agents

Trois correctifs apportés aux prompts de validation et de modélisation pour
neutraliser des biais classiques des LLM observés en conditions réelles.
Détails techniques dans `docs/07_srar_gp.md`, section « Correctifs de robustesse ».

### Corrigé

- **Préservation des données utilisateur (Validator)** — le système ne modifie
  plus les données d'entrée fournies par l'utilisateur pour « forcer » un
  résultat jugé plus plausible. Si le calcul est juste au regard des données
  fournies, le Validator valide, en émettant au besoin des réserves dans sa
  synthèse — sans jamais altérer les entrées.
- **Vérification des conversions d'unités en priorité (Validator)** — le
  Validator vérifie désormais les facteurs de conversion d'unités *avant* de
  remettre en cause la physique du Blueprint, ce qui évite les boucles de
  correction infinies déclenchées par de simples erreurs de conversion.
- **Approche macroscopique + bridage JSON (Process Engineer / Validator)** —
  les Blueprints privilégient des modèles algébriques simples et découplés pour
  les calculs d'énergie ; le passage en JSON strict bride la verbosité et
  élimine les troncatures. Temps de réponse divisés par ~3, code généré sans bug.

### Perspective (V2)

- **Validator connecté au RAG / recherche web** (architecture Actor-Critic) —
  doter le Validator d'outils de fact-checking pour vérifier la plausibilité
  d'un résultat contre la littérature, avec citations. Garde-fou impératif :
  conserver la préservation des données utilisateur — un écart avec la
  littérature est signalé, jamais imposé comme correction.

---

## [POC initial] — Architecture multi-agents PRISME

- Pipeline complet : ingestion (Docling + Qdrant), RAG hybride (BGE-M3 +
  reranker), fine-tuning LoRA (V1→V4), architecture multi-agents (6 agents,
  3 boucles d'auto-correction), évaluation (benchmark 41 questions, LLM-Judge).