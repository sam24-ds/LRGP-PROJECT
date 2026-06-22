"""
formatter.py
Formatte les réponses finales pour affichage dans Open WebUI.
Adapte le format selon la voie empruntée.

Le Blueprint est désormais produit en JSON par le Process Engineer :
  {
    "hypotheses_simplificatrices": [...],
    "donnees_manquantes": [...],
    "donnees_numeriques": {...},
    "equations_a_coder": [...],        # lignes de code Python (pas du LaTeX)
    "methode_resolution": "...",
    "resultat_attendu": "..."
  }

Le formatter lit ce JSON et le transforme en Markdown lisible :
  • résumé "Démarche" (hypothèses, équations en bloc Python, méthode)
  • section repliable reconstruite en sections numérotées
Un fallback Markdown est conservé si jamais le Blueprint revient en texte brut.
"""
import json
from srar_gp.state import SRARState


# ══════════════════════════════════════════════════════════════
# VOIE GÉNÉRAL
# ══════════════════════════════════════════════════════════════
def format_voie_general(state: SRARState) -> str:
    """Format simple pour les questions conversationnelles."""
    return state["reponse_finale"]


# ══════════════════════════════════════════════════════════════
# VOIE DOCUMENTAIRE
# ══════════════════════════════════════════════════════════════
def format_voie_documentaire(state: SRARState) -> str:
    """Format pour les questions FACTUEL / COMPARAISON."""
    reponse = state["reponse_finale"]

    sources = state.get("sources_rag", [])
    sources_web = state.get("sources_web", [])

    if not sources and not sources_web:
        return reponse

    if "Sources" not in reponse and "📚" not in reponse:
        sources_text = "\n\n---\n📚 **Sources consultées :**\n"
        for s in sources[:3]:
            sources_text += f"- [LRGP] {s.get('source', '?')[:80]}\n"
        for s in sources_web[:3]:
            url = s.get('url', '')
            titre = s.get('title', s.get('source', '?'))[:80]
            sources_text += f"- [Web] [{titre}]({url})\n" if url else f"- [Web] {titre}\n"
        reponse += sources_text

    return reponse


# ══════════════════════════════════════════════════════════════
# OUTILS BLUEPRINT (JSON prioritaire, fallback Markdown)
# ══════════════════════════════════════════════════════════════
def _parse_blueprint(blueprint_str: str):
    """Renvoie le dict JSON du Blueprint, ou None si ce n'est pas du JSON."""
    bp = (blueprint_str or "").strip()
    if bp.startswith("{"):
        try:
            return json.loads(bp)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _resume_blueprint(blueprint_str: str):
    """Retourne (hypotheses, equations, methode) prêts à afficher
    pour le résumé 'Démarche'. equations est déjà encadré si besoin."""
    bp = _parse_blueprint(blueprint_str)

    # ── Cas JSON ──
    if bp is not None:
        hyp = bp.get("hypotheses_simplificatrices", [])
        hypotheses = "\n".join(f"- {h}" for h in hyp) if isinstance(hyp, list) else str(hyp)

        eq = bp.get("equations_a_coder", [])
        code = "\n".join(str(e) for e in eq) if isinstance(eq, list) else str(eq)
        equations = f"```python\n{code}\n```" if code.strip() else ""

        methode = bp.get("methode_resolution", "")
        if isinstance(methode, list):
            methode = "\n".join(f"- {m}" for m in methode)
        return hypotheses, equations, str(methode)

    # ── Cas Markdown (fallback) ──
    hypotheses = _extraire_section(blueprint_str, "HYPOTHÈSES")
    equations = _extraire_section(blueprint_str, "ÉQUATIONS")   # LaTeX : pas de bloc code
    methode = _extraire_section(blueprint_str, "MÉTHODE")
    return hypotheses, equations, methode


def _blueprint_complet(blueprint_str: str) -> str:
    """Rend le Blueprint complet en Markdown lisible pour la section repliable.
    JSON → sections reconstruites ; Markdown → rendu tel quel."""
    bp = _parse_blueprint(blueprint_str)

    if bp is None:
        raw = (blueprint_str or "").strip()
        return raw if raw else "_Blueprint indisponible_"

    parts = []

    hyp = bp.get("hypotheses_simplificatrices", [])
    if hyp:
        corps = "\n".join(f"- {h}" for h in hyp) if isinstance(hyp, list) else str(hyp)
        parts.append("#### 1. Hypothèses simplificatrices\n" + corps)

    manquantes = bp.get("donnees_manquantes", [])
    if manquantes:
        corps = "\n".join(f"- {d}" for d in manquantes) if isinstance(manquantes, list) else str(manquantes)
        parts.append("#### ⚠ Données manquantes\n" + corps)

    donnees = bp.get("donnees_numeriques", {})
    if isinstance(donnees, dict) and donnees:
        table = "#### 2. Données numériques\n\n| Paramètre | Valeur |\n|---|---|\n"
        for k, v in donnees.items():
            table += f"| `{k}` | {v} |\n"
        parts.append(table.rstrip())

    eq = bp.get("equations_a_coder", [])
    if eq:
        code = "\n".join(str(e) for e in eq) if isinstance(eq, list) else str(eq)
        parts.append("#### 3. Équations à coder\n```python\n" + code + "\n```")

    methode = bp.get("methode_resolution", "")
    if methode:
        if isinstance(methode, list):
            methode = "\n".join(f"- {m}" for m in methode)
        parts.append("#### 4. Méthode de résolution\n" + str(methode))

    attendu = bp.get("resultat_attendu", "")
    if attendu:
        parts.append("#### 5. Résultat attendu\n" + str(attendu))

    return "\n\n".join(parts) if parts else "_Blueprint vide_"


def _extraire_section(blueprint: str, marqueur: str) -> str:
    """Fallback Markdown : extrait le contenu entre un titre ## …marqueur… et le ## suivant."""
    if not blueprint or marqueur not in blueprint:
        return ""
    idx_debut = blueprint.find(marqueur)
    debut_contenu = blueprint.find("\n", idx_debut) + 1
    idx_fin = blueprint.find("\n## ", debut_contenu)
    contenu = blueprint[debut_contenu:] if idx_fin == -1 else blueprint[debut_contenu:idx_fin]
    return contenu.strip()


# ══════════════════════════════════════════════════════════════
# VOIE CALCUL
# ══════════════════════════════════════════════════════════════
def format_voie_calcul(state: SRARState) -> str:
    """Format structuré pour les questions CALCUL avec validation."""

    # Cas 1 — Données manquantes (court-circuit)
    if "missing_data_handler" in state.get("agents_actives", []):
        return state["reponse_finale"]

    blueprint_str = state.get("blueprint", "")
    code = state.get("code_python", "")
    trace = " → ".join(state.get("agents_actives", []))

    # Cas 2 — Validation échouée
    if not state.get("validation_ok"):
        return (
            f"## ⚠ Résultat à vérifier\n\n"
            f"{state.get('reponse_finale', '')}\n\n"
            f"---\n\n"
            f"<details>\n<summary>🔍 Voir les détails techniques</summary>\n\n"
            f"### Blueprint mathématique\n\n"
            f"{_blueprint_complet(blueprint_str)}\n\n"
            f"### Code Python exécuté\n\n"
            f"```python\n{code}\n```\n\n"
            f"### Trace du parcours\n\n{trace}\n\n"
            f"</details>\n"
        )

    # Cas 3 — Validation OK
    resultat = state.get("resultat_numerique", "")
    validation_msg = state.get("validation_message", "") \
        or "Résultat cohérent avec la physique du problème."
    sources_rag = state.get("sources_rag", [])
    sources_web = state.get("sources_web", [])

    hypotheses, equations, methode = _resume_blueprint(blueprint_str)

    # ── EN-TÊTE + DÉMARCHE ──
    reponse = f"""## 📊 Résultat

**{resultat}**

## 🔬 Démarche

### Hypothèses retenues
{hypotheses if hypotheses.strip() else "_Non spécifiées_"}

### Équations utilisées
{equations if equations.strip() else "_Non spécifiées_"}

### Méthode numérique
{methode if methode.strip() else "Calcul direct"}

## ✓ Validation physique
{validation_msg}
"""

    # ── SOURCES ──
    if sources_rag:
        reponse += "\n## 📚 Sources LRGP\n"
        for s in sources_rag[:3]:
            reponse += f"- {s.get('source', '?')[:80]}\n"
    if sources_web:
        reponse += "\n## 🌐 Sources Web\n"
        for s in sources_web[:3]:
            url = s.get('url', '')
            titre = s.get('title', s.get('source', '?'))[:80]
            reponse += f"- [{titre}]({url})\n" if url else f"- {titre}\n"

    # ── DÉTAILS TECHNIQUES REPLIABLES ──
    reponse += f"""
---

<details>
<summary>🔧 Voir les détails techniques complets</summary>

### Blueprint mathématique complet

{_blueprint_complet(blueprint_str)}

### Code Python exécuté

```python
{code}
```

### Trace du parcours

{trace}

</details>
"""

    return reponse


# ══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════
def formatter_reponse(state: SRARState) -> str:
    """Formate la réponse selon la voie empruntée."""
    voie = state.get("voie", "UNKNOWN")

    if voie == "GENERAL":
        return format_voie_general(state)
    elif voie == "DOCUMENTAIRE":
        return format_voie_documentaire(state)
    elif voie == "CALCUL":
        return format_voie_calcul(state)
    else:
        return state.get("reponse_finale", "Pas de réponse générée.")