"""
formatter.py
Formatte les réponses finales pour affichage dans Open WebUI.
Adapte le format selon la voie empruntée.
"""
from srar_gp.state import SRARState


def format_voie_general(state: SRARState) -> str:
    """Format simple pour les questions conversationnelles."""
    return state["reponse_finale"]


def format_voie_documentaire(state: SRARState) -> str:
    """Format pour les questions FACTUEL / COMPARAISON."""
    reponse = state["reponse_finale"]
    
    # La réponse contient déjà les sources via le Librarian
    # On enrichit avec un en-tête optionnel
    sources = state.get("sources_rag", [])
    
    if not sources:
        return reponse
    
    # Si pas déjà inclus, ajouter les sources
    if "Sources" not in reponse and "📚" not in reponse:
        sources_text = "\n\n---\n📚 **Sources LRGP consultées :**\n"
        for s in sources[:3]:
            src_name = s.get("source", "?")[:80]
            sources_text += f"- {src_name}\n"
        reponse += sources_text
    
    return reponse


def format_voie_calcul(state: SRARState) -> str:
    """Format structuré pour les questions CALCUL avec validation."""
    
    # Cas 1 — Données manquantes (court-circuit)
    if "missing_data_handler" in state.get("agents_actives", []):
        return state["reponse_finale"]  # déjà bien formaté
    
    # Cas 2 — Validation échouée
    if not state.get("validation_ok"):
        return (
            f"## ⚠ Résultat à vérifier\n\n"
            f"{state.get('reponse_finale', '')}\n\n"
            f"---\n\n"
            f"<details>\n<summary>🔍 Voir les détails techniques</summary>\n\n"
            f"### Blueprint mathématique\n"
            f"```\n{state.get('blueprint', '')[:2000]}\n```\n\n"
            f"### Code Python exécuté\n"
            f"```python\n{state.get('code_python', '')[:1500]}\n```\n"
            f"</details>\n"
        )
    
    # Cas 3 — Validation OK — réponse complète structurée
    resultat = state.get("resultat_numerique", "")
    blueprint = state.get("blueprint", "")
    code = state.get("code_python", "")
    validation_msg = state.get("validation_message", "")
    sources = state.get("sources_rag", [])
    
    # Extraire les hypothèses du Blueprint (section 1)
    hypotheses = _extraire_section(blueprint, "1. HYPOTHÈSES")
    equations = _extraire_section(blueprint, "3. ÉQUATIONS")
    methode = _extraire_section(blueprint, "4. MÉTHODE")
    
    reponse = f"""## 📊 Résultat

**{resultat}**

## 🔬 Démarche

### Hypothèses retenues
{hypotheses if hypotheses else "_Non spécifiées_"}

### Équations utilisées
{equations if equations else "_Non spécifiées_"}

### Méthode numérique
{methode if methode else "_Calcul direct_"}

## ✓ Validation physique

{validation_msg if validation_msg else "Résultat cohérent avec la physique du problème."}
"""
    
    # Ajouter les sources si présentes
    if sources:
        reponse += "\n## 📚 Sources LRGP\n\n"
        for s in sources[:3]:
            src_name = s.get("source", "?")[:80]
            reponse += f"- {src_name}\n"
    
    # Ajouter les détails techniques repliables
    reponse += f"""

---

<details>
<summary>🔧 Voir les détails techniques complets</summary>

### Blueprint mathématique complet

{blueprint[:3000]}

### Code Python exécuté

```python
{code[:4000]}
```

### Trace du parcours
`{' → '.join(state.get('agents_actives', []))}`

</details>
"""
    
    return reponse


def _extraire_section(blueprint: str, marqueur: str) -> str:
    """Extrait une section du Blueprint entre deux titres ##."""
    if marqueur not in blueprint:
        return ""
    
    # Trouver le début
    idx_debut = blueprint.find(marqueur)
    
    # Trouver le prochain ## après
    debut_contenu = blueprint.find("\n", idx_debut) + 1
    idx_fin = blueprint.find("\n## ", debut_contenu)
    
    if idx_fin == -1:
        contenu = blueprint[debut_contenu:]
    else:
        contenu = blueprint[debut_contenu:idx_fin]
    
    return contenu.strip()[:1500]  # limiter la taille


def formatter_reponse(state: SRARState) -> str:
    """
    Point d'entrée principal — formate la réponse selon la voie.
    """
    voie = state.get("voie", "UNKNOWN")
    
    if voie == "GENERAL":
        return format_voie_general(state)
    elif voie == "DOCUMENTAIRE":
        return format_voie_documentaire(state)
    elif voie == "CALCUL":
        return format_voie_calcul(state)
    else:
        return state.get("reponse_finale", "Pas de réponse générée.")