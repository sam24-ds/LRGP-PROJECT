"""
debug_json.py
Identifie et corrige les erreurs JSON des réponses LLM.
"""
import json
import re
from pathlib import Path

# Lire le fichier brut
fichier = Path("reponse_llm.json")
texte   = fichier.read_text(encoding="utf-8")

print(f"Taille du fichier : {len(texte)} chars")
print(f"Aperçu début : {texte[:200]}")
print(f"Aperçu fin   : {texte[-200:]}")

# ── Tentative 1 : parsing direct ─────────────────────────────────
try:
    data = json.loads(texte)
    print(f"\n✓ JSON valide — {len(data)} objets")
except json.JSONDecodeError as e:
    print(f"\n✗ Erreur : {e}")

    # Afficher le contexte autour de l'erreur
    pos    = e.pos
    debut  = max(0, pos - 150)
    fin    = min(len(texte), pos + 150)
    extrait = texte[debut:fin]
    marker  = ' ' * (pos - debut) + '^^^'
    print(f"\nContexte autour de l'erreur :")
    print(extrait)
    print(marker)

    # ── Tentative 2 : extraire objets JSON valides un par un ──────
    print("\nTentative extraction objet par objet...")
    objets = []
    pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', re.DOTALL)

    for match in pattern.finditer(texte):
        try:
            obj = json.loads(match.group())
            if "question" in obj:
                objets.append(obj)
        except json.JSONDecodeError:
            pass

    print(f"Objets valides récupérés : {len(objets)}")

    if objets:
        # Sauvegarder ce qui a pu être récupéré
        sortie = Path("data/datasets/benchmark/reponse_llm_corrigee.json")
        sortie.write_text(
            json.dumps(objets, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"✓ Sauvegardé dans {sortie.name}")

        # Afficher aperçu
        for o in objets[:3]:
            print(f"\n  [{o.get('type','?')}] {o['question'][:80]}...")
    else:
        print("❌ Aucun objet valide extuvé — le JSON est trop corrompu")
        print("   → Demande au LLM de régénérer en insistant sur le format JSON strict")