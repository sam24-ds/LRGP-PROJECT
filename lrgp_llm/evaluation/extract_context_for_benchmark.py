"""
extract_context_for_benchmark.py
Extrait les chunks pertinents depuis Qdrant et prépare
les contextes + prompts à copier-coller dans le chat LLM.

Usage : python evaluation/extract_context_for_benchmark.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
import qdrant_client

THEMES = [
    {
        "domaine": "transfert_matière",
        "requete": "coefficient de transfert global K_OV résistances en série membrane contacteur",
        "n_questions": 10,
        "types": "CALCUL, FACTUEL",
    },
    {
        "domaine": "contacteurs_fibres",
        "requete": "contacteur fibres creuses flux CO2 absorption désorption hollow fiber membrane",
        "n_questions": 10,
        "types": "CALCUL, FACTUEL",
    },
    {
        "domaine": "CH4_CO2",
        "requete": "séparation CH4 CO2 biogaz perméation gazeuse sélectivité membrane",
        "n_questions": 10,
        "types": "CALCUL, COMPARAISON, FACTUEL",
    },
    {
        "domaine": "perméabilité_matériaux",
        "requete": "perméabilité PDMS PEBA membrane dense Barrer solution diffusion",
        "n_questions": 8,
        "types": "CALCUL, COMPARAISON, FACTUEL",
    },
    {
        "domaine": "modélisation_membranaire",
        "requete": "modèle module membranaire bilan matière flux perméat rétentat stage cut",
        "n_questions": 8,
        "types": "CALCUL, FACTUEL",
    },
    {
        "domaine": "absorption_CO2_amines",
        "requete": "absorption CO2 solvant aminé MEA MDEA contacteur membranaire flux transfert",
        "n_questions": 8,
        "types": "CALCUL, FACTUEL",
    },
]

PROMPT_TEMPLATE = """Tu es un expert en génie des procédés et séparation membranaire au LRGP Nancy.
À partir du contexte scientifique ci-dessous extrait du corpus du laboratoire,
génère exactement {n} questions de benchmark de qualité scientifique.

Contraintes strictes :
- Baser les questions UNIQUEMENT sur les informations présentes dans le contexte
- Types demandés : {types}
- Pour les CALCUL : inclure toutes les données numériques nécessaires dans la question
- Pour les FACTUEL : question précise avec réponse vérifiable dans le contexte
- Priorité aux niveaux N2 et N3, quelques N4
- Réponses complètes avec valeurs numériques et unités

Réponds UNIQUEMENT avec un tableau JSON valide sans texte avant ni après :
[
  {{
    "question": "...",
    "answer": "...",
    "type": "CALCUL|FACTUEL|COMPARAISON",
    "domaine": "{domaine}",
    "difficulté": "N1|N2|N3|N4"
  }}
]

Contexte scientifique :
{contexte}"""


def recuperer_contexte(requete, model, client, top_k=8):
    vec = model.encode(
        [requete], normalize_embeddings=True, convert_to_numpy=True
    )[0].tolist()

    results = client.query_points(
        collection_name="lrgp_corpus",
        query=vec,
        using="dense",
        limit=top_k,
    ).points

    parties = []
    for i, r in enumerate(results, 1):
        source = r.payload.get("source_file", "?")
        text   = r.payload.get("text", "").strip()
        if text:
            parties.append(f"[Doc {i} — Source: {source}]\n{text}")

    return "\n\n---\n\n".join(parties)


def main():
    print("\nChargement BGE-M3...", end=" ", flush=True)
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    print("✓")

    print("Connexion Qdrant...", end=" ", flush=True)
    client = qdrant_client.QdrantClient("localhost", port=6333)
    print("✓\n")

    # Dossier de sortie
    output_dir = Path("data/datasets/benchmark/prompts")
    output_dir.mkdir(parents=True, exist_ok=True)

    for theme in THEMES:
        print(f"Extraction : {theme['domaine']}...", end=" ", flush=True)

        contexte = recuperer_contexte(
            theme["requete"], model, client, top_k=12
        )

        prompt = PROMPT_TEMPLATE.format(
            n=theme["n_questions"],
            types=theme["types"],
            domaine=theme["domaine"],
            contexte=contexte,
        )

        # Sauvegarder le prompt complet dans un fichier texte
        output_file = output_dir / f"prompt_{theme['domaine']}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        print(f"✓ ({len(contexte)} chars contexte) → {output_file.name}")

    print(f"\n✓ {len(THEMES)} prompts sauvegardés dans {output_dir}")
    print("  Copie-colle chaque fichier .txt dans l'interface de chat du LLM.")
    print("  Puis ajoute le JSON retourné dans questions.jsonl\n")


if __name__ == "__main__":
    main()