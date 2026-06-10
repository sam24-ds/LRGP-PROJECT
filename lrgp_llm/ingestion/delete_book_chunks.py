"""
delete_book_chunks.py
Supprime les chunks d'un livre spécifique de Qdrant.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

COLLECTION = "lrgp_corpus"

# Le nom exact de la source (à confirmer après ÉTAPE 2)
SOURCE_A_SUPPRIMER = "Model of Vapor-Liquid Equilibria for Aqueous Acid Gas-Alkanolamine Systems. 2. Representation of H2S and C02 Solubility in Aqueous MDEA and C02 Solubility in Aqueous Mixtures of MDEA.pdf"


client = QdrantClient(host="localhost", port=6333)

# 1. État avant suppression
info = client.get_collection(COLLECTION)
total_avant = info.points_count
print(f"\nÉtat actuel Qdrant : {total_avant:,} points")

# 2. Compter les chunks du livre
result = client.count(
    collection_name=COLLECTION,
    count_filter=Filter(
        must=[FieldCondition(key="source", match=MatchValue(value=SOURCE_A_SUPPRIMER))]
    ),
    exact=True,
)
n_a_supprimer = result.count
print(f"Chunks à supprimer pour ce livre : {n_a_supprimer}")

if n_a_supprimer == 0:
    print("\n⚠ Aucun chunk trouvé avec ce nom. Vérifier le nom exact dans le JSONL.")
    exit(0)

# 3. Confirmer
confirm = input(f"\n>>> Supprimer ces {n_a_supprimer} chunks ? (oui/non) : ")
if confirm.lower() not in ("oui", "o", "yes", "y"):
    print("✗ Annulé")
    exit(0)

# 4. Supprimer
client.delete(
    collection_name=COLLECTION,
    points_selector=Filter(
        must=[FieldCondition(key="source", match=MatchValue(value=SOURCE_A_SUPPRIMER))]
    )
)
print(f"✓ {n_a_supprimer} chunks supprimés")

# 5. Vérifier
info = client.get_collection(COLLECTION)
total_apres = info.points_count
print(f"\nÉtat après suppression : {total_apres:,} points")
print(f"Différence : {total_avant - total_apres:,} chunks supprimés")