"""
librarian.py
Librarian_Agent — Extraction documentaire + génération (refactoré pour Boucle 1).

Architecture en 2 étapes pour la voie DOCUMENTAIRE :
  1. extraire_documentaire : recherche RAG seule (pas de génération)
  2. generer_reponse_documentaire : génération finale avec RAG + Web
"""
import sys
import re
import ollama
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag.chain import LRGPChain
from srar_gp.state import SRARState

# Initialisation unique
print("  [Librarian] Initialisation LRGPChain...")
_chain = LRGPChain(
    llm_backend    = "ollama",
    model_name     = "lrgp-knowledge_v5",
    ollama_url     = "http://localhost:11434",
    top_k_retrieve = 20,
    top_k_rerank   = 5,
    temperature    = 0.1,
    verbose        = False,
)
print("  [Librarian] ✓ prêt")


GEN_MODEL = "lrgp-knowledge_v5"


# ══════════════════════════════════════════════════════════════
# VOIE DOCUMENTAIRE — Phase 1 : extraction RAG
# ══════════════════════════════════════════════════════════════
def extraire_documentaire(state: SRARState) -> SRARState:
    """Voie DOCUMENTAIRE — Phase 1 : RAG uniquement."""
    print(f"\n  ┌─ [LIBRARIAN-DOC-EXTRACT] Extraction RAG documentaire...")
    
    try:
        sources = []
        if hasattr(_chain, "retriever"):
            if hasattr(_chain.retriever, "retrieve"):
                sources = _chain.retriever.retrieve(state["question"])
            elif hasattr(_chain.retriever, "get_relevant_documents"):
                sources = _chain.retriever.get_relevant_documents(state["question"])
        
        # Fallback si retriever non accessible
        if not sources:
            print(f"  │  → Fallback via .ask()")
            response = _chain.ask(state["question"])
            sources_data = response.sources if hasattr(response, 'sources') else []
            contexte = "\n\n".join(
                f"[Source: {s.source[:60]}]" for s in sources_data[:5]
            )
            print(f"  │  → {len(sources_data)} sources via ask()")
            return {
                "sources_rag": [{"source": s.source, "score": getattr(s, 'score', 0)} for s in sources_data],
                "context_rag": contexte,
                "document_pertinent": False,
                "agents_actives": ["librarian_doc_extract"],
            }
        
        # Construire le contexte avec les vrais textes
        contexte = "\n\n".join(
            f"[Source: {getattr(s, 'source', '?')[:60]}]\n{getattr(s, 'text', '')[:600]}"
            for s in sources[:5]
        )
        print(f"  │  → {len(sources)} sources extraites")
        
        return {
            "sources_rag": [
                {"source": getattr(s, 'source', '?'), "score": getattr(s, 'score', 0)}
                for s in sources
            ],
            "context_rag": contexte,
            "document_pertinent": False,
            "agents_actives": ["librarian_doc_extract"],
        }
    except Exception as e:
        print(f"  │  ⚠ Erreur extraction : {str(e)[:100]}")
        return {
            "sources_rag": [],
            "context_rag": "",
            "document_pertinent": False,
            "agents_actives": ["librarian_doc_extract_failed"],
        }


# ══════════════════════════════════════════════════════════════
# VOIE DOCUMENTAIRE — Phase 2 : génération
# ══════════════════════════════════════════════════════════════
def generer_reponse_documentaire(state: SRARState) -> SRARState:
    """Voie DOCUMENTAIRE — Phase 2 : génération avec consignes d'expert senior."""
    print(f"\n  ┌─ [LIBRARIAN-DOC-GEN] Génération de la réponse...")
    
    contexte = state.get("context_rag", "")
    sources_rag = state.get("sources_rag", [])
    sources_web = state.get("sources_web", [])
    
    # Cas 1 — Aucun contexte
    if not contexte.strip():
        print(f"  │  → Aucun contexte — fallback connaissances générales")
        prompt = f"""Tu es un expert LRGP en génie des procédés.

Le corpus LRGP ne contient aucune information sur cette question, et la
recherche web n'a rien trouvé non plus.

Réponds depuis tes connaissances générales en précisant cette limitation
au début de la réponse.

Question : {state["question"]}

Réponse :"""
        
        try:
            response = ollama.chat(
                model=GEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 2000},
            )
            contenu = response.message.content
            contenu = re.sub(r"<think>.*?</think>", "", contenu, flags=re.DOTALL).strip()
            return {
                "reponse_finale": contenu,
                "agents_actives": ["librarian_doc_fallback"],
            }
        except Exception as e:
            return {
                "reponse_finale": f"⚠ Erreur : {e}",
                "agents_actives": ["librarian_doc_failed"],
            }
    
    # Cas 2 — Génération avec contexte : PROMPT EXPERT SENIOR
    a_du_web = len(sources_web) > 0
    
    prompt = f"""Tu es un EXPERT SENIOR du LRGP en génie des procédés membranaires.
Ton interlocuteur est un ingénieur R&D — tu réponds à son niveau.

═══════════════════════════════════════════════════════════════
SOURCES DOCUMENTAIRES :
{contexte[:4500]}
═══════════════════════════════════════════════════════════════

Question : {state["question"]}

═══════════════════════════════════════════════════════════════
EXIGENCES SCIENTIFIQUES OBLIGATOIRES
═══════════════════════════════════════════════════════════════

1. VOCABULAIRE TECHNIQUE PRÉCIS
   Utilise les termes scientifiques exacts, JAMAIS les raccourcis :
   ✓ "mécanisme solution-diffusion" (PAS "transport diffusif")
   ✓ "perméabilité = solubilité × diffusivité"
   ✓ "plus perméable" (PAS "gaz rapide" — explique pourquoi)
   ✓ Préciser les matériaux : polyimides, polysulfones, PDMS, PEBA,
     CMS (Carbon Molecular Sieves), zéolithes, perovskites, MOFs...

2. ESPRIT CRITIQUE SYSTÉMATIQUE — OBLIGATOIRE
   Pour chaque procédé ou matériau mentionné, INCLUS :
   - Sa LIMITE principale (sélectivité, perméabilité, stabilité)
   - Les DIFFICULTÉS spécifiques au cas considéré
   - L'analyse comparative avec d'autres approches
   - Pourquoi le problème est techniquement DÉFIANT ou non
   
   ⚠ Si une séparation est difficile (faible sélectivité, ex: N2/CH4),
   DIS-LE EXPLICITEMENT. Ne masque pas les limites.

3. STRUCTURE DE RÉPONSE COMPLÈTE
   a. Définition du procédé (avec terminologie précise)
   b. Mécanisme physique (avec termes corrects)
   c. Matériaux applicables (au moins 3 familles distinctes)
   d. LIMITES ET DÉFIS (analyse critique — section OBLIGATOIRE)
   e. Conclusion synthétique avec recommandation pratique

4. CITATIONS
   - Cite [Source: ...] pour chaque information clé
   - Distingue sources LRGP et sources web complémentaires

═══════════════════════════════════════════════════════════════
NIVEAU ATTENDU
═══════════════════════════════════════════════════════════════
Réponse de niveau INGÉNIEUR R&D SENIOR.
Pas d'étudiant L2 qui décrit superficiellement.
Si une difficulté technique existe, NE LA MASQUE PAS — analyse-la.

{"Sources web disponibles — utilise-les pour enrichir et préciser." if a_du_web else ""}

Réponse experte (4-6 paragraphes) :"""
    
    try:
        response = ollama.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.3,
                "num_predict": 2200,  # ↑ pour permettre la profondeur
                "frequency_penalty": 1.15,  # éviter répétitions
            },
        )
        contenu = response.message.content
        contenu = re.sub(r"<think>.*?</think>", "", contenu, flags=re.DOTALL).strip()
        
        # Footer sources
        sources_text = ""
        if sources_rag:
            sources_text += "\n\n📚 **Sources LRGP :**\n"
            for s in sources_rag[:3]:
                sources_text += f"- {s.get('source', '?')[:80]}\n"
        
        if sources_web:
            sources_text += "\n\n🌐 **Sources web complémentaires :**\n"
            for s in sources_web[:3]:
                sources_text += f"- [{s.get('source', 'Web')}]({s.get('url', '')})\n"
        
        print(f"  │  → Réponse générée ({len(contenu)} chars)")
        return {
            "reponse_finale": contenu + sources_text,
            "agents_actives": ["librarian_doc_gen"],
        }
    
    except Exception as e:
        print(f"  │  ⚠ Erreur génération : {str(e)[:100]}")
        return {
            "reponse_finale": f"⚠ Erreur génération : {e}",
            "agents_actives": ["librarian_doc_gen_failed"],
        }

# ══════════════════════════════════════════════════════════════
# Ancienne fonction conservée pour rétrocompatibilité (deprecated)
# ══════════════════════════════════════════════════════════════
def rechercher_et_repondre(state: SRARState) -> SRARState:
    """DEPRECATED — utilise extraire_documentaire + generer_reponse_documentaire."""
    print(f"  ⚠ [LIBRARIAN] Méthode dépréciée — utilise nouvelle architecture en 2 phases")
    state.update(extraire_documentaire(state))
    return generer_reponse_documentaire(state)