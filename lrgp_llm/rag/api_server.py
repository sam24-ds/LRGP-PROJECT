"""
api_server.py
Expose chain.py (V4 simple) et SRAR-GP (architecture multi-agents)
comme API OpenAI-compatible pour Open WebUI.

Usage : python rag/api_server.py

Open WebUI → Settings → Connections → OpenAI API
  URL : http://localhost:8000/v1
  Key : lrgp-key (n'importe quoi)

Modèles exposés :
  - "lrgp-rag"        → V4 fine-tuné + RAG (rapide, ~12s)
  - "srar-gp"         → Architecture multi-agents complète (5-180s selon voie)
  - "srar-gp-verbose" → SRAR-GP avec parcours détaillé des agents
"""

import json
import time
import uuid
from pathlib import Path
from typing import Iterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.chain import LRGPChain
from srar_gp.main import ask_srar

app = FastAPI(title="LRGP RAG + SRAR-GP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# INITIALISATION
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Démarrage de l'API LRGP")
print("=" * 60)

# Chaîne RAG simple (V4) — pour le modèle "lrgp-rag"
print("[1/2] Initialisation LRGPChain (V4 simple)...", end=" ", flush=True)
chain = LRGPChain(
    llm_backend    = "ollama",
    model_name     = "lrgp-knowledge_v5",
    ollama_url     = "http://localhost:11434",
    top_k_retrieve = 20,
    top_k_rerank   = 5,
    temperature    = 0.1,
    verbose        = False,
)
print("✓")

# SRAR-GP : pré-chargement du graphe pour éviter latence au premier appel
print("[2/2] Pré-chargement du graphe SRAR-GP...", end=" ", flush=True)
try:
    from srar_gp.graph import get_graph
    _ = get_graph()
    print("✓")
except Exception as e:
    print(f"⚠ {e}")

print("=" * 60)
print("API prête sur http://0.0.0.0:8000/v1")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# MODÈLES PYDANTIC
# ══════════════════════════════════════════════════════════════
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "srar-gp"
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.1


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/v1/models")
def list_models():
    """Liste les modèles disponibles."""
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id":       "lrgp-rag",
                "object":   "model",
                "created":  created,
                "owned_by": "LRGP Nancy",
            },
            {
                "id":       "srar-gp",
                "object":   "model",
                "created":  created,
                "owned_by": "LRGP Nancy",
            },
            {
                "id":       "srar-gp-verbose",
                "object":   "model",
                "created":  created,
                "owned_by": "LRGP Nancy",
            },
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    # ── Extraire la dernière question utilisateur ──
    question = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            question = msg.content
            break

    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

    # ── NOUVEAU : Détecter les requêtes système Open WebUI ──
    if _est_requete_systeme_openwebui(question):
        print(f"\n[API] → Requête système Open WebUI détectée — bypass SRAR-GP")
        return _handle_requete_systeme(question, request.model, request.stream)

    # ── Aiguillage normal ──
    model_id = request.model.lower()
    
    print(f"\n{'='*60}")
    print(f"[API] Modèle : {model_id}")
    print(f"[API] Question : {question[:120]}")
    print(f"{'='*60}")
    
    if "srar" in model_id or "agentic" in model_id:
        return _handle_srar_gp(
            question, model_id, 
            stream=request.stream,
            verbose="verbose" in model_id
        )
    else:
        return _handle_rag_simple(question, model_id, stream=request.stream)


# ══════════════════════════════════════════════════════════════
# NOUVEAU : Détection et handling des requêtes système Open WebUI
# ══════════════════════════════════════════════════════════════
def _est_requete_systeme_openwebui(question: str) -> bool:
    """Détecte les requêtes parasites envoyées par Open WebUI.
    
    Open WebUI envoie automatiquement après chaque réponse :
    - Génération de follow-ups
    - Génération de titre
    - Génération de tags
    """
    debut = question[:200].lower()
    SIGNATURES = [
        "### task:",
        "suggest 3-5 relevant follow-up",
        "generate a concise, 3-5 word title",
        "generate 1-3 broad tags",
        "you are an assistant tasked with",
        "json schema",
    ]
    return any(sig in debut for sig in SIGNATURES)


def _handle_requete_systeme(question: str, model_id: str, stream: bool):
    """Traite directement les requêtes système avec Qwen 27B (rapide)."""
    import ollama
    
    try:
        response = ollama.chat(
            model="qwen3.5:27b",
            messages=[{"role": "user", "content": question}],
            think=False,
            format="json" if "json" in question.lower() else None,
            options={"temperature": 0.1, "num_predict": 500},
            keep_alive="5m",
        )
        contenu = response.message.content.strip()
        
        # Nettoyer les balises think
        import re
        contenu = re.sub(r"<think>.*?</think>", "", contenu, flags=re.DOTALL).strip()
        
        print(f"[API] → Requête système traitée ({len(contenu)} chars)")
        
        return _format_response(contenu, model_id, question, stream=stream)
    
    except Exception as e:
        print(f"[API] ✗ Erreur requête système : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# HANDLER : V4+RAG SIMPLE (modèle "lrgp-rag")
# ══════════════════════════════════════════════════════════════
def _handle_rag_simple(question: str, model_id: str, stream: bool):
    """Route V4+RAG classique."""
    try:
        response = chain.ask(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources_text = ""
    if response.sources:
        sources_uniques = list(dict.fromkeys(
            s.source[:50] for s in response.sources[:3]
        ))
        sources_text = "\n\n---\n📚 Sources : " + " · ".join(sources_uniques)

    contenu = response.answer + sources_text
    
    return _format_response(contenu, model_id, question, stream=stream)


# ══════════════════════════════════════════════════════════════
# HANDLER : SRAR-GP ARCHITECTURE MULTI-AGENTS
# ══════════════════════════════════════════════════════════════
def _handle_srar_gp(question: str, model_id: str, stream: bool, verbose: bool):
    """Route SRAR-GP — architecture multi-agents complète."""
    try:
        result = ask_srar(question)
    except Exception as e:
        print(f"[API] ✗ Erreur SRAR-GP : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur SRAR-GP : {e}")
    
    contenu = result.get("reponse_finale", "")
    parcours = result.get("agents_actives", [])
    voie = result.get("voie", "?")
    
    print(f"[API] → Voie : {voie}")
    print(f"[API] → Parcours : {' → '.join(parcours)}")
    print(f"[API] → Réponse : {len(contenu)} chars")
    
    # ── Mode verbose : ajouter le parcours en footer ──
    if verbose:
        parcours_text = "\n\n---\n🔍 **Parcours SRAR-GP** :\n"
        parcours_text += f"- **Voie** : {voie}\n"
        parcours_text += f"- **Agents traversés** : {' → '.join(parcours)}\n"
        
        if result.get("tentatives_renegociation", 0) > 0:
            parcours_text += f"- **Re-négociations** : {result['tentatives_renegociation']}\n"
        
        if result.get("sources_web"):
            parcours_text += f"- **Sources web utilisées** : {len(result['sources_web'])}\n"
        
        contenu += parcours_text
    
    return _format_response(contenu, model_id, question, stream=stream)


# ══════════════════════════════════════════════════════════════
# FORMATTER : RÉPONSE OPENAI COMPATIBLE
# ══════════════════════════════════════════════════════════════
def _format_response(contenu: str, model_id: str, question: str, stream: bool):
    """Formate la réponse au format OpenAI (streaming ou non)."""
    
    if stream:
        def generate() -> Iterator[str]:
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            
            # Chunk initial avec le rôle
            initial_chunk = {
                "id":      chunk_id,
                "object":  "chat.completion.chunk",
                "created": int(time.time()),
                "model":   model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }]
            }
            yield f"data: {json.dumps(initial_chunk)}\n\n"
            
            # Envoyer le contenu en chunks de mots
            mots = contenu.split(" ")
            for i, mot in enumerate(mots):
                chunk = {
                    "id":      chunk_id,
                    "object":  "chat.completion.chunk",
                    "created": int(time.time()),
                    "model":   model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": mot + (" " if i < len(mots)-1 else "")},
                        "finish_reason": None,
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Chunk final
            final_chunk = {
                "id":      chunk_id,
                "object":  "chat.completion.chunk",
                "created": int(time.time()),
                "model":   model_id,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    # Réponse non-streamée
    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model_id,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": contenu},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     len(question.split()),
            "completion_tokens": len(contenu.split()),
            "total_tokens":      len(question.split()) + len(contenu.split()),
        }
    }


# ══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": ["lrgp-rag", "srar-gp", "srar-gp-verbose"],
        "rag": True,
        "srar_gp": True,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=300,  # ← critique pour la voie CALCUL (jusqu'à 3 min)
    )