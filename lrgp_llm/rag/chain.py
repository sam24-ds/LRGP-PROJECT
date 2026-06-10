"""
chain.py
Pipeline RAG complet avec LangChain LCEL.

Flux :
  Question
     ↓
  Router (classifier le type)
     ↓
  Retriever (hybrid search + reranking)
     ↓
  Prompt (adapté au type)
     ↓
  LLM (Ollama local ou API cloud)
     ↓
  Réponse avec sources
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from rag.retriever import LRGPRetriever, RetrievalResult
from rag.prompts import (
    PROMPT_RAG, PROMPT_CALCUL, PROMPT_NO_CONTEXT,
    PROMPT_ROUTER, choisir_prompt, formater_historique
)


# ══════════════════════════════════════════════════════════════════
# DATACLASS — Réponse complète
# ══════════════════════════════════════════════════════════════════
@dataclass
class RAGResponse:
    question:      str
    answer:        str
    sources:       list[RetrievalResult]
    question_type: str
    model_used:    str
    context_chars: int
    history:       list = field(default_factory=list)

    def afficher(self) -> None:
        """Affiche la réponse formatée dans le terminal."""
        print(f"\n{'═'*65}")
        print(f"  Question [{self.question_type}] : {self.question}")
        print(f"  Modèle : {self.model_used}")
        print(f"{'─'*65}")
        print(f"\n{self.answer}\n")
        print(f"{'─'*65}")
        print(f"  Sources utilisées ({len(self.sources)}) :")
        for i, s in enumerate(self.sources, 1):
            print(f"    [{i}] {s.source[:55]}  "
                  f"(rerank: {s.rerank_score:.3f})")
        print(f"{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════
# CHAIN PRINCIPALE
# ══════════════════════════════════════════════════════════════════
class LRGPChain:
    """
    Chaîne RAG complète pour l'assistant LRGP.
    Supporte Ollama (local) et API cloud (OpenAI, Anthropic).
    """

    def __init__(
        self,
        llm_backend:    str = "ollama",
        model_name:     str = "qwen3.5:9b",
        ollama_url:     str = "http://localhost:11434",
        top_k_retrieve: int = 20,
        top_k_rerank:   int = 5,
        temperature:    float = 0.1,
        verbose:        bool = False,
        num_predict :   int = 2048
    ):
        self.llm_backend  = llm_backend
        self.model_name   = model_name
        self.temperature  = temperature
        self.verbose      = verbose
        self.num_predict  = num_predict

        # Retriever
        self.retriever = LRGPRetriever(
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            use_reranker=True,
        )

        # LLM
        self.llm = self._charger_llm(
            llm_backend, model_name, ollama_url, temperature,num_predict
        )

        # Router LLM (petit modèle rapide pour classifier)
        self.router_llm = self._charger_llm(
            llm_backend, model_name, ollama_url, temperature=0.0,num_predict =2048
        )

    def _charger_llm(
        self,
        backend:     str,
        model:       str,
        ollama_url:  str,
        temperature: float,
        num_predict: int
    ):
        """Charge le LLM selon le backend choisi."""
        if backend == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model,
                base_url=ollama_url,
                temperature=temperature,
                num_predict =num_predict,
                
            )

        elif backend == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                num_predict=num_predict,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        elif backend == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                num_predict=num_predict,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

        else:
            raise ValueError(f"Backend inconnu : {backend}")

    # ── Classifier le type de question ───────────────────────────
    def _classifier_question(self, question: str) -> str:
        """Retourne le type de question : CALCUL, FACTUEL, etc."""
        try:
            chain = PROMPT_ROUTER | self.router_llm | StrOutputParser()
            result = chain.invoke({"question": question})
            type_q = result.strip().upper()
            valides = {"CALCUL", "FACTUEL", "COMPARAISON",
                       "PROCEDURE", "GENERAL"}
            return type_q if type_q in valides else "FACTUEL"
        except Exception:
            return "FACTUEL"

    # ── Interface principale ──────────────────────────────────────
    def ask(
        self,
        question:      str,
        history:       Optional[list] = None,
        filter_source: Optional[str]  = None,
    ) -> RAGResponse:
        """
        Pose une question et retourne une réponse avec sources.

        Args:
            question      : question en langage naturel
            history       : historique [{"role": ..., "content": ...}]
            filter_source : filtrer le retrieval par source
        """
        history = history or []

        # 1. Classifier la question
        if self.verbose:
            print(f"  Classification...", end=" ", flush=True)
        question_type = self._classifier_question(question)
        if self.verbose:
            print(f"→ {question_type}")

        # 2. Retrieval
        if self.verbose:
            print(f"  Retrieval...", end=" ", flush=True)
        sources = self.retriever.retrieve(question, filter_source)
        if self.verbose:
            print(f"→ {len(sources)} chunks")

        # 3. Choisir le prompt
        if not sources:
            prompt   = PROMPT_NO_CONTEXT
            context  = ""
        else:
            prompt   = choisir_prompt(question_type)
            context  = self.retriever.format_context(sources)

        # 4. Construire les inputs
        inputs = {
            "question":      question,
            "context":       context,
            "context_chars": len(context),
        }
        if history:
            inputs["history"] = formater_historique(history)

        # 5. Génération
        if self.verbose:
            print(f"  Génération LLM ({self.model_name})...",
                  end=" ", flush=True)

        chain  = prompt | self.llm | StrOutputParser()
        answer = chain.invoke(inputs)

        if self.verbose:
            print(f"✓ ({len(answer)} chars)")

        return RAGResponse(
            question      = question,
            answer        = answer,
            sources       = sources,
            question_type = question_type,
            model_used    = f"{self.llm_backend}/{self.model_name}",
            context_chars = len(context),
            history       = history,
        )

    def ask_stream(
        self,
        question:      str,
        history:       Optional[list] = None,
        filter_source: Optional[str]  = None,
    ):
        """
        Version streaming — génère la réponse token par token.
        Yields des strings.
        """
        history  = history or []
        sources  = self.retriever.retrieve(question, filter_source)
        q_type   = self._classifier_question(question)
        prompt   = choisir_prompt(q_type) if sources else PROMPT_NO_CONTEXT
        context  = self.retriever.format_context(sources) if sources else ""

        inputs = {"question": question, "context": context}
        if history:
            inputs["history"] = formater_historique(history)

        chain = prompt | self.llm | StrOutputParser()
        for chunk in chain.stream(inputs):
            yield chunk