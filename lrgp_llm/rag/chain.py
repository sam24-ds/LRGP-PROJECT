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
  LLM (Ollama local via ollama.chat avec think=False)
     ↓
  Réponse avec sources

⚠ MODIFICATION (bug Qwen 3.5 thinking) :
ChatOllama ne supporte pas correctement le paramètre `think=False` nécessaire
pour Qwen 3.5. On utilise donc un wrapper Runnable personnalisé `OllamaThinkRunnable`
qui appelle ollama.chat directement et nettoie les balises <think> en sortie.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
from dataclasses import dataclass, field
from typing import Optional, Iterator, Any

import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.prompt_values import PromptValue

from rag.retriever import LRGPRetriever, RetrievalResult
from rag.prompts import (
    PROMPT_RAG, PROMPT_CALCUL, PROMPT_NO_CONTEXT,
    PROMPT_ROUTER, choisir_prompt, formater_historique
)


# ══════════════════════════════════════════════════════════════════
# WRAPPER LANGCHAIN — utilise ollama.chat avec think=False
# ══════════════════════════════════════════════════════════════════
class OllamaThinkRunnable(Runnable):
    """
    Wrapper Runnable LangChain qui appelle ollama.chat directement.
    Permet de passer le paramètre `think=False` indispensable pour Qwen 3.5,
    Qwen 3, DeepSeek-R1 et autres modèles avec mode reasoning.

    Compatible avec LCEL : prompt | llm | StrOutputParser()
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        num_predict: int = 2048,
        think: bool = False,
        **extra_options: Any,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.num_predict = num_predict
        self.think = think
        self.extra_options = extra_options
        # Client Ollama configuré une fois
        self._client = ollama.Client(host=base_url)

    # ── Helpers internes ──────────────────────────────────────────
    def _normalize_input(self, input_data: Any) -> list[dict]:  
        """Convertit l'input LCEL en liste de messages format Ollama."""
        # PromptValue → liste de messages LangChain
        if isinstance(input_data, PromptValue):
            lc_messages = input_data.to_messages()
        elif isinstance(input_data, str):
            return [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list):
            lc_messages = input_data
        else:
            # Fallback : sérialiser en str
            return [{"role": "user", "content": str(input_data)}]

        # Convertir messages LangChain → format Ollama
        messages = []
        for m in lc_messages:
            mtype = getattr(m, "type", "human")
            if mtype == "system":
                role = "system"
            elif mtype == "ai" or mtype == "assistant":
                role = "assistant"
            else:
                role = "user"
            messages.append({"role": role, "content": m.content})
        return messages

    def _clean_thinking(self, text: str) -> str:
        """Supprime les balises <think>...</think> au cas où elles fuiraient."""
        if not text:
            return ""
        # Nettoie les balises <think> complètes
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Nettoie les balises <think> non fermées (réponse tronquée)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()

    def _build_options(self) -> dict:
        """Construit le dict options pour ollama.chat."""
        options = {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
        }
        options.update(self.extra_options)
        return options

    # ── Interface Runnable (synchrone) ────────────────────────────
    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Appel synchrone — retourne un AIMessage compatible LCEL."""
        messages = self._normalize_input(input)

        response = self._client.chat(
            model=self.model,
            messages=messages,
            think=self.think,
            options=self._build_options(),
        )

        content = self._clean_thinking(response.message.content)
        return AIMessage(content=content)

    # ── Interface Runnable (streaming) ────────────────────────────
    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        """Streaming — yield des AIMessageChunk au fil de l'eau."""
        messages = self._normalize_input(input)

        stream = self._client.chat(
            model=self.model,
            messages=messages,
            think=self.think,
            options=self._build_options(),
            stream=True,
        )

        # On filtre activement les balises <think> en streaming
        inside_thinking = False
        buffer = ""

        for chunk in stream:
            content = chunk.message.content or ""
            if not content:
                continue

            buffer += content

            # Détection d'ouverture <think>
            if "<think>" in buffer and not inside_thinking:
                pre, _, rest = buffer.partition("<think>")
                if pre:
                    yield AIMessageChunk(content=pre)
                buffer = rest
                inside_thinking = True

            # Détection de fermeture </think>
            if inside_thinking and "</think>" in buffer:
                _, _, after = buffer.partition("</think>")
                buffer = after
                inside_thinking = False

            # Émission de tokens hors think
            if not inside_thinking and buffer:
                yield AIMessageChunk(content=buffer)
                buffer = ""

        # Flush du buffer final s'il reste qch hors thinking
        if not inside_thinking and buffer:
            yield AIMessageChunk(content=buffer)


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
    Supporte Ollama (local, via ollama.chat) et API cloud (OpenAI, Anthropic).
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
        num_predict:    int = 2048,
    ):
        self.llm_backend = llm_backend
        self.model_name  = model_name
        self.temperature = temperature
        self.verbose     = verbose
        self.num_predict = num_predict

        # Retriever
        self.retriever = LRGPRetriever(
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank,
            use_reranker=True,
        )

        # LLM principal (générateur)
        self.llm = self._charger_llm(
            llm_backend, model_name, ollama_url, temperature, num_predict
        )

        # Router LLM (déterministe, classification rapide)
        self.router_llm = self._charger_llm(
            llm_backend, model_name, ollama_url,
            temperature=0.0, num_predict=128,  # ↓ tokens suffisants pour un mot
        )

    def _charger_llm(
        self,
        backend:     str,
        model:       str,
        ollama_url:  str,
        temperature: float,
        num_predict: int,
    ):
        """Charge le LLM selon le backend choisi."""
        if backend == "ollama":
            # ✅ NOUVEAU : wrapper qui utilise ollama.chat avec think=False
            return OllamaThinkRunnable(
                model=model,
                base_url=ollama_url,
                temperature=temperature,
                num_predict=num_predict,
                think=False,
            )

        elif backend == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=num_predict,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        elif backend == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=num_predict,
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
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Router fallback : {e}")
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