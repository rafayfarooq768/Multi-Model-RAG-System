from __future__ import annotations

import json
import logging
import os
from urllib import error, request

from .config import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_OPENROUTER_API_BASE,
)
from .schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class RAGLLMEngine:
    """LLM engine for generating contextual answers via OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_LLM_MODEL,
        api_base: str = DEFAULT_OPENROUTER_API_BASE,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_context_chars: int = 7000,
        max_output_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self.max_output_tokens = max_output_tokens

    def is_connected(self) -> bool:
        """Check if OpenRouter credentials are available and the service responds."""
        if not self.api_key:
            return False

        try:
            req = request.Request(
                f"{self.api_base}/models",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="GET",
            )
            with request.urlopen(req, timeout=10) as response:
                body = response.read().decode("utf-8")

            parsed = json.loads(body)
            models = parsed.get("data", [])
            return isinstance(models, list) and len(models) > 0
        except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(f"OpenRouter connection failed: {exc}")
            return False

    def generate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Generate LLM answer from query and context chunks."""
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")

        if not chunks:
            return "I don't have enough information."

        context = self._format_context(chunks)
        prompt = self._build_prompt(query, context)

        try:
            model_answer, finish_reason = self._generate_with_limit(prompt, self.max_output_tokens)

            if finish_reason in {"length", "max_tokens", "MAX_TOKENS"}:
                retry_tokens = max(self.max_output_tokens * 2, 1024)
                retry_tokens = min(retry_tokens, 2048)
                retried_answer, _ = self._generate_with_limit(prompt, retry_tokens)
                if retried_answer:
                    model_answer = retried_answer

            return model_answer or "I don't have enough information."
        except error.HTTPError as exc:
            if exc.code in {429, 503}:
                logger.warning(f"LLM generation unavailable or rate-limited: {exc}")
            else:
                logger.error(f"LLM generation error: {exc}")
            raise
        except Exception as exc:
            logger.error(f"LLM generation error: {exc}")
            raise

    def _generate_with_limit(self, prompt: str, max_output_tokens: int) -> tuple[str, str | None]:
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": self._system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": max_output_tokens,
                "stream": False,
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.api_base}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with request.urlopen(req, timeout=120) as response:
            body = response.read().decode("utf-8")

        parsed = json.loads(body)
        candidates = parsed.get("candidates", [])
        choices = parsed.get("choices", [])
        if not choices:
            return "", None

        choice = choices[0]
        message = choice.get("message", {})
        answer = str(message.get("content", "")).strip()
        finish_reason = choice.get("finish_reason")
        return answer, finish_reason

    def _system_prompt(self) -> str:
        return (
            "You are a question-answering assistant. "
            "Use ONLY the provided context to answer. "
            "If the answer is not in the context, say exactly: \"I don't have enough information.\""
        )

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into context string with size limit."""
        context_parts = []
        total_chars = 0

        for idx, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "N/A")
            doc_type = chunk.metadata.get("type", "document")

            block = f"[Document {idx}: {source} (page {page}, type: {doc_type})]\n{chunk.text}\n"
            if total_chars + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - total_chars
                if remaining > 200:
                    context_parts.append(block[:remaining].rstrip() + "\n...[truncated]")
                break

            context_parts.append(block)
            total_chars += len(block)

        return "\n---\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build strict grounded RAG prompt."""
        return f"""Context:
{context}

Question:
{query}

Answer:"""

    def _format_final_answer(self, answer: str, chunks: list[RetrievedChunk]) -> str:
        """Return final answer plus normalized source list for transparency."""
        source_lines: list[str] = []
        seen: set[str] = set()

        for chunk in chunks:
            source = str(chunk.metadata.get("source", "unknown source"))
            page = chunk.metadata.get("page")
            if page is None:
                label = source
            else:
                label = f"{source} (page {page})"

            if label not in seen:
                seen.add(label)
                source_lines.append(f"- {label}")

        if not answer:
            answer = "I don't have enough information."

        if not source_lines:
            source_lines.append("- none")

        return f"Answer: {answer}\n\nSources:\n" + "\n".join(source_lines)
