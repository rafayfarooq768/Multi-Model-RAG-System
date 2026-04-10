from __future__ import annotations

from src.llm import RAGLLMEngine
from src.retrieval import expand_query, rerank_hybrid
from src.schemas import RetrievedChunk


def test_expand_query_adds_shorter_variants() -> None:
    variants = expand_query("What is this document about?")

    assert "what is this document about?" in variants
    assert "this document about?" in variants or "this document about" in variants
    assert any("document" in variant for variant in variants)


def test_rerank_hybrid_prefers_keyword_overlap_when_semantic_is_close() -> None:
    query = "invoice total due"
    chunks = [
        RetrievedChunk(
            id="1",
            text="This passage discusses project notes and planning.",
            metadata={"source": "a.pdf", "type": "pdf", "page": 1},
            distance=0.20,
        ),
        RetrievedChunk(
            id="2",
            text="Invoice total due date and payment details are listed here.",
            metadata={"source": "b.pdf", "type": "pdf", "page": 2},
            distance=0.25,
        ),
    ]

    ranked = rerank_hybrid(query, chunks)

    assert ranked[0].id == "2"


def test_prompt_is_strict_and_grounded() -> None:
    engine = RAGLLMEngine()
    prompt = engine._build_prompt("What is the summary?", "Context text here")

    assert "Use ONLY the provided context" in prompt
    assert "I don't have enough information." in prompt
    assert "Question:" in prompt
    assert "Context:" in prompt


def test_final_answer_formats_sources() -> None:
    engine = RAGLLMEngine()
    chunks = [
        RetrievedChunk(
            id="1",
            text="Alpha text",
            metadata={"source": "file.pdf", "type": "pdf", "page": 2},
        ),
        RetrievedChunk(
            id="2",
            text="Beta text",
            metadata={"source": "image.png", "type": "image", "page": 1},
        ),
    ]

    formatted = engine._format_final_answer("It is about alpha.", chunks)

    assert formatted.startswith("Answer: It is about alpha.")
    assert "Sources:" in formatted
    assert "- file.pdf (page 2)" in formatted
    assert "- image.png (page 1)" in formatted
