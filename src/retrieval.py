from __future__ import annotations

import re
from collections.abc import Sequence

from .schemas import RetrievedChunk


def build_grounded_answer(query: str, chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return f"No indexed content matched the question: {query}"

    # Prefer informative chunks over title-only headings.
    substantive = [chunk for chunk in chunks if len(" ".join(chunk.text.split())) >= 90]
    selected: list[RetrievedChunk] = list(substantive[:3])

    # Backfill with next-best chunks to provide context when substantive chunks are sparse.
    if len(selected) < 2:
        for chunk in chunks:
            if chunk in selected:
                continue
            selected.append(chunk)
            if len(selected) >= 2:
                break

    if not selected:
        selected = list(chunks[:1])

    combined_sections: list[str] = []
    for idx, chunk in enumerate(selected, start=1):
        source_label = _format_source_label(chunk.metadata)
        snippet = _trim_text(chunk.text, 420)
        combined_sections.append(f"Passage {idx} ({source_label}):\n{snippet}")

    return "Based on indexed passages, here are the most relevant details:\n\n" + "\n\n".join(combined_sections)


def format_sources(chunks: Sequence[RetrievedChunk]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        source_label = _format_source_label(chunk.metadata)
        snippet = _trim_text(chunk.text, 260)
        sources.append(f"{source_label}: {snippet}")
    return sources


def expand_query(query: str) -> list[str]:
    """Generate light query variants to improve candidate recall."""
    base = " ".join(query.lower().split())
    if not base:
        return [query]

    variants = [base]

    # Remove common question openers for a focused variant.
    stripped = base
    opener_pattern = re.compile(
        r"^(what|who|where|when|why|how|can|could|would|is|are|do|does|did)\s+"
    )
    while True:
        updated = opener_pattern.sub("", stripped).strip()
        if updated == stripped:
            break
        stripped = updated
    if stripped and stripped != base:
        variants.append(stripped)
        stripped_clean = re.sub(r"[?!.]+$", "", stripped).strip()
        if stripped_clean and stripped_clean not in variants:
            variants.append(stripped_clean)

    # Keep a keyword-only variant for mixed lexical recall.
    terms = [term for term in _tokenize(base) if len(term) > 2]
    if terms:
        keyword_variant = " ".join(terms[:8])
        if keyword_variant not in variants:
            variants.append(keyword_variant)

    return variants


def rerank_hybrid(
    query: str,
    chunks: Sequence[RetrievedChunk],
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    allowed_types: set[str] | None = None,
) -> list[RetrievedChunk]:
    """Rank by blended semantic distance and lexical overlap."""
    query_terms = set(_tokenize(query))
    ranked_items: list[tuple[float, RetrievedChunk]] = []

    for chunk in chunks:
        chunk_type = str(chunk.metadata.get("type", "")).lower()
        if allowed_types is not None and chunk_type not in allowed_types:
            continue

        semantic_score = _distance_to_similarity(chunk.distance)
        lexical_score = _keyword_overlap_score(query_terms, chunk.text)
        combined = (semantic_weight * semantic_score) + (keyword_weight * lexical_score)
        ranked_items.append((combined, chunk))

    ranked_items.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked_items]


def _format_source_label(metadata: dict[str, object]) -> str:
    source = str(metadata.get("source", "unknown source"))
    page = metadata.get("page")
    if page is None:
        return source
    return f"{source}, page {page}"


def _trim_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _distance_to_similarity(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - float(distance))


def _keyword_overlap_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0

    text_terms = set(_tokenize(text))
    if not text_terms:
        return 0.0

    overlap = query_terms.intersection(text_terms)
    return len(overlap) / len(query_terms)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
