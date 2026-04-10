from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

from .chunking import split_documents
from .config import (
    DEFAULT_CHROMA_PERSIST_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HYBRID_KEYWORD_WEIGHT,
    DEFAULT_HYBRID_SEMANTIC_WEIGHT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_QUERY_CACHE_SIZE,
    DEFAULT_QUERY_EXPANSION_ENABLED,
    DEFAULT_RETRIEVAL_CANDIDATE_MULTIPLIER,
    MAX_LLM_CONTEXT_CHARS,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_OPENROUTER_API_BASE,
    DEFAULT_TOP_K,
)
from .embeddings import SentenceTransformerEmbeddingFunction
from .ingestion import calculate_file_hash, collect_supported_files, load_documents
from .llm import RAGLLMEngine
from .retrieval import (
    build_grounded_answer,
    expand_query,
    format_sources,
    rerank_hybrid,
)
from .schemas import ChunkRecord, RetrievedChunk
from .vectorstore import ChromaStore

logger = logging.getLogger(__name__)


class LocalRAGPipeline:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        openrouter_api_key: str | None = None,
        persist_directory: str | Path = DEFAULT_CHROMA_PERSIST_DIR,
    ) -> None:
        self.embedder = SentenceTransformerEmbeddingFunction(model_name)
        self.store = ChromaStore(self.embedder, persist_directory=persist_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.query_cache: OrderedDict[tuple[str, int, bool, tuple[str, ...], bool], dict[str, object]] = OrderedDict()
        self.cache_size = DEFAULT_QUERY_CACHE_SIZE
        self.query_metrics: dict[str, float] = {
            "queries": 0,
            "cache_hits": 0,
            "total_latency_ms": 0.0,
            "last_latency_ms": 0.0,
        }
        try:
            self.llm_engine = RAGLLMEngine(
                api_key=openrouter_api_key,
                model_name=DEFAULT_LLM_MODEL,
                api_base=DEFAULT_OPENROUTER_API_BASE,
                temperature=DEFAULT_LLM_TEMPERATURE,
                max_context_chars=MAX_LLM_CONTEXT_CHARS,
                max_output_tokens=DEFAULT_LLM_MAX_TOKENS,
            )
        except Exception as error:
            logger.warning(f"LLM engine initialization failed: {error}")
            self.llm_engine = None

    def ingest_paths(
        self, paths: list[str | Path], skip_duplicates: bool = True
    ) -> int:
        """Ingest files from paths. Skip duplicates if skip_duplicates is True."""
        chunk_count = 0
        actual_paths = []

        for path in paths:
            file_hash = calculate_file_hash(path)
            if skip_duplicates and self.store.is_file_indexed(file_hash):
                logger.info(f"Skipping already indexed file: {path}")
                continue
            actual_paths.append(path)

        if not actual_paths:
            return 0

        documents = load_documents(actual_paths)
        if not documents:
            return 0

        chunk_count = self._ingest_documents(documents, actual_paths)
        if chunk_count > 0:
            # Retrieval answers depend on indexed content; drop stale cached answers.
            self.query_cache.clear()
        return chunk_count

    def ingest_directory(self, folder: str | Path) -> int:
        files = collect_supported_files(folder)
        if not files:
            return 0
        return self.ingest_paths(files)

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        use_llm: bool = True,
        filter_types: list[str] | None = None,
        use_query_expansion: bool = DEFAULT_QUERY_EXPANSION_ENABLED,
    ) -> dict[str, object]:
        started = time.perf_counter()
        effective_top_k = top_k or self.top_k
        if self._is_count_query(query):
            # Counting-style questions often need a broader evidence set.
            effective_top_k = max(effective_top_k, 12)
        allowed_types = (
            {entry.lower() for entry in filter_types}
            if filter_types
            else None
        )
        cache_key = (
            query.strip().lower(),
            effective_top_k,
            use_llm,
            tuple(sorted(allowed_types)) if allowed_types else tuple(),
            use_query_expansion,
        )

        if cache_key in self.query_cache:
            cached = dict(self.query_cache[cache_key])
            cached["cached"] = True
            cached["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            # Mark as recently used.
            self.query_cache.move_to_end(cache_key)
            self._record_query_metrics(cached["latency_ms"], cached=True)
            return cached

        retrieved = self._retrieve_ranked_chunks(
            query=query,
            top_k=effective_top_k,
            allowed_types=allowed_types,
            use_query_expansion=use_query_expansion,
        )

        deterministic = self._answer_known_count_query(query, retrieved)
        llm_failed = False
        if deterministic is not None:
            answer = deterministic
        elif use_llm and self.llm_engine is not None:
            try:
                answer = self.llm_engine.generate_answer(query, retrieved)
            except Exception as error:
                logger.warning(f"LLM failed, using grounded answer: {error}")
                llm_failed = True
                answer = build_grounded_answer(query, retrieved)
        else:
            answer = build_grounded_answer(query, retrieved)

        if llm_failed and retrieved:
            expanded_retrieved = self._retrieve_ranked_chunks(
                query=query,
                top_k=max(effective_top_k, 12),
                allowed_types=allowed_types,
                use_query_expansion=True,
            )
            if expanded_retrieved:
                retrieved = expanded_retrieved
                answer = build_grounded_answer(query, retrieved)

        if (
            use_llm
            and self.llm_engine is not None
            and retrieved
            and str(answer).strip().lower() == "i don't have enough information."
        ):
            # Retry once with broader recall before returning an empty-style response.
            expanded_retrieved = self._retrieve_ranked_chunks(
                query=query,
                top_k=max(effective_top_k, 12),
                allowed_types=allowed_types,
                use_query_expansion=True,
            )
            if expanded_retrieved:
                retrieved = expanded_retrieved
                try:
                    retried_answer = self.llm_engine.generate_answer(query, retrieved)
                except Exception as error:
                    logger.warning(f"LLM retry failed, using grounded answer: {error}")
                    retried_answer = build_grounded_answer(query, retrieved)

                if str(retried_answer).strip().lower() == "i don't have enough information.":
                    answer = build_grounded_answer(query, retrieved)
                else:
                    answer = retried_answer

        sources = format_sources(retrieved)

        result: dict[str, object] = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "chunks": retrieved,
            "cached": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

        self.query_cache[cache_key] = result
        self.query_cache.move_to_end(cache_key)
        while len(self.query_cache) > self.cache_size:
            self.query_cache.popitem(last=False)

        self._record_query_metrics(float(result["latency_ms"]), cached=False)
        return result

    def get_usage_metrics(self) -> dict[str, float]:
        """Return simple app usage metrics for UI display."""
        queries = self.query_metrics["queries"]
        cache_hits = self.query_metrics["cache_hits"]
        average_latency = (
            self.query_metrics["total_latency_ms"] / queries if queries else 0.0
        )
        hit_rate = (cache_hits / queries * 100.0) if queries else 0.0

        return {
            "queries": queries,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(hit_rate, 2),
            "average_latency_ms": round(average_latency, 2),
            "last_latency_ms": self.query_metrics["last_latency_ms"],
        }

    def document_count(self) -> int:
        return self.store.count()

    def list_indexed_files(self) -> dict[str, dict[str, object]]:
        """List all indexed files with metadata."""
        return self.store.list_indexed_files()

    def delete_indexed_file(self, source: str) -> int:
        """Delete a file and all its chunks from the index."""
        deleted_count = self.store.delete_file(source)
        if deleted_count > 0:
            # Index changed; cached answers may now reference removed chunks.
            self.query_cache.clear()
        logger.info(f"Deleted {source}: {deleted_count} chunks removed")
        return deleted_count

    @staticmethod
    def _is_count_query(query: str) -> bool:
        normalized = " ".join(query.lower().split())
        count_starts = (
            "how many",
            "number of",
            "count of",
        )
        return any(normalized.startswith(prefix) for prefix in count_starts)

    @staticmethod
    def _answer_known_count_query(query: str, chunks: list[RetrievedChunk]) -> str | None:
        normalized_query = " ".join(query.lower().split())
        if not chunks or not LocalRAGPipeline._is_count_query(normalized_query):
            return None

        if "raid" in normalized_query and ("model" in normalized_query or "level" in normalized_query):
            combined = "\n".join(chunk.text for chunk in chunks).lower()

            matches = re.findall(r"raid\s*(?:level\s*)?(\d+(?:\+\d+)?)", combined)
            if not matches:
                return None

            normalized_levels: list[str] = []
            seen: set[str] = set()
            for token in matches:
                level = token.strip()
                if level == "1+0":
                    level = "10"
                if level not in seen:
                    seen.add(level)
                    normalized_levels.append(level)

            if not normalized_levels:
                return None

            labels = [f"RAID {level}" for level in sorted(normalized_levels, key=LocalRAGPipeline._raid_sort_key)]
            return f"{len(labels)} RAID models are mentioned: " + ", ".join(labels)

        return None

    @staticmethod
    def _raid_sort_key(level: str) -> tuple[int, int]:
        parts = level.split("+")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            return (int(parts[0]), int(parts[1]))
        if level.isdigit():
            return (int(level), 0)
        return (999, 999)

    def inspect_retrieval(self, query: str, top_k: int | None = None) -> dict[str, object]:
        """Inspect retrieval results with detailed debug info."""
        retrieved = self._retrieve_ranked_chunks(
            query=query,
            top_k=top_k or self.top_k,
            allowed_types=None,
            use_query_expansion=DEFAULT_QUERY_EXPANSION_ENABLED,
        )
        return {
            "query": query,
            "chunk_count": len(retrieved),
            "chunks": [
                {
                    "id": chunk.id,
                    "source": chunk.metadata.get("source", "unknown"),
                    "type": chunk.metadata.get("type", "unknown"),
                    "page": chunk.metadata.get("page", "unknown"),
                    "distance": chunk.distance,
                    "similarity_score": self._distance_to_similarity(chunk.distance),
                    "text_preview": chunk.text[:100] + "..."
                    if len(chunk.text) > 100
                    else chunk.text,
                    "metadata": chunk.metadata,
                }
                for chunk in retrieved
            ],
        }

    def _retrieve_ranked_chunks(
        self,
        query: str,
        top_k: int,
        allowed_types: set[str] | None,
        use_query_expansion: bool,
    ) -> list[RetrievedChunk]:
        candidate_k = max(top_k, top_k * DEFAULT_RETRIEVAL_CANDIDATE_MULTIPLIER)
        variants = expand_query(query) if use_query_expansion else [query]

        best_by_id: dict[str, RetrievedChunk] = {}
        for variant in variants:
            variant_results = self.store.query(variant, top_k=candidate_k)
            for chunk in variant_results:
                previous = best_by_id.get(chunk.id)
                if previous is None:
                    best_by_id[chunk.id] = chunk
                else:
                    prev_distance = previous.distance if previous.distance is not None else float("inf")
                    new_distance = chunk.distance if chunk.distance is not None else float("inf")
                    if new_distance < prev_distance:
                        best_by_id[chunk.id] = chunk

        ranked = rerank_hybrid(
            query=query,
            chunks=list(best_by_id.values()),
            semantic_weight=DEFAULT_HYBRID_SEMANTIC_WEIGHT,
            keyword_weight=DEFAULT_HYBRID_KEYWORD_WEIGHT,
            allowed_types=allowed_types,
        )
        return ranked[:top_k]

    def check_llm_status(self) -> bool:
        """Check whether the local LLM backend is reachable."""
        return self.llm_engine.is_connected() if self.llm_engine is not None else False

    def get_llm_model_name(self) -> str | None:
        """Return active model name used by the LLM engine."""
        return self.llm_engine.model_name if self.llm_engine is not None else None

    @staticmethod
    def _distance_to_similarity(distance: float | None) -> float:
        if distance is None:
            return 0.0
        # For cosine distance in [0, 2], similarity proxy is higher when distance is lower.
        return max(0.0, 1.0 - float(distance))

    def _rank_chunks_by_similarity(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Sort chunks so highest similarity is first."""
        return sorted(
            chunks,
            key=lambda chunk: self._distance_to_similarity(chunk.distance),
            reverse=True,
        )

    def _record_query_metrics(self, latency_ms: float, cached: bool) -> None:
        self.query_metrics["queries"] += 1
        self.query_metrics["total_latency_ms"] += latency_ms
        self.query_metrics["last_latency_ms"] = latency_ms
        if cached:
            self.query_metrics["cache_hits"] += 1

    def _ingest_documents(
        self, documents: list[Document], source_paths: list[str | Path] | None = None
    ) -> int:
        if not documents:
            return 0

        chunks = split_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunk_records: list[ChunkRecord] = []
        counters: dict[tuple[str, object], int] = defaultdict(int)

        for chunk in chunks:
            metadata = dict(chunk.metadata)
            source = str(metadata.get("source", "document"))
            doc_type = str(metadata.get("type", "unknown"))
            page = metadata.get("page", 0)
            counter_key = (source, page)
            counters[counter_key] += 1
            chunk_number = counters[counter_key]
            
            # Ensure consistent metadata
            metadata["chunk_index"] = chunk_number
            metadata["source"] = source
            metadata["type"] = doc_type
            metadata["page"] = page

            chunk_id = self._build_chunk_id(source, page, chunk_number)
            chunk_records.append(
                ChunkRecord(
                    id=chunk_id,
                    text=chunk.page_content.strip(),
                    metadata=metadata,
                )
            )

        self.store.add_chunks(chunk_records)
        chunk_count = len(chunk_records)

        # Update manifest with file hashes
        if source_paths:
            for path in source_paths:
                file_hash = calculate_file_hash(path)
                source_name = Path(path).name
                # Count chunks for this source
                source_chunk_count = sum(
                    1
                    for record in chunk_records
                    if record.metadata.get("source") == source_name
                )
                self.store.mark_file_indexed(source_name, file_hash, source_chunk_count)

        return chunk_count

    @staticmethod
    def _build_chunk_id(source: str, page: object, chunk_number: int) -> str:
        safe_source = "".join(
            character if character.isalnum() or character in {".", "-", "_"} else "_"
            for character in source
        )
        page_label = str(page)
        return f"{safe_source}::p{page_label}::c{chunk_number}"
