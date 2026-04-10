from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import chromadb

from .config import DEFAULT_CHROMA_PERSIST_DIR, DEFAULT_COLLECTION_NAME, DEFAULT_TOP_K
from .embeddings import SentenceTransformerEmbeddingFunction
from .schemas import ChunkRecord, RetrievedChunk


class ChromaStore:
    def __init__(
        self,
        embedding_function: SentenceTransformerEmbeddingFunction,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str | Path = DEFAULT_CHROMA_PERSIST_DIR,
    ) -> None:
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self.manifest_path = persist_path / "manifest.json"
        self._load_manifest()

    def add_chunks(self, chunks: Sequence[ChunkRecord]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        self.collection.upsert(
            ids=[chunk.id for chunk in chunk_list],
            documents=[chunk.text for chunk in chunk_list],
            metadatas=[chunk.metadata for chunk in chunk_list],
        )
        return len(chunk_list)

    def query(self, query_text: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        result = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            retrieved.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=documents[index] if index < len(documents) else "",
                    metadata=metadatas[index] if index < len(metadatas) else {},
                    distance=distances[index] if index < len(distances) else None,
                )
            )

        return retrieved

    def count(self) -> int:
        return self.collection.count()

    def list_indexed_files(self) -> dict[str, dict[str, object]]:
        """Return manifest of all indexed files with file hash and chunk count."""
        return dict(self.manifest)

    def delete_file(self, source: str) -> int:
        """Delete all chunks from a specific source file. Returns chunk count deleted."""
        if source not in self.manifest:
            return 0

        # Find all chunks from this source
        all_metadata = self.collection.get(include=["metadatas"])
        chunk_ids_to_delete = []
        for idx, metadata in enumerate(all_metadata.get("metadatas", [])):
            if metadata.get("source") == source:
                chunk_ids_to_delete.append(all_metadata["ids"][idx])

        if chunk_ids_to_delete:
            self.collection.delete(ids=chunk_ids_to_delete)

        # Remove from manifest
        del self.manifest[source]
        self._save_manifest()
        return len(chunk_ids_to_delete)

    def is_file_indexed(self, file_hash: str) -> bool:
        """Check if a file (by hash) is already indexed."""
        for entry in self.manifest.values():
            if entry.get("file_hash") == file_hash:
                return True
        return False

    def mark_file_indexed(self, source: str, file_hash: str, chunk_count: int) -> None:
        """Mark a file as indexed in the manifest."""
        self.manifest[source] = {
            "file_hash": file_hash,
            "chunk_count": chunk_count,
        }
        self._save_manifest()

    def _load_manifest(self) -> None:
        """Load manifest from disk or initialize empty."""
        if self.manifest_path.exists():
            try:
                self.manifest = json.loads(self.manifest_path.read_text())
            except (json.JSONDecodeError, IOError):
                self.manifest = {}
        else:
            self.manifest = {}

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))
