from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from .config import DEFAULT_EMBEDDING_MODEL


class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def name(self) -> str:
        return self.model_name

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(input=input)

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:
        texts = list(input)
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, input: str | Sequence[str]) -> list[float] | list[list[float]]:
        if isinstance(input, str):
            vector = self.model.encode(
                [input],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vector[0].tolist()

        return self.embed_documents(input=input)
