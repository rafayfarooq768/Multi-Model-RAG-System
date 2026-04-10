from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))

    for chunk_index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = chunk_index
        chunk.metadata = metadata

    return chunks
