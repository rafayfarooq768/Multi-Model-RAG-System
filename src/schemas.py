from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure required metadata fields are present."""
        if "source" not in self.metadata:
            self.metadata["source"] = "unknown"
        if "type" not in self.metadata:
            self.metadata["type"] = "unknown"
        if "page" not in self.metadata:
            self.metadata["page"] = 1


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float | None = None

    def __post_init__(self) -> None:
        """Ensure required metadata fields are present."""
        if "source" not in self.metadata:
            self.metadata["source"] = "unknown"
        if "type" not in self.metadata:
            self.metadata["type"] = "unknown"
        if "page" not in self.metadata:
            self.metadata["page"] = 1
