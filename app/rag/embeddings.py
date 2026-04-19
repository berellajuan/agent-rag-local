from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.embeddings import Embeddings


@dataclass(frozen=True)
class PrefixedEmbeddings(Embeddings):
    """Embeddings wrapper for models that require different query/document prefixes.

    E5-family models are trained with prefixes such as `query: ` and `passage: `.
    Keeping this wrapper explicit avoids hiding an important retrieval concept.
    """

    base: Embeddings
    query_prefix: str = ""
    document_prefix: str = ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents(self._prefix_all(texts, self.document_prefix))

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(f"{self.query_prefix}{text}")

    @staticmethod
    def _prefix_all(texts: Iterable[str], prefix: str) -> list[str]:
        return [f"{prefix}{text}" for text in texts]
