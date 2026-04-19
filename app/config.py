from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Configuracion centralizada.

    La app falla temprano si falta GOOGLE_API_KEY, porque si no el error aparece
    tarde dentro de LangChain y se vuelve mas dificil de entender.
    """

    google_api_key: str
    google_chat_model: str = "gemini-2.5-flash"
    embedding_provider: str = "huggingface"
    google_embedding_model: str = "models/gemini-embedding-001"
    huggingface_embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_query_prefix: str = "query: "
    embedding_document_prefix: str = "passage: "
    embedding_device: str = "cpu"
    chroma_collection: str = "local_knowledge"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retriever_k: int = 4
    embedding_dimension: int = 384
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    manifest_path: Path = PROJECT_ROOT / "data" / "processed" / "manifest.json"
    chroma_persist_dir: Path = PROJECT_ROOT / "vectorstore" / "chroma"


def get_settings(require_api_key: bool = True) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise RuntimeError(
            "Falta GOOGLE_API_KEY. Copia .env.example a .env y completa tu API key de Google AI Studio."
        )

    return Settings(
        google_api_key=api_key,
        google_chat_model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash").strip(),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower(),
        google_embedding_model=os.getenv(
            "GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"
        ).strip(),
        huggingface_embedding_model=os.getenv(
            "HUGGINGFACE_EMBEDDING_MODEL",
            "intfloat/multilingual-e5-small",
        ).strip(),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu").strip(),
        embedding_query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", "query: "),
        embedding_document_prefix=os.getenv("EMBEDDING_DOCUMENT_PREFIX", "passage: "),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "local_knowledge").strip(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        retriever_k=int(os.getenv("RETRIEVER_K", "4")),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
    )
