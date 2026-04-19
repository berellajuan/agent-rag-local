from __future__ import annotations

from langchain_chroma import Chroma

from app.config import Settings, get_settings
from app.models import get_embedding_model


def get_vector_store(settings: Settings | None = None) -> Chroma:
    settings = settings or get_settings(require_api_key=False)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embedding_model(settings),
        persist_directory=str(settings.chroma_persist_dir),
    )
