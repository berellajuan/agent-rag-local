from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Settings, get_settings
from app.rag.embeddings import PrefixedEmbeddings


def get_chat_model(settings: Settings | None = None, *, temperature: float = 0.0):
    settings = settings or get_settings(require_api_key=True)
    return ChatGoogleGenerativeAI(
        model=settings.google_chat_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )


def get_embedding_model(settings: Settings | None = None):
    settings = settings or get_settings(require_api_key=False)

    if settings.embedding_provider == "google":
        if not settings.google_api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=google requiere GOOGLE_API_KEY. "
                "Si queres embeddings gratis/locales, usa EMBEDDING_PROVIDER=huggingface."
            )
        return GoogleGenerativeAIEmbeddings(
            model=settings.google_embedding_model,
            google_api_key=settings.google_api_key,
        )

    if settings.embedding_provider == "huggingface":
        base_embeddings = HuggingFaceEmbeddings(
            model_name=settings.huggingface_embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        return PrefixedEmbeddings(
            base=base_embeddings,
            query_prefix=settings.embedding_query_prefix,
            document_prefix=settings.embedding_document_prefix,
        )

    raise ValueError(
        f"EMBEDDING_PROVIDER invalido: {settings.embedding_provider!r}. "
        "Valores soportados: 'huggingface' o 'google'."
    )
