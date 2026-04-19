from __future__ import annotations

from langchain_core.tools import tool

from app.rag.retriever import retrieve


@tool
def retrieve_context(query: str) -> str:
    """Busca informacion relevante en los documentos locales indexados con RAG."""

    context = retrieve(query)
    sources = "\n".join(f"- {source}" for source in context.sources) or "- sin fuentes"
    return f"Contexto recuperado:\n{context.to_prompt_context()}\n\nFuentes:\n{sources}"
