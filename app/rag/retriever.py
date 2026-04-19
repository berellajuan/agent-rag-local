from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from app.console import StepTracer
from app.config import Settings, get_settings
from app.rag.vectorstore import get_vector_store


@dataclass(frozen=True)
class RetrievedContext:
    query: str
    documents: list[Document]

    @property
    def sources(self) -> list[str]:
        seen: set[str] = set()
        sources: list[str] = []
        for doc in self.documents:
            source = str(doc.metadata.get("relative_path") or doc.metadata.get("source") or "unknown")
            if source not in seen:
                seen.add(source)
                sources.append(source)
        return sources

    def to_prompt_context(self) -> str:
        if not self.documents:
            return "No se recupero contexto relevante."

        blocks: list[str] = []
        for index, doc in enumerate(self.documents, start=1):
            source = doc.metadata.get("relative_path") or doc.metadata.get("source") or "unknown"
            chunk = doc.metadata.get("chunk_index", "?")
            blocks.append(
                f"<document index=\"{index}\" source=\"{source}\" chunk=\"{chunk}\">\n"
                f"{doc.page_content}\n"
                "</document>"
            )
        return "\n\n".join(blocks)


def retrieve(
    query: str,
    *,
    k: int | None = None,
    settings: Settings | None = None,
    verbose: bool = False,
) -> RetrievedContext:
    tracer = StepTracer(verbose)
    settings = settings or get_settings(require_api_key=False)
    active_k = k or settings.retriever_k

    tracer.title("RAG Search")
    tracer.step("Cargo configuracion del vector store y del modelo de embeddings.")
    tracer.detail("embedding_provider", settings.embedding_provider)
    tracer.detail(
        "embedding_model",
        settings.huggingface_embedding_model
        if settings.embedding_provider == "huggingface"
        else settings.google_embedding_model,
    )
    tracer.detail("query_prefix", repr(settings.embedding_query_prefix))
    tracer.detail("retriever_k", active_k)
    tracer.concept("La pregunta se convierte en vector y se buscan chunks semanticamente parecidos.")

    vector_store = get_vector_store(settings)
    tracer.step("Ejecuto similarity_search en Chroma.")
    docs = vector_store.similarity_search(query, k=active_k)
    tracer.result(f"Se recuperaron {len(docs)} chunks.")
    for index, doc in enumerate(docs, start=1):
        tracer.detail(
            f"chunk_{index}",
            {
                "source": doc.metadata.get("relative_path") or doc.metadata.get("source"),
                "chunk_index": doc.metadata.get("chunk_index"),
            },
        )
    return RetrievedContext(query=query, documents=docs)


def main() -> None:
    question = input("Pregunta para buscar en el RAG: ").strip()
    if not question:
        raise SystemExit("No ingresaste una pregunta.")

    context = retrieve(question, verbose=True)
    print("\nChunks recuperados:\n")
    print(context.to_prompt_context())
    print("\nFuentes:")
    for source in context.sources:
        print(f"- {source}")


if __name__ == "__main__":
    main()
