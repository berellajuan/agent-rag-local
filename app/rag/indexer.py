from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.console import StepTracer
from app.config import Settings, get_settings
from app.rag.loader import load_local_documents
from app.rag.splitter import split_documents
from app.rag.vectorstore import get_vector_store


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def chunk_id(relative_path: str, content: str, index: int) -> str:
    digest = sha256_text(f"{relative_path}\n{index}\n{content}")[:24]
    safe_path = relative_path.replace("\\", "/")
    return f"{safe_path}::chunk-{index}::{digest}"


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _file_hashes(documents: list[Document]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for doc in documents:
        relative_path = str(doc.metadata["relative_path"])
        hashes[relative_path] = sha256_text(doc.page_content)
    return hashes


def _prepare_chunks(settings: Settings, documents: list[Document]) -> tuple[list[Document], list[str]]:
    splits = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    per_file_index: dict[str, int] = {}
    prepared_docs: list[Document] = []
    ids: list[str] = []
    now = datetime.now(UTC).isoformat()

    for split in splits:
        relative_path = str(split.metadata["relative_path"])
        index = per_file_index.get(relative_path, 0)
        per_file_index[relative_path] = index + 1

        content_hash = sha256_text(split.page_content)
        doc_id = chunk_id(relative_path, split.page_content, index)
        metadata = {
            **split.metadata,
            "chunk_id": doc_id,
            "chunk_index": index,
            "created_at": now,
            "content_hash": content_hash,
        }
        prepared_docs.append(Document(page_content=split.page_content, metadata=metadata))
        ids.append(doc_id)

    return prepared_docs, ids


def index_documents(settings: Settings | None = None, *, verbose: bool = False) -> dict[str, Any]:
    """Indexacion incremental: add/update/delete sin duplicar chunks."""

    tracer = StepTracer(verbose)
    tracer.title("Indexacion incremental RAG")
    tracer.concept(
        "Indexar significa convertir documentos en chunks, chunks en embeddings, y guardarlos en un vector store."
    )

    settings = settings or get_settings(require_api_key=False)
    tracer.step("Cargo configuracion.")
    tracer.detail("raw_data_dir", settings.raw_data_dir)
    tracer.detail("manifest_path", settings.manifest_path)
    tracer.detail("chroma_persist_dir", settings.chroma_persist_dir)
    tracer.detail("embedding_provider", settings.embedding_provider)
    tracer.detail(
        "embedding_model",
        settings.huggingface_embedding_model
        if settings.embedding_provider == "huggingface"
        else settings.google_embedding_model,
    )
    tracer.detail("embedding_dimension", settings.embedding_dimension)
    tracer.detail("chunk_size", settings.chunk_size)
    tracer.detail("chunk_overlap", settings.chunk_overlap)

    tracer.step("Leo documentos desde data/raw.")
    raw_documents = load_local_documents(settings.raw_data_dir)
    tracer.result(f"Documentos cargados: {len(raw_documents)}")
    for doc in raw_documents:
        tracer.detail("documento", doc.metadata.get("relative_path"))

    tracer.step("Calculo hashes de archivos para detectar cambios.")
    current_hashes = _file_hashes(raw_documents)
    manifest = load_manifest(settings.manifest_path)
    tracer.detail("manifest_existe", settings.manifest_path.exists())
    manifest.setdefault("version", 1)
    manifest.setdefault("files", {})

    previous_files: dict[str, Any] = manifest.get("files", {})
    current_paths = set(current_hashes)
    previous_paths = set(previous_files)
    active_embedding_model = (
        settings.huggingface_embedding_model
        if settings.embedding_provider == "huggingface"
        else settings.google_embedding_model
    )

    deleted_paths = sorted(previous_paths - current_paths)
    tracer.step("Comparo estado actual contra manifest anterior.")
    changed_paths = sorted(
        path
        for path in current_paths
        if previous_files.get(path, {}).get("file_hash") != current_hashes[path]
        or previous_files.get(path, {}).get("embedding_provider") != settings.embedding_provider
        or previous_files.get(path, {}).get("embedding_model") != active_embedding_model
        or previous_files.get(path, {}).get("embedding_dimension") != settings.embedding_dimension
        or previous_files.get(path, {}).get("chunk_size") != settings.chunk_size
        or previous_files.get(path, {}).get("chunk_overlap") != settings.chunk_overlap
        or previous_files.get(path, {}).get("embedding_query_prefix") != settings.embedding_query_prefix
        or previous_files.get(path, {}).get("embedding_document_prefix") != settings.embedding_document_prefix
    )
    tracer.detail("archivos_cambiados_o_nuevos", changed_paths)
    tracer.detail("archivos_borrados", deleted_paths)

    tracer.step("Abro vector store Chroma.")
    vector_store = get_vector_store(settings)

    deleted_chunk_ids: list[str] = []
    for path in deleted_paths:
        deleted_chunk_ids.extend(previous_files.get(path, {}).get("chunk_ids", []))
        previous_files.pop(path, None)

    for path in changed_paths:
        deleted_chunk_ids.extend(previous_files.get(path, {}).get("chunk_ids", []))

    if deleted_chunk_ids:
        tracer.step("Borro chunks viejos del vector store.")
        tracer.detail("chunks_a_borrar", len(deleted_chunk_ids))
        vector_store.delete(ids=deleted_chunk_ids)

    changed_documents = [
        doc for doc in raw_documents if str(doc.metadata["relative_path"]) in set(changed_paths)
    ]
    tracer.step("Divido documentos cambiados en chunks.")
    prepared_docs, prepared_ids = _prepare_chunks(settings, changed_documents)
    tracer.result(f"Chunks preparados: {len(prepared_docs)}")

    if prepared_docs:
        tracer.step("Genero embeddings y guardo chunks en Chroma.")
        tracer.concept(
            "Con E5, los documentos se embebben como 'passage: ...'. El wrapper lo hace automaticamente."
        )
        vector_store.add_documents(documents=prepared_docs, ids=prepared_ids)

    ids_by_path: dict[str, list[str]] = {path: [] for path in changed_paths}
    for doc_id, doc in zip(prepared_ids, prepared_docs, strict=True):
        ids_by_path[str(doc.metadata["relative_path"])].append(doc_id)

    now = datetime.now(UTC).isoformat()
    for path in changed_paths:
        previous_files[path] = {
            "file_hash": current_hashes[path],
            "chunk_ids": ids_by_path.get(path, []),
            "indexed_at": now,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": active_embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_query_prefix": settings.embedding_query_prefix,
            "embedding_document_prefix": settings.embedding_document_prefix,
        }

    manifest["files"] = dict(sorted(previous_files.items()))
    manifest["updated_at"] = now
    manifest["embedding_provider"] = settings.embedding_provider
    manifest["embedding_model"] = active_embedding_model
    manifest["embedding_dimension"] = settings.embedding_dimension
    manifest["embedding_query_prefix"] = settings.embedding_query_prefix
    manifest["embedding_document_prefix"] = settings.embedding_document_prefix
    tracer.step("Guardo manifest actualizado.")
    save_manifest(settings.manifest_path, manifest)
    tracer.result("Indexacion finalizada.")

    return {
        "indexed_files": changed_paths,
        "deleted_files": deleted_paths,
        "added_or_updated_chunks": len(prepared_ids),
        "deleted_chunks": len(deleted_chunk_ids),
        "manifest_path": str(settings.manifest_path),
    }


def main() -> None:
    result = index_documents(verbose=True)
    print("Resultado de indexacion incremental:")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
