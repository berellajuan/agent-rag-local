from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}


def load_local_documents(raw_dir: Path) -> list[Document]:
    """Carga documentos de texto locales desde data/raw.

    Mantengo esto simple a proposito: antes de meter PDFs, Word o web scraping,
    hay que entender bien el pipeline base.
    """

    if not raw_dir.exists():
        return []

    documents: list[Document] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "relative_path": str(path.relative_to(raw_dir)),
                },
            )
        )

    return documents
