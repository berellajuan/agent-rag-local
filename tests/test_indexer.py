from app.rag.indexer import chunk_id, sha256_text


def test_sha256_text_is_stable():
    assert sha256_text("hola") == sha256_text("hola")
    assert sha256_text("hola") != sha256_text("chau")


def test_chunk_id_is_deterministic_and_changes_with_content():
    first = chunk_id("doc.md", "contenido", 0)
    second = chunk_id("doc.md", "contenido", 0)
    changed = chunk_id("doc.md", "otro contenido", 0)

    assert first == second
    assert first != changed
    assert first.startswith("doc.md::chunk-0::")
