"""
Tests para format_structured_context (F.3/DAM-8).

Verifica que el contexto estructurado incluye las 3 secciones
(entidades, relaciones, chunks) y respeta el budget de caracteres.
"""

import json

from sandbox_mteb.retrieval_executor import (
    format_context,
    format_structured_context,
)


def test_structured_context_includes_all_sections():
    """Las 3 secciones aparecen cuando hay entidades, relaciones y chunks."""
    entities = [
        {"entity": "Alice", "type": "PERSON", "description": "Protagonist"},
    ]
    relations = [
        {"source": "Alice", "target": "Bob", "relation": "knows", "description": "Friends"},
    ]
    contents = ["Document text here."]

    result = format_structured_context(contents, entities, relations, max_length=5000)

    assert "Knowledge Graph Data (Entity):" in result
    assert "Knowledge Graph Data (Relationship):" in result
    assert "Document Chunks:" in result
    assert '"Alice"' in result
    assert '"knows"' in result
    assert "Document text here." in result


def test_structured_context_entities_as_json():
    """Entidades se formatean como JSON lines."""
    entities = [
        {"entity": "Alice", "type": "PERSON", "description": "Main character"},
        {"entity": "Bob", "type": "PERSON", "description": "Friend"},
    ]
    result = format_structured_context(["text"], entities, [], max_length=5000)

    # Cada entidad debe ser JSON parseable
    lines = []
    in_entity_block = False
    for line in result.split("\n"):
        if line.startswith("```json"):
            in_entity_block = True
            continue
        if line.startswith("```"):
            in_entity_block = False
            continue
        if in_entity_block and line.strip():
            parsed = json.loads(line)
            lines.append(parsed)
            break  # solo verifico la primera

    assert lines[0]["entity"] == "Alice"


def test_structured_context_without_kg_returns_chunks_only():
    """Sin entidades ni relaciones, solo muestra chunks."""
    result = format_structured_context(["Content A", "Content B"], [], [], max_length=5000)

    assert "Knowledge Graph Data" not in result
    assert "Document Chunks:" in result
    assert "Content A" in result


def test_structured_context_respects_max_length():
    """Chunks se truncan si exceden el budget restante."""
    entities = [{"entity": "X", "type": "CONCEPT", "description": "d" * 200}]
    # Chunks muy largos
    big_content = "word " * 1000  # ~5000 chars
    contents = [big_content, big_content, big_content]

    result = format_structured_context(contents, entities, [], max_length=1000)

    assert len(result) <= 1500  # algo de overhead por formato


def test_structured_context_empty_everything():
    """Sin nada, retorna mensaje por defecto."""
    result = format_structured_context([], [], [], max_length=5000)
    assert result == "[No se encontraron documentos]"


def test_structured_context_no_fallback_for_plain():
    """format_context sigue funcionando como antes (sin KG)."""
    result = format_context(["Content A", "Content B"], max_length=5000)
    assert "[Doc 1]" in result
    assert "[Doc 2]" in result
    assert "Knowledge Graph" not in result
