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


# =====================================================================
# Divergencia #7: presupuestos proporcionales por seccion segun modo
# =====================================================================

def test_structured_context_mode_local_excludes_relations():
    """En modo 'local', las relaciones no reciben budget (se omiten)."""
    entities = [{"entity": "Alice", "type": "PERSON", "description": "p"}]
    relations = [
        {"source": "A", "target": "B", "relation": "knows", "description": "d"},
    ]
    contents = ["doc content"]

    result = format_structured_context(
        contents, entities, relations, max_length=5000, mode="local",
    )

    assert "Knowledge Graph Data (Entity):" in result
    assert "Knowledge Graph Data (Relationship):" not in result
    assert "Document Chunks:" in result


def test_structured_context_mode_global_excludes_entities():
    """En modo 'global', las entidades no reciben budget (se omiten)."""
    entities = [{"entity": "Alice", "type": "PERSON", "description": "p"}]
    relations = [
        {"source": "A", "target": "B", "relation": "knows", "description": "d"},
    ]
    contents = ["doc content"]

    result = format_structured_context(
        contents, entities, relations, max_length=5000, mode="global",
    )

    assert "Knowledge Graph Data (Entity):" not in result
    assert "Knowledge Graph Data (Relationship):" in result
    assert "Document Chunks:" in result


def test_structured_context_kg_cannot_starve_chunks():
    """KG voluminoso no puede consumir todo el budget — chunks tienen espacio reservado."""
    # 30 entidades largas (cada una ~250 chars en JSON)
    big_entities = [
        {"entity": f"E{i}", "type": "CONCEPT", "description": "x" * 200}
        for i in range(30)
    ]
    # 30 relaciones largas
    big_relations = [
        {"source": f"A{i}", "target": f"B{i}", "relation": "r",
         "description": "y" * 200}
        for i in range(30)
    ]
    chunk_text = "important chunk content here" * 3  # chunk mediano
    contents = [chunk_text, chunk_text, chunk_text]

    # Con hybrid: 20% entidades, 20% relaciones, 60% chunks
    # max_length=5000 → chunks tienen garantizados ~3000 chars
    result = format_structured_context(
        contents, big_entities, big_relations, max_length=5000, mode="hybrid",
    )

    assert "Document Chunks:" in result
    assert "important chunk content here" in result
    # Las secciones KG se truncan para no desbordar
    assert len(result) <= 5500  # buffer razonable por headers/separadores


def test_structured_context_unused_kg_budget_redistributes_to_chunks():
    """Si KG usa menos que su budget, el sobrante va a chunks."""
    # Entidades pequenas — no consumen todo su budget
    small_entities = [
        {"entity": "X", "type": "T", "description": "d"},
    ]
    # Chunks grandes — necesitan el espacio sobrante
    big_chunk = "word " * 500  # ~2500 chars
    contents = [big_chunk, big_chunk]

    # max_length=4000, hybrid (chunks base = 4000 - 800 - 800 - 100 = 2300)
    # Con redistribucion: chunks reciben ~3800 chars (casi todo)
    result = format_structured_context(
        contents, small_entities, [], max_length=4000, mode="hybrid",
    )

    # Con redistribucion, al menos 1 chunk debe caber
    assert "Document Chunks:" in result
    assert "word" in result


def test_structured_context_unknown_mode_defaults_hybrid():
    """Modo desconocido cae a hybrid por defecto."""
    entities = [{"entity": "A", "type": "T", "description": "d"}]
    relations = [{"source": "A", "target": "B", "relation": "r", "description": "d"}]

    result = format_structured_context(
        ["content"], entities, relations, max_length=5000, mode="garbage",
    )

    # Con hybrid, ambas secciones KG deben aparecer
    assert "Knowledge Graph Data (Entity):" in result
    assert "Knowledge Graph Data (Relationship):" in result


def test_structured_context_mode_parameter_default_is_hybrid():
    """Sin mode explicito, usa hybrid (retrocompatibilidad con callers antiguos)."""
    entities = [{"entity": "A", "type": "T", "description": "d"}]
    relations = [{"source": "A", "target": "B", "relation": "r", "description": "d"}]

    # Llamada sin mode (como los tests pre-existentes)
    result = format_structured_context(["content"], entities, relations, max_length=5000)

    assert "Knowledge Graph Data (Entity):" in result
    assert "Knowledge Graph Data (Relationship):" in result
