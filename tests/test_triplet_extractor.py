"""
Tests unitarios para TripletExtractor (DTm-23, Fase 1.2 + DTm-16, Fase 4).

Cobertura:
  TE1. _parse_extraction_json con JSON valido.
  TE2. _parse_extraction_json con markdown code block.
  TE3. _parse_extraction_json con JSON malformado -> ([], []).
  TE4. _parse_extraction_json con campos faltantes.
  TE5. Entity type invalido -> normalizado a OTHER (DTm-16).
  TE6. Truncation: doc > max_text_chars se trunca.
  TE7. extract_from_doc_async: LLM lanza exception -> ([], []).
  TE8. extract_from_doc_async: texto vacio -> ([], []).
  TE9. _parse_keywords_json con JSON valido.
  TE10. _parse_keywords_json con JSON malformado -> ([], []).
  TE11. Entity name de 1 char aceptada (DTm-27: MIN_ENTITY_NAME_LEN=1).
  TE12. Entity name vacio rechazada (DTm-16).
  TE13. Entity description truncada a MAX_DESCRIPTION_CHARS (DTm-16).
  TE14. Relation description truncada a MAX_DESCRIPTION_CHARS (DTm-16).
  TE15. Todos los VALID_ENTITY_TYPES aceptados (DTm-16).
  TE16. Entity type case-insensitive (DTm-16).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.retrieval.knowledge_graph import KGEntity, KGRelation
from shared.retrieval.triplet_extractor import TripletExtractor


# =============================================================================
# Helpers
# =============================================================================

def _make_extractor(mock_llm=None, max_text_chars=3000):
    """Crea TripletExtractor con LLM mockeado."""
    llm = mock_llm or MagicMock()
    ext = object.__new__(TripletExtractor)
    ext._llm = llm
    ext._max_text_chars = max_text_chars
    ext._stats = {
        "docs_processed": 0, "docs_success": 0, "docs_failed": 0,
        "docs_empty_input": 0, "docs_empty_result": 0,
        "total_entities": 0, "total_relations": 0,
    }
    return ext


# =============================================================================
# TE1: JSON valido
# =============================================================================

def test_parse_valid_json():
    """JSON bien formado -> entidades + relaciones."""
    ext = _make_extractor()
    raw = '''{
        "entities": [
            {"name": "Alice", "type": "PERSON", "description": "a researcher"},
            {"name": "MIT", "type": "ORG", "description": "university"}
        ],
        "relations": [
            {"source": "Alice", "target": "MIT", "relation": "works at", "description": "employed"}
        ]
    }'''

    entities, relations = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 2
    assert len(relations) == 1
    assert entities[0].name == "Alice"
    assert entities[0].entity_type == "PERSON"
    assert relations[0].source == "Alice"
    assert relations[0].target == "MIT"
    assert relations[0].source_doc_id == "doc1"


# =============================================================================
# TE2: Markdown code block
# =============================================================================

def test_parse_json_with_markdown():
    """```json ... ``` -> strip y parse correctamente."""
    ext = _make_extractor()
    raw = '''```json
{
    "entities": [{"name": "Bob", "type": "PERSON", "description": ""}],
    "relations": []
}
```'''

    entities, relations = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 1
    assert entities[0].name == "Bob"
    assert relations == []


# =============================================================================
# TE3: JSON malformado
# =============================================================================

def test_parse_malformed_json():
    """JSON roto -> ([], []), no exception."""
    ext = _make_extractor()
    entities, relations = ext._parse_extraction_json("not json at all", "doc1")
    assert entities == []
    assert relations == []


def test_parse_truncated_json():
    """JSON truncado (incompleto) -> ([], [])."""
    ext = _make_extractor()
    raw = '{"entities": [{"name": "Alice"'  # truncado
    entities, relations = ext._parse_extraction_json(raw, "doc1")
    assert entities == []
    assert relations == []


# =============================================================================
# TE4: Campos faltantes
# =============================================================================

def test_parse_missing_relations():
    """JSON sin 'relations' -> entities ok, relations=[]."""
    ext = _make_extractor()
    raw = '{"entities": [{"name": "Alice", "type": "PERSON"}]}'
    entities, relations = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 1
    assert relations == []


def test_parse_missing_entities():
    """JSON sin 'entities' -> entities=[], relations ok."""
    ext = _make_extractor()
    raw = '{"relations": [{"source": "A", "target": "B", "relation": "knows"}]}'
    entities, relations = ext._parse_extraction_json(raw, "doc1")
    assert entities == []
    assert len(relations) == 1


def test_parse_entity_without_name():
    """Entity sin 'name' se ignora (isinstance check)."""
    ext = _make_extractor()
    raw = '{"entities": [{"type": "PERSON"}, {"name": "Alice", "type": "ORG"}], "relations": []}'
    entities, relations = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 1
    assert entities[0].name == "Alice"


def test_parse_relation_without_source():
    """Relation sin 'source' se ignora."""
    ext = _make_extractor()
    raw = '{"entities": [], "relations": [{"target": "B", "relation": "knows"}]}'
    entities, relations = ext._parse_extraction_json(raw, "doc1")
    assert relations == []


# =============================================================================
# TE5: Entity type no-enum
# =============================================================================

def test_parse_invalid_entity_type_normalized_to_other():
    """Entity con type invalido se normaliza a OTHER (DTm-16)."""
    ext = _make_extractor()
    raw = '{"entities": [{"name": "SomeThing", "type": "CUSTOM_TYPE"}], "relations": []}'
    entities, _ = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 1
    assert entities[0].entity_type == "OTHER"


# =============================================================================
# TE6: Truncation
# =============================================================================

def test_truncation():
    """Doc > max_text_chars se trunca antes de enviar al LLM."""
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value='{"entities":[],"relations":[]}')
    ext = _make_extractor(mock_llm=mock_llm, max_text_chars=100)

    long_text = "x" * 500

    asyncio.run(ext.extract_from_doc_async("doc1", long_text))

    # Verificar que el prompt enviado contiene texto truncado
    call_args = mock_llm.invoke_async.call_args
    prompt_sent = call_args[0][0]
    # El texto en el prompt no debe contener los 500 chars completos
    assert "x" * 500 not in prompt_sent
    assert "x" * 100 in prompt_sent


# =============================================================================
# TE7: LLM exception
# =============================================================================

def test_extract_from_doc_llm_error():
    """LLM lanza exception -> ([], []), no propaga."""
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(side_effect=RuntimeError("API down"))
    ext = _make_extractor(mock_llm=mock_llm)

    entities, relations = asyncio.run(
        ext.extract_from_doc_async("doc1", "some text")
    )
    assert entities == []
    assert relations == []


# =============================================================================
# TE8: Texto vacio
# =============================================================================

def test_extract_from_doc_empty_text():
    """Texto vacio -> ([], []) sin llamar al LLM."""
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock()
    ext = _make_extractor(mock_llm=mock_llm)

    entities, relations = asyncio.run(
        ext.extract_from_doc_async("doc1", "")
    )
    assert entities == []
    assert relations == []
    mock_llm.invoke_async.assert_not_called()


def test_extract_from_doc_whitespace_only():
    """Texto solo whitespace -> ([], []) sin llamar al LLM."""
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock()
    ext = _make_extractor(mock_llm=mock_llm)

    entities, relations = asyncio.run(
        ext.extract_from_doc_async("doc1", "   \n  ")
    )
    assert entities == []
    assert relations == []
    mock_llm.invoke_async.assert_not_called()


# =============================================================================
# TE9: _parse_keywords_json valido
# =============================================================================

def test_parse_keywords_valid():
    """JSON de keywords bien formado."""
    ext = _make_extractor()
    raw = '{"low_level": ["Alice", "MIT"], "high_level": ["research", "AI"]}'
    low, high = ext._parse_keywords_json(raw)
    assert low == ["Alice", "MIT"]
    assert high == ["research", "AI"]


def test_parse_keywords_missing_field():
    """JSON de keywords sin high_level -> default []."""
    ext = _make_extractor()
    raw = '{"low_level": ["Alice"]}'
    low, high = ext._parse_keywords_json(raw)
    assert low == ["Alice"]
    assert high == []


# =============================================================================
# TE10: _parse_keywords_json malformado
# =============================================================================

def test_parse_keywords_malformed():
    """JSON de keywords roto -> ([], [])."""
    ext = _make_extractor()
    low, high = ext._parse_keywords_json("not json")
    assert low == []
    assert high == []


# =============================================================================
# TE11-TE16: Validacion post-parse (DTm-16)
# =============================================================================

def test_entity_name_min_length_accepted():
    """Entity con nombre de 1 char se acepta (DTm-27: MIN_ENTITY_NAME_LEN=1)."""
    ext = _make_extractor()
    raw = '{"entities": [{"name": "A", "type": "PERSON"}, {"name": "Bob", "type": "PERSON"}], "relations": []}'
    entities, _ = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 2
    assert entities[0].name == "A"
    assert entities[1].name == "Bob"


def test_entity_empty_name_rejected():
    """Entity con nombre vacio se rechaza (DTm-16)."""
    ext = _make_extractor()
    raw = '{"entities": [{"name": "", "type": "PERSON"}, {"name": "  ", "type": "PERSON"}], "relations": []}'
    entities, _ = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 0


def test_entity_description_truncated():
    """Description > MAX_DESCRIPTION_CHARS se trunca (DTm-16)."""
    ext = _make_extractor()
    long_desc = "x" * 500
    raw = f'{{"entities": [{{"name": "Alice", "type": "PERSON", "description": "{long_desc}"}}], "relations": []}}'
    entities, _ = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 1
    assert len(entities[0].description) == 200


def test_relation_description_truncated():
    """Relation description > MAX_DESCRIPTION_CHARS se trunca (DTm-16)."""
    ext = _make_extractor()
    long_desc = "y" * 500
    raw = f'{{"entities": [], "relations": [{{"source": "AA", "target": "BB", "relation": "knows", "description": "{long_desc}"}}]}}'
    _, relations = ext._parse_extraction_json(raw, "doc1")
    assert len(relations) == 1
    assert len(relations[0].description) == 200


def test_valid_entity_types_accepted():
    """Todos los VALID_ENTITY_TYPES se aceptan sin modificar (DTm-16)."""
    ext = _make_extractor()
    from shared.retrieval.triplet_extractor import VALID_ENTITY_TYPES
    for etype in VALID_ENTITY_TYPES:
        raw = f'{{"entities": [{{"name": "Test Entity", "type": "{etype}"}}], "relations": []}}'
        entities, _ = ext._parse_extraction_json(raw, "doc1")
        assert entities[0].entity_type == etype


def test_entity_type_case_insensitive():
    """Entity type se normaliza a uppercase (DTm-16)."""
    ext = _make_extractor()
    raw = '{"entities": [{"name": "Alice", "type": "person"}], "relations": []}'
    entities, _ = ext._parse_extraction_json(raw, "doc1")
    assert entities[0].entity_type == "PERSON"


# =============================================================================
# TE17-TE21: Estadisticas de extraccion (DTm-33)
# =============================================================================

def test_stats_initial_zero():
    """TE17: get_stats() devuelve contadores a cero al iniciar (DTm-33)."""
    ext = _make_extractor()
    stats = ext.get_stats()
    assert all(v == 0 for v in stats.values())
    assert "docs_failed" in stats
    assert "docs_empty_input" in stats


def test_stats_success_counted():
    """TE18: Extraccion exitosa incrementa docs_success y totales (DTm-33)."""
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value='{"entities": [{"name": "Alice", "type": "PERSON"}], "relations": []}')
    ext = _make_extractor(mock_llm=mock_llm)

    asyncio.run(ext.extract_from_doc_async("doc1", "Alice is a researcher."))

    stats = ext.get_stats()
    assert stats["docs_processed"] == 1
    assert stats["docs_success"] == 1
    assert stats["docs_failed"] == 0
    assert stats["total_entities"] == 1


def test_stats_failure_counted():
    """TE19: Excepcion en LLM incrementa docs_failed (DTm-33)."""
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(side_effect=RuntimeError("LLM down"))
    ext = _make_extractor(mock_llm=mock_llm)

    asyncio.run(ext.extract_from_doc_async("doc1", "Some text"))

    stats = ext.get_stats()
    assert stats["docs_processed"] == 1
    assert stats["docs_failed"] == 1
    assert stats["docs_success"] == 0


def test_stats_empty_input_counted():
    """TE20: Texto vacio incrementa docs_empty_input (DTm-33)."""
    ext = _make_extractor()

    asyncio.run(ext.extract_from_doc_async("doc1", "   "))

    stats = ext.get_stats()
    assert stats["docs_processed"] == 1
    assert stats["docs_empty_input"] == 1
    assert stats["docs_success"] == 0


def test_stats_reset():
    """TE21: reset_stats() pone todos los contadores a cero (DTm-33)."""
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value='{"entities": [{"name": "Bob", "type": "PERSON"}], "relations": []}')
    ext = _make_extractor(mock_llm=mock_llm)

    asyncio.run(ext.extract_from_doc_async("doc1", "Bob works here."))
    assert ext.get_stats()["docs_success"] == 1

    ext.reset_stats()
    stats = ext.get_stats()
    assert all(v == 0 for v in stats.values())
