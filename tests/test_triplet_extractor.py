"""
Tests unitarios para TripletExtractor (DTm-23, Fase 1.2 + DTm-16, Fase 4).

Cobertura:
  TE1. _parse_extraction_json con JSON valido.
  TE2. _parse_extraction_json con markdown code block.
  TE3. _parse_extraction_json con JSON malformado -> re-raises exception.
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
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.retrieval.lightrag.knowledge_graph import KGEntity, KGRelation
from shared.retrieval.lightrag.triplet_extractor import TripletExtractor

from tests.helpers import make_extractor


# =============================================================================
# TE1: JSON valido
# =============================================================================

def test_parse_valid_json():
    """JSON bien formado -> entidades + relaciones."""
    ext = make_extractor()
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
    ext = make_extractor()
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
    """JSON roto -> re-raises para que el caller (extract_from_doc_async) lo maneje."""
    ext = make_extractor()
    with pytest.raises((json.JSONDecodeError, ValueError)):
        ext._parse_extraction_json("not json at all", "doc1")


def test_parse_truncated_json():
    """JSON truncado (incompleto) -> re-raises."""
    ext = make_extractor()
    raw = '{"entities": [{"name": "Alice"'  # truncado
    with pytest.raises((json.JSONDecodeError, ValueError)):
        ext._parse_extraction_json(raw, "doc1")


# =============================================================================
# TE4: Campos faltantes
# =============================================================================

def test_parse_missing_relations():
    """JSON sin 'relations' -> entities ok, relations=[]."""
    ext = make_extractor()
    raw = '{"entities": [{"name": "Alice", "type": "PERSON"}]}'
    entities, relations = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 1
    assert relations == []


def test_parse_missing_entities():
    """JSON sin 'entities' -> entities=[], relations ok."""
    ext = make_extractor()
    raw = '{"relations": [{"source": "A", "target": "B", "relation": "knows"}]}'
    entities, relations = ext._parse_extraction_json(raw, "doc1")
    assert entities == []
    assert len(relations) == 1


def test_parse_entity_without_name():
    """Entity sin 'name' se ignora (isinstance check)."""
    ext = make_extractor()
    raw = '{"entities": [{"type": "PERSON"}, {"name": "Alice", "type": "ORG"}], "relations": []}'
    entities, relations = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 1
    assert entities[0].name == "Alice"


def test_parse_relation_without_source():
    """Relation sin 'source' se ignora."""
    ext = make_extractor()
    raw = '{"entities": [], "relations": [{"target": "B", "relation": "knows"}]}'
    entities, relations = ext._parse_extraction_json(raw, "doc1")
    assert relations == []


# =============================================================================
# TE5: Entity type no-enum
# =============================================================================

def test_parse_invalid_entity_type_normalized_to_other():
    """Entity con type invalido se normaliza a OTHER (DTm-16)."""
    ext = make_extractor()
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
    ext = make_extractor(mock_llm=mock_llm, max_text_chars=100)

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
    ext = make_extractor(mock_llm=mock_llm)

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
    ext = make_extractor(mock_llm=mock_llm)

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
    ext = make_extractor(mock_llm=mock_llm)

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
    ext = make_extractor()
    raw = '{"low_level": ["Alice", "MIT"], "high_level": ["research", "AI"]}'
    low, high = ext._parse_keywords_json(raw)
    assert low == ["Alice", "MIT"]
    assert high == ["research", "AI"]


def test_parse_keywords_missing_field():
    """JSON de keywords sin high_level -> default []."""
    ext = make_extractor()
    raw = '{"low_level": ["Alice"]}'
    low, high = ext._parse_keywords_json(raw)
    assert low == ["Alice"]
    assert high == []


# =============================================================================
# TE10: _parse_keywords_json malformado
# =============================================================================

def test_parse_keywords_malformed():
    """JSON de keywords roto -> ([], [])."""
    ext = make_extractor()
    low, high = ext._parse_keywords_json("not json")
    assert low == []
    assert high == []


# =============================================================================
# TE11-TE16: Validacion post-parse (DTm-16)
# =============================================================================

def test_entity_name_min_length_accepted():
    """Entity con nombre de 1 char se acepta (DTm-27: MIN_ENTITY_NAME_LEN=1)."""
    ext = make_extractor()
    raw = '{"entities": [{"name": "A", "type": "PERSON"}, {"name": "Bob", "type": "PERSON"}], "relations": []}'
    entities, _ = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 2
    assert entities[0].name == "A"
    assert entities[1].name == "Bob"


def test_entity_empty_name_rejected():
    """Entity con nombre vacio se rechaza (DTm-16)."""
    ext = make_extractor()
    raw = '{"entities": [{"name": "", "type": "PERSON"}, {"name": "  ", "type": "PERSON"}], "relations": []}'
    entities, _ = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 0


def test_entity_description_truncated():
    """Description > MAX_DESCRIPTION_CHARS se trunca (DTm-16)."""
    ext = make_extractor()
    long_desc = "x" * 500
    raw = f'{{"entities": [{{"name": "Alice", "type": "PERSON", "description": "{long_desc}"}}], "relations": []}}'
    entities, _ = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 1
    assert len(entities[0].description) == 200


def test_relation_description_truncated():
    """Relation description > MAX_DESCRIPTION_CHARS se trunca (DTm-16)."""
    ext = make_extractor()
    long_desc = "y" * 500
    raw = f'{{"entities": [], "relations": [{{"source": "AA", "target": "BB", "relation": "knows", "description": "{long_desc}"}}]}}'
    _, relations = ext._parse_extraction_json(raw, "doc1")
    assert len(relations) == 1
    assert len(relations[0].description) == 200


def test_valid_entity_types_accepted():
    """Todos los VALID_ENTITY_TYPES se aceptan sin modificar (DTm-16)."""
    ext = make_extractor()
    from shared.retrieval.lightrag.triplet_extractor import VALID_ENTITY_TYPES
    for etype in VALID_ENTITY_TYPES:
        raw = f'{{"entities": [{{"name": "Test Entity", "type": "{etype}"}}], "relations": []}}'
        entities, _ = ext._parse_extraction_json(raw, "doc1")
        assert entities[0].entity_type == etype


def test_entity_type_case_insensitive():
    """Entity type se normaliza a uppercase (DTm-16)."""
    ext = make_extractor()
    raw = '{"entities": [{"name": "Alice", "type": "person"}], "relations": []}'
    entities, _ = ext._parse_extraction_json(raw, "doc1")
    assert entities[0].entity_type == "PERSON"


# =============================================================================
# TE17-TE21: Estadisticas de extraccion (DTm-33)
# =============================================================================

def test_stats_initial_zero():
    """TE17: get_stats() devuelve contadores a cero al iniciar (DTm-33)."""
    ext = make_extractor()
    stats = ext.get_stats()
    assert all(v == 0 for v in stats.values())
    assert "docs_failed" in stats
    assert "docs_empty_input" in stats


def test_stats_success_counted():
    """TE18: Extraccion exitosa incrementa docs_success y totales (DTm-33)."""
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value='{"entities": [{"name": "Alice", "type": "PERSON"}], "relations": []}')
    ext = make_extractor(mock_llm=mock_llm)

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
    ext = make_extractor(mock_llm=mock_llm)

    asyncio.run(ext.extract_from_doc_async("doc1", "Some text"))

    stats = ext.get_stats()
    assert stats["docs_processed"] == 1
    assert stats["docs_failed"] == 1
    assert stats["docs_success"] == 0


def test_stats_empty_input_counted():
    """TE20: Texto vacio incrementa docs_empty_input (DTm-33)."""
    ext = make_extractor()

    asyncio.run(ext.extract_from_doc_async("doc1", "   "))

    stats = ext.get_stats()
    assert stats["docs_processed"] == 1
    assert stats["docs_empty_input"] == 1
    assert stats["docs_success"] == 0


def test_stats_reset():
    """TE21: reset_stats() pone todos los contadores a cero (DTm-33)."""
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value='{"entities": [{"name": "Bob", "type": "PERSON"}], "relations": []}')
    ext = make_extractor(mock_llm=mock_llm)

    asyncio.run(ext.extract_from_doc_async("doc1", "Bob works here."))
    assert ext.get_stats()["docs_success"] == 1

    ext.reset_stats()
    stats = ext.get_stats()
    assert all(v == 0 for v in stats.values())


# =============================================================================
# TE22-TE28: Multi-doc batch extraction (P2)
# =============================================================================

def test_group_docs_for_batch_basic():
    """TE22: _group_docs_for_batch agrupa docs cortos en mini-batches."""
    ext = make_extractor(max_text_chars=3000)
    docs = [{"doc_id": f"d{i}", "content": "short text"} for i in range(12)]
    groups = ext._group_docs_for_batch(docs, batch_docs_per_call=5)
    # 12 docs / 5 per batch = 3 groups (5, 5, 2)
    assert len(groups) == 3
    assert len(groups[0]) == 5
    assert len(groups[1]) == 5
    assert len(groups[2]) == 2


def test_group_docs_for_batch_budget_limit():
    """TE23: Docs largos se agrupan respetando presupuesto de chars."""
    ext = make_extractor(max_text_chars=100)
    # Cada doc tiene 80 chars, budget = 100*3 = 300 chars
    # 3 docs = 240 chars OK, 4th would be 320 > 300
    docs = [{"doc_id": f"d{i}", "content": "x" * 80} for i in range(7)]
    groups = ext._group_docs_for_batch(docs, batch_docs_per_call=3)
    # All groups should have at most 3 docs
    for g in groups:
        assert len(g) <= 3


def test_build_batch_prompt():
    """TE24: _build_batch_prompt construye prompt con marcadores [DOC N]."""
    ext = make_extractor(max_text_chars=100)
    docs = [
        {"doc_id": "doc1", "content": "Alice works at MIT"},
        {"doc_id": "doc2", "content": "Bob studies physics"},
    ]
    prompt = ext._build_batch_prompt(docs)
    assert '[DOC 1 id="doc1"]' in prompt
    assert '[DOC 2 id="doc2"]' in prompt
    assert "Alice works at MIT" in prompt
    assert "Bob studies physics" in prompt
    assert "[/DOC 1]" in prompt
    assert "[/DOC 2]" in prompt


def test_parse_batch_extraction_json_valid():
    """TE25: _parse_batch_extraction_json parsea respuesta multi-doc."""
    ext = make_extractor()
    raw = json.dumps({
        "documents": [
            {
                "doc_id": "doc1",
                "entities": [{"name": "Alice", "type": "PERSON", "description": "researcher"}],
                "relations": [{"source": "Alice", "target": "MIT", "relation": "works at"}],
            },
            {
                "doc_id": "doc2",
                "entities": [{"name": "Bob", "type": "PERSON", "description": "student"}],
                "relations": [],
            },
        ]
    })
    docs = [{"doc_id": "doc1", "content": "..."}, {"doc_id": "doc2", "content": "..."}]
    result = ext._parse_batch_extraction_json(raw, docs)
    assert result is not None
    assert "doc1" in result
    assert "doc2" in result
    assert len(result["doc1"][0]) == 1  # 1 entity
    assert result["doc1"][0][0].name == "Alice"
    assert len(result["doc1"][1]) == 1  # 1 relation
    assert len(result["doc2"][0]) == 1  # 1 entity


def test_parse_batch_extraction_json_invalid_returns_none():
    """TE26: Non-batch JSON format returns None (triggers fallback)."""
    ext = make_extractor()
    # Single-doc format (no "documents" key)
    raw = '{"entities": [{"name": "Alice", "type": "PERSON"}], "relations": []}'
    result = ext._parse_batch_extraction_json(raw, [])
    assert result is None


def test_extract_multi_doc_async_success():
    """TE27: _extract_multi_doc_async processes multiple docs in one LLM call."""
    batch_response = json.dumps({
        "documents": [
            {
                "doc_id": "doc1",
                "entities": [{"name": "Alice", "type": "PERSON"}],
                "relations": [],
            },
            {
                "doc_id": "doc2",
                "entities": [{"name": "Bob", "type": "PERSON"}],
                "relations": [],
            },
        ]
    })
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value=batch_response)
    ext = make_extractor(mock_llm=mock_llm)

    docs = [
        {"doc_id": "doc1", "content": "Alice is a researcher."},
        {"doc_id": "doc2", "content": "Bob is a student."},
    ]
    results = asyncio.run(ext._extract_multi_doc_async(docs))

    assert "doc1" in results
    assert "doc2" in results
    assert results["doc1"][0][0].name == "Alice"
    assert results["doc2"][0][0].name == "Bob"
    # Only 1 LLM call for 2 docs
    assert mock_llm.invoke_async.call_count == 1


def test_extract_multi_doc_async_fallback_on_bad_response():
    """TE28: Falls back to single-doc extraction when batch parse fails."""
    single_response = '{"entities": [{"name": "Alice", "type": "PERSON"}], "relations": []}'
    mock_llm = MagicMock()
    # First call returns non-batch format, subsequent calls return single-doc format
    mock_llm.invoke_async = AsyncMock(return_value=single_response)
    ext = make_extractor(mock_llm=mock_llm)

    docs = [
        {"doc_id": "doc1", "content": "Alice is a researcher."},
        {"doc_id": "doc2", "content": "Bob is a student."},
    ]
    results = asyncio.run(ext._extract_multi_doc_async(docs))

    assert "doc1" in results
    assert "doc2" in results
    # 1 batch call (failed parse) + 2 individual calls = 3 total
    assert mock_llm.invoke_async.call_count == 3


def test_extract_batch_async_multi_doc_mode():
    """TE29: extract_batch_async with batch_docs_per_call > 1 uses multi-doc."""
    batch_response = json.dumps({
        "documents": [
            {"doc_id": "d0", "entities": [{"name": "X", "type": "CONCEPT"}], "relations": []},
            {"doc_id": "d1", "entities": [{"name": "Y", "type": "CONCEPT"}], "relations": []},
        ]
    })
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value=batch_response)
    ext = make_extractor(mock_llm=mock_llm)

    docs = [
        {"doc_id": "d0", "content": "Doc about X."},
        {"doc_id": "d1", "content": "Doc about Y."},
    ]
    results = asyncio.run(ext.extract_batch_async(docs, batch_docs_per_call=5))

    assert "d0" in results
    assert "d1" in results
    # 2 docs in 1 group -> 1 LLM call
    assert mock_llm.invoke_async.call_count == 1


def test_extract_batch_async_legacy_mode():
    """TE30: extract_batch_async with batch_docs_per_call=1 uses one call per doc."""
    single_response = '{"entities": [{"name": "A", "type": "CONCEPT"}], "relations": []}'
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value=single_response)
    ext = make_extractor(mock_llm=mock_llm)

    docs = [
        {"doc_id": "d0", "content": "Doc A."},
        {"doc_id": "d1", "content": "Doc B."},
    ]
    results = asyncio.run(ext.extract_batch_async(docs, batch_docs_per_call=1))

    assert "d0" in results
    assert "d1" in results
    # 2 docs -> 2 LLM calls (one per doc)
    assert mock_llm.invoke_async.call_count == 2
