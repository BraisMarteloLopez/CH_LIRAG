"""
Tests unitarios para glean_from_doc_async() (Fase I.5, DTm-77).

Cobertura:
  GL1. Gleaning con mock LLM retorna entidades adicionales
  GL2. Gleaning con previous_entities vacio retorna ([], [])
  GL3. Gleaning con texto vacio retorna ([], [])
  GL4. Gleaning prompt incluye previous_entities
  GL5. Gleaning con LLM que falla retorna ([], [])
  GL6. Gleaning con rounds=0 no se invoca (test de integracion con retriever)
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.retrieval.lightrag.knowledge_graph import KGEntity, KGRelation
from shared.retrieval.lightrag.triplet_extractor import TripletExtractor
from shared.llm import run_sync


def _make_extractor(mock_llm=None):
    """Crea TripletExtractor con LLM mockeado."""
    llm = mock_llm or MagicMock()
    ext = object.__new__(TripletExtractor)
    ext._llm = llm
    ext._max_text_chars = 3000
    ext._keyword_max_tokens = 1024
    ext._extraction_max_tokens = 4096
    ext._batch_size = 64
    ext._stats = {
        "docs_processed": 0, "docs_success": 0, "docs_failed": 0,
        "docs_empty_input": 0, "docs_empty_result": 0,
        "docs_json_recovered": 0,
    }
    return ext


def _make_entity(name: str, entity_type: str = "PERSON") -> KGEntity:
    return KGEntity(name=name, entity_type=entity_type, description=f"desc of {name}")


# =============================================================================
# GL1: Gleaning retorna entidades adicionales
# =============================================================================

def test_gleaning_returns_additional_entities():
    """Gleaning con LLM que retorna JSON valido produce entidades."""
    response_json = json.dumps({
        "entities": [
            {"name": "Hidden Entity", "type": "CONCEPT", "description": "was missed"},
        ],
        "relations": [
            {"source": "Hidden Entity", "target": "Einstein", "relation": "related_to", "description": "link"},
        ],
    })
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value=response_json)

    ext = _make_extractor(mock_llm)
    previous = [_make_entity("Einstein"), _make_entity("Ulm")]

    entities, relations = run_sync(
        ext.glean_from_doc_async("doc1", "Some text about Einstein and Ulm.", previous)
    )

    assert len(entities) == 1
    assert entities[0].name == "Hidden Entity"  # extractor preserva case
    assert len(relations) == 1
    assert relations[0].source == "Hidden Entity"


# =============================================================================
# GL2: Gleaning con previous_entities vacio
# =============================================================================

def test_gleaning_empty_previous_entities():
    """Gleaning con previous_entities vacio retorna ([], [])."""
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value="should not be called")

    ext = _make_extractor(mock_llm)

    entities, relations = run_sync(
        ext.glean_from_doc_async("doc1", "Some text", [])
    )

    assert entities == []
    assert relations == []
    mock_llm.invoke_async.assert_not_called()


# =============================================================================
# GL3: Gleaning con texto vacio
# =============================================================================

def test_gleaning_empty_text():
    """Gleaning con texto vacio retorna ([], [])."""
    mock_llm = MagicMock()
    ext = _make_extractor(mock_llm)

    entities, relations = run_sync(
        ext.glean_from_doc_async("doc1", "", [_make_entity("X")])
    )

    assert entities == []
    assert relations == []


def test_gleaning_whitespace_text():
    """Gleaning con texto solo whitespace retorna ([], [])."""
    mock_llm = MagicMock()
    ext = _make_extractor(mock_llm)

    entities, relations = run_sync(
        ext.glean_from_doc_async("doc1", "   \n  ", [_make_entity("X")])
    )

    assert entities == []
    assert relations == []


# =============================================================================
# GL4: Gleaning prompt incluye previous_entities
# =============================================================================

def test_gleaning_prompt_includes_previous_entities():
    """El prompt de gleaning lista las entidades previas."""
    response_json = json.dumps({"entities": [], "relations": []})
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(return_value=response_json)

    ext = _make_extractor(mock_llm)
    previous = [_make_entity("Albert Einstein"), _make_entity("Ulm")]

    run_sync(ext.glean_from_doc_async("doc1", "text", previous))

    call_args = mock_llm.invoke_async.call_args
    prompt = call_args[0][0]
    assert "Albert Einstein" in prompt
    assert "Ulm" in prompt


# =============================================================================
# GL5: Gleaning con LLM que falla
# =============================================================================

def test_gleaning_llm_failure():
    """Si LLM falla, gleaning retorna ([], []) sin crash."""
    mock_llm = MagicMock()
    mock_llm.invoke_async = AsyncMock(side_effect=RuntimeError("NIM timeout"))

    ext = _make_extractor(mock_llm)
    previous = [_make_entity("X")]

    entities, relations = run_sync(
        ext.glean_from_doc_async("doc1", "text", previous)
    )

    assert entities == []
    assert relations == []
