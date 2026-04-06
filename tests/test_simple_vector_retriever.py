"""
Tests: SimpleVectorRetriever en shared/retrieval/core.py (lineas 171-347)

Cubre retrieve(), retrieve_by_vector(), index_documents(), clear_index(),
get_documents_by_ids() con vector store mockeado.

Patron: object.__new__() como documenta TESTS.md (lineas 96-101).

Coverage IDs:
  SVR1: retrieve() sin vector store -> RetrievalResult vacio
  SVR2: retrieve() con resultados -> extrae doc_ids, contents, scores
  SVR3: retrieve() con excepcion -> retorna vacio con metadata.error
  SVR4: retrieve_by_vector() sin vector store -> vacio
  SVR5: retrieve_by_vector() con resultados -> extrae correctamente
  SVR6: index_documents() exitoso -> retorna True, _is_indexed=True
  SVR7: index_documents() con excepcion -> retorna False
  SVR8: clear_index() -> delega a vector_store, _is_indexed=False
  SVR9: get_documents_by_ids() sin vector store -> dict vacio
  SVR10: get_documents_by_ids() delega a vector store
"""

from unittest.mock import MagicMock, patch

import pytest

from shared.retrieval.core import (
    RetrievalConfig,
    RetrievalStrategy,
    SimpleVectorRetriever,
)


def _make_retriever(with_store=False):
    """Crea SimpleVectorRetriever via object.__new__() sin infra."""
    r = object.__new__(SimpleVectorRetriever)
    r.config = RetrievalConfig()
    r._vector_store = None
    r._is_indexed = False
    r.embedding_model = MagicMock()
    r.collection_name = "test_col"
    r.embedding_batch_size = 50

    if with_store:
        r._vector_store = MagicMock()
    return r


def _fake_doc(doc_id, content="content"):
    """Simula un Document de langchain con metadata y page_content."""
    doc = MagicMock()
    doc.metadata = {"doc_id": doc_id, "title": f"Title {doc_id}"}
    doc.page_content = content
    return doc


# ---------------------------------------------------------------------------
# SVR1: retrieve() sin vector store
# ---------------------------------------------------------------------------
def test_retrieve_no_store_returns_empty():
    """SVR1: uninitialized vector store -> empty RetrievalResult."""
    r = _make_retriever(with_store=False)
    result = r.retrieve("query")
    assert result.doc_ids == []
    assert result.contents == []
    assert result.scores == []
    assert result.strategy_used == RetrievalStrategy.SIMPLE_VECTOR


# ---------------------------------------------------------------------------
# SVR2: retrieve() con resultados
# ---------------------------------------------------------------------------
def test_retrieve_extracts_results():
    """SVR2: vector store returns docs -> correct doc_ids, contents, scores."""
    r = _make_retriever(with_store=True)
    r._vector_store.similarity_search_with_score.return_value = [
        (_fake_doc("d1", "text1"), 0.95),
        (_fake_doc("d2", "text2"), 0.80),
    ]
    result = r.retrieve("query", top_k=5)
    assert result.doc_ids == ["d1", "d2"]
    assert result.contents == ["text1", "text2"]
    assert result.scores == [0.95, 0.80]
    assert result.vector_scores == [0.95, 0.80]
    assert result.strategy_used == RetrievalStrategy.SIMPLE_VECTOR
    assert result.retrieval_time_ms > 0


# ---------------------------------------------------------------------------
# SVR3: retrieve() con excepcion
# ---------------------------------------------------------------------------
def test_retrieve_exception_returns_empty_with_error():
    """SVR3: vector store raises -> empty result with error in metadata."""
    r = _make_retriever(with_store=True)
    r._vector_store.similarity_search_with_score.side_effect = RuntimeError("DB down")
    result = r.retrieve("query")
    assert result.doc_ids == []
    assert "DB down" in result.metadata["error"]


# ---------------------------------------------------------------------------
# SVR4: retrieve_by_vector() sin vector store
# ---------------------------------------------------------------------------
def test_retrieve_by_vector_no_store_returns_empty():
    """SVR4: no vector store -> empty result."""
    r = _make_retriever(with_store=False)
    result = r.retrieve_by_vector("query", [0.1, 0.2, 0.3])
    assert result.doc_ids == []
    assert result.strategy_used == RetrievalStrategy.SIMPLE_VECTOR


# ---------------------------------------------------------------------------
# SVR5: retrieve_by_vector() con resultados
# ---------------------------------------------------------------------------
def test_retrieve_by_vector_extracts_results():
    """SVR5: vector store returns docs via pre-computed embedding."""
    r = _make_retriever(with_store=True)
    r._vector_store.similarity_search_by_vector_with_score.return_value = [
        (_fake_doc("d3", "vec_text"), 0.88),
    ]
    result = r.retrieve_by_vector("query", [0.1, 0.2], top_k=3)
    assert result.doc_ids == ["d3"]
    assert result.contents == ["vec_text"]
    assert result.scores == [0.88]
    assert result.retrieval_time_ms > 0


# ---------------------------------------------------------------------------
# SVR6: index_documents() exitoso
# ---------------------------------------------------------------------------
@patch("shared.retrieval.core.SimpleVectorRetriever._init_vector_store")
def test_index_documents_success(mock_init):
    """SVR6: index_documents -> creates Documents, calls add_documents, returns True."""
    r = _make_retriever(with_store=False)

    # _init_vector_store es patched; simulamos que crea el store
    def fake_init(name):
        r._vector_store = MagicMock()

    mock_init.side_effect = fake_init

    docs = [
        {"doc_id": "d1", "content": "Hello", "title": "T1"},
        {"doc_id": "d2", "content": "World", "title": "T2"},
    ]
    result = r.index_documents(docs, collection_name="my_col")
    assert result is True
    assert r._is_indexed is True

    # Verify add_documents was called with 2 documents
    # Document is from langchain_core (possibly mocked by conftest),
    # so we verify the call happened with correct count.
    r._vector_store.add_documents.assert_called_once()
    call_args = r._vector_store.add_documents.call_args[0][0]
    assert len(call_args) == 2

    # Verify _init_vector_store received the collection name
    mock_init.assert_called_once_with("my_col")


# ---------------------------------------------------------------------------
# SVR7: index_documents() con excepcion
# ---------------------------------------------------------------------------
@patch("shared.retrieval.core.SimpleVectorRetriever._init_vector_store")
def test_index_documents_exception_returns_false(mock_init):
    """SVR7: exception during indexing -> returns False, no crash."""
    r = _make_retriever(with_store=False)
    mock_init.side_effect = RuntimeError("ChromaDB unavailable")

    result = r.index_documents([{"doc_id": "d1", "content": "text"}])
    assert result is False


# ---------------------------------------------------------------------------
# SVR8: clear_index()
# ---------------------------------------------------------------------------
def test_clear_index_delegates_and_resets():
    """SVR8: clear_index -> calls delete_all_documents, sets _is_indexed=False."""
    r = _make_retriever(with_store=True)
    r._is_indexed = True
    r.clear_index()
    r._vector_store.delete_all_documents.assert_called_once()
    assert r._is_indexed is False


# ---------------------------------------------------------------------------
# SVR9: get_documents_by_ids() sin store
# ---------------------------------------------------------------------------
def test_get_documents_by_ids_no_store():
    """SVR9: no vector store -> empty dict."""
    r = _make_retriever(with_store=False)
    assert r.get_documents_by_ids(["d1", "d2"]) == {}


# ---------------------------------------------------------------------------
# SVR10: get_documents_by_ids() delega
# ---------------------------------------------------------------------------
def test_get_documents_by_ids_delegates():
    """SVR10: delegates to vector_store.get_documents_by_ids."""
    r = _make_retriever(with_store=True)
    r._vector_store.get_documents_by_ids.return_value = {"d1": "content1"}
    result = r.get_documents_by_ids(["d1"])
    assert result == {"d1": "content1"}
    r._vector_store.get_documents_by_ids.assert_called_once_with(["d1"])
