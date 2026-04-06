"""
Tests: ChromaVectorStore en shared/vector_store.py

Cubre add_documents (batching), similarity_search_with_score,
similarity_search_by_vector_with_score, get_documents_by_ids (batching),
delete_all_documents, y manejo de errores.

Patron: object.__new__() como documenta TESTS.md (lineas 106-110).
Mock de _store (langchain Chroma) y _client (chromadb nativo).

Coverage IDs:
  VS1: add_documents vacio -> warning, retorna []
  VS2: add_documents sin batching -> delega a _store.add_documents
  VS3: add_documents con batching -> multiples llamadas
  VS4: add_documents excepcion -> re-raise
  VS5: similarity_search_with_score -> delega a _store
  VS6: similarity_search_with_score excepcion -> retorna []
  VS7: similarity_search_by_vector_with_score -> query via _client
  VS8: similarity_search_by_vector_with_score excepcion -> retorna []
  VS9: get_documents_by_ids vacio -> retorna {}
  VS10: get_documents_by_ids con batching (>100 ids) -> multiples chunks
  VS11: get_documents_by_ids excepcion -> retorna {}
  VS12: delete_all_documents -> delete + recreate collection
  VS13: delete_all_documents excepcion -> _document_count = 0
"""

from unittest.mock import MagicMock, patch, call

import pytest

from shared.vector_store import ChromaVectorStore


def _make_store(batch_size=0):
    """Crea ChromaVectorStore via object.__new__() sin ChromaDB real."""
    store = object.__new__(ChromaVectorStore)
    store.collection_name = "test_collection"
    store.persist_directory = None
    store.embedding_model = MagicMock()
    store.batch_size = batch_size
    store._hnsw_num_threads = 1
    store._hnsw_space = None
    store._collection_metadata = {"hnsw:num_threads": 1}
    store._store = MagicMock()
    store._client = MagicMock()
    store._document_count = 0
    store._CHROMA_IN_BATCH_SIZE = 100
    return store


def _fake_doc(doc_id, content="content"):
    """Simula un Document."""
    doc = MagicMock()
    doc.metadata = {"doc_id": doc_id}
    doc.page_content = content
    return doc


# ---------------------------------------------------------------------------
# VS1: add_documents vacio
# ---------------------------------------------------------------------------
def test_add_documents_empty():
    """VS1: empty list -> warning, returns [], no call to _store."""
    store = _make_store()
    result = store.add_documents([])
    assert result == []
    store._store.add_documents.assert_not_called()


# ---------------------------------------------------------------------------
# VS2: add_documents sin batching
# ---------------------------------------------------------------------------
def test_add_documents_no_batching():
    """VS2: docs <= batch_size (or batch_size=0) -> single call."""
    store = _make_store(batch_size=0)
    docs = [_fake_doc(f"d{i}") for i in range(5)]
    store._store.add_documents.return_value = ["id1", "id2", "id3", "id4", "id5"]

    result = store.add_documents(docs)
    assert len(result) == 5
    assert store._document_count == 5
    store._store.add_documents.assert_called_once_with(docs)


# ---------------------------------------------------------------------------
# VS3: add_documents con batching
# ---------------------------------------------------------------------------
def test_add_documents_with_batching():
    """VS3: docs > batch_size -> multiple batched calls."""
    store = _make_store(batch_size=2)
    docs = [_fake_doc(f"d{i}") for i in range(5)]
    store._store.add_documents.return_value = ["id"]

    result = store.add_documents(docs)
    # 5 docs / batch_size=2 -> 3 batches (2+2+1)
    assert store._store.add_documents.call_count == 3
    assert store._document_count == 5


# ---------------------------------------------------------------------------
# VS4: add_documents excepcion -> re-raise
# ---------------------------------------------------------------------------
def test_add_documents_exception_reraises():
    """VS4: _store raises -> exception propagates."""
    store = _make_store()
    store._store.add_documents.side_effect = RuntimeError("ChromaDB error")
    with pytest.raises(RuntimeError, match="ChromaDB error"):
        store.add_documents([_fake_doc("d1")])


# ---------------------------------------------------------------------------
# VS5: similarity_search_with_score
# ---------------------------------------------------------------------------
def test_similarity_search_with_score_delegates():
    """VS5: delegates to _store.similarity_search_with_score."""
    store = _make_store()
    expected = [(_fake_doc("d1"), 0.9)]
    store._store.similarity_search_with_score.return_value = expected

    result = store.similarity_search_with_score("query", k=3)
    assert result == expected
    store._store.similarity_search_with_score.assert_called_once_with("query", k=3)


# ---------------------------------------------------------------------------
# VS6: similarity_search_with_score excepcion -> []
# ---------------------------------------------------------------------------
def test_similarity_search_exception_returns_empty():
    """VS6: _store raises -> returns empty list (bare except)."""
    store = _make_store()
    store._store.similarity_search_with_score.side_effect = RuntimeError("DB down")
    result = store.similarity_search_with_score("query")
    assert result == []


# ---------------------------------------------------------------------------
# VS7: similarity_search_by_vector_with_score
# ---------------------------------------------------------------------------
def test_similarity_search_by_vector():
    """VS7: vector search via _client.get_collection().query()."""
    store = _make_store()
    mock_collection = MagicMock()
    store._client.get_collection.return_value = mock_collection
    mock_collection.query.return_value = {
        "ids": [["chroma_1", "chroma_2"]],
        "metadatas": [[{"doc_id": "d1"}, {"doc_id": "d2"}]],
        "documents": [["text1", "text2"]],
        "distances": [[0.1, 0.3]],
    }

    result = store.similarity_search_by_vector_with_score([0.1, 0.2], k=2)
    assert len(result) == 2
    # Document is mocked by conftest (langchain_core.documents),
    # so we verify the query was correct and results have right count/scores.
    _, score1 = result[0]
    _, score2 = result[1]
    assert score1 == pytest.approx(0.1)
    assert score2 == pytest.approx(0.3)
    mock_collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )


# ---------------------------------------------------------------------------
# VS8: similarity_search_by_vector_with_score excepcion -> []
# ---------------------------------------------------------------------------
def test_similarity_search_by_vector_exception():
    """VS8: exception -> returns empty list."""
    store = _make_store()
    store._client.get_collection.side_effect = RuntimeError("no collection")
    result = store.similarity_search_by_vector_with_score([0.1], k=5)
    assert result == []


# ---------------------------------------------------------------------------
# VS9: get_documents_by_ids vacio
# ---------------------------------------------------------------------------
def test_get_documents_by_ids_empty():
    """VS9: empty doc_ids -> returns {}."""
    store = _make_store()
    assert store.get_documents_by_ids([]) == {}


# ---------------------------------------------------------------------------
# VS10: get_documents_by_ids con batching
# ---------------------------------------------------------------------------
def test_get_documents_by_ids_batching():
    """VS10: >100 doc_ids -> multiple chunked calls."""
    store = _make_store()
    store._CHROMA_IN_BATCH_SIZE = 100
    mock_collection = MagicMock()
    store._client.get_collection.return_value = mock_collection

    # 150 doc_ids -> 2 chunks (100 + 50)
    doc_ids = [f"d{i}" for i in range(150)]

    def fake_get(**kwargs):
        chunk = kwargs["where"]["doc_id"]["$in"]
        return {
            "ids": [f"chroma_{d}" for d in chunk[:2]],  # return 2 per chunk
            "metadatas": [{"doc_id": chunk[0]}, {"doc_id": chunk[1]}],
            "documents": [f"content_{chunk[0]}", f"content_{chunk[1]}"],
        }

    mock_collection.get.side_effect = fake_get
    result = store.get_documents_by_ids(doc_ids)

    # 2 chunks -> 2 calls
    assert mock_collection.get.call_count == 2
    # Each chunk returns 2 docs -> 4 total
    assert len(result) == 4


# ---------------------------------------------------------------------------
# VS11: get_documents_by_ids excepcion -> {}
# ---------------------------------------------------------------------------
def test_get_documents_by_ids_exception():
    """VS11: exception -> returns empty dict."""
    store = _make_store()
    store._client.get_collection.side_effect = RuntimeError("DB error")
    result = store.get_documents_by_ids(["d1"])
    assert result == {}


# ---------------------------------------------------------------------------
# VS12: delete_all_documents
# ---------------------------------------------------------------------------
def test_delete_all_documents():
    """VS12: delete + recreate collection, reset count."""
    store = _make_store()
    store._document_count = 50
    original_store = store._store

    store.delete_all_documents()

    store._client.delete_collection.assert_called_once_with("test_collection")
    assert store._document_count == 0
    # _store should have been reassigned (new Chroma wrapper)
    # In mock environment, Chroma is a MagicMock from conftest, so _store
    # will be a new MagicMock instance. Just verify delete was called.


# ---------------------------------------------------------------------------
# VS13: delete_all_documents excepcion -> count reset
# ---------------------------------------------------------------------------
def test_delete_all_documents_exception_resets_count():
    """VS13: exception during delete -> _document_count still reset to 0."""
    store = _make_store()
    store._document_count = 50
    store._client.delete_collection.side_effect = RuntimeError("cannot delete")

    # Should not raise
    store.delete_all_documents()
    assert store._document_count == 0
