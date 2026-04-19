"""
Tests para cambios de los grupos A y B de la review:

A3. ChromaVectorStore.get_documents_by_ids parte en batches de _CHROMA_IN_BATCH_SIZE.
B1. Reranker propaga vector_scores en path exitoso.
B2. Reranker propaga vector_scores en path de error (fallback).

Nota: A1 (delega al vector store) y A2 (sin store / lista vacia) viven en
test_simple_vector_retriever.py::test_get_documents_by_ids_delegates y
::test_get_documents_by_ids_no_store. El caso "lista vacia" permanece aqui
porque ejercita una rama distinta (store inicializado, input vacio).
"""
from unittest.mock import MagicMock

from shared.retrieval.core import (
    RetrievalResult,
    RetrievalStrategy,
)

from tests.helpers import make_reranker, make_retriever, make_vector_store


# =============================================================================
# Helpers
# =============================================================================

class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def _make_retrieval_result_with_vector_scores(n=6):
    return RetrievalResult(
        doc_ids=[f"doc_{i}" for i in range(n)],
        contents=[f"content_{i}" for i in range(n)],
        scores=[1.0 - i * 0.1 for i in range(n)],
        vector_scores=[0.9 - i * 0.05 for i in range(n)],
        strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
    )


def _make_reranker_with_mock_compress(fake_docs):
    reranker = make_reranker()
    reranker._reranker.compress_documents = MagicMock(return_value=fake_docs)
    return reranker


# =============================================================================
# A2: get_documents_by_ids con lista vacia (store inicializado)
# =============================================================================

def test_get_documents_by_ids_empty_list():
    """Lista vacia retorna {} sin llamar al store."""
    retriever = make_retriever(with_store=True)

    result = retriever.get_documents_by_ids([])
    assert result == {}
    retriever._vector_store.get_documents_by_ids.assert_not_called()


# =============================================================================
# A3: Batching en ChromaVectorStore.get_documents_by_ids
# =============================================================================

def test_chroma_batching():
    """Con >_CHROMA_IN_BATCH_SIZE ids, se hacen multiples llamadas collection.get."""
    store = make_vector_store()
    mock_collection = MagicMock()
    store._client.get_collection.return_value = mock_collection

    # Generar 250 doc_ids (deberia hacer 3 batches: 100+100+50)
    n_docs = 250
    doc_ids = [f"doc_{i}" for i in range(n_docs)]

    # Cada llamada a collection.get retorna subset de docs
    def fake_get(**kwargs):
        chunk = kwargs["where"]["doc_id"]["$in"]
        return {
            "ids": [f"chroma_{i}" for i in range(len(chunk))],
            "documents": [f"content_{did}" for did in chunk],
            "metadatas": [{"doc_id": did} for did in chunk],
        }

    mock_collection.get.side_effect = fake_get

    result = store.get_documents_by_ids(doc_ids)

    # Debe hacer 3 llamadas (100 + 100 + 50)
    assert mock_collection.get.call_count == 3, (
        f"Esperado 3 batches, obtenido {mock_collection.get.call_count}"
    )
    # Debe recuperar todos los docs
    assert len(result) == n_docs, (
        f"Esperado {n_docs} docs, obtenido {len(result)}"
    )


def test_chroma_single_batch():
    """Con <=100 ids, se hace una sola llamada."""
    store = make_vector_store()
    mock_collection = MagicMock()
    store._client.get_collection.return_value = mock_collection

    doc_ids = [f"doc_{i}" for i in range(50)]

    def fake_get(**kwargs):
        chunk = kwargs["where"]["doc_id"]["$in"]
        return {
            "ids": [f"chroma_{i}" for i in range(len(chunk))],
            "documents": [f"content_{did}" for did in chunk],
            "metadatas": [{"doc_id": did} for did in chunk],
        }

    mock_collection.get.side_effect = fake_get

    result = store.get_documents_by_ids(doc_ids)

    assert mock_collection.get.call_count == 1
    assert len(result) == 50


# =============================================================================
# B1: Reranker propaga vector_scores (path exitoso)
# =============================================================================

def test_reranker_propagates_vector_scores():
    """Rerank exitoso preserva vector_scores remapeados por doc_id."""
    # Reranker devuelve docs en orden inverso: doc_2, doc_1, doc_0
    fake_docs = [
        FakeDocument("content_2", {"doc_id": "doc_2", "relevance_score": 0.9}),
        FakeDocument("content_1", {"doc_id": "doc_1", "relevance_score": 0.7}),
        FakeDocument("content_0", {"doc_id": "doc_0", "relevance_score": 0.3}),
    ]

    reranker = _make_reranker_with_mock_compress(fake_docs)
    original = _make_retrieval_result_with_vector_scores(n=6)
    result = reranker.rerank("query", original, top_n=3)

    assert result.vector_scores is not None, "vector_scores no deberia ser None"
    assert len(result.vector_scores) == 3

    # vector_scores originales: doc_0=0.9, doc_1=0.85, doc_2=0.8
    # Reranked order: doc_2, doc_1, doc_0
    assert result.vector_scores[0] == 0.8, (  # doc_2
        f"Esperado 0.8 para doc_2, obtenido {result.vector_scores[0]}"
    )
    assert result.vector_scores[1] == 0.85, (  # doc_1
        f"Esperado 0.85 para doc_1, obtenido {result.vector_scores[1]}"
    )
    assert result.vector_scores[2] == 0.9, (  # doc_0
        f"Esperado 0.9 para doc_0, obtenido {result.vector_scores[2]}"
    )


# =============================================================================
# B2: Reranker propaga vector_scores (path error/fallback)
# =============================================================================

def test_reranker_propagates_vector_scores_on_error():
    """Rerank fallido preserva vector_scores[:top_n] del original."""
    reranker = make_reranker()
    reranker._reranker.compress_documents.side_effect = RuntimeError("API down")

    original = _make_retrieval_result_with_vector_scores(n=6)
    result = reranker.rerank("query", original, top_n=3)

    assert result.metadata.get("reranked") is False
    assert result.vector_scores is not None
    assert result.vector_scores == original.vector_scores[:3], (
        f"Esperado {original.vector_scores[:3]}, obtenido {result.vector_scores}"
    )


def test_reranker_no_vector_scores():
    """Si el original no tiene vector_scores, el resultado tampoco."""
    fake_docs = [
        FakeDocument("c_0", {"doc_id": "doc_0", "relevance_score": 0.9}),
    ]

    reranker = _make_reranker_with_mock_compress(fake_docs)
    original = RetrievalResult(
        doc_ids=["doc_0", "doc_1"],
        contents=["c_0", "c_1"],
        scores=[1.0, 0.9],
        strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
    )

    result = reranker.rerank("query", original, top_n=1)
    assert result.vector_scores is None, (
        f"Esperado None, obtenido {result.vector_scores}"
    )
