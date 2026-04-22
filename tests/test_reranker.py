"""Tests unitarios para shared/retrieval/reranker.py."""

from unittest.mock import MagicMock, patch

from shared.retrieval.core import RetrievalResult, RetrievalStrategy


def _make_reranker():
    from shared.retrieval.reranker import CrossEncoderReranker

    with patch("shared.retrieval.reranker.NVIDIARerank") as MockRerank:
        MockRerank.return_value = MagicMock()
        return CrossEncoderReranker(
            base_url="http://fake:8000/v1", model_name="test-reranker"
        )


def _make_result(n=5, with_vector_scores=True) -> RetrievalResult:
    doc_ids = [f"d{i}" for i in range(n)]
    return RetrievalResult(
        doc_ids=doc_ids,
        contents=[f"content {d}" for d in doc_ids],
        scores=[1.0 - i * 0.1 for i in range(n)],
        vector_scores=[0.9 - i * 0.05 for i in range(n)] if with_vector_scores else None,
        retrieval_time_ms=10.0,
        strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
    )


def _make_reranked_doc(doc_id, score, with_score=True):
    doc = MagicMock()
    doc.page_content = f"content {doc_id}"
    doc.metadata = {"doc_id": doc_id}
    if with_score:
        doc.metadata["relevance_score"] = score
    return doc


def test_empty_result_passthrough():
    empty = RetrievalResult(
        doc_ids=[], contents=[], scores=[],
        strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
    )
    assert _make_reranker().rerank("query", empty, top_n=5).doc_ids == []


def test_sorts_by_relevance_descending():
    reranker = _make_reranker()
    reranker._reranker.compress_documents.return_value = [
        _make_reranked_doc("d2", 0.3),
        _make_reranked_doc("d0", 0.9),
        _make_reranked_doc("d1", 0.6),
    ]
    result = reranker.rerank("query", _make_result(3), top_n=3)
    assert result.doc_ids == ["d0", "d1", "d2"]
    assert result.scores == [0.9, 0.6, 0.3]


def test_truncates_to_top_n():
    reranker = _make_reranker()
    reranker._reranker.compress_documents.return_value = [
        _make_reranked_doc(f"d{i}", 1.0 - i * 0.1) for i in range(5)
    ]
    assert len(reranker.rerank("query", _make_result(5), top_n=2).doc_ids) == 2


def test_preserves_original_vector_scores():
    reranker = _make_reranker()
    original = _make_result(3, with_vector_scores=True)
    reranker._reranker.compress_documents.return_value = [
        _make_reranked_doc("d1", 0.9),
        _make_reranked_doc("d0", 0.8),
    ]
    result = reranker.rerank("query", original, top_n=2)
    assert result.vector_scores == [original.vector_scores[1], original.vector_scores[0]]


def test_no_vector_scores_returns_none():
    reranker = _make_reranker()
    reranker._reranker.compress_documents.return_value = [_make_reranked_doc("d0", 0.9)]
    result = reranker.rerank("query", _make_result(2, with_vector_scores=False), top_n=1)
    assert result.vector_scores is None


def test_identical_scores():
    reranker = _make_reranker()
    reranker._reranker.compress_documents.return_value = [
        _make_reranked_doc(d, 0.7) for d in ("a", "b", "c")
    ]
    result = reranker.rerank("query", _make_result(3), top_n=3)
    assert len(result.doc_ids) == 3
    assert all(s == 0.7 for s in result.scores)


def test_missing_relevance_score_sinks_to_end():
    reranker = _make_reranker()
    reranker._reranker.compress_documents.return_value = [
        _make_reranked_doc("no_score", 0.0, with_score=False),
        _make_reranked_doc("high", 0.9),
        _make_reranked_doc("mid", 0.5),
    ]
    result = reranker.rerank("query", _make_result(3), top_n=3)
    assert result.doc_ids[-1] == "no_score"
    assert result.scores[-1] == 0.0


def test_error_returns_fallback():
    reranker = _make_reranker()
    reranker._reranker.compress_documents.side_effect = RuntimeError("NIM down")
    result = reranker.rerank("query", _make_result(5), top_n=3)
    assert result.doc_ids == ["d0", "d1", "d2"]
    assert result.metadata["reranked"] is False
    assert "NIM down" in result.metadata["rerank_error"]


def test_success_metadata():
    reranker = _make_reranker()
    reranker._reranker.compress_documents.return_value = [_make_reranked_doc("d0", 0.9)]
    result = reranker.rerank("query", _make_result(2), top_n=1)
    assert result.metadata["reranked"] is True
    assert result.metadata["reranker_model"] == "test-reranker"
    assert "rerank_time_ms" in result.metadata
