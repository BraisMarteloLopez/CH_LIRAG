"""
Tests unitarios para shared/retrieval/reranker.py (Audit Fase 3 — A3.4).

Cobertura:
  K1. rerank() con resultado vacio retorna mismo resultado
  K2. rerank() ordena por relevance_score descendente
  K3. rerank() trunca a top_n
  K4. rerank() preserva vector_scores originales
  K5. rerank() en error retorna fallback sin rerank
  K6. rerank() metadata incluye reranked=True/False
"""

from unittest.mock import MagicMock, patch

import pytest

from shared.retrieval.core import RetrievalResult, RetrievalStrategy


def _make_reranker():
    """Crea CrossEncoderReranker con mock de NVIDIARerank."""
    from shared.retrieval.reranker import CrossEncoderReranker

    with patch("shared.retrieval.reranker.NVIDIARerank") as MockRerank:
        mock_instance = MagicMock()
        MockRerank.return_value = mock_instance
        reranker = CrossEncoderReranker(
            base_url="http://fake:8000/v1",
            model_name="test-reranker",
        )
    return reranker


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


def _make_reranked_doc(doc_id, content, score):
    """Simula un Document retornado por compress_documents."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"doc_id": doc_id, "relevance_score": score}
    return doc


class TestRerankEmpty:
    """K1: resultado vacio."""

    def test_empty_result_passthrough(self):
        reranker = _make_reranker()
        empty = RetrievalResult(
            doc_ids=[], contents=[], scores=[],
            strategy_used=RetrievalStrategy.SIMPLE_VECTOR,
        )
        result = reranker.rerank("query", empty, top_n=5)
        assert result.doc_ids == []


class TestRerankOrdering:
    """K2: ordenamiento por relevance_score."""

    def test_sorts_by_relevance_descending(self):
        reranker = _make_reranker()
        original = _make_result(3)

        # Simulate reranker returning docs in wrong order
        reranker._reranker.compress_documents.return_value = [
            _make_reranked_doc("d2", "content d2", 0.3),
            _make_reranked_doc("d0", "content d0", 0.9),
            _make_reranked_doc("d1", "content d1", 0.6),
        ]

        result = reranker.rerank("query", original, top_n=3)
        assert result.doc_ids == ["d0", "d1", "d2"]
        assert result.scores == [0.9, 0.6, 0.3]


class TestRerankTopN:
    """K3: truncamiento a top_n."""

    def test_truncates_to_top_n(self):
        reranker = _make_reranker()
        original = _make_result(5)

        reranker._reranker.compress_documents.return_value = [
            _make_reranked_doc(f"d{i}", f"content d{i}", 1.0 - i * 0.1)
            for i in range(5)
        ]

        result = reranker.rerank("query", original, top_n=2)
        assert len(result.doc_ids) == 2


class TestRerankVectorScores:
    """K4: preserva vector_scores."""

    def test_preserves_original_vector_scores(self):
        reranker = _make_reranker()
        original = _make_result(3, with_vector_scores=True)

        reranker._reranker.compress_documents.return_value = [
            _make_reranked_doc("d1", "content d1", 0.9),
            _make_reranked_doc("d0", "content d0", 0.8),
        ]

        result = reranker.rerank("query", original, top_n=2)
        # d1 had vector_score 0.85, d0 had 0.9
        assert result.vector_scores is not None
        assert result.vector_scores[0] == original.vector_scores[1]  # d1's score
        assert result.vector_scores[1] == original.vector_scores[0]  # d0's score

    def test_no_vector_scores_returns_none(self):
        reranker = _make_reranker()
        original = _make_result(2, with_vector_scores=False)

        reranker._reranker.compress_documents.return_value = [
            _make_reranked_doc("d0", "content d0", 0.9),
        ]

        result = reranker.rerank("query", original, top_n=1)
        assert result.vector_scores is None


class TestRerankSortEdges:
    """Ordering edge cases (scores identicos / sin relevance_score)."""

    def test_identical_scores(self):
        """Docs con scores identicos no producen error."""
        reranker = _make_reranker()
        reranker._reranker.compress_documents.return_value = [
            _make_reranked_doc("a", "c_a", 0.7),
            _make_reranked_doc("b", "c_b", 0.7),
            _make_reranked_doc("c", "c_c", 0.7),
        ]
        result = reranker.rerank("query", _make_result(3), top_n=3)
        assert len(result.doc_ids) == 3
        assert all(s == 0.7 for s in result.scores)

    def test_missing_relevance_score(self):
        """Doc sin relevance_score en metadata -> default 0.0, queda al final."""
        no_score = MagicMock()
        no_score.page_content = "c_no_score"
        no_score.metadata = {"doc_id": "no_score"}  # sin relevance_score
        reranker = _make_reranker()
        reranker._reranker.compress_documents.return_value = [
            no_score,
            _make_reranked_doc("high", "c_high", 0.9),
            _make_reranked_doc("mid", "c_mid", 0.5),
        ]
        result = reranker.rerank("query", _make_result(3), top_n=3)
        assert result.doc_ids[-1] == "no_score"
        assert result.scores[-1] == 0.0


class TestRerankError:
    """K5/K6: error fallback y metadata."""

    def test_error_returns_fallback(self):
        reranker = _make_reranker()
        original = _make_result(5)

        reranker._reranker.compress_documents.side_effect = RuntimeError("NIM down")

        result = reranker.rerank("query", original, top_n=3)
        # Fallback: top_n from original
        assert len(result.doc_ids) == 3
        assert result.doc_ids == ["d0", "d1", "d2"]
        assert result.metadata["reranked"] is False
        assert "NIM down" in result.metadata["rerank_error"]

    def test_success_metadata(self):
        reranker = _make_reranker()
        original = _make_result(2)

        reranker._reranker.compress_documents.return_value = [
            _make_reranked_doc("d0", "content d0", 0.9),
        ]

        result = reranker.rerank("query", original, top_n=1)
        assert result.metadata["reranked"] is True
        assert result.metadata["reranker_model"] == "test-reranker"
        assert "rerank_time_ms" in result.metadata
