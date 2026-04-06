"""
Tests: QueryRetrievalDetail._calculate_all_metrics() y _calculate_generation_metrics()

Produccion: shared/types.py (lineas 290-335)

Verifica las formulas NDCG@K, MRR, Hit@K, Recall@K y generation_hit/recall
con valores calculados manualmente. Tests puros, sin mocks.

Coverage IDs:
  RM1: Doc relevante en posicion 1 — MRR=1.0, Hit@1=1.0
  RM2: Doc relevante en posicion 3 — MRR=0.333, Hit@1=0.0, Hit@5=1.0
  RM3: expected_doc_ids vacio — todo 0.0, sin division por cero
  RM4: retrieved_doc_ids vacio — __post_init__ no calcula nada
  RM5: Multiples docs relevantes — Recall@K progresivo
  RM6: NDCG con 1 doc relevante en posicion 1 — NDCG@1=1.0
  RM7: NDCG con doc relevante fuera de top-K — NDCG@1=0.0
  RM8: Todos los retrieved son relevantes — Recall=1.0, NDCG=1.0
  RM9: MRR con ningun doc relevante — MRR=0.0
  RM10: generation_hit y generation_recall — subset de expected
  RM11: generation con expected vacio — recall=0.0
  RM12: generation sin overlap — hit=0.0, recall=0.0
"""

import math

import pytest

from shared.types import QueryRetrievalDetail


def _make_detail(
    retrieved_doc_ids,
    expected_doc_ids,
    scores=None,
    generation_doc_ids=None,
):
    """Helper: crea QueryRetrievalDetail con contenidos placeholder."""
    n = len(retrieved_doc_ids)
    return QueryRetrievalDetail(
        retrieved_doc_ids=retrieved_doc_ids,
        retrieved_contents=[f"content_{i}" for i in range(n)],
        retrieval_scores=scores or [1.0 - i * 0.1 for i in range(n)],
        expected_doc_ids=expected_doc_ids,
        generation_doc_ids=generation_doc_ids or [],
        generation_contents=[f"gen_{d}" for d in (generation_doc_ids or [])],
    )


# ---------------------------------------------------------------------------
# RM1: Doc relevante en posicion 1
# ---------------------------------------------------------------------------
def test_relevant_at_position_1():
    """RM1: relevant doc at rank 1 -> MRR=1.0, Hit@1=1.0, NDCG@1=1.0."""
    d = _make_detail(
        retrieved_doc_ids=["d1", "d2", "d3"],
        expected_doc_ids=["d1"],
    )
    assert d.mrr == pytest.approx(1.0)
    assert d.hit_at_k[1] == 1.0
    assert d.hit_at_k[5] == 1.0
    assert d.recall_at_k[1] == pytest.approx(1.0)
    assert d.ndcg_at_k[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RM2: Doc relevante en posicion 3
# ---------------------------------------------------------------------------
def test_relevant_at_position_3():
    """RM2: relevant doc at rank 3 -> MRR=1/3, Hit@1=0, Hit@5=1."""
    d = _make_detail(
        retrieved_doc_ids=["d1", "d2", "d3", "d4", "d5"],
        expected_doc_ids=["d3"],
    )
    assert d.mrr == pytest.approx(1.0 / 3.0)
    assert d.hit_at_k[1] == 0.0
    assert d.hit_at_k[3] == 1.0
    assert d.hit_at_k[5] == 1.0
    assert d.recall_at_k[1] == 0.0
    assert d.recall_at_k[3] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RM3: expected_doc_ids vacio — no division por cero
# ---------------------------------------------------------------------------
def test_empty_expected_no_crash():
    """RM3: empty expected -> __post_init__ skips calculation, defaults 0."""
    d = _make_detail(
        retrieved_doc_ids=["d1", "d2"],
        expected_doc_ids=[],
    )
    # __post_init__ guard: if self.expected_doc_ids is falsy, no calc
    assert d.mrr == 0.0
    assert d.hit_at_k == {}
    assert d.recall_at_k == {}
    assert d.ndcg_at_k == {}


# ---------------------------------------------------------------------------
# RM4: retrieved_doc_ids vacio — no calcula nada
# ---------------------------------------------------------------------------
def test_empty_retrieved_no_crash():
    """RM4: empty retrieved -> skips calculation, defaults 0."""
    d = _make_detail(
        retrieved_doc_ids=[],
        expected_doc_ids=["d1"],
    )
    assert d.mrr == 0.0
    assert d.hit_at_k == {}


# ---------------------------------------------------------------------------
# RM5: Multiples docs relevantes — Recall@K progresivo
# ---------------------------------------------------------------------------
def test_multiple_relevant_progressive_recall():
    """RM5: 2 relevant docs at positions 1 and 5 -> Recall@1=0.5, Recall@5=1.0."""
    d = _make_detail(
        retrieved_doc_ids=["r1", "x1", "x2", "x3", "r2", "x4"],
        expected_doc_ids=["r1", "r2"],
    )
    assert d.mrr == pytest.approx(1.0)  # first relevant at rank 1
    assert d.recall_at_k[1] == pytest.approx(0.5)
    assert d.recall_at_k[3] == pytest.approx(0.5)  # only r1 in top 3
    assert d.recall_at_k[5] == pytest.approx(1.0)  # r1 + r2 in top 5
    assert d.hit_at_k[1] == 1.0  # at least one relevant in top 1


# ---------------------------------------------------------------------------
# RM6: NDCG con 1 doc relevante en posicion 1
# ---------------------------------------------------------------------------
def test_ndcg_perfect_single():
    """RM6: 1 relevant doc at rank 1 -> NDCG@K=1.0 for all K."""
    d = _make_detail(
        retrieved_doc_ids=["r1", "x1", "x2"],
        expected_doc_ids=["r1"],
    )
    # DCG = 1/log2(2) = 1.0, IDCG = 1/log2(2) = 1.0 -> NDCG = 1.0
    assert d.ndcg_at_k[1] == pytest.approx(1.0)
    assert d.ndcg_at_k[5] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RM7: NDCG con doc relevante fuera de top-1
# ---------------------------------------------------------------------------
def test_ndcg_relevant_at_position_2():
    """RM7: relevant at rank 2 -> NDCG@1=0.0, NDCG@3 < 1.0."""
    d = _make_detail(
        retrieved_doc_ids=["x1", "r1", "x2"],
        expected_doc_ids=["r1"],
    )
    assert d.ndcg_at_k[1] == 0.0  # r1 not in top 1
    # DCG@3 = 1/log2(3) ≈ 0.6309, IDCG@3 = 1/log2(2) = 1.0
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert d.ndcg_at_k[3] == pytest.approx(expected_ndcg)


# ---------------------------------------------------------------------------
# RM8: Todos retrieved son relevantes — perfect retrieval
# ---------------------------------------------------------------------------
def test_all_retrieved_are_relevant():
    """RM8: all retrieved docs are relevant -> Recall=1.0, NDCG=1.0."""
    d = _make_detail(
        retrieved_doc_ids=["r1", "r2", "r3"],
        expected_doc_ids=["r1", "r2", "r3"],
    )
    assert d.recall_at_k[3] == pytest.approx(1.0)
    assert d.ndcg_at_k[3] == pytest.approx(1.0)
    assert d.mrr == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RM9: MRR sin ningun doc relevante
# ---------------------------------------------------------------------------
def test_mrr_no_relevant():
    """RM9: no relevant doc in retrieved -> MRR=0.0."""
    d = _make_detail(
        retrieved_doc_ids=["x1", "x2", "x3"],
        expected_doc_ids=["r1"],
    )
    assert d.mrr == 0.0
    assert d.hit_at_k[1] == 0.0
    assert d.hit_at_k[3] == 0.0


# ---------------------------------------------------------------------------
# RM10: generation_hit y generation_recall
# ---------------------------------------------------------------------------
def test_generation_metrics_partial_overlap():
    """RM10: generation_doc_ids has 1 of 2 expected -> hit=1.0, recall=0.5."""
    d = _make_detail(
        retrieved_doc_ids=["d1", "d2"],
        expected_doc_ids=["r1", "r2"],
        generation_doc_ids=["r1", "x1"],
    )
    assert d.generation_hit == 1.0
    assert d.generation_recall == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RM11: generation con expected vacio
# ---------------------------------------------------------------------------
def test_generation_metrics_empty_expected():
    """RM11: empty expected -> generation metrics not calculated (guard)."""
    d = _make_detail(
        retrieved_doc_ids=["d1"],
        expected_doc_ids=[],
        generation_doc_ids=["d1"],
    )
    # __post_init__ guard: if not self.expected_doc_ids -> skip
    assert d.generation_recall == 0.0
    assert d.generation_hit == 0.0


# ---------------------------------------------------------------------------
# RM12: generation sin overlap con expected
# ---------------------------------------------------------------------------
def test_generation_metrics_no_overlap():
    """RM12: generation_doc_ids disjoint from expected -> hit=0, recall=0."""
    d = _make_detail(
        retrieved_doc_ids=["d1"],
        expected_doc_ids=["r1", "r2"],
        generation_doc_ids=["x1", "x2"],
    )
    assert d.generation_hit == 0.0
    assert d.generation_recall == 0.0
