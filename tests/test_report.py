"""
Tests unitarios para shared/report.py (Audit Fase 3 — A3.3).

Cobertura:
  X1. to_json() genera JSON valido con estructura correcta
  X2. to_summary_csv() genera CSV con headers y una fila
  X3. to_detail_csv() genera CSV con una fila por query
  X4. to_detail_csv() sin query_results genera archivo vacio
  X5. export() genera los 3 archivos
  X6. to_summary_csv() incluye recall_at_k y ndcg_at_k columns
  X7. to_detail_csv() detecta reranker y LIGHT_RAG columns
"""

import csv
import json
from pathlib import Path

import pytest

from shared.types import (
    DatasetType,
    EvaluationRun,
    EvaluationStatus,
    GenerationResult,
    MetricType,
    QueryEvaluationResult,
    QueryRetrievalDetail,
)
from shared.report import RunExporter


def _make_retrieval(doc_ids=None, expected=None) -> QueryRetrievalDetail:
    doc_ids = doc_ids or ["d1", "d2"]
    return QueryRetrievalDetail(
        retrieved_doc_ids=doc_ids,
        retrieved_contents=["content " + d for d in doc_ids],
        retrieval_scores=[0.9, 0.8],
        expected_doc_ids=expected or ["d1"],
        retrieval_time_ms=10.0,
    )


def _make_qr(query_id="q1", with_gen=True, reranked=False, graph_meta=False) -> QueryEvaluationResult:
    gen = GenerationResult("answer text", 42.0) if with_gen else None
    metadata = {}
    if reranked:
        metadata["reranked"] = True

    retrieval = _make_retrieval()
    if graph_meta:
        retrieval.retrieval_metadata = {
            "graph_candidates": 5,
            "vector_candidates": 10,
            "graph_only_candidates": 2,
            "graph_resolved": 2,
            "query_keywords": {"low": ["physics"]},
        }

    return QueryEvaluationResult(
        query_id=query_id,
        query_text=f"What is {query_id}?",
        dataset_name="hotpotqa",
        dataset_type=DatasetType.HYBRID,
        status=EvaluationStatus.COMPLETED,
        expected_response="expected",
        primary_metric_type=MetricType.F1_SCORE,
        primary_metric_value=0.85,
        secondary_metrics={"exact_match": 1.0},
        metadata=metadata,
        retrieval=retrieval,
        generation=gen,
    )


def _make_run(query_results=None) -> EvaluationRun:
    qrs = query_results if query_results is not None else [_make_qr("q1"), _make_qr("q2")]
    return EvaluationRun(
        run_id="test_run_001",
        dataset_name="hotpotqa",
        embedding_model="nvidia/test-model",
        retrieval_strategy="SIMPLE_VECTOR",
        config_snapshot={
            "retrieval_k": 20,
            "reranker_top_n": 5,
            "corpus_shuffle_seed": 42,
            "corpus_indexed": 1000,
            "corpus_total_available": 5000,
            "gen_zero_count": 0,
            "gen_nonzero_count": 2,
        },
        status=EvaluationStatus.COMPLETED,
        num_queries_evaluated=2,
        num_queries_failed=0,
        total_documents=1000,
        avg_hit_rate_at_5=0.9,
        avg_mrr=0.85,
        avg_recall_at_k={5: 0.8, 10: 0.9},
        avg_ndcg_at_k={5: 0.75, 10: 0.85},
        retrieval_complement_recall_at_k={5: 0.2, 10: 0.1},
        avg_retrieved_count=15.0,
        avg_expected_count=2.0,
        avg_generation_score=0.85,
        execution_time_seconds=120.0,
        query_results=qrs,
    )


class TestToJson:
    """Tests para to_json()."""

    def test_generates_valid_json(self, tmp_path):
        exporter = RunExporter(output_dir=tmp_path)
        run = _make_run()
        path = exporter.to_json(run)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["run_id"] == "test_run_001"
        assert data["dataset_name"] == "hotpotqa"
        assert "query_results" in data

    def test_custom_filename(self, tmp_path):
        exporter = RunExporter(output_dir=tmp_path)
        run = _make_run()
        path = exporter.to_json(run, filename="custom.json")
        assert path.name == "custom.json"


class TestToSummaryCsv:
    """Tests para to_summary_csv()."""

    def test_generates_csv_with_one_row(self, tmp_path):
        exporter = RunExporter(output_dir=tmp_path)
        run = _make_run()
        path = exporter.to_summary_csv(run)

        assert path.exists()
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "test_run_001"
        assert rows[0]["mrr"] == "0.85"

    def test_includes_recall_ndcg_columns(self, tmp_path):
        exporter = RunExporter(output_dir=tmp_path)
        run = _make_run()
        path = exporter.to_summary_csv(run)

        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "recall_at_5" in row
        assert "recall_at_10" in row
        assert "ndcg_at_5" in row
        assert "ndcg_at_10" in row

    def test_config_snapshot_fields(self, tmp_path):
        exporter = RunExporter(output_dir=tmp_path)
        run = _make_run()
        path = exporter.to_summary_csv(run)

        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["retrieval_k"] == "20"
        assert row["corpus_indexed"] == "1000"


class TestToDetailCsv:
    """Tests para to_detail_csv()."""

    def test_one_row_per_query(self, tmp_path):
        exporter = RunExporter(output_dir=tmp_path)
        run = _make_run()
        path = exporter.to_detail_csv(run)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["query_id"] == "q1"
        assert rows[1]["query_id"] == "q2"

    def test_empty_results_creates_empty_file(self, tmp_path):
        exporter = RunExporter(output_dir=tmp_path)
        run = _make_run(query_results=[])
        path = exporter.to_detail_csv(run)

        assert path.exists()
        assert path.stat().st_size == 0

    def test_secondary_metrics_columns(self, tmp_path):
        exporter = RunExporter(output_dir=tmp_path)
        run = _make_run()
        path = exporter.to_detail_csv(run)

        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "sec_exact_match" in row
        assert row["sec_exact_match"] == "1.0"

    def test_lightrag_columns_when_graph_meta(self, tmp_path):
        exporter = RunExporter(output_dir=tmp_path)
        qrs = [_make_qr("q1", graph_meta=True), _make_qr("q2", graph_meta=True)]
        run = _make_run(query_results=qrs)
        path = exporter.to_detail_csv(run)

        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert "graph_candidates" in row
        assert row["graph_candidates"] == "5"


class TestExport:
    """Tests para export()."""

    def test_generates_three_files(self, tmp_path):
        exporter = RunExporter(output_dir=tmp_path)
        run = _make_run()
        paths = exporter.export(run)

        assert "json" in paths
        assert "summary_csv" in paths
        assert "detail_csv" in paths
        for p in paths.values():
            assert p.exists()

    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "nested" / "results"
        exporter = RunExporter(output_dir=out)
        assert out.exists()
