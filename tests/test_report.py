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


def _make_qr(
    query_id="q1",
    with_gen=True,
    reranked=False,
    graph_meta=False,
    kg_fallback=None,
    synthesis_error=None,
) -> QueryEvaluationResult:
    gen = GenerationResult("answer text", 42.0) if with_gen else None
    metadata = {}
    if reranked:
        metadata["reranked"] = True

    retrieval = _make_retrieval()
    if graph_meta:
        # Claves actuales emitidas por LightRAGRetriever y por
        # GenerationExecutor.synthesis. El retriever ya no emite las claves
        # legacy (graph_candidates, etc.); el guard del exporter usa las
        # vigentes para activar las columnas KG en el CSV.
        retrieval.retrieval_metadata = {
            "lightrag_mode": "hybrid",
            "kg_entities": [{"name": "physics"}, {"name": "quantum"}],
            "kg_relations": [{"source": "physics", "target": "quantum"}],
            "kg_chunk_keyword_matches": 3,
            "kg_synthesis_used": synthesis_error is None,
        }
        if kg_fallback is not None:
            retrieval.retrieval_metadata["kg_fallback"] = kg_fallback
        if synthesis_error is not None:
            retrieval.retrieval_metadata["kg_synthesis_error"] = synthesis_error

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
            "retrieval": {"retrieval_k": 20},
            "reranker": {"top_n": 5, "enabled": False},
            "corpus_shuffle_seed": 42,
            "_runtime": {
                "corpus_indexed": 1000,
                "corpus_total_available": 5000,
                "gen_zero_count": 0,
                "gen_nonzero_count": 2,
            },
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

    def test_retrieval_metadata_serialized_when_lightrag(self, tmp_path):
        """Deuda #15 cerrada: el JSON serializa el subset de retrieval_metadata
        con conteos de entidades/relaciones + flags de synthesis."""
        exporter = RunExporter(output_dir=tmp_path)
        qrs = [_make_qr("q1", graph_meta=True, synthesis_error="timeout")]
        run = _make_run(query_results=qrs)
        path = exporter.to_json(run)

        data = json.loads(path.read_text())
        qr_json = data["query_results"][0]
        assert "retrieval_metadata" in qr_json
        rm = qr_json["retrieval_metadata"]
        assert rm["lightrag_mode"] == "hybrid"
        assert rm["kg_entities_count"] == 2
        assert rm["kg_relations_count"] == 1
        assert rm["kg_chunk_keyword_matches"] == 3
        assert rm["kg_synthesis_used"] is False
        assert rm["kg_synthesis_error"] == "timeout"

    def test_retrieval_metadata_absent_for_simple_vector(self, tmp_path):
        """Regresion guard: sin retrieval_metadata LightRAG, la clave
        retrieval_metadata no aparece en el JSON de la query."""
        exporter = RunExporter(output_dir=tmp_path)
        qrs = [_make_qr("q1", graph_meta=False)]
        run = _make_run(query_results=qrs)
        path = exporter.to_json(run)

        data = json.loads(path.read_text())
        qr_json = data["query_results"][0]
        assert "retrieval_metadata" not in qr_json

    def test_retrieval_metadata_with_kg_fallback(self, tmp_path):
        """kg_fallback (p.ej. 'no_doc_ids') se preserva per-query en JSON."""
        exporter = RunExporter(output_dir=tmp_path)
        qrs = [_make_qr("q1", graph_meta=True, kg_fallback="no_doc_ids")]
        run = _make_run(query_results=qrs)
        path = exporter.to_json(run)

        data = json.loads(path.read_text())
        rm = data["query_results"][0]["retrieval_metadata"]
        assert rm["kg_fallback"] == "no_doc_ids"

    def test_retrieval_metadata_new_observables_in_json(self, tmp_path):
        """Divergencias #9 y #4+5: los nuevos observables llegan al JSON."""
        exporter = RunExporter(output_dir=tmp_path)
        qr = _make_qr("q1", graph_meta=True)
        qr.retrieval.retrieval_metadata["kg_entities_with_neighbors"] = 2
        qr.retrieval.retrieval_metadata["kg_mean_neighbors_per_entity"] = 3.5
        qr.retrieval.retrieval_metadata["kg_budget_cap_triggered"] = True
        run = _make_run(query_results=[qr])
        path = exporter.to_json(run)

        data = json.loads(path.read_text())
        rm = data["query_results"][0]["retrieval_metadata"]
        assert rm["kg_entities_with_neighbors"] == 2
        assert rm["kg_mean_neighbors_per_entity"] == 3.5
        assert rm["kg_budget_cap_triggered"] is True

    def test_citation_refs_fields_in_json(self, tmp_path):
        """Divergencia #7: los 14 campos citation_refs_{synth,gen}_* llegan al JSON."""
        exporter = RunExporter(output_dir=tmp_path)
        qr = _make_qr("q1", graph_meta=True)
        # Los 7 synth + 7 gen.
        synth_values = {
            "total": 5, "valid": 4, "malformed": 1, "in_range": 3,
            "out_of_range": 1, "distinct": 2, "coverage_ratio": 0.667,
        }
        gen_values = {
            "total": 2, "valid": 2, "malformed": 0, "in_range": 2,
            "out_of_range": 0, "distinct": 2, "coverage_ratio": 1.0,
        }
        for k, v in synth_values.items():
            qr.retrieval.retrieval_metadata[f"citation_refs_synth_{k}"] = v
        for k, v in gen_values.items():
            qr.retrieval.retrieval_metadata[f"citation_refs_gen_{k}"] = v

        run = _make_run(query_results=[qr])
        path = exporter.to_json(run)

        data = json.loads(path.read_text())
        rm = data["query_results"][0]["retrieval_metadata"]
        for k, v in synth_values.items():
            assert rm[f"citation_refs_synth_{k}"] == v
        for k, v in gen_values.items():
            assert rm[f"citation_refs_gen_{k}"] == v

    def test_citation_refs_absent_when_not_emitted(self, tmp_path):
        """Regresion guard: SIMPLE_VECTOR o LIGHT_RAG sin synthesis no
        inflan el JSON con 14 campos en 0."""
        exporter = RunExporter(output_dir=tmp_path)
        qr = _make_qr("q1", graph_meta=True)  # KG data pero sin citation_refs_*
        run = _make_run(query_results=[qr])
        path = exporter.to_json(run)

        data = json.loads(path.read_text())
        rm = data["query_results"][0]["retrieval_metadata"]
        # Ninguno de los 14 campos debe aparecer en el JSON si no estaban
        # en el retrieval_metadata original.
        for prefix in ("synth", "gen"):
            for k in ("total", "valid", "malformed", "in_range",
                     "out_of_range", "distinct", "coverage_ratio"):
                assert f"citation_refs_{prefix}_{k}" not in rm


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
        """Con retrieval_metadata LightRAG presente, detail.csv expone las
        columnas KG actuales."""
        exporter = RunExporter(output_dir=tmp_path)
        qrs = [_make_qr("q1", graph_meta=True), _make_qr("q2", graph_meta=True)]
        run = _make_run(query_results=qrs)
        path = exporter.to_detail_csv(run)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Claves actuales del retriever/synthesis
        assert "lightrag_mode" in rows[0]
        assert "kg_entities_count" in rows[0]
        assert "kg_relations_count" in rows[0]
        assert "kg_chunk_keyword_matches" in rows[0]
        assert "kg_synthesis_used" in rows[0]
        assert "kg_synthesis_error" in rows[0]
        assert "kg_fallback" in rows[0]
        assert rows[0]["lightrag_mode"] == "hybrid"
        assert rows[0]["kg_entities_count"] == "2"
        assert rows[0]["kg_relations_count"] == "1"
        assert rows[0]["kg_chunk_keyword_matches"] == "3"
        assert rows[0]["kg_synthesis_used"] == "true"
        # Claves legacy eliminadas: `graph_candidates` ya no aparece.
        assert "graph_candidates" not in rows[0]

    def test_lightrag_columns_absent_for_simple_vector_run(self, tmp_path):
        """Regresion guard: sin retrieval_metadata LightRAG, ninguna columna KG
        aparece en el header (SIMPLE_VECTOR run)."""
        exporter = RunExporter(output_dir=tmp_path)
        qrs = [_make_qr("q1", graph_meta=False), _make_qr("q2", graph_meta=False)]
        run = _make_run(query_results=qrs)
        path = exporter.to_detail_csv(run)

        with open(path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        assert "lightrag_mode" not in fieldnames
        assert "kg_fallback" not in fieldnames
        assert "kg_entities_count" not in fieldnames

    def test_lightrag_fallback_and_error_serialized(self, tmp_path):
        """kg_fallback y kg_synthesis_error se escriben en el CSV cuando existen."""
        exporter = RunExporter(output_dir=tmp_path)
        qrs = [
            _make_qr(
                "q_timeout", graph_meta=True,
                kg_fallback=None, synthesis_error="timeout",
            ),
            _make_qr(
                "q_fallback", graph_meta=True,
                kg_fallback="no_doc_ids", synthesis_error=None,
            ),
        ]
        run = _make_run(query_results=qrs)
        path = exporter.to_detail_csv(run)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["kg_synthesis_error"] == "timeout"
        assert rows[0]["kg_synthesis_used"] == "false"
        assert rows[1]["kg_fallback"] == "no_doc_ids"
        assert rows[1]["kg_synthesis_used"] == "true"

    def test_new_observables_serialized(self, tmp_path):
        """Observables nuevos de divergencias #9 y #4+5 llegan al CSV."""
        exporter = RunExporter(output_dir=tmp_path)
        qr = _make_qr("q1", graph_meta=True)
        qr.retrieval.retrieval_metadata["kg_entities_with_neighbors"] = 3
        qr.retrieval.retrieval_metadata["kg_mean_neighbors_per_entity"] = 1.5
        qr.retrieval.retrieval_metadata["kg_budget_cap_triggered"] = False
        run = _make_run(query_results=[qr])
        path = exporter.to_detail_csv(run)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["kg_entities_with_neighbors"] == "3"
        assert rows[0]["kg_mean_neighbors_per_entity"] == "1.5"
        assert rows[0]["kg_budget_cap_triggered"] == "false"

    def test_citation_refs_fields_in_csv(self, tmp_path):
        """Divergencia #7: los 14 campos citation_refs_{synth,gen}_* llegan al CSV."""
        exporter = RunExporter(output_dir=tmp_path)
        qr = _make_qr("q1", graph_meta=True)
        synth_values = {
            "total": 5, "valid": 4, "malformed": 1, "in_range": 3,
            "out_of_range": 1, "distinct": 2, "coverage_ratio": 0.667,
        }
        gen_values = {
            "total": 2, "valid": 2, "malformed": 0, "in_range": 2,
            "out_of_range": 0, "distinct": 2, "coverage_ratio": 1.0,
        }
        for k, v in synth_values.items():
            qr.retrieval.retrieval_metadata[f"citation_refs_synth_{k}"] = v
        for k, v in gen_values.items():
            qr.retrieval.retrieval_metadata[f"citation_refs_gen_{k}"] = v

        run = _make_run(query_results=[qr])
        path = exporter.to_detail_csv(run)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        row = rows[0]
        # Los 14 campos estan presentes con sus valores stringificados.
        assert row["citation_refs_synth_total"] == "5"
        assert row["citation_refs_synth_valid"] == "4"
        assert row["citation_refs_synth_malformed"] == "1"
        assert row["citation_refs_synth_out_of_range"] == "1"
        assert row["citation_refs_synth_coverage_ratio"] == "0.667"
        assert row["citation_refs_gen_total"] == "2"
        assert row["citation_refs_gen_out_of_range"] == "0"
        assert row["citation_refs_gen_coverage_ratio"] == "1.0"

    def test_citation_refs_absent_columns_when_not_emitted(self, tmp_path):
        """Queries sin los campos citation_refs_* obtienen string vacio
        en esas 14 columnas (no ``0``, para discriminar 'no aplica' de
        'cero validos')."""
        exporter = RunExporter(output_dir=tmp_path)
        qr = _make_qr("q1", graph_meta=True)  # KG meta pero sin citation_refs_*
        run = _make_run(query_results=[qr])
        path = exporter.to_detail_csv(run)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        row = rows[0]
        # Las columnas existen en el header (por que hay LightRAG data) pero
        # las celdas de los 14 campos nuevos estan vacias.
        for prefix in ("synth", "gen"):
            for k in ("total", "valid", "malformed", "in_range",
                     "out_of_range", "distinct", "coverage_ratio"):
                assert row[f"citation_refs_{prefix}_{k}"] == ""


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
