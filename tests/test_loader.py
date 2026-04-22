"""
Tests unitarios para MinIOLoader.

Cobertura:
  LO1. check_connection exitosa
  LO2. check_connection falla
  LO3-LO12. _populate_from_dataframes: queries, corpus, qrels, edge cases
            (None/empty, answer_type inferido, question_type metadata,
             comparison auto-conversion)
  LO13. load_dataset con error retorna LoadedDataset con status error
"""

from unittest.mock import MagicMock, patch
from pathlib import Path

from shared.types import (
    LoadedDataset,
    DatasetType,
    MetricType,
)
from sandbox_mteb.config import MinIOStorageConfig


# =============================================================================
# Helpers
# =============================================================================

class _FakeClientError(Exception):
    """Excepcion real que simula botocore.exceptions.ClientError.

    Necesaria porque conftest.py mockea botocore como MagicMock cuando
    no esta instalado, y `except MagicMock` no funciona en Python.
    """
    def __init__(self, error_response, operation_name):
        self.response = error_response
        self.operation_name = operation_name
        super().__init__(f"{operation_name}: {error_response}")


def _make_storage_config():
    return MinIOStorageConfig(
        minio_endpoint="http://fake:9000",
        minio_access_key="test",
        minio_secret_key="test",
        minio_bucket="test-bucket",
        s3_datasets_prefix="datasets/eval",
        datasets_cache_dir=Path("/tmp/test_cache"),
    )


class _MockDataFrame:
    """DataFrame minimo compatible con iterrows()."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, _MockRow(row)


class _MockRow(dict):
    """Row que soporta .get() como dict."""

    def get(self, key, default=""):
        return super().get(key, default)


def _make_empty_result():
    return LoadedDataset(
        name="test",
        dataset_type=DatasetType.HYBRID,
        primary_metric=MetricType.F1_SCORE,
    )


# =============================================================================
# LO1: check_connection exitosa
# =============================================================================

@patch("sandbox_mteb.loader.boto3")
def test_check_connection_success(mock_boto3):
    """head_bucket sin error -> True."""
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    from sandbox_mteb.loader import MinIOLoader
    loader = MinIOLoader(_make_storage_config())
    assert loader.check_connection() is True
    # Verify head_bucket was called (connection was actually checked)
    mock_client.head_bucket.assert_called_once()


# =============================================================================
# LO2: check_connection falla
# =============================================================================

@patch("sandbox_mteb.loader.boto3")
@patch("sandbox_mteb.loader.ClientError", _FakeClientError)
def test_check_connection_failure(mock_boto3):
    """head_bucket lanza ClientError -> False."""
    mock_client = MagicMock()
    mock_client.head_bucket.side_effect = _FakeClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
    )
    mock_boto3.client.return_value = mock_client

    from tests.helpers import make_loader
    loader = make_loader(mock_client=mock_client)

    assert loader.check_connection() is False


# =============================================================================
# LO3-LO12: _populate_from_dataframes (queries, corpus, qrels, edge cases)
# =============================================================================

def test_populate_basic():
    """Popula queries, corpus y qrels correctamente."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _MockDataFrame([
        {"query_id": "q1", "text": "What is AI?", "answer": "Artificial Intelligence",
         "answer_type": "text", "question_type": "bridge", "level": "easy"},
        {"query_id": "q2", "text": "Is Python good?", "answer": "yes",
         "answer_type": "label", "question_type": "comparison", "level": "medium"},
    ])
    corpus_df = _MockDataFrame([
        {"doc_id": "d1", "title": "AI Intro", "text": "AI is the simulation of intelligence."},
        {"doc_id": "d2", "title": "Python", "text": "Python is a programming language."},
    ])
    qrels_df = _MockDataFrame([
        {"query_id": "q1", "doc_id": "d1"},
        {"query_id": "q2", "doc_id": "d2"},
    ])

    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, queries_df, corpus_df, qrels_df)

    assert len(result.queries) == 2
    assert result.queries[0].query_id == "q1"
    assert result.queries[0].query_text == "What is AI?"
    assert result.queries[0].expected_answer == "Artificial Intelligence"
    assert result.queries[0].answer_type == "text"
    assert result.queries[1].answer_type == "label"

    assert len(result.corpus) == 2
    assert result.corpus["d1"].title == "AI Intro"
    assert "simulation" in result.corpus["d1"].content

    assert result.queries[0].relevant_doc_ids == ["d1"]
    assert result.queries[1].relevant_doc_ids == ["d2"]
    assert result.total_queries == 2
    assert result.total_corpus == 2


def test_populate_multiple_qrels_per_query():
    """Una query con multiples docs relevantes."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _MockDataFrame([{"query_id": "q1", "text": "Multi-hop question"}])
    corpus_df = _MockDataFrame([
        {"doc_id": "d1", "title": "Doc A", "text": "Content A"},
        {"doc_id": "d2", "title": "Doc B", "text": "Content B"},
    ])
    qrels_df = _MockDataFrame([
        {"query_id": "q1", "doc_id": "d1"},
        {"query_id": "q1", "doc_id": "d2"},
    ])

    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, queries_df, corpus_df, qrels_df)
    assert set(result.queries[0].relevant_doc_ids) == {"d1", "d2"}


def test_populate_none_dataframes_no_crash():
    """DataFrames None no causan error."""
    from sandbox_mteb.loader import MinIOLoader

    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, None, None, None)
    assert len(result.queries) == 0
    assert len(result.corpus) == 0


def test_populate_empty_dataframes():
    """DataFrames vacios producen listas vacias."""
    from sandbox_mteb.loader import MinIOLoader

    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(
        result, _MockDataFrame([]), _MockDataFrame([]), _MockDataFrame([])
    )
    assert len(result.queries) == 0
    assert len(result.corpus) == 0


def test_populate_answer_type_inferred_when_missing():
    """Si answer_type ausente pero answer presente, se infiere 'text'."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _MockDataFrame([
        {"query_id": "q1", "text": "Question?", "answer": "Some answer"},
    ])
    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, queries_df, _MockDataFrame([]), None)
    assert result.queries[0].answer_type == "text"


def test_populate_no_answer_no_answer_type():
    """Sin answer ni answer_type, expected_answer es None."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _MockDataFrame([{"query_id": "q1", "text": "Question?"}])
    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, queries_df, _MockDataFrame([]), None)
    assert result.queries[0].expected_answer is None
    assert result.queries[0].answer_type is None


def test_populate_question_type_metadata():
    """question_type se guarda en metadata."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _MockDataFrame([
        {"query_id": "q1", "text": "Q?", "question_type": "bridge", "level": "hard"},
    ])
    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, queries_df, _MockDataFrame([]), None)
    assert result.queries[0].metadata["question_type"] == "bridge"
    assert result.queries[0].metadata["level"] == "hard"


def test_populate_question_type_fallback_to_type_field():
    """Si question_type no existe, usa campo 'type'."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _MockDataFrame([
        {"query_id": "q1", "text": "Q?", "type": "comparison"},
    ])
    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, queries_df, _MockDataFrame([]), None)
    assert result.queries[0].metadata["question_type"] == "comparison"


def test_populate_query_without_qrels_empty_relevant_ids():
    """Query sin qrels correspondiente tiene relevant_doc_ids vacio."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _MockDataFrame([{"query_id": "q1", "text": "Orphan query"}])
    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(
        result, queries_df, _MockDataFrame([]), _MockDataFrame([])
    )
    assert result.queries[0].relevant_doc_ids == []


def test_populate_comparison_query_forces_label():
    """question_type=comparison fuerza answer_type=label; ya-label es idempotente."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _MockDataFrame([
        {"query_id": "q1", "text": "Is A taller?", "answer": "yes",
         "answer_type": "text", "question_type": "comparison"},
        {"query_id": "q2", "text": "Is B taller?", "answer": "no",
         "answer_type": "label", "question_type": "comparison"},
    ])
    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, queries_df, _MockDataFrame([]), None)
    assert result.queries[0].answer_type == "label"
    assert result.queries[1].answer_type == "label"


def test_populate_non_comparison_preserves_answer_type():
    """question_type!=comparison no toca answer_type."""
    from sandbox_mteb.loader import MinIOLoader

    queries_df = _MockDataFrame([
        {"query_id": "q1", "text": "What is X?", "answer": "some answer",
         "answer_type": "text", "question_type": "bridge"},
    ])
    result = _make_empty_result()
    MinIOLoader._populate_from_dataframes(result, queries_df, _MockDataFrame([]), None)
    assert result.queries[0].answer_type == "text"


# =============================================================================
# LO13: load_dataset con error
# =============================================================================

@patch("sandbox_mteb.loader.boto3")
@patch("sandbox_mteb.loader.ClientError", _FakeClientError)
def test_load_dataset_download_error(mock_boto3):
    """Error en descarga -> LoadedDataset con status error."""
    mock_client = MagicMock()
    mock_client.get_object.side_effect = _FakeClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject"
    )
    mock_boto3.client.return_value = mock_client

    from sandbox_mteb.loader import MinIOLoader
    loader = MinIOLoader(_make_storage_config())

    result = loader.load_dataset("nonexistent_dataset", use_cache=False)
    assert result.load_status == "error"
    assert result.error_message is not None
