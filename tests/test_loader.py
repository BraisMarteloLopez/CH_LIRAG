"""Tests unitarios para MinIOLoader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sandbox_mteb.config import MinIOStorageConfig
from sandbox_mteb.loader import MinIOLoader
from shared.types import DatasetType, LoadedDataset, MetricType


class _FakeClientError(Exception):
    """Simula botocore.exceptions.ClientError. conftest mockea botocore
    como MagicMock y `except MagicMock` no funciona."""

    def __init__(self, error_response, operation_name):
        self.response = error_response
        self.operation_name = operation_name
        super().__init__(f"{operation_name}: {error_response}")


def _storage_config():
    return MinIOStorageConfig(
        minio_endpoint="http://fake:9000",
        minio_access_key="test",
        minio_secret_key="test",
        minio_bucket="test-bucket",
        s3_datasets_prefix="datasets/eval",
        datasets_cache_dir=Path("/tmp/test_cache"),
    )


class _MockDataFrame:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def iterrows(self):
        for i, row in enumerate(self._rows):
            yield i, _MockRow(row)


class _MockRow(dict):
    def get(self, key, default=""):
        return super().get(key, default)


def _empty_result():
    return LoadedDataset(
        name="test",
        dataset_type=DatasetType.HYBRID,
        primary_metric=MetricType.F1_SCORE,
    )


def _populate(queries, corpus=None, qrels=None):
    result = _empty_result()
    MinIOLoader._populate_from_dataframes(
        result,
        _MockDataFrame(queries) if queries is not None else None,
        _MockDataFrame(corpus) if corpus is not None else None,
        _MockDataFrame(qrels) if qrels is not None else None,
    )
    return result


# =============================================================================
# check_connection
# =============================================================================

@patch("sandbox_mteb.loader.boto3")
def test_check_connection_success(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    assert MinIOLoader(_storage_config()).check_connection() is True
    mock_client.head_bucket.assert_called_once()


@patch("sandbox_mteb.loader.boto3")
@patch("sandbox_mteb.loader.ClientError", _FakeClientError)
def test_check_connection_failure(mock_boto3):
    mock_client = MagicMock()
    mock_client.head_bucket.side_effect = _FakeClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
    )
    mock_boto3.client.return_value = mock_client

    from tests.helpers import make_loader
    assert make_loader(mock_client=mock_client).check_connection() is False


# =============================================================================
# _populate_from_dataframes
# =============================================================================

def test_populate_basic():
    result = _populate(
        queries=[
            {"query_id": "q1", "text": "What is AI?", "answer": "Artificial Intelligence",
             "answer_type": "text", "question_type": "bridge", "level": "easy"},
            {"query_id": "q2", "text": "Is Python good?", "answer": "yes",
             "answer_type": "label", "question_type": "comparison", "level": "medium"},
        ],
        corpus=[
            {"doc_id": "d1", "title": "AI Intro", "text": "AI is the simulation of intelligence."},
            {"doc_id": "d2", "title": "Python", "text": "Python is a programming language."},
        ],
        qrels=[{"query_id": "q1", "doc_id": "d1"}, {"query_id": "q2", "doc_id": "d2"}],
    )
    assert [q.query_id for q in result.queries] == ["q1", "q2"]
    assert result.queries[0].query_text == "What is AI?"
    assert result.queries[0].expected_answer == "Artificial Intelligence"
    assert [q.answer_type for q in result.queries] == ["text", "label"]
    assert result.corpus["d1"].title == "AI Intro"
    assert "simulation" in result.corpus["d1"].content
    assert result.queries[0].relevant_doc_ids == ["d1"]
    assert result.queries[1].relevant_doc_ids == ["d2"]
    assert result.total_queries == 2
    assert result.total_corpus == 2


def test_populate_multiple_qrels_per_query():
    result = _populate(
        queries=[{"query_id": "q1", "text": "Multi-hop"}],
        corpus=[{"doc_id": "d1", "title": "A", "text": "CA"},
                {"doc_id": "d2", "title": "B", "text": "CB"}],
        qrels=[{"query_id": "q1", "doc_id": "d1"}, {"query_id": "q1", "doc_id": "d2"}],
    )
    assert set(result.queries[0].relevant_doc_ids) == {"d1", "d2"}


@pytest.mark.parametrize("queries,corpus,qrels", [
    (None, None, None),
    ([], [], []),
])
def test_populate_empty_or_none_no_crash(queries, corpus, qrels):
    result = _populate(queries, corpus, qrels)
    assert len(result.queries) == 0
    assert len(result.corpus) == 0


def test_populate_answer_type_inferred_when_missing():
    """answer presente sin answer_type → 'text'."""
    result = _populate([{"query_id": "q1", "text": "Q?", "answer": "Some answer"}], [])
    assert result.queries[0].answer_type == "text"


def test_populate_no_answer_no_answer_type():
    result = _populate([{"query_id": "q1", "text": "Q?"}], [])
    assert result.queries[0].expected_answer is None
    assert result.queries[0].answer_type is None


@pytest.mark.parametrize("row,expected_type", [
    ({"query_id": "q1", "text": "Q?", "question_type": "bridge", "level": "hard"}, "bridge"),
    ({"query_id": "q1", "text": "Q?", "type": "comparison"}, "comparison"),
])
def test_populate_question_type_metadata(row, expected_type):
    """question_type (o fallback a 'type') se guarda en metadata."""
    result = _populate([row], [])
    assert result.queries[0].metadata["question_type"] == expected_type


def test_populate_question_type_level_captured():
    result = _populate(
        [{"query_id": "q1", "text": "Q?", "question_type": "bridge", "level": "hard"}], []
    )
    assert result.queries[0].metadata["level"] == "hard"


def test_populate_query_without_qrels_empty_relevant_ids():
    result = _populate([{"query_id": "q1", "text": "Orphan"}], [], [])
    assert result.queries[0].relevant_doc_ids == []


@pytest.mark.parametrize("question_type,initial_answer_type,expected", [
    ("comparison", "text", "label"),   # comparison fuerza label
    ("comparison", "label", "label"),  # idempotente
    ("bridge", "text", "text"),        # no-comparison preserva
])
def test_populate_comparison_forces_label(question_type, initial_answer_type, expected):
    result = _populate([{
        "query_id": "q1", "text": "Q?", "answer": "a",
        "answer_type": initial_answer_type, "question_type": question_type,
    }], [])
    assert result.queries[0].answer_type == expected


# =============================================================================
# load_dataset error path
# =============================================================================

@patch("sandbox_mteb.loader.boto3")
@patch("sandbox_mteb.loader.ClientError", _FakeClientError)
def test_load_dataset_download_error(mock_boto3):
    mock_client = MagicMock()
    mock_client.get_object.side_effect = _FakeClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject"
    )
    mock_boto3.client.return_value = mock_client

    result = MinIOLoader(_storage_config()).load_dataset("nonexistent", use_cache=False)
    assert result.load_status == "error"
    assert result.error_message is not None
