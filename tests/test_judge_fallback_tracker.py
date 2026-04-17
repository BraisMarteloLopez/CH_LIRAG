"""
Tests deuda tecnica #4: instrumentacion de tasa de fallback del LLM judge.

Verifica que el tracker contabiliza correctamente:
  - invocaciones totales
  - fallos de parseo JSON (parse_failures)
  - retornos de 0.5 por defecto (default_returns, evento critico)
y que max_judge_default_return_rate identifica la metrica peor.
"""
from __future__ import annotations

import pytest

from shared.metrics import (
    MetricType,
    _extract_score_fallback,
    _extract_score_fallback_with_status,
    _parse_judge_result,
    get_judge_fallback_stats,
    max_judge_default_return_rate,
    reset_judge_fallback_stats,
)


@pytest.fixture(autouse=True)
def _clean_tracker():
    """Cada test arranca con contadores limpios."""
    reset_judge_fallback_stats()
    yield
    reset_judge_fallback_stats()


# -----------------------------------------------------------------------------
# _extract_score_fallback_with_status
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected_score,expected_default", [
    ("score: 0.85", 0.85, False),          # decimal extraido
    ("I rate this 8/10", 0.8, False),      # fraccion extraida
    ("score: 7", 0.7, False),              # int 1-10 normalizado
    ("I cannot provide a score", 0.5, True),  # default
    ("", 0.5, True),                       # vacio, default
])
def test_extract_with_status_reports_default(text, expected_score, expected_default):
    """El status refleja si el 0.5 viene de regex o de fallback total."""
    score, was_default = _extract_score_fallback_with_status(text)
    assert abs(score - expected_score) < 0.001
    assert was_default is expected_default


def test_extract_wrapper_compat():
    """_extract_score_fallback sigue siendo funcion publica sin cambios."""
    assert _extract_score_fallback("score: 0.8") == 0.8
    assert _extract_score_fallback("no score here") == 0.5


# -----------------------------------------------------------------------------
# Tracker via _parse_judge_result (integracion real)
# -----------------------------------------------------------------------------


def test_tracker_empty_when_no_calls():
    """Sin invocaciones, el snapshot esta vacio."""
    stats = get_judge_fallback_stats()
    assert stats == {}


def test_tracker_counts_valid_json_as_no_fallback():
    """Respuesta JSON valida no cuenta como parse_failure ni default."""
    # Simulamos lo que hace _invoke_judge, pero sin invocar LLM:
    # llamamos directo a _parse_judge_result con un JSON valido y
    # registramos la invocacion manualmente (para reflejar el flujo real).
    from shared.metrics import _judge_fallback_tracker
    _judge_fallback_tracker.record_invocation(MetricType.FAITHFULNESS)

    result = _parse_judge_result(
        '{"score": 0.9, "justification": "OK"}', MetricType.FAITHFULNESS
    )
    assert result.value == 0.9

    stats = get_judge_fallback_stats()
    s = stats["faithfulness"]
    assert s["invocations"] == 1
    assert s["parse_failures"] == 0
    assert s["default_returns"] == 0
    assert s["parse_failure_rate"] == 0.0
    assert s["default_return_rate"] == 0.0


def test_tracker_counts_parse_failure_but_regex_hit():
    """JSON invalido + regex extrae score: parse_failure si, default no."""
    from shared.metrics import _judge_fallback_tracker
    _judge_fallback_tracker.record_invocation(MetricType.FAITHFULNESS)

    # Respuesta no parseable como JSON pero con un decimal extraible
    _parse_judge_result("The score is 0.75 overall.", MetricType.FAITHFULNESS)

    stats = get_judge_fallback_stats()
    s = stats["faithfulness"]
    assert s["invocations"] == 1
    assert s["parse_failures"] == 1
    assert s["default_returns"] == 0


def test_tracker_counts_default_return_as_critical():
    """JSON invalido + regex falla: parse_failure Y default_return."""
    from shared.metrics import _judge_fallback_tracker
    _judge_fallback_tracker.record_invocation(MetricType.FAITHFULNESS)

    result = _parse_judge_result("I cannot evaluate this.", MetricType.FAITHFULNESS)
    assert result.value == 0.5
    assert result.details["fallback_default_used"] is True

    stats = get_judge_fallback_stats()
    s = stats["faithfulness"]
    assert s["invocations"] == 1
    assert s["parse_failures"] == 1
    assert s["default_returns"] == 1
    assert s["parse_failure_rate"] == 1.0
    assert s["default_return_rate"] == 1.0


def test_tracker_rates_are_per_metric():
    """Cada MetricType tiene su propio conjunto de contadores."""
    from shared.metrics import _judge_fallback_tracker

    # 2 faithfulness: 1 OK, 1 default
    _judge_fallback_tracker.record_invocation(MetricType.FAITHFULNESS)
    _parse_judge_result('{"score": 0.8, "justification": "x"}', MetricType.FAITHFULNESS)
    _judge_fallback_tracker.record_invocation(MetricType.FAITHFULNESS)
    _parse_judge_result("no score", MetricType.FAITHFULNESS)

    # 4 answer_relevance: todas OK
    for _ in range(4):
        _judge_fallback_tracker.record_invocation(MetricType.ANSWER_RELEVANCE)
        _parse_judge_result('{"score": 0.5, "justification": "x"}', MetricType.ANSWER_RELEVANCE)

    stats = get_judge_fallback_stats()
    assert stats["faithfulness"]["invocations"] == 2
    assert stats["faithfulness"]["default_returns"] == 1
    assert stats["faithfulness"]["default_return_rate"] == 0.5

    assert stats["answer_relevance"]["invocations"] == 4
    assert stats["answer_relevance"]["default_returns"] == 0
    assert stats["answer_relevance"]["default_return_rate"] == 0.0


def test_reset_clears_all_counters():
    """reset_judge_fallback_stats() limpia todo el estado."""
    from shared.metrics import _judge_fallback_tracker
    _judge_fallback_tracker.record_invocation(MetricType.FAITHFULNESS)
    _parse_judge_result("garbage", MetricType.FAITHFULNESS)

    assert get_judge_fallback_stats() != {}

    reset_judge_fallback_stats()
    assert get_judge_fallback_stats() == {}


# -----------------------------------------------------------------------------
# max_judge_default_return_rate
# -----------------------------------------------------------------------------


def test_max_default_return_rate_empty_stats():
    """Sin stats, retorna (None, 0.0)."""
    worst, rate = max_judge_default_return_rate({})
    assert worst is None
    assert rate == 0.0


def test_max_default_return_rate_identifies_worst_metric():
    """Devuelve la metrica con mayor default_return_rate."""
    stats = {
        "faithfulness": {
            "invocations": 100, "parse_failures": 5, "default_returns": 3,
            "parse_failure_rate": 0.05, "default_return_rate": 0.03,
        },
        "answer_relevance": {
            "invocations": 100, "parse_failures": 1, "default_returns": 1,
            "parse_failure_rate": 0.01, "default_return_rate": 0.01,
        },
    }
    worst, rate = max_judge_default_return_rate(stats)
    assert worst == "faithfulness"
    assert abs(rate - 0.03) < 1e-9


def test_max_default_return_rate_all_zero():
    """Cuando ninguna metrica tiene fallbacks, retorna (None, 0.0)."""
    stats = {
        "faithfulness": {
            "invocations": 50, "parse_failures": 0, "default_returns": 0,
            "parse_failure_rate": 0.0, "default_return_rate": 0.0,
        },
    }
    worst, rate = max_judge_default_return_rate(stats)
    assert worst is None
    assert rate == 0.0


# -----------------------------------------------------------------------------
# Evaluator threshold validation
# -----------------------------------------------------------------------------


def _make_minimal_config(threshold: float):
    from sandbox_mteb.config import MTEBConfig, MinIOStorageConfig
    from shared.config_base import InfraConfig, RerankerConfig
    from shared.retrieval.core import RetrievalConfig, RetrievalStrategy

    return MTEBConfig(
        infra=InfraConfig(
            embedding_base_url="http://fake:8000/v1",
            embedding_model_name="test-model",
            llm_base_url="http://fake:8000/v1",
            llm_model_name="test-llm",
        ),
        storage=MinIOStorageConfig(
            minio_endpoint="http://fake:9000",
            minio_access_key="x",
            minio_secret_key="x",
            minio_bucket="b",
        ),
        retrieval=RetrievalConfig(strategy=RetrievalStrategy.SIMPLE_VECTOR),
        reranker=RerankerConfig(enabled=False),
        judge_fallback_threshold=threshold,
    )


def test_evaluator_threshold_disabled_never_fails():
    """threshold=0.0 desactiva la validacion: no lanza aunque haya fallbacks."""
    from sandbox_mteb.evaluator import MTEBEvaluator
    from shared.metrics import _judge_fallback_tracker

    ev = MTEBEvaluator(_make_minimal_config(threshold=0.0))

    # Simular 10 invocaciones, todas default_returns (100% fallback)
    for _ in range(10):
        _judge_fallback_tracker.record_invocation(MetricType.FAITHFULNESS)
        _judge_fallback_tracker.record_parse_failure(MetricType.FAITHFULNESS)
        _judge_fallback_tracker.record_default_return(MetricType.FAITHFULNESS)

    # No lanza
    ev._validate_judge_fallback_threshold("test_run")


def test_evaluator_threshold_passes_below_limit():
    """Tasa bajo umbral: no lanza."""
    from sandbox_mteb.evaluator import MTEBEvaluator
    from shared.metrics import _judge_fallback_tracker

    ev = MTEBEvaluator(_make_minimal_config(threshold=0.05))

    # 100 invocaciones, 2 default_returns (2% < 5%)
    for i in range(100):
        _judge_fallback_tracker.record_invocation(MetricType.FAITHFULNESS)
        if i < 2:
            _judge_fallback_tracker.record_parse_failure(MetricType.FAITHFULNESS)
            _judge_fallback_tracker.record_default_return(MetricType.FAITHFULNESS)

    ev._validate_judge_fallback_threshold("test_run")  # No lanza


def test_evaluator_threshold_raises_above_limit():
    """Tasa sobre umbral: lanza RuntimeError con mensaje informativo."""
    from sandbox_mteb.evaluator import MTEBEvaluator
    from shared.metrics import _judge_fallback_tracker

    ev = MTEBEvaluator(_make_minimal_config(threshold=0.02))

    # 100 invocaciones, 10 default_returns (10% > 2%)
    for i in range(100):
        _judge_fallback_tracker.record_invocation(MetricType.FAITHFULNESS)
        if i < 10:
            _judge_fallback_tracker.record_parse_failure(MetricType.FAITHFULNESS)
            _judge_fallback_tracker.record_default_return(MetricType.FAITHFULNESS)

    with pytest.raises(RuntimeError):
        ev._validate_judge_fallback_threshold("test_run")


def test_evaluator_threshold_uses_worst_metric():
    """El check se aplica a la metrica peor, no al promedio."""
    from sandbox_mteb.evaluator import MTEBEvaluator
    from shared.metrics import _judge_fallback_tracker

    ev = MTEBEvaluator(_make_minimal_config(threshold=0.05))

    # faithfulness: 100 inv, 10 defaults (10%) -> sobre umbral
    for i in range(100):
        _judge_fallback_tracker.record_invocation(MetricType.FAITHFULNESS)
        if i < 10:
            _judge_fallback_tracker.record_parse_failure(MetricType.FAITHFULNESS)
            _judge_fallback_tracker.record_default_return(MetricType.FAITHFULNESS)

    # answer_relevance: 100 inv, 0 defaults (0%) -> bajo umbral
    for _ in range(100):
        _judge_fallback_tracker.record_invocation(MetricType.ANSWER_RELEVANCE)

    with pytest.raises(RuntimeError, match="faithfulness"):
        ev._validate_judge_fallback_threshold("test_run")
