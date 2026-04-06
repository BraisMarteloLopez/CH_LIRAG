"""
Tests unitarios para shared/llm.py (Fase I.2).

Cobertura:
  L1. LLMMetrics.record_request incrementa contadores
  L2. LLMMetrics.avg_latency_ms y success_rate
  L3. LLMMetrics.summary formato
  L4. _strip_thinking_tags via invoke (regex en _invoke_with_retry)
  L5. _ThinkingExhaustedError cuando respuesta es solo <think>
  L6. invoke_async construye mensajes correctamente
  L7. invoke sync wrapper llama invoke_async
  L8. load_embedding_model valida parametros
  L9. Retry logic: max_retries intentos antes de fallar
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from shared.llm import LLMMetrics, run_sync


# =============================================================================
# L1-L3: LLMMetrics
# =============================================================================

def test_metrics_record_request_success():
    """record_request incrementa successful y total."""
    m = LLMMetrics()
    run_sync(m.record_request(True, 100.0, 0))
    assert m.total_requests == 1
    assert m.successful_requests == 1
    assert m.failed_requests == 0
    assert m.total_latency_ms == 100.0


def test_metrics_record_request_failure():
    """record_request incrementa failed y retries."""
    m = LLMMetrics()
    run_sync(m.record_request(False, 200.0, 2))
    assert m.total_requests == 1
    assert m.failed_requests == 1
    assert m.retries_total == 2


def test_metrics_multiple_requests():
    """Multiples requests acumulan correctamente."""
    m = LLMMetrics()
    run_sync(m.record_request(True, 100.0, 0))
    run_sync(m.record_request(True, 200.0, 1))
    run_sync(m.record_request(False, 300.0, 2))

    assert m.total_requests == 3
    assert m.successful_requests == 2
    assert m.failed_requests == 1
    assert m.retries_total == 3


def test_metrics_avg_latency():
    """avg_latency_ms calcula correctamente."""
    m = LLMMetrics()
    run_sync(m.record_request(True, 100.0))
    run_sync(m.record_request(True, 300.0))
    assert m.avg_latency_ms == 200.0


def test_metrics_avg_latency_zero_requests():
    """avg_latency_ms con 0 requests retorna 0."""
    m = LLMMetrics()
    assert m.avg_latency_ms == 0


def test_metrics_success_rate():
    """success_rate calcula correctamente."""
    m = LLMMetrics()
    run_sync(m.record_request(True, 50.0))
    run_sync(m.record_request(False, 50.0))
    assert m.success_rate == 0.5


def test_metrics_success_rate_zero_requests():
    """success_rate con 0 requests retorna 0."""
    m = LLMMetrics()
    assert m.success_rate == 0


def test_metrics_summary_format():
    """summary() produce string con formato esperado."""
    m = LLMMetrics()
    run_sync(m.record_request(True, 100.0, 1))
    s = m.summary()
    assert "Requests: 1" in s
    assert "100.0%" in s
    assert "100ms" in s
    assert "Retries: 1" in s


# =============================================================================
# L4-L5: Thinking tag stripping (via regex en _invoke_with_retry)
# =============================================================================

@patch("shared.llm.HAS_NVIDIA", True)
@patch("shared.llm.ChatNVIDIA")
def test_strip_thinking_tags(mock_chat_cls):
    """Respuesta con <think>...</think> se limpia."""
    from shared.llm import AsyncLLMService

    mock_response = MagicMock()
    mock_response.content = '<think>reasoning here</think>{"entities": []}'

    mock_client = AsyncMock()
    mock_client.ainvoke = AsyncMock(return_value=mock_response)
    mock_chat_cls.return_value = mock_client

    service = AsyncLLMService(
        base_url="http://fake:8000/v1",
        model_name="test",
        max_retries=0,
    )
    result = run_sync(service.invoke_async("test prompt"))
    assert "<think>" not in result
    assert '{"entities": []}' in result


@patch("shared.llm.HAS_NVIDIA", True)
@patch("shared.llm.ChatNVIDIA")
def test_strip_unclosed_thinking_tags(mock_chat_cls):
    """Respuesta con <think> sin cerrar (truncada) se limpia."""
    from shared.llm import AsyncLLMService

    mock_response = MagicMock()
    mock_response.content = "<think>reasoning without closing tag"

    mock_client = AsyncMock()
    mock_client.ainvoke = AsyncMock(return_value=mock_response)
    mock_chat_cls.return_value = mock_client

    service = AsyncLLMService(
        base_url="http://fake:8000/v1",
        model_name="test",
        max_retries=0,
    )
    # Should raise because after stripping, content is empty
    with pytest.raises(RuntimeError, match="Inferencia fallida"):
        run_sync(service.invoke_async("test prompt"))


# =============================================================================
# L6-L7: invoke_async / invoke
# =============================================================================

@patch("shared.llm.HAS_NVIDIA", True)
@patch("shared.llm.ChatNVIDIA")
def test_invoke_async_with_system_prompt(mock_chat_cls):
    """invoke_async con system_prompt construye [SystemMessage, HumanMessage]."""
    from shared.llm import AsyncLLMService

    mock_response = MagicMock()
    mock_response.content = "response text"

    mock_client = AsyncMock()
    mock_client.ainvoke = AsyncMock(return_value=mock_response)
    mock_chat_cls.return_value = mock_client

    service = AsyncLLMService(
        base_url="http://fake:8000/v1",
        model_name="test",
        max_retries=0,
    )
    result = service.invoke("user prompt", system_prompt="system prompt")
    assert result == "response text"

    # Verify messages structure
    call_args = mock_client.ainvoke.call_args
    messages = call_args[0][0]
    assert len(messages) == 2


@patch("shared.llm.HAS_NVIDIA", True)
@patch("shared.llm.ChatNVIDIA")
def test_invoke_async_without_system_prompt(mock_chat_cls):
    """invoke_async sin system_prompt construye solo [HumanMessage]."""
    from shared.llm import AsyncLLMService

    mock_response = MagicMock()
    mock_response.content = "response"

    mock_client = AsyncMock()
    mock_client.ainvoke = AsyncMock(return_value=mock_response)
    mock_chat_cls.return_value = mock_client

    service = AsyncLLMService(
        base_url="http://fake:8000/v1",
        model_name="test",
        max_retries=0,
    )
    result = service.invoke("user prompt")
    assert result == "response"

    messages = mock_client.ainvoke.call_args[0][0]
    assert len(messages) == 1


# =============================================================================
# L8: load_embedding_model validation
# =============================================================================

@patch("shared.llm.HAS_NVIDIA_EMBEDDINGS", False)
def test_load_embedding_model_no_nvidia():
    """load_embedding_model sin langchain-nvidia lanza ImportError."""
    from shared.llm import load_embedding_model
    with pytest.raises(ImportError):
        load_embedding_model("http://fake:8000/v1", "model")


@patch("shared.llm.HAS_NVIDIA_EMBEDDINGS", True)
def test_load_embedding_model_empty_params():
    """load_embedding_model con params vacios lanza ValueError."""
    from shared.llm import load_embedding_model
    with pytest.raises(ValueError, match="base_url y model_name son requeridos"):
        load_embedding_model("", "")


@patch("shared.llm.HAS_NVIDIA_EMBEDDINGS", True)
def test_load_embedding_model_invalid_type():
    """load_embedding_model con model_type invalido lanza ValueError."""
    from shared.llm import load_embedding_model
    with pytest.raises(ValueError, match="no valido"):
        load_embedding_model("http://fake:8000/v1", "model", model_type="invalid")


# =============================================================================
# L9: Retry logic
# =============================================================================

@patch("shared.llm.HAS_NVIDIA", True)
@patch("shared.llm.ChatNVIDIA")
def test_retry_exhaustion(mock_chat_cls):
    """Tras max_retries+1 intentos fallidos, lanza RuntimeError."""
    from shared.llm import AsyncLLMService

    mock_client = AsyncMock()
    mock_client.ainvoke = AsyncMock(side_effect=RuntimeError("NIM down"))
    mock_chat_cls.return_value = mock_client

    service = AsyncLLMService(
        base_url="http://fake:8000/v1",
        model_name="test",
        max_retries=1,
    )
    with pytest.raises(RuntimeError, match="Inferencia fallida tras 2 intentos"):
        run_sync(service.invoke_async("prompt"))

    # Should have called ainvoke 2 times (initial + 1 retry)
    assert mock_client.ainvoke.call_count == 2
