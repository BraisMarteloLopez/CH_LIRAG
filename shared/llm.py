"""
Modulo: LLM & Embeddings Service
Descripcion: Servicios NIM para inferencia LLM y embeddings.

Ubicacion: shared/llm.py

Consolida llm.py + embeddings.py.

Notas de diseno:
  - _PersistentLoop: un unico event loop en thread daemon, elimina el ciclo
    crear/destruir loops de asyncio.run() que causaba "Semaphore bound to a
    different event loop".
  - LLMMetrics._lock: threading.Lock en vez de asyncio.Lock (no hay awaits
    dentro de la seccion critica, y elimina el binding a event loop).
  - run_sync() usa run_coroutine_threadsafe al loop persistente.
  - load_embedding_model acepta solo parametros explicitos (sin fallback).

Contrato externo (NVIDIA NIM, API compatible OpenAI):
  - POST {LLM_BASE_URL}/chat/completions — cuerpo `{model, messages,
    temperature, max_tokens}`; respuesta `choices[0].message.content`.
  - GET  {LLM_BASE_URL}/models — `data[0].max_model_len` en tokens;
    consumido por `embedding_service.resolve_max_context_chars` (deuda
    [#5](../CLAUDE.md#dt-5)).
  - POST {EMBEDDING_BASE_URL}/embeddings — cuerpo `{input, model,
    input_type?}`; `input_type=query|passage` solo si
    EMBEDDING_MODEL_TYPE=asymmetric.

Acople event loop: TODAS las invocaciones sync->async cruzan
`_PersistentLoop._loop` via `asyncio.run_coroutine_threadsafe`. Un segundo
loop rompe el binding del semaforo `_concurrency_sem`. Ver deuda
[#9](../CLAUDE.md#dt-9) (lock-in NIM) y [#12](../CLAUDE.md#dt-12) (tests
acoplados a mensajes de error).
"""

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Coroutine, Dict, Optional, TypeVar

from shared.types import EmbeddingModelProtocol

logger = logging.getLogger(__name__)

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_core.messages import HumanMessage, SystemMessage
    HAS_NVIDIA = True
except ImportError:
    HAS_NVIDIA = False
    ChatNVIDIA = None
    HumanMessage = None
    SystemMessage = None


class _ThinkingExhaustedError(ValueError):
    """Model used all tokens for <think> reasoning, producing no content."""


# =============================================================================
# PERSISTENT EVENT LOOP
# =============================================================================

class _PersistentLoop:
    """Singleton: un unico event loop en un thread daemon.

    Problema original: run_sync() llamaba asyncio.run() repetidamente,
    creando y destruyendo loops. Los asyncio.Semaphore/Lock se vinculan
    al primer loop que los usa, y al cambiar de loop lanzan
    "bound to a different event loop".

    Solucion: un unico loop persistente. Todas las coroutines se envian
    via run_coroutine_threadsafe(), que es thread-safe por diseno.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def _run_loop(self) -> None:
        """Target del thread daemon: crea loop y lo ejecuta forever."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        """Arranca el thread+loop si no existe. Thread-safe."""
        if self._loop is not None and self._loop.is_running():
            return self._loop

        with self._lock:
            # Double-check dentro del lock
            if self._loop is not None and self._loop.is_running():
                return self._loop

            self._ready.clear()
            self._thread = threading.Thread(
                target=self._run_loop, daemon=True, name="persistent-async-loop"
            )
            self._thread.start()
            self._ready.wait()
            assert self._loop is not None
            return self._loop

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Envia una coroutine al loop persistente y bloquea hasta resultado.

        Thread-safe: puede llamarse desde cualquier thread.
        Raises si se llama desde el thread del loop (deadlock).
        """
        loop = self._ensure_started()

        if threading.current_thread() is self._thread:
            raise RuntimeError(
                "run_sync() llamado desde dentro del loop persistente. "
                "Usa 'await' directamente en codigo async."
            )

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()


# Singleton global
_persistent_loop = _PersistentLoop()


# =============================================================================
# METRICAS
# =============================================================================

@dataclass
class LLMMetrics:
    """Metricas de rendimiento del servicio LLM.

    Usa threading.Lock en vez de asyncio.Lock: la seccion critica no
    contiene awaits, y threading.Lock no tiene binding a event loop.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    retries_total: int = 0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    async def record_request(
        self, success: bool, latency_ms: float, retries: int = 0
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            self.retries_total += retries
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.successful_requests / self.total_requests

    def summary(self) -> str:
        return (
            f"Requests: {self.total_requests} | "
            f"Success: {self.success_rate:.1%} | "
            f"Avg Latency: {self.avg_latency_ms:.0f}ms | "
            f"Retries: {self.retries_total}"
        )

    def __copy__(self) -> "LLMMetrics":
        return LLMMetrics(
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            total_latency_ms=self.total_latency_ms,
            retries_total=self.retries_total,
        )

    def __deepcopy__(self, memo: dict) -> "LLMMetrics":
        return self.__copy__()


# =============================================================================
# HELPER: ejecutar coroutine de forma segura
# =============================================================================

_T = TypeVar("_T")


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """
    Ejecuta una coroutine de forma sincrona.

    Envia la coroutine al loop persistente (_PersistentLoop) via
    run_coroutine_threadsafe. Funciona desde cualquier thread,
    con o sin event loop activo (CLI, Jupyter, frameworks async).

    El loop persistente garantiza que asyncio.Semaphore/Lock siempre
    se usan desde el mismo event loop, eliminando el error
    "bound to a different event loop".
    """
    return _persistent_loop.run(coro)  # type: ignore[no-any-return]


# =============================================================================
# SERVICIO LLM ASINCRONO
# =============================================================================

class AsyncLLMService:
    """
    Servicio asincrono para inferencia de texto con NVIDIA NIM.

    Uso:
        service = AsyncLLMService(
            base_url="http://nim:8080/v1",
            model_name="meta/llama-3.1-70b-instruct",
        )
        response = await service.invoke_async("prompt")
        response = service.invoke("prompt")  # sync wrapper
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_concurrent: int = 32,
        timeout_seconds: int = 120,
        max_retries: int = 3,
        temperature: float = 0.1,
    ):
        if not HAS_NVIDIA:
            raise ImportError("pip install langchain-nvidia-ai-endpoints")

        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature

        self._max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_loop: Optional[asyncio.AbstractEventLoop] = None
        self._client = ChatNVIDIA(
            base_url=base_url,
            model=model_name,
            temperature=temperature,
        )
        self.metrics = LLMMetrics()

        logger.info(
            f"AsyncLLMService: {model_name} @ {base_url} "
            f"(max_concurrent={max_concurrent})"
        )

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return a Semaphore bound to the current event loop.

        With the persistent loop, the loop never changes, so this creates
        the semaphore exactly once. The loop-check is kept as a safety net.
        """
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
            self._semaphore_loop = loop
        return self._semaphore

    async def __aenter__(self) -> "AsyncLLMService":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    # -------------------------------------------------------------------------
    # INFERENCIA
    # -------------------------------------------------------------------------

    async def _invoke_with_retry(
        self, messages: list, max_tokens: int = 4096,
        timing_out: Optional[Dict[str, float]] = None,
    ) -> str:
        """Invoca el LLM con reintentos y reporta timing opcional.

        Si `timing_out` (dict) se proporciona, se popula incrementalmente:
          - `queue_wait_ms`: tiempo desde inicio del intento hasta adquirir
            el semaforo de concurrencia. Se escribe justo tras el acquire.
          - `llm_ms`: tiempo desde el acquire hasta el retorno de
            `client.ainvoke`. Se escribe tras completar la llamada.
        Con reintentos, el dict refleja los valores del **ultimo intento**
        completado. Si `wait_for(...)` cancela esta coroutine mid-call, las
        claves presentes indican hasta donde llego (util para diagnosticar
        timeouts: `queue_wait_ms` presente sin `llm_ms` => cancelada dentro
        de la llamada al LLM; ambas ausentes => cancelada antes del acquire).
        """
        last_error = None
        retries = 0

        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                semaphore = self._get_semaphore()
                async with semaphore:
                    queue_ms = (time.perf_counter() - start_time) * 1000
                    if timing_out is not None:
                        timing_out["queue_wait_ms"] = queue_ms
                    llm_start = time.perf_counter()
                    response = await self._client.ainvoke(
                        messages,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                    )
                    llm_ms = (time.perf_counter() - llm_start) * 1000
                    if timing_out is not None:
                        timing_out["llm_ms"] = llm_ms

                latency_ms = (time.perf_counter() - start_time) * 1000
                await self.metrics.record_request(True, latency_ms, retries)

                content = response.content
                if isinstance(content, list):
                    content = "\n".join(
                        part.get("text", str(part))
                        if isinstance(part, dict)
                        else str(part)
                        for part in content
                    )

                # Strip reasoning tags from thinking-mode models
                # (e.g. nemotron-3-nano). Two passes: closed tags, then
                # unclosed tags (model hit max_tokens mid-thought).
                content = str(content)
                original_len = len(content)
                content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
                content = re.sub(r'<think>[\s\S]*$', '', content).strip()
                stripped = original_len - len(content)
                if stripped > 0:
                    logger.debug(
                        f"Stripped {stripped} chars of <think> tags "
                        f"({original_len} -> {len(content)})"
                    )

                if not content:
                    raise _ThinkingExhaustedError(
                        "LLM returned empty content after stripping reasoning tags"
                    )
                return content

            except Exception as e:
                last_error = e
                retries += 1
                latency_ms = (time.perf_counter() - start_time) * 1000

                # If model used all tokens for thinking, double
                # max_tokens on next attempt to leave room for the
                # actual response (capped at 16384).
                if isinstance(e, _ThinkingExhaustedError):
                    new_limit = min(max_tokens * 2, 32768)
                    if new_limit > max_tokens:
                        logger.debug(
                            f"Bumping max_tokens {max_tokens} -> {new_limit} "
                            f"after thinking exhaustion"
                        )
                        max_tokens = new_limit

                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Intento {attempt + 1}/{self.max_retries + 1} fallo: {e}. "
                        f"Reintentando en {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    await self.metrics.record_request(
                        False, latency_ms, retries
                    )

        raise RuntimeError(
            f"Inferencia fallida tras {self.max_retries + 1} intentos. "
            f"Ultimo error: {last_error}"
        )

    async def invoke_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        timing_out: Optional[Dict[str, float]] = None,
    ) -> str:
        """API async principal.

        `timing_out`: si se proporciona, el dict se popula con
        `queue_wait_ms` y `llm_ms` del ultimo intento completado
        (ver `_invoke_with_retry` para semantica exacta bajo cancelaciones
        y reintentos).
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return await self._invoke_with_retry(
            messages, max_tokens, timing_out=timing_out,
        )

    def invoke(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        """Wrapper sincrono. Usa run_sync para compatibilidad con loops activos."""
        return run_sync(
            self.invoke_async(user_prompt, system_prompt, max_tokens)
        )

    def get_metrics_summary(self) -> str:
        return self.metrics.summary()

    def reset_metrics(self) -> None:
        self.metrics = LLMMetrics()


# =============================================================================
# EMBEDDINGS (consolidado desde embeddings.py)
# =============================================================================

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    HAS_NVIDIA_EMBEDDINGS = True
except ImportError:
    HAS_NVIDIA_EMBEDDINGS = False
    NVIDIAEmbeddings = None


def load_embedding_model(
    base_url: str,
    model_name: str,
    model_type: str = "symmetric",
) -> EmbeddingModelProtocol:
    """
    Carga modelo de embeddings NIM para uso con LangChain/ChromaDB.

    Acepta SOLO parametros explicitos. Sin fallback a config ni env vars.
    El caller (cada sandbox) pasa los valores desde su propio config.

    Args:
        base_url: URL base del servidor NIM.
        model_name: Nombre del modelo de embedding.
        model_type: "symmetric" (gRPC/Triton) o "asymmetric" (REST OpenAI-compatible).

    Returns:
        Instancia NVIDIAEmbeddings compatible con LangChain.
    """
    if not HAS_NVIDIA_EMBEDDINGS:
        raise ImportError("pip install langchain-nvidia-ai-endpoints")

    if not base_url or not model_name:
        raise ValueError(
            "base_url y model_name son requeridos. "
            "Verificar configuracion en .env del sandbox."
        )

    if model_type not in ("symmetric", "asymmetric"):
        raise ValueError(
            f"model_type='{model_type}' no valido. Usar 'symmetric' o 'asymmetric'"
        )

    logger.info(
        f"Cargando embedding NIM: {model_name} @ {base_url} [tipo={model_type}]"
    )

    if model_type == "asymmetric":
        return NVIDIAEmbeddings(  # type: ignore[no-any-return]
            model=model_name,
            base_url=base_url,
            truncate="END",
        )
    else:
        return NVIDIAEmbeddings(  # type: ignore[no-any-return]
            model=model_name,
            base_url=base_url,
            truncate="END",
            mode="nim",
        )


__all__ = [
    "AsyncLLMService",
    "LLMMetrics",
    "HAS_NVIDIA",
    "run_sync",
    "load_embedding_model",
    "HAS_NVIDIA_EMBEDDINGS",
]
