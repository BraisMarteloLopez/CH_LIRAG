"""
Generation executor: generacion LLM + calculo de metricas.

Extraido de evaluator.py para reducir su tamano (Fase B descomposicion).

Contrato observable producido per-query (llega a JSON/CSV via
`result_builder` y a `config_snapshot._runtime`):
  - `kg_synthesis_stats` (solo LIGHT_RAG + `KG_SYNTHESIS_ENABLED=true`):
    counters `{invocations, successes, errors, empty_returns,
    truncations, timeouts, fallback_rate}` + timing `p50/p95/max_{total,
    queue, llm}_ms`. Poblado por `_KGSynthesisTracker.snapshot()`;
    reset al inicio de cada run por `reset_kg_synthesis_stats()`.
    Guardrail: `fallback_rate > 10%` es senal roja (warning en logs,
    sin bloqueo activo — ver CLAUDE.md §Observabilidad).
  - `citation_refs_synth_*` y `citation_refs_gen_*` (14 campos,
    [div #7](../CLAUDE.md#div-7)): parser `shared/citation_parser.py`
    invocado dos veces en `_process_single_async` sobre narrativa synth
    y respuesta final. Contadores: `total, valid, malformed, in_range,
    out_of_range, distinct, coverage_ratio`. `out_of_range > 0` o
    `malformed > 0` son senales rojas equivalentes a
    `judge.default_return_rate > 2%` ([deuda #18](../CLAUDE.md#dt-18)).

Fallback graceful: error/vacio/timeout en synthesis -> contexto
estructurado original; faithfulness se evalua siempre contra el
estructurado, no contra la narrativa (control anti-fabricacion).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict

# Tipos cerrados (R2): dominio de `kg_synthesis_error` per-query, tambien
# emitido por `_synthesize_kg_context_async`. `None` indica synthesis
# exitosa; el resto son razones de fallback graceful.
SynthesisErrorReason = Literal["timeout", "error", "empty"]


class KGSynthesisStats(TypedDict):
    """Stats agregadas de la capa de synthesis KG (R5/R14).

    Counters + timing per-invocacion (3 categorias: total/queue/llm).
    `n_*_samples` reportan cuantas invocaciones contribuyeron al timing:
    queue/llm pueden tener menos que total cuando un timeout cancelo la
    coroutine antes del acquire o de la respuesta del LLM.
    """

    invocations: int
    successes: int
    errors: int
    empty_returns: int
    truncations: int
    timeouts: int
    fallback_rate: float
    p50_total_ms: float
    p95_total_ms: float
    max_total_ms: float
    n_total_samples: int
    p50_queue_ms: float
    p95_queue_ms: float
    max_queue_ms: float
    n_queue_samples: int
    p50_llm_ms: float
    p95_llm_ms: float
    max_llm_ms: float
    n_llm_samples: int

from shared.types import (
    NormalizedQuery,
    QueryRetrievalDetail,
    GenerationResult,
    DatasetType,
    MetricType,
    get_dataset_config,
)
from shared.llm import AsyncLLMService, InvokeTiming
from shared.metrics import MetricsCalculator, MetricResult
from shared.operational_tracker import record_operational_event

from shared.constants import GENERATION_QUERY_TIMEOUT_S
from .config import (
    GENERATION_PROMPTS,
    KG_SYNTHESIS_SYSTEM_PROMPT,
    KG_SYNTHESIS_USER_TEMPLATE,
)
from shared.citation_parser import parse_citation_refs

from .retrieval_executor import (
    format_context,
    format_structured_context,
    format_structured_context_with_stats,
    is_kg_budget_cap_triggered,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tracker de KG synthesis.
#
# La synthesis es un paso opcional que reescribe el contexto multi-seccion
# como narrativa coherente. Si falla, hacemos fallback al contexto
# estructurado y el run continua. El tracker permite auditar si el feature
# esta funcionando correctamente o degradando silenciosamente.
#
# Eventos:
#   - invocations: queries donde se intento la synthesis (KG data + flag on).
#   - successes: synthesis devolvio narrativa no vacia dentro del budget.
#   - errors: el LLM lanzo excepcion.
#   - empty_returns: el LLM devolvio string vacio o solo whitespace.
#   - truncations: la respuesta excedio max_chars y fue truncada.
#   - timeouts: la llamada excedio kg_synthesis_timeout_s.
# -----------------------------------------------------------------------------


def _percentile(values: List[float], p: float) -> float:
    """Percentil simple (nearest-rank). Retorna 0.0 si la lista esta vacia.

    Implementacion local para evitar dependencia en `statistics.quantiles`
    (requiere n>=2) y tratar el caso vacio como 0 en vez de excepcion.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
    return sorted_vals[idx]


class _KGSynthesisTracker:
    """Contador + timing thread-safe de eventos de KG synthesis.

    Ademas de contadores, acumula tres listas paralelas de duraciones por
    invocacion:
      - `total_ms`: tiempo total de `_synthesize_kg_context_async` desde
        que entra a `asyncio.wait_for` hasta que decide el outcome
        (exito/truncacion/empty/error/timeout). Incluye cola + LLM + parsing.
      - `queue_ms`: tiempo esperando el semaforo de concurrencia de
        `AsyncLLMService`. Solo presente si el intento llego a acquire.
      - `llm_ms`: tiempo desde acquire hasta respuesta del LLM. Solo
        presente si la llamada retorno (puede faltar en timeouts).

    `snapshot()` expone p50/p95/max para cada lista — permite discriminar
    saturacion de cola vs llamadas LLM lentas en runs con `fallback_rate`
    alto (diagnostico documentado en CLAUDE.md, seccion observabilidad).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter = Counter()
        self._timings: Dict[str, List[float]] = {
            "total_ms": [],
            "queue_ms": [],
            "llm_ms": [],
        }

    def record(self, event: str) -> None:
        with self._lock:
            self._counters[event] += 1

    def record_timing(
        self,
        total_ms: float,
        queue_ms: Optional[float],
        llm_ms: Optional[float],
    ) -> None:
        """Registra timing de una invocacion individual.

        Los valores opcionales (`queue_ms`, `llm_ms`) pueden faltar cuando
        la invocacion cayo antes de poblar la clave correspondiente (p.ej.
        timeout mientras aun estaba esperando el semaforo => solo
        `total_ms` disponible).
        """
        with self._lock:
            self._timings["total_ms"].append(total_ms)
            if queue_ms is not None:
                self._timings["queue_ms"].append(queue_ms)
            if llm_ms is not None:
                self._timings["llm_ms"].append(llm_ms)

    def snapshot(self) -> KGSynthesisStats:
        with self._lock:
            n = self._counters.get("invocations", 0)
            fallbacks = (
                self._counters.get("errors", 0)
                + self._counters.get("empty_returns", 0)
                + self._counters.get("timeouts", 0)
            )
            total = self._timings["total_ms"]
            queue = self._timings["queue_ms"]
            llm = self._timings["llm_ms"]
            return KGSynthesisStats(
                invocations=n,
                successes=self._counters.get("successes", 0),
                errors=self._counters.get("errors", 0),
                empty_returns=self._counters.get("empty_returns", 0),
                truncations=self._counters.get("truncations", 0),
                timeouts=self._counters.get("timeouts", 0),
                fallback_rate=(fallbacks / n) if n else 0.0,
                p50_total_ms=round(_percentile(total, 0.50), 1),
                p95_total_ms=round(_percentile(total, 0.95), 1),
                max_total_ms=round(max(total) if total else 0.0, 1),
                n_total_samples=len(total),
                p50_queue_ms=round(_percentile(queue, 0.50), 1),
                p95_queue_ms=round(_percentile(queue, 0.95), 1),
                max_queue_ms=round(max(queue) if queue else 0.0, 1),
                n_queue_samples=len(queue),
                p50_llm_ms=round(_percentile(llm, 0.50), 1),
                p95_llm_ms=round(_percentile(llm, 0.95), 1),
                max_llm_ms=round(max(llm) if llm else 0.0, 1),
                n_llm_samples=len(llm),
            )

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            for values in self._timings.values():
                values.clear()


_kg_synthesis_tracker = _KGSynthesisTracker()


def get_kg_synthesis_stats() -> KGSynthesisStats:
    """Snapshot del tracker de KG synthesis.

    Claves devueltas:
      - invocations: queries donde se intento la synthesis
      - successes, errors, empty_returns, truncations, timeouts
      - fallback_rate: (errors + empty_returns + timeouts) / invocations
      - timing per-categoria (total/queue/llm): p50/p95/max_*_ms + n_*_samples
    """
    return _kg_synthesis_tracker.snapshot()


def reset_kg_synthesis_stats() -> None:
    """Resetea contadores. Llamar al inicio de cada run."""
    _kg_synthesis_tracker.reset()


class GenMetricsResult:
    """Resultado interno del pipeline async (generacion + metricas)."""
    __slots__ = ("generation", "primary_metric_value", "primary_metric_type", "secondary_metrics")

    def __init__(
        self,
        generation: GenerationResult,
        primary_metric_value: float,
        primary_metric_type: MetricType = MetricType.F1_SCORE,
        secondary_metrics: Optional[Dict[str, float]] = None,
    ):
        self.generation = generation
        self.primary_metric_value = primary_metric_value
        self.primary_metric_type = primary_metric_type
        self.secondary_metrics = secondary_metrics or {}


# Dispatch table R6: resolucion de la metrica primaria por DatasetType.
# Cada resolver retorna (MetricType, kwargs) listo para `calculate_async()`.
# Mantener en sync con `DatasetType` (shared/types.py) y las metricas
# declaradas en `DATASET_CONFIG`.

def _resolve_primary_hybrid(
    *, answer_type: Optional[str], expected_answer: Optional[str],
    query_text: str, context: str, calc: MetricsCalculator,
) -> Tuple[MetricType, Dict[str, Any]]:
    # HYBRID: ACCURACY para clasificacion yes/no, F1_SCORE para extractiva.
    # Para 1 token F1 ~ EM; para 2+ tokens F1 captura aciertos parciales.
    if answer_type == "label":
        return MetricType.ACCURACY, {"expected": expected_answer}
    return MetricType.F1_SCORE, {"expected": expected_answer}


def _resolve_primary_adapted(
    *, answer_type: Optional[str], expected_answer: Optional[str],
    query_text: str, context: str, calc: MetricsCalculator,
) -> Tuple[MetricType, Dict[str, Any]]:
    # ADAPTED: SEMANTIC_SIMILARITY si hay expected + embedding_model;
    # ANSWER_RELEVANCE via LLM-judge en caso contrario.
    if expected_answer and calc.embedding_model:
        return MetricType.SEMANTIC_SIMILARITY, {"expected": expected_answer}
    return MetricType.ANSWER_RELEVANCE, {"query": query_text}


def _resolve_primary_retrieval_only(
    *, answer_type: Optional[str], expected_answer: Optional[str],
    query_text: str, context: str, calc: MetricsCalculator,
) -> Tuple[MetricType, Dict[str, Any]]:
    # RETRIEVAL_ONLY: FAITHFULNESS del generado contra el contexto.
    return MetricType.FAITHFULNESS, {"context": context}


_PRIMARY_METRIC_RESOLVERS: Dict[
    DatasetType,
    Callable[..., Tuple[MetricType, Dict[str, Any]]],
] = {
    DatasetType.HYBRID: _resolve_primary_hybrid,
    DatasetType.ADAPTED: _resolve_primary_adapted,
    DatasetType.RETRIEVAL_ONLY: _resolve_primary_retrieval_only,
}


class GenerationExecutor:
    """Ejecuta generacion LLM + calculo de metricas para queries."""

    def __init__(
        self,
        llm_service: AsyncLLMService,
        metrics_calculator: MetricsCalculator,
        max_context_chars: int,
        query_timeout_s: float = GENERATION_QUERY_TIMEOUT_S,
        kg_synthesis_enabled: bool = False,
        kg_synthesis_max_chars: int = 0,
        kg_synthesis_timeout_s: float = 30.0,
    ):
        self._llm_service = llm_service
        self._metrics_calculator = metrics_calculator
        self._max_context_chars = max_context_chars
        self._query_timeout_s = query_timeout_s
        self._kg_synthesis_enabled = kg_synthesis_enabled
        # Si max_chars es 0, usar el mismo limite que la generacion.
        self._kg_synthesis_max_chars = (
            kg_synthesis_max_chars if kg_synthesis_max_chars > 0
            else max_context_chars
        )
        self._kg_synthesis_timeout_s = kg_synthesis_timeout_s

    async def batch_generate_and_evaluate(
        self,
        queries: List[NormalizedQuery],
        retrievals: List[QueryRetrievalDetail],
        ds_config: Dict[str, Any],
        dataset_name: str,
    ) -> list:
        """Lanza generacion+metricas en paralelo para todas las queries."""
        tasks = [
            asyncio.wait_for(
                self._process_single_async(query, retrieval, ds_config, dataset_name),
                timeout=self._query_timeout_s,
            )
            for query, retrieval in zip(queries, retrievals)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_single_async(
        self,
        query: NormalizedQuery,
        retrieval_detail: QueryRetrievalDetail,
        ds_config: Dict[str, Any],
        dataset_name: str,
    ) -> GenMetricsResult:
        """Procesa una query: generacion async + metricas async."""
        # Contexto estructurado si hay datos KG disponibles.
        # El modo LightRAG determina presupuestos por seccion.
        kg_entities = retrieval_detail.retrieval_metadata.get("kg_entities")
        kg_relations = retrieval_detail.retrieval_metadata.get("kg_relations")
        has_kg_data = bool(kg_entities or kg_relations)
        n_chunks_emitted = 0
        if has_kg_data:
            lightrag_mode = retrieval_detail.retrieval_metadata.get(
                "lightrag_mode", "hybrid"
            )
            structured_context, n_chunks_emitted = format_structured_context_with_stats(
                retrieval_detail.get_generation_contents(),
                kg_entities or [],
                kg_relations or [],
                self._max_context_chars,
                mode=lightrag_mode,
            )
            # Registrar si el cap al 50% del budget total escalo las
            # secciones KG. Con modelos de contexto >= 112000 chars el cap
            # no dispara; con ventanas mas pequenas discrimina "chunks
            # starved por KG" vs "budgets intactos".
            retrieval_detail.retrieval_metadata["kg_budget_cap_triggered"] = (
                is_kg_budget_cap_triggered(self._max_context_chars, lightrag_mode)
            )
        else:
            structured_context = format_context(
                retrieval_detail.get_generation_contents(),
                self._max_context_chars,
            )

        # Synthesis del contexto KG: reescribe las secciones como narrativa
        # coherente via LLM. Solo si hay datos KG y la flag esta activa. Si
        # la synthesis falla, degradamos al contexto estructurado original.
        # `generation_context` es lo que ve el LLM generador.
        # `structured_context` es siempre el original; faithfulness se evalua
        # contra este para penalizar cualquier alucinacion introducida por
        # la propia capa de synthesis.
        if has_kg_data and self._kg_synthesis_enabled:
            generation_context, synth_error = await self._synthesize_kg_context_async(
                query.query_text, structured_context
            )
            # Deuda #15: persistir outcome per-query para auditoria en JSON/CSV.
            # synth_error is None => la narrativa fue la entregada al LLM.
            retrieval_detail.retrieval_metadata["kg_synthesis_used"] = (
                synth_error is None
            )
            if synth_error is not None:
                retrieval_detail.retrieval_metadata["kg_synthesis_error"] = synth_error

            # Divergencia #7: observable de citaciones `[ref:N]` en la narrativa
            # synthesized. El prompt KG_SYNTHESIS_SYSTEM_PROMPT instruye al LLM
            # a emitir el formato; el parser cuenta cuantas respeto, cuantas
            # estan fuera de rango (apuntan a chunks truncados o inventadas),
            # cuantas son duplicados, etc. Si synth fallo, generation_context
            # es structured_context verbatim => el parser produce 0 validly
            # (no hay `[ref:N]` en el JSON crudo, solo `reference_id`).
            synth_stats = parse_citation_refs(generation_context, n_chunks_emitted)
            for metric_name, value in synth_stats.items():
                retrieval_detail.retrieval_metadata[
                    f"citation_refs_synth_{metric_name}"
                ] = value
        else:
            generation_context = structured_context

        # 1. Generacion async (usa el contexto sintetizado si lo hay)
        generation = await self._execute_generation_async(
            query.query_text, generation_context, dataset_name
        )

        # Divergencia #7: observable de citaciones en la respuesta final del
        # generador, emitido bajo el mismo gate que synth_* (has_kg_data +
        # synthesis enabled). Comparado con synth_*, discrimina el trasvase
        # narrativa -> respuesta:
        #   gen_valid > synth_valid  => generador invento citas (alarma)
        #   gen_out_of_range > 0 y synth_out_of_range == 0 => alucinacion
        #                                                     solo en gen
        if has_kg_data and self._kg_synthesis_enabled:
            gen_stats = parse_citation_refs(
                generation.generated_response, n_chunks_emitted,
            )
            for metric_name, value in gen_stats.items():
                retrieval_detail.retrieval_metadata[
                    f"citation_refs_gen_{metric_name}"
                ] = value

        # 2. Metricas async (faithfulness se evalua contra structured_context
        # original, no contra la narrativa del synthesis)
        primary_result, secondary_results = await self._calculate_metrics_async(
            generated=generation.generated_response,
            expected_answer=query.expected_answer,
            answer_type=query.answer_type,
            context=structured_context,
            query_text=query.query_text,
            dataset_type=ds_config["type"],
            dataset_name=dataset_name,
        )

        # Convertir secondary a dict
        secondary_dict = {}
        for sr in secondary_results:
            secondary_dict[sr.metric_type.value] = sr.value

        return GenMetricsResult(
            generation=generation,
            primary_metric_value=primary_result.value,
            primary_metric_type=primary_result.metric_type,
            secondary_metrics=secondary_dict,
        )

    async def _execute_generation_async(
        self, query_text: str, context: str, dataset_name: str
    ) -> GenerationResult:
        """Genera respuesta con LLM via invoke_async."""
        start = time.time()
        prompts = GENERATION_PROMPTS.get(
            dataset_name, GENERATION_PROMPTS["default"]
        )
        user_prompt = prompts["user_template"].format(
            context=context, query=query_text
        )

        try:
            response = await self._llm_service.invoke_async(
                user_prompt, system_prompt=prompts["system"]
            )
            return GenerationResult(
                generated_response=str(response).strip(),
                generation_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.warning(f"Error generacion async: {e}")
            record_operational_event("generation_error")
            return GenerationResult(f"[ERROR: {e}]", 0.0)

    async def _synthesize_kg_context_async(
        self,
        query_text: str,
        structured_context: str,
    ) -> Tuple[str, Optional[SynthesisErrorReason]]:
        """Reescribe el contexto KG multi-seccion como narrativa coherente.

        Usa el LLM (`self._llm_service`) con un prompt query-aware que pide
        narrativa con citas [ref:N] preservadas.

        Degradacion graceful: si el LLM falla, devuelve string vacio o
        timeout, hacemos fallback al `structured_context` original. El run
        nunca se rompe por culpa de la synthesis. Todos los modos de fallo
        se contabilizan en `_kg_synthesis_tracker`.

        Returns:
            Tupla (contexto_para_generacion, error_code):
              - contexto_para_generacion: narrativa sintetizada (puede estar
                truncada) o el `structured_context` original como fallback.
              - error_code: None en caso de exito o truncacion; en fallback
                uno de "timeout" | "error" | "empty". Usado por
                `_process_single_async` para poblar
                `retrieval_metadata.kg_synthesis_used` /
                `kg_synthesis_error` per-query.
        """
        _kg_synthesis_tracker.record("invocations")

        max_chars = self._kg_synthesis_max_chars
        system_prompt = KG_SYNTHESIS_SYSTEM_PROMPT.format(max_chars=max_chars)
        user_prompt = KG_SYNTHESIS_USER_TEMPLATE.format(
            query=query_text,
            structured_context=structured_context,
        )

        # Timing per-invocacion para discriminar saturacion de cola vs
        # llamada LLM lenta. `timing_out` se popula dentro de invoke_async
        # cuando llega al semaforo / completa la llamada.
        timing_out: InvokeTiming = {}
        t_start = time.perf_counter()

        def _record_timing() -> None:
            total_ms = (time.perf_counter() - t_start) * 1000
            _kg_synthesis_tracker.record_timing(
                total_ms,
                timing_out.get("queue_wait_ms"),
                timing_out.get("llm_ms"),
            )

        try:
            response = await asyncio.wait_for(
                self._llm_service.invoke_async(
                    user_prompt,
                    system_prompt=system_prompt,
                    timing_out=timing_out,
                ),
                timeout=self._kg_synthesis_timeout_s,
            )
        except asyncio.TimeoutError:
            _kg_synthesis_tracker.record("timeouts")
            _record_timing()
            logger.warning(
                "KG synthesis timeout (%.1fs) para query '%s...'; "
                "usando contexto estructurado.",
                self._kg_synthesis_timeout_s,
                query_text[:60],
            )
            return structured_context, "timeout"
        except Exception as e:
            _kg_synthesis_tracker.record("errors")
            _record_timing()
            logger.warning(
                "KG synthesis error: %s (query: '%s...'); "
                "usando contexto estructurado.",
                e,
                query_text[:60],
            )
            return structured_context, "error"

        narrative = str(response).strip()
        if not narrative:
            _kg_synthesis_tracker.record("empty_returns")
            _record_timing()
            logger.warning(
                "KG synthesis devolvio respuesta vacia para query '%s...'; "
                "usando contexto estructurado.",
                query_text[:60],
            )
            return structured_context, "empty"

        if len(narrative) > max_chars:
            _kg_synthesis_tracker.record("truncations")
            narrative = narrative[:max_chars]

        _kg_synthesis_tracker.record("successes")
        _record_timing()
        return narrative, None

    async def _calculate_metrics_async(
        self,
        generated: str,
        expected_answer: Optional[str],
        answer_type: Optional[str],
        context: str,
        query_text: str,
        dataset_type: DatasetType,
        dataset_name: str,
    ) -> Tuple[MetricResult, List[MetricResult]]:
        """
        Calcula metricas usando calculate_async para LLM-judge.

        Metrica primaria adaptativa para HYBRID datasets:
          - answer_type == "label": ACCURACY (clasificacion yes/no)
          - extractiva (cualquier longitud): F1_SCORE
            Para 1 token, F1 y EM son equivalentes (0 o 1).
            Para 2+ tokens, F1 captura aciertos parciales.
        EM se computa siempre como secundaria para analisis post-hoc.
        """
        ds_config = get_dataset_config(dataset_name)
        calc = self._metrics_calculator

        # Metrica principal via dispatch table (R6).
        try:
            try:
                resolver = _PRIMARY_METRIC_RESOLVERS[dataset_type]
            except KeyError:
                raise ValueError(
                    f"DatasetType sin resolver de metrica primaria: {dataset_type}"
                )
            metric_type, extra_kwargs = resolver(
                answer_type=answer_type,
                expected_answer=expected_answer,
                query_text=query_text,
                context=context,
                calc=calc,
            )
            primary = await calc.calculate_async(
                metric_type, generated=generated, **extra_kwargs,
            )
        except Exception as e:
            logger.warning(
                "Primary metric %s calculation failed: %s",
                ds_config["primary_metric"].value, e,
            )
            primary = MetricResult(
                metric_type=ds_config["primary_metric"],
                value=0.0,
                error=str(e),
            )

        # Metricas secundarias: siempre calcular todas las disponibles
        secondary = []

        # Para HYBRID: siempre calcular F1, EM, y Faithfulness como secundarias
        secondary_types = []
        if dataset_type == DatasetType.HYBRID and expected_answer:
            secondary_types = [MetricType.F1_SCORE, MetricType.EXACT_MATCH]
            # Excluir la que ya es primary
            secondary_types = [
                mt for mt in secondary_types
                if mt != primary.metric_type
            ]
        # Tambien incluir las configuradas en el dataset
        for mt in ds_config.get("secondary_metrics", []):
            if mt != primary.metric_type and mt not in secondary_types:
                secondary_types.append(mt)

        for mt in secondary_types:
            try:
                r = await calc.calculate_async(
                    mt,
                    generated=generated,
                    expected=expected_answer,
                    context=context,
                    query=query_text,
                )
                secondary.append(r)
            except Exception as e:
                # Registrar fallo y propagar MetricResult con error.
                logger.warning(
                    f"Metrica secundaria {mt.value} fallo: {e} "
                    f"(query: '{query_text[:80]}...')"
                )
                secondary.append(MetricResult(
                    metric_type=mt,
                    value=0.0,
                    error=f"secondary_metric_error: {e}",
                ))

        return primary, secondary


__all__ = [
    "GenerationExecutor",
    "GenMetricsResult",
    "KGSynthesisStats",
    "SynthesisErrorReason",
    "get_kg_synthesis_stats",
    "reset_kg_synthesis_stats",
]
