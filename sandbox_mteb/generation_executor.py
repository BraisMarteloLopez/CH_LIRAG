"""
Generation executor: generacion LLM + calculo de metricas.

Extraido de evaluator.py para reducir su tamano (Fase B descomposicion).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from shared.types import (
    NormalizedQuery,
    QueryRetrievalDetail,
    GenerationResult,
    DatasetType,
    MetricType,
    get_dataset_config,
)
from shared.llm import AsyncLLMService
from shared.metrics import MetricsCalculator, MetricResult

from shared.constants import GENERATION_QUERY_TIMEOUT_S
from .config import (
    GENERATION_PROMPTS,
    KG_SYNTHESIS_SYSTEM_PROMPT,
    KG_SYNTHESIS_USER_TEMPLATE,
)
from .retrieval_executor import format_context, format_structured_context

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tracker de KG synthesis (divergencia LightRAG #2).
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


class _KGSynthesisTracker:
    """Contador thread-safe de eventos de KG synthesis."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter = Counter()

    def record(self, event: str) -> None:
        with self._lock:
            self._counters[event] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            n = self._counters.get("invocations", 0)
            fallbacks = (
                self._counters.get("errors", 0)
                + self._counters.get("empty_returns", 0)
                + self._counters.get("timeouts", 0)
            )
            return {
                "invocations": n,
                "successes": self._counters.get("successes", 0),
                "errors": self._counters.get("errors", 0),
                "empty_returns": self._counters.get("empty_returns", 0),
                "truncations": self._counters.get("truncations", 0),
                "timeouts": self._counters.get("timeouts", 0),
                "fallback_rate": (fallbacks / n) if n else 0.0,
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()


_kg_synthesis_tracker = _KGSynthesisTracker()


def get_kg_synthesis_stats() -> Dict[str, Any]:
    """Snapshot del tracker de KG synthesis (divergencia LightRAG #2).

    Claves devueltas:
      - invocations: queries donde se intento la synthesis
      - successes, errors, empty_returns, truncations, timeouts
      - fallback_rate: (errors + empty_returns + timeouts) / invocations
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
        # F.3/DAM-8: Contexto estructurado si hay datos KG disponibles.
        # Divergencia #7: modo LightRAG determina presupuestos por seccion.
        kg_entities = retrieval_detail.retrieval_metadata.get("kg_entities")
        kg_relations = retrieval_detail.retrieval_metadata.get("kg_relations")
        has_kg_data = bool(kg_entities or kg_relations)
        if has_kg_data:
            lightrag_mode = retrieval_detail.retrieval_metadata.get(
                "lightrag_mode", "hybrid"
            )
            structured_context = format_structured_context(
                retrieval_detail.get_generation_contents(),
                kg_entities or [],
                kg_relations or [],
                self._max_context_chars,
                mode=lightrag_mode,
            )
        else:
            structured_context = format_context(
                retrieval_detail.get_generation_contents(),
                self._max_context_chars,
            )

        # Divergencia LightRAG #2: synthesis del contexto KG.
        # Solo si hay datos KG y la flag esta activa. Si la synthesis falla,
        # degradamos al contexto estructurado original (no rompemos el run).
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
        else:
            generation_context = structured_context

        # 1. Generacion async (usa el contexto sintetizado si lo hay)
        generation = await self._execute_generation_async(
            query.query_text, generation_context, dataset_name
        )

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
            return GenerationResult(f"[ERROR: {e}]", 0.0)

    async def _synthesize_kg_context_async(
        self,
        query_text: str,
        structured_context: str,
    ) -> Tuple[str, Optional[str]]:
        """Reescribe el contexto KG multi-seccion como narrativa coherente.

        Divergencia LightRAG #2. Usa el LLM (`self._llm_service`) con un
        prompt query-aware que pide narrativa con citas [ref:N] preservadas.

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
                `kg_synthesis_error` per-query (deuda #15).
        """
        _kg_synthesis_tracker.record("invocations")

        max_chars = self._kg_synthesis_max_chars
        system_prompt = KG_SYNTHESIS_SYSTEM_PROMPT.format(max_chars=max_chars)
        user_prompt = KG_SYNTHESIS_USER_TEMPLATE.format(
            query=query_text,
            structured_context=structured_context,
        )

        try:
            response = await asyncio.wait_for(
                self._llm_service.invoke_async(
                    user_prompt, system_prompt=system_prompt
                ),
                timeout=self._kg_synthesis_timeout_s,
            )
        except asyncio.TimeoutError:
            _kg_synthesis_tracker.record("timeouts")
            logger.warning(
                "KG synthesis timeout (%.1fs) para query '%s...'; "
                "usando contexto estructurado.",
                self._kg_synthesis_timeout_s,
                query_text[:60],
            )
            return structured_context, "timeout"
        except Exception as e:
            _kg_synthesis_tracker.record("errors")
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

        # Metrica principal
        try:
            if dataset_type == DatasetType.HYBRID:
                if answer_type == "label":
                    primary = await calc.calculate_async(
                        MetricType.ACCURACY,
                        generated=generated,
                        expected=expected_answer,
                    )
                else:
                    primary = await calc.calculate_async(
                        MetricType.F1_SCORE,
                        generated=generated,
                        expected=expected_answer,
                    )
            elif dataset_type == DatasetType.ADAPTED:
                if expected_answer and calc.embedding_model:
                    primary = await calc.calculate_async(
                        MetricType.SEMANTIC_SIMILARITY,
                        generated=generated,
                        expected=expected_answer,
                    )
                else:
                    primary = await calc.calculate_async(
                        MetricType.ANSWER_RELEVANCE,
                        generated=generated,
                        query=query_text,
                    )
            else:  # RETRIEVAL_ONLY
                primary = await calc.calculate_async(
                    MetricType.FAITHFULNESS,
                    generated=generated,
                    context=context,
                )
        except Exception as e:
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
                # FIX DTm-5: registrar fallo y propagar MetricResult con error.
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
    "get_kg_synthesis_stats",
    "reset_kg_synthesis_stats",
]
