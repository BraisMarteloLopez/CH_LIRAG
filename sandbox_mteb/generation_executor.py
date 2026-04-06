"""
Generation executor: generacion LLM + calculo de metricas.

Extraido de evaluator.py para reducir su tamano (Fase B descomposicion).
"""

from __future__ import annotations

import asyncio
import logging
import time
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

from .config import GENERATION_PROMPTS
from .retrieval_executor import format_context, format_structured_context

logger = logging.getLogger(__name__)


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
    ):
        self._llm_service = llm_service
        self._metrics_calculator = metrics_calculator
        self._max_context_chars = max_context_chars

    async def batch_generate_and_evaluate(
        self,
        queries: List[NormalizedQuery],
        retrievals: List[QueryRetrievalDetail],
        ds_config: Dict[str, Any],
        dataset_name: str,
    ) -> list:
        """Lanza generacion+metricas en paralelo para todas las queries."""
        tasks = [
            self._process_single_async(query, retrieval, ds_config, dataset_name)
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
        # F.3/DAM-8: Contexto estructurado si hay datos KG disponibles
        kg_entities = retrieval_detail.retrieval_metadata.get("kg_entities")
        kg_relations = retrieval_detail.retrieval_metadata.get("kg_relations")
        if kg_entities or kg_relations:
            context = format_structured_context(
                retrieval_detail.get_generation_contents(),
                kg_entities or [],
                kg_relations or [],
                self._max_context_chars,
            )
        else:
            context = format_context(
                retrieval_detail.get_generation_contents(),
                self._max_context_chars,
            )

        # 1. Generacion async
        generation = await self._execute_generation_async(
            query.query_text, context, dataset_name
        )

        # 2. Metricas async
        primary_result, secondary_results = await self._calculate_metrics_async(
            generated=generation.generated_response,
            expected_answer=query.expected_answer,
            answer_type=query.answer_type,
            context=context,
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


__all__ = ["GenerationExecutor", "GenMetricsResult"]
