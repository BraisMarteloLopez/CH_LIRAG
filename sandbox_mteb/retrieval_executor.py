"""
Retrieval executor: ejecuta retrieval + reranking opcional.

Extraido de evaluator.py para reducir su tamano (Fase B descomposicion).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from shared.types import QueryRetrievalDetail
from shared.retrieval.core import BaseRetriever
from shared.retrieval.reranker import CrossEncoderReranker

from .config import MTEBConfig

logger = logging.getLogger(__name__)


class RetrievalExecutor:
    """Ejecuta retrieval + reranking opcional para queries individuales."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever],
        reranker: Optional[CrossEncoderReranker],
        config: MTEBConfig,
    ):
        self._retriever = retriever
        self._reranker = reranker
        self._config = config
        self._rerank_failures: int = 0
        self._strategy_mismatches: int = 0

    @property
    def rerank_failures(self) -> int:
        return self._rerank_failures

    @property
    def strategy_mismatches(self) -> int:
        return self._strategy_mismatches

    def execute(
        self, query_text: str, expected_doc_ids: List[str],
        query_vector: Optional[List[float]] = None,
    ) -> Tuple[QueryRetrievalDetail, Optional[bool]]:
        """
        Ejecuta retrieval + reranking opcional.

        Si query_vector se proporciona, usa retrieve_by_vector() para
        evitar la llamada REST al NIM de embeddings (pre-embebido en batch).

        Separacion de flujos cuando reranker activo:
          - Metricas de retrieval: sobre los top RETRIEVAL_K docs del retriever
            (pre-rerank). Mide la capacidad del retriever.
          - Generacion: sobre los top RERANKER_TOP_N docs post-rerank.
            Mide la calidad del contexto que recibe el LLM.

        Sin reranker: ambos flujos usan los mismos docs (RETRIEVAL_K).

        Returns:
            Tupla (QueryRetrievalDetail, reranked_status):
              - reranked_status: None si no hay reranker, True si rerank OK,
                False si fallback sin rerank.
              Retrieval metadata viaja en QueryRetrievalDetail.retrieval_metadata.
        """
        if self._retriever is None:
            return QueryRetrievalDetail(
                retrieved_doc_ids=[], retrieved_contents=[],
                retrieval_scores=[], expected_doc_ids=expected_doc_ids,
            ), None

        try:
            retrieval_k = self._config.retrieval.retrieval_k
            configured_strategy = self._config.retrieval.strategy

            # Seleccionar metodo de retrieval
            def _do_retrieve(top_k: int):
                if query_vector is not None:
                    return self._retriever.retrieve_by_vector(
                        query_text, query_vector, top_k=top_k
                    )
                return self._retriever.retrieve(query_text, top_k=top_k)

            def _check_strategy(result) -> None:
                """Detecta discrepancia entre estrategia configurada y ejecutada (DTm-38)."""
                actual = result.strategy_used.name
                expected = configured_strategy.name
                if actual != expected:
                    self._strategy_mismatches += 1
                    if self._strategy_mismatches == 1:
                        logger.error(
                            f"STRATEGY MISMATCH: configurado={expected}, "
                            f"ejecutado={actual}. Los resultados no representan "
                            f"la estrategia configurada."
                        )

            if self._reranker:
                fetch_k = self._config.reranker.fetch_k or (self._config.reranker.top_n * 3)
                # Garantizar que fetch_k >= retrieval_k para que las metricas
                # pre-rerank tengan suficientes candidatos (DTm-35).
                fetch_k = max(fetch_k, retrieval_k)
                logger.debug(
                    f"  fetch_k={fetch_k} (retrieval_k={retrieval_k}, "
                    f"reranker.top_n={self._config.reranker.top_n})"
                )
                full_result = _do_retrieve(fetch_k)
                _check_strategy(full_result)

                # Metricas de retrieval: top RETRIEVAL_K del retriever (pre-rerank)
                metric_doc_ids = full_result.doc_ids[:retrieval_k]
                metric_contents = full_result.contents[:retrieval_k]
                metric_scores = full_result.scores[:retrieval_k]

                # Generacion: reranker reordena los candidatos completos
                reranked = self._reranker.rerank(
                    query=query_text,
                    retrieval_result=full_result,
                    top_n=self._config.reranker.top_n,
                )

                # FIX DT-7: detectar fallback silencioso del reranker
                reranked_ok = reranked.metadata.get("reranked", True)
                if not reranked_ok:
                    self._rerank_failures += 1
                    logger.warning(
                        f"  Rerank fallback (sin reorder) en query: "
                        f"'{query_text[:80]}...' | "
                        f"Error: {reranked.metadata.get('rerank_error', 'unknown')}"
                    )

                return QueryRetrievalDetail(
                    retrieved_doc_ids=metric_doc_ids,
                    retrieved_contents=metric_contents,
                    retrieval_scores=metric_scores,
                    expected_doc_ids=expected_doc_ids,
                    retrieval_time_ms=reranked.retrieval_time_ms,
                    generation_doc_ids=reranked.doc_ids,
                    generation_contents=reranked.contents,
                    # FIX DT-5: almacenar IDs de todos los candidatos pre-rerank
                    # para trazabilidad post-hoc (solo IDs, ~3KB/query)
                    pre_rerank_candidate_ids=full_result.doc_ids,
                    retrieval_metadata=full_result.metadata,
                ), reranked_ok
            else:
                # Sin reranker: retrieval_k docs para metricas y generacion
                result = _do_retrieve(retrieval_k)
                _check_strategy(result)

                return QueryRetrievalDetail(
                    retrieved_doc_ids=result.doc_ids,
                    retrieved_contents=result.contents,
                    retrieval_scores=result.scores,
                    expected_doc_ids=expected_doc_ids,
                    retrieval_time_ms=result.retrieval_time_ms,
                    retrieval_metadata=result.metadata,
                ), None

        except Exception as e:
            logger.warning(f"Error retrieval: {e}")
            return QueryRetrievalDetail(
                retrieved_doc_ids=[], retrieved_contents=[],
                retrieval_scores=[], expected_doc_ids=expected_doc_ids,
            ), None


def format_context(contents: List[str], max_length: int) -> str:
    """Formatea documentos recuperados como contexto para generacion."""
    if not contents:
        return "[No se encontraron documentos]"

    separator = "\n\n"
    parts: List[str] = []
    length = 0
    for i, content in enumerate(contents, 1):
        header = f"[Doc {i}]\n"
        part_len = len(header) + len(content)
        sep_len = len(separator) if parts else 0
        if length + sep_len + part_len > max_length:
            break
        parts.append(f"{header}{content}")
        length += sep_len + part_len

    if len(parts) < len(contents):
        logger.debug(
            f"Contexto truncado: {len(parts)}/{len(contents)} docs "
            f"({length}/{max_length} chars)"
        )

    return separator.join(parts)


__all__ = ["RetrievalExecutor", "format_context"]
