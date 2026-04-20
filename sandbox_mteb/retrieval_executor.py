"""
Retrieval executor: ejecuta retrieval + reranking opcional.

Extraido de evaluator.py para reducir su tamano (Fase B descomposicion).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from shared.types import QueryRetrievalDetail
from shared.retrieval.core import BaseRetriever, RetrievalStrategy
from shared.retrieval.reranker import CrossEncoderReranker
from shared.constants import KG_MAX_ENTITY_CONTEXT_CHARS, KG_MAX_RELATION_CONTEXT_CHARS

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
            retriever = self._retriever  # narrowed: not None (guarded above)
            def _do_retrieve(top_k: int):
                if query_vector is not None:
                    return retriever.retrieve_by_vector(
                        query_text, query_vector, top_k=top_k
                    )
                return retriever.retrieve(query_text, top_k=top_k)

            def _check_strategy(result) -> None:
                """Detecta discrepancia entre estrategia configurada y ejecutada."""
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

            # Paper LightRAG no usa reranker — el cross-encoder single-hop
            # penalizaria los docs multi-hop que aporta el KG.
            use_reranker = (
                self._reranker
                and configured_strategy != RetrievalStrategy.LIGHT_RAG
            )
            if use_reranker:
                fetch_k = self._config.reranker.fetch_k or (self._config.reranker.top_n * 3)
                # Garantizar fetch_k >= retrieval_k para que las metricas
                # pre-rerank tengan suficientes candidatos.
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

                # Detectar fallback silencioso del reranker.
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
                    # Almacenar IDs de candidatos pre-rerank para
                    # trazabilidad post-hoc (~3KB/query).
                    pre_rerank_candidate_ids=full_result.doc_ids,
                    retrieval_metadata=full_result.metadata,
                ), reranked_ok
            else:
                # Sin reranker: retrieval_k docs para metricas.
                result = _do_retrieve(retrieval_k)
                _check_strategy(result)

                # LIGHT_RAG: separar docs de generacion (top-N por KG score)
                # de docs de metricas (todos retrieval_k). Analogo a
                # reranker.top_n para SV: el KG scoring ya rankeo los chunks,
                # seleccionamos los top-N para el LLM generador.
                gen_top_n = self._config.retrieval.lightrag_generation_top_n
                if (
                    gen_top_n > 0
                    and configured_strategy == RetrievalStrategy.LIGHT_RAG
                    and len(result.doc_ids) > gen_top_n
                ):
                    return QueryRetrievalDetail(
                        retrieved_doc_ids=result.doc_ids,
                        retrieved_contents=result.contents,
                        retrieval_scores=result.scores,
                        expected_doc_ids=expected_doc_ids,
                        retrieval_time_ms=result.retrieval_time_ms,
                        generation_doc_ids=result.doc_ids[:gen_top_n],
                        generation_contents=result.contents[:gen_top_n],
                        retrieval_metadata=result.metadata,
                    ), None

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


# Paper-aligned: presupuestos fijos por seccion (HKUDS/LightRAG, EMNLP 2025).
# Paper: MAX_ENTITY_TOKENS=6000, MAX_RELATION_TOKENS=8000.
# Chunks reciben el budget restante (max_length - entity - relation).
# Modo determina que secciones se activan, no los budgets por seccion.
_LIGHTRAG_MODE_SECTIONS: Dict[str, Tuple[bool, bool]] = {
    # (use_entities, use_relations)
    "hybrid": (True,  True),
    "local":  (True,  False),
    "global": (False, True),
    "naive":  (False, False),
}

# Overhead fijo del header de una seccion KG: "<label>:\n\n```json\n" + "\n```"
_KG_SECTION_OVERHEAD = 60


def _build_kg_section(
    label: str,
    items: List[Dict[str, Any]],
    budget: int,
) -> str:
    """Construye una seccion KG respetando el budget. Retorna '' si no cabe nada."""
    import json

    content_budget = budget - _KG_SECTION_OVERHEAD
    if content_budget <= 0 or not items:
        return ""

    lines: List[str] = []
    used = 0
    for item in items:
        line = json.dumps(item, ensure_ascii=False)
        sep = 1 if lines else 0  # newline separator entre lineas
        if used + sep + len(line) > content_budget:
            break
        lines.append(line)
        used += sep + len(line)

    if not lines:
        return ""

    body = "\n".join(lines)
    return f"{label}:\n\n```json\n{body}\n```"


def format_structured_context(
    contents: List[str],
    kg_entities: List[Dict[str, Any]],
    kg_relations: List[Dict[str, Any]],
    max_length: int,
    mode: str = "hybrid",
) -> str:
    """Wrapper retrocompatible de `format_structured_context_with_stats`.

    Devuelve solo el string formateado. Para auditar cuantos chunks llegaron
    al LLM (requerido por el observable de citaciones, divergencia #7) usar
    `format_structured_context_with_stats` directamente.
    """
    text, _ = format_structured_context_with_stats(
        contents, kg_entities, kg_relations, max_length, mode,
    )
    return text


def format_structured_context_with_stats(
    contents: List[str],
    kg_entities: List[Dict[str, Any]],
    kg_relations: List[Dict[str, Any]],
    max_length: int,
    mode: str = "hybrid",
) -> Tuple[str, int]:
    """Formatea contexto con secciones KG estructuradas y reporta chunks emitidos.

    Identica a `format_structured_context` en todo excepto el retorno:
    `(texto, n_chunks_emitted)` donde `n_chunks_emitted` es el numero de
    chunks que cupieron en el budget y fueron serializados como JSON-lines.
    Con budget holgado, `n_chunks_emitted == len(contents)`; con budget
    apretado, es menor.

    Este dato es requerido por `parse_citation_refs` (divergencia #7) para
    validar el rango `[1, n_chunks_emitted]` de las referencias `[ref:N]`
    emitidas por el LLM. Sin este numero, citas a chunks truncados
    apareceran como `in_range` cuando realmente son `out_of_range`.
    """
    import json

    use_entities, use_relations = _LIGHTRAG_MODE_SECTIONS.get(
        mode, _LIGHTRAG_MODE_SECTIONS["hybrid"]
    )
    # Budgets fijos del paper, capeados proporcionalmente si exceden 50%
    # del max_length (chunks siempre reciben al menos 50%).
    entity_raw = KG_MAX_ENTITY_CONTEXT_CHARS if use_entities else 0
    relation_raw = KG_MAX_RELATION_CONTEXT_CHARS if use_relations else 0
    total_kg_raw = entity_raw + relation_raw
    kg_budget_cap = max_length // 2
    if total_kg_raw > kg_budget_cap and total_kg_raw > 0:
        scale = kg_budget_cap / total_kg_raw
        entity_budget = int(entity_raw * scale)
        relation_budget = int(relation_raw * scale)
    else:
        entity_budget = entity_raw
        relation_budget = relation_raw
    base_chunk_budget = max_length - entity_budget - relation_budget - 100  # buffer separadores

    parts: List[str] = []
    unused_budget = 0

    # Seccion 1: Entidades (budget propio)
    if kg_entities and entity_budget > 0:
        section = _build_kg_section(
            "Knowledge Graph Data (Entity)", kg_entities, entity_budget,
        )
        if section:
            parts.append(section)
            unused_budget += max(0, entity_budget - len(section))
        else:
            unused_budget += entity_budget
    else:
        unused_budget += entity_budget

    # Seccion 2: Relaciones (budget propio)
    if kg_relations and relation_budget > 0:
        section = _build_kg_section(
            "Knowledge Graph Data (Relationship)", kg_relations, relation_budget,
        )
        if section:
            parts.append(section)
            unused_budget += max(0, relation_budget - len(section))
        else:
            unused_budget += relation_budget
    else:
        unused_budget += relation_budget

    # Seccion 3: Chunks (budget base + redistribuido)
    n_chunks_emitted = 0
    chunk_budget = base_chunk_budget + unused_budget
    if chunk_budget > 0 and contents:
        chunk_parts: List[str] = []
        chunk_used = 0
        for i, content in enumerate(contents, 1):
            entry = json.dumps(
                {"reference_id": i, "content": content},
                ensure_ascii=False,
            )
            if chunk_used + len(entry) + 1 > chunk_budget:
                break
            chunk_parts.append(entry)
            chunk_used += len(entry) + 1  # +1 for newline

        n_chunks_emitted = len(chunk_parts)
        if chunk_parts:
            chunks_str = "\n".join(chunk_parts)
            section = (
                "Document Chunks:\n\n"
                f"```json\n{chunks_str}\n```"
            )
            parts.append(section)

    if not parts:
        return "[No se encontraron documentos]", n_chunks_emitted

    return "\n\n".join(parts), n_chunks_emitted


def is_kg_budget_cap_triggered(max_length: int, mode: str) -> bool:
    """Indica si el cap del 50% del presupuesto KG se dispara.

    Retorna `True` si la suma de budgets brutos KG (entity + relation segun
    el modo) excede `max_length // 2` y por tanto se escalaron
    proporcionalmente. Logica identica a `format_structured_context`, pero
    expuesta para anotar el observable sin formatear contexto.

    Notas:
    - Con la config por defecto del paper (entity=24000, relation=32000
      chars; total=56000) y `max_length >= 112000`, el cap NO dispara y
      chunks reciben el budget base completo.
    - Con `max_length < 112000` en modo `hybrid`, o analogos en otros
      modos, el cap SI dispara y las 2 secciones KG se reducen.
    """
    use_entities, use_relations = _LIGHTRAG_MODE_SECTIONS.get(
        mode, _LIGHTRAG_MODE_SECTIONS["hybrid"]
    )
    entity_raw = KG_MAX_ENTITY_CONTEXT_CHARS if use_entities else 0
    relation_raw = KG_MAX_RELATION_CONTEXT_CHARS if use_relations else 0
    total_kg_raw = entity_raw + relation_raw
    kg_budget_cap = max_length // 2
    return total_kg_raw > kg_budget_cap and total_kg_raw > 0


__all__ = [
    "RetrievalExecutor",
    "format_context",
    "format_structured_context",
    "format_structured_context_with_stats",
    "is_kg_budget_cap_triggered",
]
