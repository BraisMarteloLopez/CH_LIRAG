"""
MTEBEvaluator: ejecucion unica de evaluacion RAG.

Flujo:
    load -> index -> pre-embed queries -> retrieve (sync, local) -> generate+metrics (async) -> EvaluationRun

Optimizacion v3.2: pre-embed queries en batch via REST NIM.
    - Batch embed: N queries en batches de EMBEDDING_BATCH_SIZE -> N vectores.
    - Retrieval: busqueda local ChromaDB por vector pre-computado (sin llamada NIM).
    - Resultado: fase de retrieval pasa de O(N * RTT_NIM) a O(N/batch * RTT_NIM + N * local_search).
"""

from __future__ import annotations

import gc
import logging
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.types import (
    EvaluationRun,
    EvaluationStatus,
    LoadedDataset,
    NormalizedQuery,
    QueryRetrievalDetail,
    QueryEvaluationResult,
    EmbeddingModelProtocol,
    get_dataset_config,
)
from shared.llm import AsyncLLMService, load_embedding_model, run_sync
from shared.metrics import (
    MetricsCalculator,
    get_judge_fallback_stats,
    max_judge_default_return_rate,
    reset_judge_fallback_stats,
)
from shared.operational_tracker import reset_operational_stats
from shared.retrieval import get_retriever, RetrievalStrategy
from shared.retrieval.core import BaseRetriever
from shared.retrieval.reranker import CrossEncoderReranker, HAS_NVIDIA_RERANK
from shared.structured_logging import structured_log

from .config import MTEBConfig
from .loader import MinIOLoader
from .result_builder import build_run
from .retrieval_executor import RetrievalExecutor, format_context

_CHUNK_SIZE = 50  # queries por chunk (progreso + memoria)
from .generation_executor import (
    GenerationExecutor,
    GenMetricsResult,
    get_kg_synthesis_stats,
    reset_kg_synthesis_stats,
)
from .embedding_service import batch_embed_queries, resolve_max_context_chars

logger = logging.getLogger(__name__)


def _format_query_exc(exc: BaseException) -> str:
    """Formato consistente para error_message de queries fallidas.

    Algunas excepciones (p.ej. `asyncio.TimeoutError` instanciado sin args)
    tienen `str(exc) == ""`, lo que dejaba `error_message` vacio y perdia
    el diagnostico. Incluimos siempre el tipo y caemos a repr() si el
    mensaje esta vacio.
    """
    return f"{type(exc).__name__}: {str(exc) or repr(exc)}"


# -----------------------------------------------------------------
# EVALUADOR
# -----------------------------------------------------------------

class MTEBEvaluator:
    """
    Evaluador single-run para datasets MTEB/BeIR.

    Un run = un dataset + un embedding + una estrategia.
    No hay bucle multi-modelo ni results_matrix.
    """

    def __init__(self, config: MTEBConfig):
        self.config = config
        self._embedding_model: Optional[EmbeddingModelProtocol] = None
        self._retriever: Optional[BaseRetriever] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._retrieval_executor: Optional[RetrievalExecutor] = None
        self._generation_executor: Optional[GenerationExecutor] = None
        self._llm_service: Optional[AsyncLLMService] = None
        self._metrics_calculator: Optional[MetricsCalculator] = None
        self._max_context_chars: int = 4000  # fallback por defecto

    def run(self) -> EvaluationRun:
        """Ejecuta una evaluacion completa y devuelve el EvaluationRun."""
        start_time = time.time()
        run_id = f"mteb_{self.config.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"

        self._log_run_start(run_id)

        # Resetear tracker del judge (judge_fallback_stats) al inicio de cada
        # run para que las tasas reflejen solo este run y no estado acumulado
        # de runs previos en el mismo proceso.
        reset_judge_fallback_stats()
        # Reset equivalente para el tracker de synthesis (kg_synthesis_stats).
        reset_kg_synthesis_stats()
        # Reset del tracker de eventos operacionales (operational_stats):
        # contadores de degradaciones silenciosas en 7 puntos del pipeline.
        reset_operational_stats()

        try:
            self._init_components()

            dataset, queries, corpus = self._load_and_prepare()

            self._index_documents(dataset.name, corpus, run_id)

            self._retrieval_executor = RetrievalExecutor(
                retriever=self._retriever,
                reranker=self._reranker,
                config=self.config,
            )

            ds_config = get_dataset_config(dataset.name)
            query_results = self._evaluate_queries(
                queries, ds_config, dataset.name, run_id=run_id
            )

            elapsed = time.time() - start_time
            evaluation_run = self._build_run(
                run_id=run_id,
                dataset=dataset,
                query_results=query_results,
                elapsed_seconds=elapsed,
                indexed_corpus_size=len(corpus),
            )

            self._log_run_complete(run_id, elapsed, evaluation_run)

            # 6. Validar tasa de fallback del judge (judge_fallback_stats).
            # Se hace DESPUES de construir el run para que las stats queden
            # persistidas en `config_snapshot._runtime` incluso si fallamos.
            self._validate_judge_fallback_threshold(run_id)

            return evaluation_run

        except Exception as e:
            logger.error(f"RUN FALLIDO: {e}")
            structured_log("run_failed", run_id=run_id, error=str(e))
            raise
        finally:
            self._cleanup()

    # -----------------------------------------------------------------
    # RUN PHASES
    # -----------------------------------------------------------------

    def _log_run_start(self, run_id: str) -> None:
        """Log header y evento estructurado de inicio."""
        logger.info("=" * 60)
        logger.info("MTEB EVALUATION RUN")
        logger.info(f"  Run ID:     {run_id}")
        logger.info(f"  Dataset:    {self.config.dataset_name}")
        logger.info(f"  Embedding:  {self.config.infra.embedding_model_name}")
        logger.info(f"  Strategy:   {self.config.retrieval.strategy.name}")
        if self.config.dev_mode:
            logger.info(f"  Mode:       DEV_MODE ({self.config.dev_queries} queries, {self.config.dev_corpus_size} corpus)")
        else:
            logger.info(f"  Queries:    {self.config.max_queries if self.config.max_queries > 0 else 'ALL'}")
            logger.info(f"  Corpus:     {self.config.max_corpus if self.config.max_corpus > 0 else 'ALL'}")
        logger.info("=" * 60)

        structured_log(
            "run_start",
            run_id=run_id,
            dataset=self.config.dataset_name,
            embedding=self.config.infra.embedding_model_name,
            strategy=self.config.retrieval.strategy.name,
            dev_mode=self.config.dev_mode,
        )

    def _load_and_prepare(self) -> tuple:
        """Carga dataset y selecciona subset de queries/corpus."""
        dataset = self._load_dataset()

        if self.config.dev_mode:
            if self.config.max_queries > 0 or self.config.max_corpus > 0:
                logger.info(
                    "  DEV_MODE activo, ignorando EVAL_MAX_QUERIES/EVAL_MAX_CORPUS"
                )
            queries, corpus = self._select_subset_dev(dataset)
        else:
            queries, corpus = self._select_subset_standard(dataset)

        logger.info(
            f"  Usando {len(queries)} queries, {len(corpus)} docs"
        )
        return dataset, queries, corpus

    def _select_subset_dev(
        self, dataset: LoadedDataset,
    ) -> Tuple[List[NormalizedQuery], Dict[str, Any]]:
        """DEV_MODE: gold docs garantizados + distractores aleatorios."""
        seed = self.config.corpus_shuffle_seed or 42
        rng = random.Random(seed)
        all_queries = list(dataset.queries)
        rng.shuffle(all_queries)
        queries = all_queries[: self.config.dev_queries]

        gold_ids = set()
        for q in queries:
            gold_ids.update(q.relevant_doc_ids)
        available_gold = gold_ids & set(dataset.corpus.keys())
        if len(available_gold) > self.config.dev_corpus_size:
            raise ValueError(
                f"DEV_MODE: gold docs ({len(available_gold)}) > "
                f"DEV_CORPUS_SIZE ({self.config.dev_corpus_size})."
            )

        corpus = {k: dataset.corpus[k] for k in available_gold}
        non_gold = [k for k in dataset.corpus if k not in gold_ids]
        rng.shuffle(non_gold)
        for doc_id in non_gold[: self.config.dev_corpus_size - len(corpus)]:
            corpus[doc_id] = dataset.corpus[doc_id]
        return queries, corpus

    def _select_subset_standard(
        self, dataset: LoadedDataset,
    ) -> Tuple[List[NormalizedQuery], Dict[str, Any]]:
        """Modo estandar: max_queries y max_corpus con shuffle (0 = todo)."""
        queries = (
            dataset.queries[: self.config.max_queries]
            if self.config.max_queries > 0 else dataset.queries
        )
        corpus_ids = list(dataset.corpus.keys())
        seed = self.config.corpus_shuffle_seed
        if seed is not None:
            random.Random(seed).shuffle(corpus_ids)
        else:
            logger.warning(
                "  CORPUS_SHUFFLE_SEED no configurado (riesgo de sesgo de orden)."
            )
        if self.config.max_corpus > 0:
            corpus_ids = corpus_ids[: self.config.max_corpus]
        return queries, {k: dataset.corpus[k] for k in corpus_ids}

    def _log_run_complete(
        self, run_id: str, elapsed: float, evaluation_run: EvaluationRun,
    ) -> None:
        """Log resumen y evento estructurado de fin de run."""
        logger.info(
            f"RUN COMPLETADO en {elapsed:.1f}s | "
            f"Hit@5={evaluation_run.avg_hit_rate_at_5:.4f} | "
            f"MRR={evaluation_run.avg_mrr:.4f}"
        )

        # Reporte visible de judge_fallback_stats (tasa de fallback del judge).
        judge_stats = get_judge_fallback_stats()
        if judge_stats:
            for metric_name, s in judge_stats.items():
                logger.info(
                    "  Judge fallback (%s): invocations=%d, "
                    "parse_failure_rate=%.2f%%, default_return_rate=%.2f%%",
                    metric_name,
                    s["invocations"],
                    s["parse_failure_rate"] * 100,
                    s["default_return_rate"] * 100,
                )

        # Reporte visible de kg_synthesis_stats (capa de synthesis del KG).
        synthesis_stats = get_kg_synthesis_stats()
        if synthesis_stats["invocations"] > 0:
            logger.info(
                "  KG synthesis: invocations=%d, successes=%d, "
                "errors=%d, empty=%d, truncations=%d, timeouts=%d, "
                "fallback_rate=%.2f%%",
                synthesis_stats["invocations"],
                synthesis_stats["successes"],
                synthesis_stats["errors"],
                synthesis_stats["empty_returns"],
                synthesis_stats["truncations"],
                synthesis_stats["timeouts"],
                synthesis_stats["fallback_rate"] * 100,
            )

        structured_log(
            "run_complete",
            run_id=run_id,
            elapsed_s=round(elapsed, 2),
            queries_evaluated=evaluation_run.num_queries_evaluated,
            queries_failed=evaluation_run.num_queries_failed,
            hit_at_5=round(evaluation_run.avg_hit_rate_at_5, 4),
            mrr=round(evaluation_run.avg_mrr, 4),
            avg_generation_score=(
                round(evaluation_run.avg_generation_score, 4)
                if evaluation_run.avg_generation_score is not None
                else None
            ),
            judge_fallback_stats=judge_stats,
            kg_synthesis_stats=synthesis_stats,
        )

    def _validate_judge_fallback_threshold(self, run_id: str) -> None:
        """Falla el run si algun judge metric supera el threshold configurado.

        Guardrail de `judge_fallback_stats`: protege P2 (experimento 3) y
        cualquier run sobre corpus especializados de reportar deltas artefacto
        causados por el judge devolviendo 0.5 por defecto. Las stats ya estan
        persistidas en `evaluation_run.config_snapshot._runtime` antes de
        llegar aqui, asi que el usuario siempre puede inspeccionarlas.
        """
        threshold = self.config.judge_fallback_threshold
        if threshold <= 0.0:
            return  # Validacion desactivada explicitamente

        stats = get_judge_fallback_stats()
        worst_metric, worst_rate = max_judge_default_return_rate(stats)

        if worst_metric is not None and worst_rate > threshold:
            msg = (
                f"Tasa de fallback del judge excedida: metrica "
                f"'{worst_metric}' devolvio 0.5 por defecto en "
                f"{worst_rate * 100:.2f}% de invocaciones "
                f"(umbral={threshold * 100:.2f}%). Las metricas del judge "
                f"estan sesgadas hacia el centro y los deltas entre "
                f"estrategias no son interpretables. "
                f"Sube JUDGE_FALLBACK_THRESHOLD (no recomendado) o "
                f"investiga por que el judge no produce JSON parseable."
            )
            logger.error(msg)
            structured_log(
                "run_judge_threshold_exceeded",
                run_id=run_id,
                metric=worst_metric,
                default_return_rate=round(worst_rate, 4),
                threshold=threshold,
            )
            raise RuntimeError(msg)

    # -----------------------------------------------------------------
    # INICIALIZACION
    # -----------------------------------------------------------------

    def _init_components(self) -> None:
        """Inicializa embedding, LLM, reranker, metrics."""
        from shared.retrieval.core import RetrievalStrategy

        # Embedding
        self._embedding_model = load_embedding_model(
            base_url=self.config.infra.embedding_base_url,
            model_name=self.config.infra.embedding_model_name,
            model_type=self.config.infra.embedding_model_type,
        )
        logger.info(f"  Embedding cargado: {self.config.infra.embedding_model_name}")

        # LLM: requerido si generacion activa O si LIGHT_RAG (triplet extraction)
        needs_llm = (
            self.config.generation_enabled
            or self.config.retrieval.strategy == RetrievalStrategy.LIGHT_RAG
        )
        if needs_llm:
            self._llm_service = AsyncLLMService(
                base_url=self.config.infra.llm_base_url,
                model_name=self.config.infra.llm_model_name,
                max_concurrent=self.config.infra.nim_max_concurrent,
                timeout_seconds=self.config.infra.nim_timeout,
                max_retries=self.config.infra.nim_max_retries,
            )

        # Metrics calculator
        if self._llm_service and self.config.generation_enabled:
            self._metrics_calculator = MetricsCalculator(
                llm_judge=self._llm_service,
                embedding_model=self._embedding_model,
            )
        else:
            self._metrics_calculator = MetricsCalculator(
                embedding_model=self._embedding_model,
            )

        # Reranker (opcional). LightRAG no usa reranker: el cross-encoder
        # single-hop penaliza chunks multi-hop del KG. Guard aqui para que
        # no se instancie (conexion HTTP al NIM desperdiciada) cuando la
        # estrategia lo ignora.
        if self.config.reranker.enabled:
            if self.config.retrieval.strategy == RetrievalStrategy.LIGHT_RAG:
                logger.warning(
                    "  Reranker habilitado en .env pero estrategia es LIGHT_RAG; "
                    "omitiendo inicializacion (LightRAG no usa reranker)."
                )
            elif not HAS_NVIDIA_RERANK:
                logger.warning("Reranker habilitado pero NVIDIARerank no disponible")
            elif self.config.reranker.base_url and self.config.reranker.model_name:
                try:
                    self._reranker = CrossEncoderReranker(
                        base_url=self.config.reranker.base_url,
                        model_name=self.config.reranker.model_name,
                    )
                    logger.info(f"  Reranker activo: {self.config.reranker.model_name}")
                except Exception as e:
                    logger.error(f"  Error inicializando reranker: {e}")

        # Limite de contexto para generacion
        if self.config.generation_enabled and self._llm_service:
            self._max_context_chars = resolve_max_context_chars(self.config)

            # Synthesis del contexto KG: solo tiene sentido para LIGHT_RAG
            # (SIMPLE_VECTOR no produce kg_entities/kg_relations).
            kg_synthesis_enabled = (
                self.config.kg_synthesis_enabled
                and self.config.retrieval.strategy == RetrievalStrategy.LIGHT_RAG
            )

            # Generation executor
            self._generation_executor = GenerationExecutor(
                llm_service=self._llm_service,
                metrics_calculator=self._metrics_calculator,
                max_context_chars=self._max_context_chars,
                kg_synthesis_enabled=kg_synthesis_enabled,
                kg_synthesis_max_chars=self.config.kg_synthesis_max_chars,
                kg_synthesis_timeout_s=self.config.kg_synthesis_timeout_s,
            )

    # -----------------------------------------------------------------
    # DATA LOADING
    # -----------------------------------------------------------------

    def _load_dataset(self) -> LoadedDataset:
        """Carga dataset desde MinIO."""
        loader = MinIOLoader(self.config.storage)

        if not loader.check_connection():
            raise ConnectionError(
                f"No se puede conectar a MinIO: {self.config.storage.minio_endpoint}"
            )

        dataset = loader.load_dataset(self.config.dataset_name)

        if dataset.load_status != "success":
            raise RuntimeError(
                f"Fallo carga de '{self.config.dataset_name}': "
                f"{dataset.error_message}"
            )

        return dataset

    # -----------------------------------------------------------------
    # INDEXACION
    # -----------------------------------------------------------------

    def _index_documents(
        self, dataset_name: str, corpus: Dict[str, Any],
        run_id: str = "",
    ) -> None:
        """Crea retriever e indexa el corpus."""
        # Usar run_id (unico y determinista) en lugar de uuid corto (8 hex
        # = 32 bits) para eliminar riesgo de colision en ejecuciones paralelas.
        collection_name = f"eval_{run_id}" if run_id else f"eval_{dataset_name}_{uuid.uuid4().hex[:8]}"

        assert self._embedding_model is not None, "_init_components must set _embedding_model"
        self._retriever = get_retriever(
            config=self.config.retrieval,
            embedding_model=self._embedding_model,
            collection_name=collection_name,
            embedding_batch_size=self.config.infra.embedding_batch_size,
            llm_service=self._llm_service,
        )

        documents = [
            {
                "doc_id": doc_id,
                "content": doc.get_full_text(),
                "title": doc.title or "",
            }
            for doc_id, doc in corpus.items()
        ]

        n_docs = len(documents)
        if self.config.retrieval.strategy == RetrievalStrategy.LIGHT_RAG:
            kg_cache = self.config.retrieval.kg_cache_dir
            if kg_cache:
                logger.info(
                    f"  LIGHT_RAG: KG cache habilitado en '{kg_cache}'. "
                    f"Se reutilizara el grafo si el corpus coincide."
                )
            else:
                concurrent = self.config.infra.nim_max_concurrent
                avg_latency_s = 2.0  # Estimacion conservadora por llamada LLM
                est_minutes = (n_docs / max(concurrent, 1)) * avg_latency_s / 60
                logger.warning(
                    f"  LIGHT_RAG: indexacion hara ~{n_docs} llamadas LLM "
                    f"para extraccion de tripletas. "
                    f"Estimacion: ~{est_minutes:.0f} min "
                    f"({n_docs} docs, {concurrent} concurrentes, ~{avg_latency_s:.0f}s/llamada). "
                    f"Tip: configura KG_CACHE_DIR para persistir el grafo entre runs."
                )
        logger.info(f"  Indexando {n_docs} documentos...")
        success = self._retriever.index_documents(
            documents, collection_name=collection_name
        )
        if not success:
            raise RuntimeError("Fallo indexacion de documentos")

        logger.info("  Indexacion completada")

    # -----------------------------------------------------------------
    # EVALUACION DE QUERIES (PIPELINE ASYNC)
    # -----------------------------------------------------------------

    def _evaluate_queries(
        self,
        queries: List[NormalizedQuery],
        ds_config: Dict[str, Any],
        dataset_name: str,
        run_id: str = "",
    ) -> List[QueryEvaluationResult]:
        """Pipeline de evaluacion: pre-embed + retrieval/generation/metrics por chunks."""
        n = len(queries)

        # --- Fase 0: Pre-embed queries ---
        query_texts = [q.query_text for q in queries]
        query_vectors = batch_embed_queries(query_texts, self.config)
        use_preembed = len(query_vectors) == n
        if not use_preembed:
            logger.warning(
                "  Pre-embed fallido o incompleto. "
                "Usando retrieval con embedding por query (lento)."
            )

        # --- Fase 0b: Pre-extract query keywords (LIGHT_RAG) ---
        if self.config.retrieval.strategy == RetrievalStrategy.LIGHT_RAG:
            from shared.retrieval.lightrag.retriever import LightRAGRetriever
            if isinstance(self._retriever, LightRAGRetriever):
                logger.info(f"  Pre-extrayendo keywords de {n} queries (batch LLM)...")
                t_kw = time.time()
                self._retriever.pre_extract_query_keywords(query_texts)
                logger.info(f"  Keywords pre-extraidas en {time.time() - t_kw:.1f}s")

        # --- Chunked processing: Retrieval + Generation + Metrics ---
        all_results: List[QueryEvaluationResult] = []

        for chunk_start in range(0, n, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, n)
            chunk_queries = queries[chunk_start:chunk_end]
            chunk_vectors = query_vectors[chunk_start:chunk_end] if use_preembed else []
            n_chunk = len(chunk_queries)

            logger.info(
                f"  Chunk {chunk_start // _CHUNK_SIZE + 1}: "
                f"queries {chunk_start + 1}-{chunk_end}/{n} ({n_chunk} queries)"
            )

            # Retrieval (sync)
            retrievals: List[QueryRetrievalDetail] = []
            rerank_statuses: List[Optional[bool]] = []
            for i, query in enumerate(chunk_queries):
                vector = chunk_vectors[i] if use_preembed else None
                assert self._retrieval_executor is not None
                detail, reranked_ok = self._retrieval_executor.execute(
                    query.query_text, query.relevant_doc_ids,
                    query_vector=vector,
                )
                retrievals.append(detail)
                rerank_statuses.append(reranked_ok)

            # Generation + Metrics (async)
            gen_metrics_results: List[Optional[GenMetricsResult]] = [None] * n_chunk
            gen_errors: List[Optional[BaseException]] = [None] * n_chunk
            if self.config.generation_enabled and self._generation_executor:
                raw = run_sync(
                    self._generation_executor.batch_generate_and_evaluate(
                        chunk_queries, retrievals, ds_config, dataset_name
                    )
                )
                for idx, item in enumerate(raw):
                    if isinstance(item, BaseException):
                        gen_errors[idx] = item
                        logger.warning(
                            f"  Error async query {chunk_queries[idx].query_id}: "
                            f"{_format_query_exc(item)}"
                        )
                    else:
                        gen_metrics_results[idx] = item

            chunk_results = self._assemble_results(
                chunk_queries, retrievals, gen_metrics_results,
                rerank_statuses, ds_config, dataset_name,
                gen_errors=gen_errors,
            )
            all_results.extend(chunk_results)

        assert self._retrieval_executor is not None
        if self._retrieval_executor.strategy_mismatches > 0:
            logger.error(
                f"  STRATEGY MISMATCH en {self._retrieval_executor.strategy_mismatches}/{n} queries: "
                f"configurado={self.config.retrieval.strategy.name}, "
                f"ejecutado distinto. Resultados NO representan la estrategia configurada."
            )
        if self._reranker and self._retrieval_executor.rerank_failures > 0:
            logger.warning(
                f"  Rerank failures: {self._retrieval_executor.rerank_failures}/{n} queries "
                f"usaron fallback sin reranking"
            )

        completed = sum(1 for r in all_results if r.status == EvaluationStatus.COMPLETED)
        failed = sum(1 for r in all_results if r.status == EvaluationStatus.FAILED)
        if failed:
            logger.warning(f"  Queries: {completed} completadas, {failed} fallidas")

        return all_results

    def _assemble_results(
        self,
        queries: List[NormalizedQuery],
        retrievals: List[QueryRetrievalDetail],
        gen_metrics_results: List[Optional[GenMetricsResult]],
        rerank_statuses: List[Optional[bool]],
        ds_config: Dict[str, Any],
        dataset_name: str,
        gen_errors: Optional[List[Optional[BaseException]]] = None,
    ) -> List[QueryEvaluationResult]:
        """Ensambla QueryEvaluationResult desde retrieval + generation results.

        gen_errors: lista paralela a gen_metrics_results con la excepcion
        capturada (si la hubo) para cada query. Se usa para poblar
        error_message en los FAILED.
        """
        if gen_errors is None:
            gen_errors = [None] * len(queries)
        results: List[QueryEvaluationResult] = []
        for query, retrieval, gm, reranked_status, exc in zip(
            queries, retrievals, gen_metrics_results, rerank_statuses, gen_errors,
        ):
            # Passthrough generico de metadata de la query.
            qr_metadata: Dict[str, Any] = {
                k: v for k, v in query.metadata.items() if v
            }
            if reranked_status is not None:
                qr_metadata["reranked"] = reranked_status

            if gm is not None:
                results.append(QueryEvaluationResult(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    dataset_name=dataset_name,
                    dataset_type=ds_config["type"],
                    retrieval=retrieval,
                    generation=gm.generation,
                    expected_response=query.expected_answer,
                    primary_metric_type=gm.primary_metric_type,
                    primary_metric_value=gm.primary_metric_value,
                    secondary_metrics=gm.secondary_metrics,
                    status=EvaluationStatus.COMPLETED,
                    metadata=qr_metadata,
                ))
            elif self.config.generation_enabled:
                # Si hubo excepcion, preservar tipo+mensaje (diagnostico post-run).
                error_message = (
                    _format_query_exc(exc) if exc is not None
                    else "Error en generacion/metricas async"
                )
                results.append(QueryEvaluationResult(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    dataset_name=dataset_name,
                    dataset_type=ds_config["type"],
                    retrieval=retrieval,
                    status=EvaluationStatus.FAILED,
                    error_message=error_message,
                    metadata=qr_metadata,
                ))
            else:
                results.append(QueryEvaluationResult(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    dataset_name=dataset_name,
                    dataset_type=ds_config["type"],
                    retrieval=retrieval,
                    expected_response=query.expected_answer,
                    status=EvaluationStatus.COMPLETED,
                    metadata=qr_metadata,
                ))
        return results

    def _build_run(
        self,
        run_id: str,
        dataset: LoadedDataset,
        query_results: List[QueryEvaluationResult],
        elapsed_seconds: float,
        indexed_corpus_size: int = 0,
    ) -> EvaluationRun:
        return build_run(
            config=self.config,
            run_id=run_id,
            dataset=dataset,
            query_results=query_results,
            elapsed_seconds=elapsed_seconds,
            indexed_corpus_size=indexed_corpus_size,
            max_context_chars=self._max_context_chars,
            rerank_failures=self._retrieval_executor.rerank_failures if self._retrieval_executor else 0,
            strategy_mismatches=self._retrieval_executor.strategy_mismatches if self._retrieval_executor else 0,
        )

    # -----------------------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------------------

    def _cleanup(self) -> None:
        if self._retriever:
            try:
                self._retriever.clear_index()
            except Exception as e:
                logger.debug("Error en cleanup de retriever (no fatal): %s", e)
            self._retriever = None
        self._embedding_model = None
        self._llm_service = None
        gc.collect()


__all__ = ["MTEBEvaluator"]
