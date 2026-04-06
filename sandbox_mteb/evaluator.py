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
import time
import uuid
from typing import Any, Dict, List, Optional, Set

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
from shared.metrics import MetricsCalculator
from shared.retrieval import get_retriever, RetrievalStrategy
from shared.retrieval.core import BaseRetriever
from shared.retrieval.reranker import CrossEncoderReranker, HAS_NVIDIA_RERANK
from shared.structured_logging import structured_log

from .config import MTEBConfig
from .loader import MinIOLoader
from .checkpoint import (
    CHECKPOINT_CHUNK_SIZE,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint,
)
from .result_builder import build_run
from .subset_selection import select_subset_dev, select_subset_standard
from .retrieval_executor import RetrievalExecutor, format_context
from .generation_executor import GenerationExecutor, GenMetricsResult
from .embedding_service import batch_embed_queries, resolve_max_context_chars

logger = logging.getLogger(__name__)


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

    def run(self, resume_run_id: Optional[str] = None) -> EvaluationRun:
        """
        Ejecuta una evaluacion completa.

        Args:
            resume_run_id: Si se proporciona, reanuda un run previo desde
                su checkpoint. Re-indexa (idempotente) pero salta queries
                ya evaluadas.

        Returns:
            EvaluationRun con todos los resultados.
        """
        start_time = time.time()
        run_id = resume_run_id or f"mteb_{self.config.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"

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

        # DT-3: evento estructurado de inicio de run
        structured_log(
            "run_start",
            run_id=run_id,
            dataset=self.config.dataset_name,
            embedding=self.config.infra.embedding_model_name,
            strategy=self.config.retrieval.strategy.name,
            dev_mode=self.config.dev_mode,
        )

        try:
            # 1. Inicializar componentes
            self._init_components()

            # 2. Cargar dataset
            dataset = self._load_dataset()

            # 3. Seleccionar subset
            if self.config.dev_mode:
                if self.config.max_queries > 0 or self.config.max_corpus > 0:
                    logger.info(
                        "  DEV_MODE activo, ignorando EVAL_MAX_QUERIES/EVAL_MAX_CORPUS"
                    )
                queries, corpus = select_subset_dev(dataset, self.config)
            else:
                queries, corpus = select_subset_standard(dataset, self.config)

            logger.info(
                f"  Usando {len(queries)} queries, {len(corpus)} docs"
            )

            # 4. Indexar documentos
            self._index_documents(dataset.name, corpus, run_id)

            # 4b. Crear retrieval executor
            self._retrieval_executor = RetrievalExecutor(
                retriever=self._retriever,
                reranker=self._reranker,
                config=self.config,
            )

            # 5. Evaluar queries (pipeline async)
            ds_config = get_dataset_config(dataset.name)
            query_results = self._evaluate_queries(
                queries, ds_config, dataset.name, run_id=run_id
            )

            # 6. Construir EvaluationRun
            elapsed = time.time() - start_time
            evaluation_run = self._build_run(
                run_id=run_id,
                dataset=dataset,
                query_results=query_results,
                elapsed_seconds=elapsed,
                indexed_corpus_size=len(corpus),
            )

            logger.info(
                f"RUN COMPLETADO en {elapsed:.1f}s | "
                f"Hit@5={evaluation_run.avg_hit_rate_at_5:.4f} | "
                f"MRR={evaluation_run.avg_mrr:.4f}"
            )

            # DT-3: evento estructurado de fin de run
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
            )

            return evaluation_run

        except Exception as e:
            logger.error(f"RUN FALLIDO: {e}")
            structured_log("run_failed", run_id=run_id, error=str(e))
            raise
        finally:
            self._cleanup()

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

        # Reranker (opcional)
        if self.config.reranker.enabled:
            if not HAS_NVIDIA_RERANK:
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

            # Generation executor
            self._generation_executor = GenerationExecutor(
                llm_service=self._llm_service,
                metrics_calculator=self._metrics_calculator,
                max_context_chars=self._max_context_chars,
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
        # FIX DTm-10: usar run_id (unico y determinista) en lugar de uuid
        # corto (8 hex = 32 bits) para eliminar riesgo de colision en
        # ejecuciones paralelas.
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
        """
        Pipeline de evaluacion con checkpoint/resume (DTm-36).

        Fases:
          0. Pre-embed: batch embed queries via REST NIM
          1. Retrieval + Generation + Metrics en chunks de _CHECKPOINT_CHUNK_SIZE
             Con checkpoint despues de cada chunk.

        Si existe un checkpoint previo para run_id, las queries ya evaluadas
        se saltan y sus resultados se reutilizan.
        """
        n = len(queries)

        # --- Resume: cargar checkpoint previo ---
        checkpoint_results: List[QueryEvaluationResult] = []
        evaluated_ids: Set[str] = set()
        if run_id:
            loaded = load_checkpoint(str(self.config.storage.evaluation_results_dir), run_id)
            if loaded:
                evaluated_ids, checkpoint_results = loaded
                logger.info(
                    f"  Resumiendo: {len(evaluated_ids)}/{n} queries ya evaluadas, "
                    f"{n - len(evaluated_ids)} pendientes"
                )

        # Filtrar queries pendientes
        pending_queries = [q for q in queries if q.query_id not in evaluated_ids]
        n_pending = len(pending_queries)

        if n_pending == 0:
            logger.info("  Todas las queries ya evaluadas (checkpoint completo)")
            return checkpoint_results

        # --- Fase 0: Pre-embed queries pendientes ---
        pending_texts = [q.query_text for q in pending_queries]
        query_vectors = batch_embed_queries(pending_texts, self.config)
        use_preembed = len(query_vectors) == n_pending
        if not use_preembed:
            logger.warning(
                "  Pre-embed fallido o incompleto. "
                "Usando retrieval con embedding por query (lento)."
            )

        # --- Fase 0b: Pre-extract query keywords (LIGHT_RAG) ---
        if self.config.retrieval.strategy == RetrievalStrategy.LIGHT_RAG:
            from shared.retrieval.lightrag.retriever import LightRAGRetriever
            if isinstance(self._retriever, LightRAGRetriever):
                logger.info(
                    f"  Pre-extrayendo keywords de {n_pending} queries (batch LLM)..."
                )
                t_kw = time.time()
                self._retriever.pre_extract_query_keywords(pending_texts)
                logger.info(
                    f"  Keywords pre-extraidas en {time.time() - t_kw:.1f}s"
                )

        # --- Chunked processing: Retrieval + Generation + Metrics ---
        chunk_size = CHECKPOINT_CHUNK_SIZE
        all_results = list(checkpoint_results)  # start with resumed results

        for chunk_start in range(0, n_pending, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_pending)
            chunk_queries = pending_queries[chunk_start:chunk_end]
            chunk_vectors = query_vectors[chunk_start:chunk_end] if use_preembed else []
            n_chunk = len(chunk_queries)

            logger.info(
                f"  Chunk {chunk_start // chunk_size + 1}: "
                f"queries {len(evaluated_ids) + chunk_start + 1}-"
                f"{len(evaluated_ids) + chunk_end}/{n} "
                f"({n_chunk} queries)"
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
            if self.config.generation_enabled and self._generation_executor:
                raw = run_sync(
                    self._generation_executor.batch_generate_and_evaluate(
                        chunk_queries, retrievals, ds_config, dataset_name
                    )
                )
                for idx, item in enumerate(raw):
                    if isinstance(item, Exception):
                        logger.warning(
                            f"  Error async query {chunk_queries[idx].query_id}: {item}"
                        )
                    else:
                        gen_metrics_results[idx] = item

            # Assemble chunk results
            chunk_results = self._assemble_results(
                chunk_queries, retrievals, gen_metrics_results,
                rerank_statuses, ds_config, dataset_name,
            )
            all_results.extend(chunk_results)

            # Update evaluated IDs and save checkpoint
            for qr in chunk_results:
                evaluated_ids.add(qr.query_id)

            if run_id:
                save_checkpoint(str(self.config.storage.evaluation_results_dir), run_id, evaluated_ids, all_results)

        # Log summary
        assert self._retrieval_executor is not None  # initialized in _init_components
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

        # Clean up checkpoint on successful completion
        if run_id:
            delete_checkpoint(str(self.config.storage.evaluation_results_dir), run_id)

        return all_results

    def _assemble_results(
        self,
        queries: List[NormalizedQuery],
        retrievals: List[QueryRetrievalDetail],
        gen_metrics_results: List[Optional[GenMetricsResult]],
        rerank_statuses: List[Optional[bool]],
        ds_config: Dict[str, Any],
        dataset_name: str,
    ) -> List[QueryEvaluationResult]:
        """Ensambla QueryEvaluationResult desde retrieval + generation results."""
        results: List[QueryEvaluationResult] = []
        for query, retrieval, gm, reranked_status in zip(
            queries, retrievals, gen_metrics_results, rerank_statuses,
        ):
            # DTm-20: passthrough generico de metadata de la query
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
                results.append(QueryEvaluationResult(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    dataset_name=dataset_name,
                    dataset_type=ds_config["type"],
                    retrieval=retrieval,
                    status=EvaluationStatus.FAILED,
                    error_message="Error en generacion/metricas async",
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

    # -----------------------------------------------------------------
    # BUILD RUN (delegado a result_builder.py — DTm-36 fase 4)
    # -----------------------------------------------------------------

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
