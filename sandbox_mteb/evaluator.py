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

import asyncio
import gc
import json
import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.types import (
    EvaluationRun,
    EvaluationStatus,
    LoadedDataset,
    NormalizedQuery,
    QueryRetrievalDetail,
    GenerationResult,
    QueryEvaluationResult,
    DatasetType,
    MetricType,
    LLMJudgeProtocol,
    EmbeddingModelProtocol,
    get_dataset_config,
)
from shared.llm import AsyncLLMService, load_embedding_model, run_sync
from shared.metrics import MetricsCalculator, MetricResult
from shared.retrieval import get_retriever, RetrievalStrategy
from shared.retrieval.core import BaseRetriever
from shared.retrieval.reranker import CrossEncoderReranker, HAS_NVIDIA_RERANK
from shared.structured_logging import structured_log

from .config import MTEBConfig, GENERATION_PROMPTS
from .loader import MinIOLoader

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# TIPO INTERNO
# -----------------------------------------------------------------

class _GenMetricsResult:
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
        self._llm_service: Optional[AsyncLLMService] = None
        self._metrics_calculator: Optional[MetricsCalculator] = None
        self._max_context_chars: int = 4000  # fallback por defecto
        self._rerank_failures: int = 0
        self._strategy_mismatches: int = 0

    # -----------------------------------------------------------------
    # CONTEXT WINDOW DETECTION
    # -----------------------------------------------------------------

    # Constantes para derivar limite de contexto desde model context window.
    _CHARS_PER_TOKEN: float = 4.0
    _OVERHEAD_TOKENS: int = 1024  # system prompt + user template + max_output

    def _query_model_context_window(self) -> Optional[int]:
        """
        Consulta GET /v1/models del LLM NIM para obtener max_model_len.

        Returns:
            max_model_len en tokens, o None si no se puede obtener.
        """
        import urllib.request
        import json

        base_url = self.config.infra.llm_base_url.rstrip("/")
        url = f"{base_url}/models"

        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            models = data.get("data", [])
            if not models:
                logger.warning("GET /v1/models: respuesta vacia")
                return None

            max_model_len = models[0].get("max_model_len")
            if max_model_len and isinstance(max_model_len, (int, float)):
                logger.info(
                    f"  LLM context window: {int(max_model_len)} tokens "
                    f"(modelo: {models[0].get('id', 'unknown')})"
                )
                return int(max_model_len)

            logger.warning("GET /v1/models: max_model_len no encontrado en respuesta")
            return None

        except Exception as e:
            logger.warning(f"No se pudo consultar context window del LLM: {e}")
            return None

    def _resolve_max_context_chars(self) -> int:
        """
        Determina el limite de caracteres para contexto de generacion.

        Prioridad:
          1. Override manual via GENERATION_MAX_CONTEXT_CHARS > 0
          2. Derivado del context window del modelo via /v1/models
          3. Fallback hardcodeado: 4000 chars
        """
        fallback = 4000

        # 1. Override manual
        if self.config.generation_max_context_chars > 0:
            logger.info(
                f"  Context chars: {self.config.generation_max_context_chars} "
                "(override manual via GENERATION_MAX_CONTEXT_CHARS)"
            )
            return self.config.generation_max_context_chars

        # 2. Derivar del modelo
        max_model_len = self._query_model_context_window()
        if max_model_len is not None:
            available_tokens = max_model_len - self._OVERHEAD_TOKENS
            if available_tokens <= 0:
                logger.warning(
                    f"  Context window ({max_model_len}) menor que overhead "
                    f"({self._OVERHEAD_TOKENS}). Usando fallback={fallback}"
                )
                return fallback

            derived_chars = int(available_tokens * self._CHARS_PER_TOKEN)
            logger.info(
                f"  Context chars: {derived_chars} "
                f"(derivado: {max_model_len} tokens - {self._OVERHEAD_TOKENS} overhead "
                f"= {available_tokens} tokens * {self._CHARS_PER_TOKEN} chars/token)"
            )
            return derived_chars

        # 3. Fallback
        logger.info(f"  Context chars: {fallback} (fallback por defecto)")
        return fallback

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
                queries, corpus = self._select_subset_dev(dataset)
            else:
                # --- flujo estandar (codigo original) ---
                # max_queries=0 / max_corpus=0 significa "usar todo"
                if self.config.max_queries > 0:
                    queries = dataset.queries[: self.config.max_queries]
                else:
                    queries = dataset.queries

                # Shuffle corpus antes de slice para evitar sesgo de orden.
                # Sin shuffle, corpus[0:N] puede contener artificialmente
                # todos los docs relevantes para queries[0:M] (alineacion
                # por posicion detectada en HotpotQA Parquet).
                corpus_ids = list(dataset.corpus.keys())
                corpus_seed = self.config.corpus_shuffle_seed
                if corpus_seed is not None:
                    # Instancia aislada para no contaminar RNG global (DTm-32)
                    rng_corpus = random.Random(corpus_seed)
                    rng_corpus.shuffle(corpus_ids)
                    logger.info(f"  Corpus shuffled con seed={corpus_seed}")
                else:
                    logger.warning(
                        "  CORPUS_SHUFFLE_SEED no configurado. "
                        "Corpus NO shuffled (riesgo de sesgo de orden)."
                    )

                if self.config.max_corpus > 0:
                    corpus_ids = corpus_ids[: self.config.max_corpus]
                # else: usar todo el corpus (max_corpus=0)

                corpus = {k: dataset.corpus[k] for k in corpus_ids}

            logger.info(
                f"  Usando {len(queries)} queries, {len(corpus)} docs"
            )

            # 4. Indexar documentos
            self._index_documents(dataset.name, corpus, run_id)

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
            self._max_context_chars = self._resolve_max_context_chars()

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
    # SUBSET SELECTION (DEV_MODE)
    # -----------------------------------------------------------------

    def _select_subset_dev(
        self, dataset: LoadedDataset,
    ) -> Tuple[List[NormalizedQuery], Dict[str, Any]]:
        """
        Subset para DEV_MODE: gold docs garantizados en corpus.

        1. Shuffle queries con seed, tomar dev_queries
        2. Recopilar gold doc_ids de queries seleccionadas
        3. Incluir gold docs en corpus
        4. Rellenar con distractores aleatorios hasta dev_corpus_size
        """
        seed = self.config.corpus_shuffle_seed or 42
        rng = random.Random(seed)
        dev_queries = self.config.dev_queries
        dev_corpus_size = self.config.dev_corpus_size

        # 1. Seleccionar queries
        all_queries = list(dataset.queries)
        rng.shuffle(all_queries)
        if dev_queries >= len(all_queries):
            logger.warning(
                f"  DEV_MODE: dev_queries ({dev_queries}) >= total queries "
                f"({len(all_queries)}). Usando todas."
            )
        queries = all_queries[:dev_queries]

        # 2. Recopilar gold docs
        gold_ids = set()
        for q in queries:
            gold_ids.update(q.relevant_doc_ids)

        available_gold = gold_ids & set(dataset.corpus.keys())
        missing = gold_ids - available_gold
        if missing:
            logger.warning(
                f"  DEV_MODE: {len(missing)} gold docs ausentes en corpus"
            )

        if len(available_gold) > dev_corpus_size:
            raise ValueError(
                f"DEV_MODE: gold docs ({len(available_gold)}) > "
                f"DEV_CORPUS_SIZE ({dev_corpus_size}). Aumentar DEV_CORPUS_SIZE."
            )

        # 3. Corpus: gold obligatorios + distractores aleatorios
        corpus = {k: dataset.corpus[k] for k in available_gold}

        non_gold = [k for k in dataset.corpus if k not in gold_ids]
        rng.shuffle(non_gold)
        n_distractors = dev_corpus_size - len(corpus)
        for doc_id in non_gold[:n_distractors]:
            corpus[doc_id] = dataset.corpus[doc_id]

        n_gold = len(available_gold)
        logger.info(
            f"  DEV_MODE: {len(queries)} queries, "
            f"{n_gold} gold docs, "
            f"{len(corpus) - n_gold} distractores, "
            f"{len(corpus)} corpus total"
        )

        return queries, corpus

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
    # PRE-EMBED QUERIES (batch REST al NIM)
    # -----------------------------------------------------------------

    def _batch_embed_queries(
        self, query_texts: List[str]
    ) -> List[List[float]]:
        """
        Embebe todas las queries en batch via REST al NIM de embeddings.

        Usa input_type=query para modelos asimetricos (el NIM distingue
        entre query y passage). Para modelos simetricos, input_type se omite.

        Returns:
            Lista de vectores, uno por query. Si falla, retorna lista vacia
            y el caller debe hacer fallback a retrieval sin pre-embed.
        """
        import json
        import urllib.request

        n = len(query_texts)
        batch_size = self.config.infra.embedding_batch_size or 5
        base_url = self.config.infra.embedding_base_url.rstrip("/")
        model_name = self.config.infra.embedding_model_name
        model_type = self.config.infra.embedding_model_type
        url = f"{base_url}/embeddings"

        all_vectors: List[List[float]] = []

        logger.info(
            f"  Pre-embedding {n} queries (batch={batch_size}, "
            f"type={model_type})..."
        )
        t0 = time.time()

        for batch_start in range(0, n, batch_size):
            batch = query_texts[batch_start : batch_start + batch_size]

            payload: Dict[str, Any] = {
                "input": batch,
                "model": model_name,
            }
            # Modelos asimetricos requieren input_type
            if model_type == "asymmetric":
                payload["input_type"] = "query"

            # FIX DTm-6: retry por batch antes de abandonar todo el pre-embed.
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    body = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(
                        url, data=body, method="POST",
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))

                    # Ordenar por index (la API puede devolver desordenado)
                    items = sorted(data["data"], key=lambda x: x["index"])
                    for item in items:
                        all_vectors.append(item["embedding"])
                    break  # batch OK

                except Exception as e:
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        logger.warning(
                            f"  Batch embed retry {attempt + 1}/{max_retries} "
                            f"(offset={batch_start}): {e}. "
                            f"Reintentando en {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        logger.warning(
                            f"  Error en batch embed (offset={batch_start}) "
                            f"tras {max_retries + 1} intentos: {e}. "
                            "Fallback a retrieval sin pre-embed."
                        )
                        return []

            batch_end = batch_start + len(batch)
            if batch_end % 500 == 0 or batch_end == n:
                logger.info(f"  Queries embebidas: {batch_end}/{n}")

        elapsed = time.time() - t0
        logger.info(
            f"  Pre-embedding completado: {n} queries en {elapsed:.1f}s "
            f"({n / elapsed:.0f} queries/s)"
        )
        return all_vectors

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
            loaded = self._load_checkpoint(run_id)
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
        query_vectors = self._batch_embed_queries(pending_texts)
        use_preembed = len(query_vectors) == n_pending
        if not use_preembed:
            logger.warning(
                "  Pre-embed fallido o incompleto. "
                "Usando retrieval con embedding por query (lento)."
            )

        # --- Fase 0b: Pre-extract query keywords (LIGHT_RAG) ---
        if self.config.retrieval.strategy == RetrievalStrategy.LIGHT_RAG:
            from shared.retrieval.lightrag_retriever import LightRAGRetriever
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
        chunk_size = self._CHECKPOINT_CHUNK_SIZE
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
                detail, reranked_ok = self._execute_retrieval(
                    query.query_text, query.relevant_doc_ids,
                    query_vector=vector,
                )
                retrievals.append(detail)
                rerank_statuses.append(reranked_ok)

            # Generation + Metrics (async)
            gen_metrics_results: List[Optional[_GenMetricsResult]] = [None] * n_chunk
            if self.config.generation_enabled and self._llm_service:
                raw = run_sync(
                    self._batch_generate_and_evaluate(
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
                self._save_checkpoint(run_id, evaluated_ids, all_results)

        # Log summary
        if self._strategy_mismatches > 0:
            logger.error(
                f"  STRATEGY MISMATCH en {self._strategy_mismatches}/{n} queries: "
                f"configurado={self.config.retrieval.strategy.name}, "
                f"ejecutado distinto. Resultados NO representan la estrategia configurada."
            )
        if self._reranker and self._rerank_failures > 0:
            logger.warning(
                f"  Rerank failures: {self._rerank_failures}/{n} queries "
                f"usaron fallback sin reranking"
            )

        completed = sum(1 for r in all_results if r.status == EvaluationStatus.COMPLETED)
        failed = sum(1 for r in all_results if r.status == EvaluationStatus.FAILED)
        if failed:
            logger.warning(f"  Queries: {completed} completadas, {failed} fallidas")

        # Clean up checkpoint on successful completion
        if run_id:
            self._delete_checkpoint(run_id)

        return all_results

    def _assemble_results(
        self,
        queries: List[NormalizedQuery],
        retrievals: List[QueryRetrievalDetail],
        gen_metrics_results: List[Optional[_GenMetricsResult]],
        rerank_statuses: List[Optional[bool]],
        ds_config: Dict[str, Any],
        dataset_name: str,
    ) -> List[QueryEvaluationResult]:
        """Ensambla QueryEvaluationResult desde retrieval + generation results."""
        results: List[QueryEvaluationResult] = []
        for query, retrieval, gm, reranked_status in zip(
            queries, retrievals, gen_metrics_results, rerank_statuses,
        ):
            qr_metadata: Dict[str, Any] = {}
            if reranked_status is not None:
                qr_metadata["reranked"] = reranked_status
            qt = query.metadata.get("question_type", "")
            if qt:
                qr_metadata["question_type"] = qt

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
    # BATCH ASYNC: Generacion + Metricas
    # -----------------------------------------------------------------

    async def _batch_generate_and_evaluate(
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
    ) -> _GenMetricsResult:
        """Procesa una query: generacion async + metricas async."""
        context = self._format_context(retrieval_detail.get_generation_contents())

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

        return _GenMetricsResult(
            generation=generation,
            primary_metric_value=primary_result.value,
            primary_metric_type=primary_result.metric_type,
            secondary_metrics=secondary_dict,
        )

    # -----------------------------------------------------------------
    # RETRIEVAL (sync)
    # -----------------------------------------------------------------

    def _execute_retrieval(
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
            retrieval_k = self.config.retrieval.retrieval_k

            configured_strategy = self.config.retrieval.strategy

            # Seleccionar metodo de retrieval
            def _do_retrieve(top_k: int) -> "RetrievalResult":
                if query_vector is not None:
                    return self._retriever.retrieve_by_vector(
                        query_text, query_vector, top_k=top_k
                    )
                return self._retriever.retrieve(query_text, top_k=top_k)

            def _check_strategy(result: "RetrievalResult") -> None:
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
                fetch_k = self.config.reranker.fetch_k or (self.config.reranker.top_n * 3)
                # Garantizar que fetch_k >= retrieval_k para que las metricas
                # pre-rerank tengan suficientes candidatos (DTm-35).
                fetch_k = max(fetch_k, retrieval_k)
                logger.debug(
                    f"  fetch_k={fetch_k} (retrieval_k={retrieval_k}, "
                    f"reranker.top_n={self.config.reranker.top_n})"
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
                    top_n=self.config.reranker.top_n,
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

    # -----------------------------------------------------------------
    # GENERACION (async)
    # -----------------------------------------------------------------

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

    def _format_context(self, contents: List[str]) -> str:
        max_length = self._max_context_chars
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

    # -----------------------------------------------------------------
    # METRICAS (async)
    # -----------------------------------------------------------------

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
                # Sin esto, la metrica desaparece del dict y report.py la muestra
                # como 0.0 via .get(sk, 0.0), indistinguible de un score legitimo.
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

    # -----------------------------------------------------------------
    # BUILD RUN
    # -----------------------------------------------------------------

    def _build_run(
        self,
        run_id: str,
        dataset: LoadedDataset,
        query_results: List[QueryEvaluationResult],
        elapsed_seconds: float,
        indexed_corpus_size: int = 0,
    ) -> EvaluationRun:
        """Construye EvaluationRun plano a partir de resultados."""
        completed = [
            qr for qr in query_results
            if qr.status == EvaluationStatus.COMPLETED
        ]
        failed = [
            qr for qr in query_results
            if qr.status == EvaluationStatus.FAILED
        ]

        # Agregar metricas de retrieval
        avg_hit5 = 0.0
        avg_mrr = 0.0
        recall_sums: Dict[int, float] = {}
        ndcg_sums: Dict[int, float] = {}

        if completed:
            for qr in completed:
                avg_hit5 += qr.retrieval.hit_at_k.get(5, 0.0)
                avg_mrr += qr.retrieval.mrr
                for k, v in qr.retrieval.recall_at_k.items():
                    recall_sums[k] = recall_sums.get(k, 0.0) + v
                for k, v in qr.retrieval.ndcg_at_k.items():
                    ndcg_sums[k] = ndcg_sums.get(k, 0.0) + v

            nc = len(completed)
            avg_hit5 /= nc
            avg_mrr /= nc
            avg_recall = {k: v / nc for k, v in recall_sums.items()}
            avg_ndcg = {k: v / nc for k, v in ndcg_sums.items()}
            complement_recall = {k: 1.0 - v for k, v in avg_recall.items()}
            avg_retrieved = sum(
                len(qr.retrieval.retrieved_doc_ids) for qr in completed
            ) / nc
            avg_expected = sum(
                len(qr.retrieval.expected_doc_ids) for qr in completed
            ) / nc
        else:
            avg_recall = {}
            avg_ndcg = {}
            complement_recall = {}
            avg_retrieved = 0.0
            avg_expected = 0.0

        # Metricas de retrieval efectivo (post-rerank)
        avg_gen_recall: Optional[float] = None
        avg_gen_hit: Optional[float] = None
        rescue_count = 0
        if completed:
            with_gen = [
                qr for qr in completed if qr.retrieval.generation_doc_ids
            ]
            if with_gen:
                retrieval_k = self.config.retrieval.retrieval_k
                avg_gen_recall = sum(
                    qr.retrieval.generation_recall for qr in with_gen
                ) / len(with_gen)
                avg_gen_hit = sum(
                    qr.retrieval.generation_hit for qr in with_gen
                ) / len(with_gen)
                rescue_count = sum(
                    1 for qr in with_gen
                    if qr.retrieval.generation_recall
                    > qr.retrieval.recall_at_k.get(retrieval_k, 0.0)
                )

        # Generacion promedio - INCLUYE ZEROS (fix DT-002)
        avg_gen = None
        gen_zero_count = 0
        gen_nonzero_count = 0
        if self.config.generation_enabled and completed:
            all_gen_values = [
                qr.primary_metric_value
                for qr in completed
            ]
            gen_zero_count = sum(1 for v in all_gen_values if v == 0.0)
            gen_nonzero_count = sum(1 for v in all_gen_values if v > 0.0)
            if all_gen_values:
                avg_gen = sum(all_gen_values) / len(all_gen_values)

        # Config snapshot
        config_snapshot = {
            "retrieval_strategy": self.config.retrieval.strategy.name,
            "retrieval_k": self.config.retrieval.retrieval_k,
            "pre_fusion_k": self.config.retrieval.pre_fusion_k,
            "bm25_weight": self.config.retrieval.bm25_weight,
            "vector_weight": self.config.retrieval.vector_weight,
            "rrf_k": self.config.retrieval.rrf_k,
            "corpus_shuffle_seed": self.config.corpus_shuffle_seed,
            "max_queries": self.config.max_queries,
            "max_corpus": self.config.max_corpus,
            "generation_enabled": self.config.generation_enabled,
            "max_context_chars": self._max_context_chars,
            "reranker_enabled": self.config.reranker.enabled,
            "reranker_top_n": self.config.reranker.top_n if self.config.reranker.enabled else None,
            "rerank_failures": self._rerank_failures if self.config.reranker.enabled else None,
            "strategy_mismatches": self._strategy_mismatches,
            "corpus_total_available": len(dataset.corpus),
            "corpus_indexed": indexed_corpus_size,
            "gen_zero_count": gen_zero_count,
            "gen_nonzero_count": gen_nonzero_count,
            "dev_mode": self.config.dev_mode,
            # DTm-38: estrategia real vs configurada
            "strategy_actual": (
                self.config.retrieval.strategy.name
                if self._strategy_mismatches == 0
                else "FALLBACK_SIMPLE_VECTOR"
            ),
        }
        if self.config.dev_mode:
            config_snapshot["dev_queries"] = self.config.dev_queries
            config_snapshot["dev_corpus_size"] = self.config.dev_corpus_size
        if self.config.retrieval.strategy == RetrievalStrategy.LIGHT_RAG:
            config_snapshot["kg_max_hops"] = self.config.retrieval.kg_max_hops
            config_snapshot["kg_max_text_chars"] = self.config.retrieval.kg_max_text_chars
            config_snapshot["kg_max_entities"] = self.config.retrieval.kg_max_entities
            config_snapshot["kg_graph_weight"] = self.config.retrieval.kg_graph_weight
            config_snapshot["kg_vector_weight"] = self.config.retrieval.kg_vector_weight
            config_snapshot["max_graph_expansion"] = self.config.retrieval.max_graph_expansion
            config_snapshot["kg_cache_dir"] = self.config.retrieval.kg_cache_dir or None

        return EvaluationRun(
            run_id=run_id,
            dataset_name=self.config.dataset_name,
            embedding_model=self.config.infra.embedding_model_name,
            retrieval_strategy=self.config.retrieval.strategy.name,
            config_snapshot=config_snapshot,
            status=EvaluationStatus.COMPLETED,
            num_queries_evaluated=len(completed),
            num_queries_failed=len(failed),
            total_documents=indexed_corpus_size,
            avg_hit_rate_at_5=avg_hit5,
            avg_mrr=avg_mrr,
            avg_recall_at_k=avg_recall,
            avg_ndcg_at_k=avg_ndcg,
            retrieval_complement_recall_at_k=complement_recall,
            avg_retrieved_count=avg_retrieved,
            avg_expected_count=avg_expected,
            avg_generation_recall=avg_gen_recall,
            avg_generation_hit=avg_gen_hit,
            reranker_rescue_count=rescue_count,
            avg_generation_score=avg_gen,
            execution_time_seconds=elapsed_seconds,
            query_results=query_results,
        )

    # -----------------------------------------------------------------
    # CHECKPOINT / RESUME (DTm-36)
    # -----------------------------------------------------------------

    _CHECKPOINT_CHUNK_SIZE = 50  # queries per checkpoint

    def _checkpoint_path(self, run_id: str) -> Path:
        return Path(self.config.storage.evaluation_results_dir) / f"{run_id}_checkpoint.json"

    def _save_checkpoint(
        self,
        run_id: str,
        evaluated_query_ids: Set[str],
        results: List[QueryEvaluationResult],
    ) -> None:
        """Persiste resultados parciales a disco."""
        path = self._checkpoint_path(run_id)
        data = {
            "run_id": run_id,
            "evaluated_query_ids": sorted(evaluated_query_ids),
            "num_results": len(results),
            "results": [self._serialize_query_result(r) for r in results],
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.rename(path)
        logger.info(
            f"  Checkpoint guardado: {len(evaluated_query_ids)} queries → {path.name}"
        )

    def _load_checkpoint(
        self, run_id: str
    ) -> Optional[Tuple[Set[str], List[QueryEvaluationResult]]]:
        """Carga checkpoint previo si existe."""
        path = self._checkpoint_path(run_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            evaluated_ids = set(data["evaluated_query_ids"])
            results = [self._deserialize_query_result(r) for r in data["results"]]
            logger.info(
                f"  Checkpoint cargado: {len(evaluated_ids)} queries desde {path.name}"
            )
            return evaluated_ids, results
        except Exception as e:
            logger.warning(f"  Checkpoint corrupto ({path.name}): {e}. Ignorando.")
            return None

    def _delete_checkpoint(self, run_id: str) -> None:
        path = self._checkpoint_path(run_id)
        if path.exists():
            path.unlink()
            logger.info(f"  Checkpoint eliminado: {path.name}")

    @staticmethod
    def _serialize_query_result(qr: QueryEvaluationResult) -> Dict[str, Any]:
        """Serializa un QueryEvaluationResult a dict JSON-compatible."""
        r = qr.retrieval
        gen = qr.generation
        return {
            "query_id": qr.query_id,
            "query_text": qr.query_text,
            "dataset_name": qr.dataset_name,
            "dataset_type": qr.dataset_type.value,
            "status": qr.status.value,
            "error_message": qr.error_message,
            "expected_response": qr.expected_response,
            "primary_metric_type": qr.primary_metric_type.value,
            "primary_metric_value": qr.primary_metric_value,
            "secondary_metrics": qr.secondary_metrics,
            "metadata": qr.metadata,
            "retrieval": {
                "retrieved_doc_ids": r.retrieved_doc_ids,
                "retrieved_contents": r.retrieved_contents,
                "retrieval_scores": r.retrieval_scores,
                "expected_doc_ids": r.expected_doc_ids,
                "retrieval_time_ms": r.retrieval_time_ms,
                "generation_doc_ids": r.generation_doc_ids,
                "generation_contents": r.generation_contents,
                "pre_rerank_candidate_ids": r.pre_rerank_candidate_ids,
                "retrieval_metadata": r.retrieval_metadata,
            },
            "generation": {
                "generated_response": gen.generated_response,
                "generation_time_ms": gen.generation_time_ms,
            } if gen else None,
        }

    @staticmethod
    def _deserialize_query_result(data: Dict[str, Any]) -> QueryEvaluationResult:
        """Deserializa un dict a QueryEvaluationResult."""
        rd = data["retrieval"]
        gen_data = data.get("generation")
        return QueryEvaluationResult(
            query_id=data["query_id"],
            query_text=data["query_text"],
            dataset_name=data["dataset_name"],
            dataset_type=DatasetType(data["dataset_type"]),
            status=EvaluationStatus(data["status"]),
            error_message=data.get("error_message"),
            expected_response=data.get("expected_response"),
            primary_metric_type=MetricType(data["primary_metric_type"]),
            primary_metric_value=data.get("primary_metric_value", 0.0),
            secondary_metrics=data.get("secondary_metrics", {}),
            metadata=data.get("metadata", {}),
            retrieval=QueryRetrievalDetail(
                retrieved_doc_ids=rd["retrieved_doc_ids"],
                retrieved_contents=rd["retrieved_contents"],
                retrieval_scores=rd["retrieval_scores"],
                expected_doc_ids=rd["expected_doc_ids"],
                retrieval_time_ms=rd.get("retrieval_time_ms", 0.0),
                generation_doc_ids=rd.get("generation_doc_ids", []),
                generation_contents=rd.get("generation_contents", []),
                pre_rerank_candidate_ids=rd.get("pre_rerank_candidate_ids", []),
                retrieval_metadata=rd.get("retrieval_metadata", {}),
            ),
            generation=GenerationResult(
                generated_response=gen_data["generated_response"],
                generation_time_ms=gen_data.get("generation_time_ms", 0.0),
            ) if gen_data else None,
        )

    # -----------------------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------------------

    def _cleanup(self) -> None:
        if self._retriever:
            try:
                self._retriever.clear_index()
            except Exception:
                pass
            self._retriever = None
        self._embedding_model = None
        self._llm_service = None
        gc.collect()


__all__ = ["MTEBEvaluator"]
