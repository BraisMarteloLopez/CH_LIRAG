# TESTS.md

Referencia interna para Claude Code. Describe la suite de tests, patrones de mock, atributos requeridos por `object.__new__()`, y trampas conocidas. Consultar antes de modificar tests o codigo de produccion que afecte firmas/atributos.

## Infraestructura de mocks (conftest.py)

`tests/conftest.py` inyecta `MagicMock()` para modulos de infra **solo si no estan instalados**:

```
boto3, botocore, botocore.exceptions,
langchain_nvidia_ai_endpoints, langchain_core, langchain_core.messages,
langchain_core.documents, langchain_core.embeddings, langchain_chroma, chromadb
```

### Trampa critica: botocore.exceptions como MagicMock

Cuando `botocore` no esta instalado, `botocore.exceptions.ClientError` es un `MagicMock`. Esto significa:
- `except ClientError` en produccion **no captura** instancias del mock (MagicMock no es una clase exception real)
- Tests que necesitan simular `ClientError` deben usar `@patch("sandbox_mteb.loader.ClientError", _FakeClientError)` con una exception real (ver `test_loader.py`)
- Tests que usan `except Exception` generico funcionan por casualidad

**Regla**: nunca importar `ClientError` directamente de `botocore.exceptions` en tests. Siempre patchear el nombre en el modulo de produccion.

## Configuracion pytest

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = ["integration: requiere NIM y MinIO reales"]
```

Ejecutar unit tests: `pytest tests/ -m "not integration"`

## Patron object.__new__() — atributos requeridos

Varios tests crean objetos sin pasar por `__init__` usando `object.__new__(Class)`. **Cada atributo nuevo en produccion requiere actualizar estos helpers.** Lista exhaustiva de atributos por clase:

### LightRAGRetriever

Archivos: `test_lightrag_fusion.py:_make_lightrag()`, `test_dtm38_strategy_guardrail.py:_make_lightrag()`

```python
retriever.config = RetrievalConfig()
retriever._graph_weight = 0.3
retriever._vector_weight = 0.7
retriever._kg_max_hops = 2
retriever._GRAPH_OVERFETCH_FACTOR = 2
retriever._kg = MagicMock(spec=KnowledgeGraph)  # o None
retriever._extractor = MagicMock()               # o None
retriever._has_graph = True
retriever._lightrag_mode = "hybrid"
retriever._query_keywords_cache = OrderedDict()
retriever._cache_lock = threading.Lock()
retriever._QUERY_CACHE_MAX_SIZE = 10_000
retriever._vector_retriever = MagicMock()
retriever._kg_fusion_method = "rrf"
retriever._kg_rrf_k = 60
retriever._entities_vdb = None
retriever._relationships_vdb = None
retriever._fusion_overlap_threshold = 0.3   # DTm-62
retriever._fusion_graph_only_cap = 0.2      # DTm-62
```

### TripletExtractor

Archivos: `test_triplet_extractor.py:_make_extractor()`, `test_gleaning.py:_make_extractor()`

```python
ext._llm_service = MagicMock()
ext._max_text_chars = 3000
ext._max_tokens = 4096
ext._keyword_max_tokens = 1024
ext._batch_docs_per_call = 5
ext._gleaning_rounds = 0
ext._stats = {
    "docs_processed": 0, "docs_failed": 0,
    "docs_empty": 0, "total_entities": 0,
    "total_relations": 0,
}
```

### CrossEncoderReranker

Archivos: `test_dt8_09_10_11_reranker_sort.py`, `test_group_a_b_review.py`

```python
reranker._reranker = MagicMock()  # langchain NVIDIARerank mock
reranker._top_n = 5
```

### SimpleVectorRetriever

Archivo: `test_group_a_b_review.py`

```python
retriever.config = RetrievalConfig()
retriever._vector_store = MagicMock()  # o None
retriever._embedding_model = MagicMock()
```

### ChromaVectorStore

Archivo: `test_group_a_b_review.py`

```python
store._collection = MagicMock()
store._CHROMA_IN_BATCH_SIZE = 100
```

### MinIOLoader

Archivo: `test_loader.py`

```python
loader.endpoint = "http://fake:9000"
loader.bucket = "test-bucket"
loader.prefix = "datasets/eval"
loader.cache_dir = Path("/tmp/test_cache")
loader.client = MagicMock()
loader._manifest = None
```

## Mapa test → produccion

### shared/ (libreria core)

| Test | Produccion | Tests | Que cubre |
|------|-----------|-------|-----------|
| test_metrics_reference_based.py | shared/metrics.py | 15 | normalize_text, f1_score, exact_match, accuracy |
| test_semantic_similarity.py | shared/metrics.py | 9 | semantic_similarity coseno, vector cero, empty input, numpy guard |
| test_dt6_context_truncation.py | shared/metrics.py | 3 | context pass-through sync/async, empty context |
| test_dt9_extract_score_fallback.py | shared/metrics.py | 6 | _extract_score_fallback regex |
| test_llm.py | shared/llm.py | 16 | LLMMetrics, thinking tags, invoke_async, load_embedding_model, retry |
| test_knowledge_graph.py | shared/retrieval/lightrag/knowledge_graph.py | 65 | CRUD, BFS weighted, keywords, persistence, VDB, stats, eviction, co-occurrence |
| test_triplet_extractor.py | shared/retrieval/lightrag/triplet_extractor.py | 36 | parsing, validation, batch, stats |
| test_gleaning.py | shared/retrieval/lightrag/triplet_extractor.py | 6 | glean_from_doc_async |
| test_lightrag_fusion.py | shared/retrieval/lightrag/retriever.py | 45 | _fuse_with_graph, fingerprint, VDBs, modes, DTm-62 conditional fusion |
| test_simple_vector_retriever.py | shared/retrieval/core.py | 10 | retrieve, retrieve_by_vector, index_documents, clear_index, get_documents_by_ids |
| test_dtm4_rrf.py | shared/retrieval/core.py | 11 | reciprocal_rank_fusion |
| test_dt8_09_10_11_reranker_sort.py | shared/retrieval/reranker.py | 3 | rerank sorting |
| test_reranker.py | shared/retrieval/reranker.py | 8 | empty passthrough, ordering, top_n, vector_scores, error fallback, metadata |
| test_report.py | shared/report.py | 13 | to_json, to_summary_csv, to_detail_csv, export, LIGHT_RAG columns |
| test_group_a_b_review.py | shared/retrieval/core.py, reranker.py, vector_store.py | 8 | get_documents_by_ids, batching, vector_scores |
| test_vector_store.py | shared/vector_store.py | 13 | add_documents batching, search, get_by_ids chunking, delete+recreate, error paths |
| test_config_validation.py | shared/config_base.py | 11 | InfraConfig.validate(), RerankerConfig.validate() error paths |
| test_retrieval_metrics_formulas.py | shared/types.py | 12 | NDCG, MRR, Hit@K, Recall@K, generation_hit/recall con valores exactos |
| test_dt7_08_csv_reranked.py | shared/report.py | 1 | reranked column CSV |
| test_dtm17_generation_retrieval_metrics.py | shared/types.py, shared/report.py | 14 | generation_recall/hit, CSV columns |

### sandbox_mteb/ (pipeline)

| Test | Produccion | Tests | Que cubre |
|------|-----------|-------|-----------|
| test_evaluator.py | sandbox_mteb/evaluator.py | 9 | _init_components, _cleanup, _assemble_results, run validation |
| test_dtm4_build_run_aggregation.py | sandbox_mteb/evaluator.py → result_builder.py | 18 | _build_run aggregation, config_snapshot |
| test_embedding_service.py | sandbox_mteb/embedding_service.py | 8 | batch_embed_queries, resolve_max_context_chars |
| test_format_context.py | sandbox_mteb/retrieval_executor.py | 9 | format_context truncation |
| test_structured_context.py | sandbox_mteb/retrieval_executor.py | 6 | format_structured_context |
| test_dt5_pre_rerank_traceability.py | sandbox_mteb/retrieval_executor.py | 3 | pre_rerank_candidate_ids |
| test_dt7_05_06_rerank_status.py | sandbox_mteb/retrieval_executor.py | 2 | rerank success/failure |
| test_dt7_07_no_reranker.py | sandbox_mteb/retrieval_executor.py, result_builder.py | 1 | sin reranker path |
| test_dtm38_strategy_guardrail.py | sandbox_mteb/retrieval_executor.py, result_builder.py | 8 | strategy mismatch, config_snapshot |
| test_dtm5_12_13_secondary_metric_errors.py | sandbox_mteb/generation_executor.py | 3 | secondary metric errors |
| test_generation_executor.py | sandbox_mteb/generation_executor.py | 8 | generation async, metrics HYBRID, structured context, batch |
| test_run_cli.py | sandbox_mteb/run.py | 11 | parse_args, setup_logging, main (dry-run, full, errors) |
| test_preflight.py | sandbox_mteb/preflight.py | 8 | _check wrapper, dependencies, lock_file, config, main |
| test_checkpoint.py | sandbox_mteb/checkpoint.py | 11 | save/load/delete checkpoint |
| test_loader.py | sandbox_mteb/loader.py | 6 | check_connection, _populate_from_dataframes |
| test_dtm4_loader_populate.py | sandbox_mteb/loader.py | 9 | _populate_from_dataframes detallado |
| test_dtm4_subset_selection.py | sandbox_mteb/subset_selection.py | 9 | select_subset_dev |

### E2E

| Test | Tests | Que cubre |
|------|-------|-----------|
| test_pipeline_e2e.py | 2 | MTEBEvaluator.run() completo (SIMPLE_VECTOR + LIGHT_RAG), assertions con metricas exactas |

## Modulos sin tests dedicados

| Modulo | Riesgo |
|--------|--------|
| shared/structured_logging.py | Bajo — utilidad de logging |

## Gaps de cobertura conocidos

| Area | Detalle |
|------|---------|
| loader.py:_safe_str() | Utility helper para None/NaN coercion, cubierta indirectamente por test_loader |
| loader.py:175-176 | Auto-conversion `question_type == "comparison"` → `answer_type = "label"` no testeada |
| Modos lightrag en retrieve_by_vector | Solo `retrieve()` tiene tests de modo; `retrieve_by_vector()` comparte logica pero no tiene tests de modo dedicados |

## Reglas para modificar tests

1. **Atributo nuevo en clase con `object.__new__()`**: actualizar TODOS los helpers listados arriba. Buscar con `grep -r "object.__new__(ClassName)" tests/`
2. **Nuevo campo en RetrievalConfig**: propagado automaticamente via `RetrievalConfig()` default. Sin accion en tests salvo que el field necesite valor no-default
3. **Nuevo campo en MTEBConfig**: actualizar helpers `_make_config()` en test_dtm4_subset_selection.py, test_embedding_service.py, test_pipeline_e2e.py si el field es required
4. **Nuevo campo en QueryRetrievalDetail o QueryEvaluationResult**: actualizar `_make_qr()` en test_checkpoint.py si el field es required
5. **Import de botocore.exceptions**: nunca directo en tests. Usar `_FakeClientError` + `@patch`
6. **Cada assertion debe usar `assert`**: nunca dejar expresiones booleanas sueltas (`x is None` sin assert)
