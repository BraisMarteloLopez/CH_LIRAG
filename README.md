# CH_LIRAG

Sistema de evaluacion RAG (Retrieval-Augmented Generation) para benchmarking de pipelines de recuperacion y generacion sobre datasets MTEB/BeIR (HotpotQA) con infraestructura NVIDIA NIM.

## Estrategias de retrieval

| Estrategia | Indexacion | Busqueda | Reranker |
|---|---|---|---|
| `SIMPLE_VECTOR` | Embedding directo (NIM) | Cosine similarity (ChromaDB) | Opcional |
| `LIGHT_RAG` | LLM triplet extraction + chunk keywords + KG + Embedding (chunks, entidades, relaciones) | Chunks via KG: tres canales (Entity VDB, Relationship VDB, Chunk Keywords VDB) agregados al mismo scoring; fallback a vector search; LLM synthesis del contexto | Off (auto-desactivado) |

**LIGHT_RAG** implementa la arquitectura de [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (EMNLP 2025; [arxiv](https://arxiv.org/abs/2410.05779)), con adaptaciones operativas propias del entorno (cache de KG a disco, fallbacks ante errores del LLM/igraph, instrumentacion de observabilidad). LightRAG extrae via LLM entidades, relaciones y high-level keywords por chunk para construir un knowledge graph + indices tematicos; las queries se resuelven en dos niveles (entidades concretas y temas abstractos) contra tres VDBs dedicadas — Entity VDB (entidades), Relationship VDB (relaciones) y Chunk Keywords VDB (temas high-level por chunk, divergencia #10) — cuyos resultados agregan al mismo scoring de chunks. Chunks seleccionados via `source_doc_ids` del KG y matching directo a temas (paper-aligned), con fallback a vector search directo cuando el KG no produce doc_ids. 4 modos configurables via `LIGHTRAG_MODE`: `hybrid` (default), `local`, `global`, `naive`. Tras el retrieval, una capa de synthesis LLM (query-aware, `KG_SYNTHESIS_ENABLED`) reescribe el contexto multi-seccion (entidades + relaciones + chunks) como narrativa coherente antes de la generacion final. Sin `igraph` o sin LLM → degrada a vector search puro; fallos de synthesis → degrada al contexto estructurado.

> **Estado frente al paper**: divergencias #2, #4+5, #6, #7, #8, #9 resueltas. **#10 (keywords high-level por chunk durante indexacion) implementada en codigo y con tests unitarios, pendiente validacion end-to-end con NIM + MinIO reales** antes de marcarse como "Resuelta". Detalle y criticidad en [CLAUDE.md — Divergencias con el paper original](CLAUDE.md#divergencias-con-el-paper-original--evaluacion-de-criticidad). El gate activo es Pre-P0 (completitud arquitectural + ejecucion estable) antes de P0 (replicacion empirica) — ver [CLAUDE.md — Proximos pasos](CLAUDE.md#proximos-pasos).

## Arquitectura

```
shared/                          # Libreria compartida
  types.py                       # NormalizedQuery, LoadedDataset, EvaluationRun, Protocols
  metrics.py                     # F1, ExactMatch, Accuracy, Faithfulness (LLM-judge)
  llm.py                         # AsyncLLMService, load_embedding_model
  config_base.py                 # InfraConfig, RerankerConfig, helpers _env_*
  report.py                      # RunExporter: JSON + CSV summary + CSV detail
  vector_store.py                # ChromaVectorStore
  structured_logging.py          # Logging JSONL estructurado
  retrieval/
    __init__.py                  # Factory get_retriever()
    core.py                      # BaseRetriever, SimpleVectorRetriever, RetrievalConfig
    reranker.py                  # CrossEncoderReranker (NVIDIARerank)
    lightrag/
      retriever.py               # LightRAGRetriever: vector + KG dual-level
      knowledge_graph.py         # KG in-memory (igraph): entidades, relaciones, traversal
      triplet_extractor.py       # Extraccion tripletas y query keywords via LLM

sandbox_mteb/                    # Pipeline de evaluacion MTEB/BeIR
  config.py                      # MTEBConfig: .env -> dataclass validada
  evaluator.py                   # Pipeline orquestador (<600 LOC)
  run.py                         # Entry point (--dry-run, -v, --resume)
  loader.py                      # MinIO/Parquet -> LoadedDataset
  retrieval_executor.py          # Loop retrieval sync + reranking
  generation_executor.py         # Generacion async + metricas
  embedding_service.py           # Pre-embed queries batch (NIM REST)
  checkpoint.py                  # Checkpoint/resume cada N queries
  result_builder.py              # Construccion EvaluationRun final
  preflight.py                   # Validacion pre-run (deps, NIM, MinIO)
  env.example                    # Plantilla .env

tests/                           # pytest (~465 unit + ~15 integration tests, cifras orientativas; ver CLAUDE.md "Test coverage")
```

## Pipeline

```
.env -> MTEBConfig -> MinIO/cache(Parquet) -> LoadedDataset
     -> shuffle(seed) -> slice(max_corpus)
     -> [LLM triplet extraction + KG build si LIGHT_RAG]
     -> index(ChromaDB)
     -> pre-embed queries (batch REST NIM)
     -> [pre-extract query keywords (batch LLM) si LIGHT_RAG]
     -> retrieve (sync) -> [rerank si SIMPLE_VECTOR]
     -> [LLM synthesis del contexto KG si LIGHT_RAG + kg_synthesis_enabled]
     -> generate + metrics (async)
     -> EvaluationRun -> JSON + CSV
```

## Setup

```bash
pip install -r requirements.txt
cp sandbox_mteb/env.example sandbox_mteb/.env
# Editar .env con endpoints NIM y MinIO
```

## Uso

```bash
python -m sandbox_mteb.run                  # Run con .env
python -m sandbox_mteb.run --dry-run        # Solo validar config
python -m sandbox_mteb.run --env /path/.env # .env alternativo
python -m sandbox_mteb.run -v               # Verbose (DEBUG)
python -m sandbox_mteb.run --resume RUN_ID  # Reanudar desde checkpoint

python -m sandbox_mteb.preflight            # Preflight check (deps, NIM, MinIO)
```

## Tests

```bash
pytest tests/                      # Unit + integracion
pytest tests/ -m "not integration" # Solo unit (sin NIM/MinIO)
pytest tests/integration/ -v       # Solo integracion (requiere NIM + MinIO)
```

## Configuracion (.env)

Referencia completa en `sandbox_mteb/env.example`. Variables principales:

```bash
# Embedding (NIM)
EMBEDDING_MODEL_NAME=nvidia/llama-3.2-nv-embedqa-1b-v2
EMBEDDING_BASE_URL=http://<nim-host>:8000/v1
EMBEDDING_BATCH_SIZE=5

# LLM (NIM) — requerido para LIGHT_RAG y generacion
LLM_BASE_URL=http://<nim-host>:8000/v1
LLM_MODEL_NAME=nvidia/nemotron-3-nano
NIM_MAX_CONCURRENT_REQUESTS=32

# Retrieval
RETRIEVAL_STRATEGY=SIMPLE_VECTOR      # SIMPLE_VECTOR | LIGHT_RAG
RETRIEVAL_K=20

# Knowledge Graph (solo LIGHT_RAG)
LIGHTRAG_MODE=hybrid                  # hybrid | local | global | naive
KG_MAX_HOPS=1                         # Profundidad BFS (1-hop como el original)
KG_EXTRACTION_MAX_TOKENS=4096         # Max tokens para triplet extraction
KG_BATCH_DOCS_PER_CALL=5             # Docs por LLM call en batch
KG_MAX_ENTITIES=0                     # Cap entidades (0 = default interno 100K)
KG_CACHE_DIR=                         # Directorio para persistir KG (vacio = sin cache)
KG_DESCRIPTION_SYNTHESIS=false        # LLM synthesis para descripciones multi-doc (DAM-4)
KG_SYNTHESIS_CHAR_THRESHOLD=200       # Chars minimos para trigger LLM synthesis (DAM-4)

# Chunk high-level keywords (divergencia LightRAG #10)
KG_CHUNK_KEYWORDS_ENABLED=true        # Piggyback extraccion de temas por chunk + 3er canal en retrieval
KG_CHUNK_KEYWORDS_TOP_K=20            # Top-K devueltos por keyword al consultar Chunk Keywords VDB

# KG context synthesis en generacion (divergencia LightRAG #2)
KG_SYNTHESIS_ENABLED=true             # LLM reescribe contexto multi-seccion como narrativa
KG_SYNTHESIS_MAX_CHARS=0              # 0 = usar max_context_chars del run
KG_SYNTHESIS_TIMEOUT_S=30.0

# Reranker (opcional, desactivado automaticamente para LIGHT_RAG)
RERANKER_ENABLED=false
RERANKER_TOP_N=5

# Dataset
MTEB_DATASET_NAME=hotpotqa
EVAL_MAX_QUERIES=50
EVAL_MAX_CORPUS=1000
GENERATION_ENABLED=true

# LLM judge instrumentation
JUDGE_FALLBACK_THRESHOLD=0.02         # Max tasa de fallback a 0.5 permitida (0 = desactiva). Ver CLAUDE.md "Observabilidad de runs"

# DEV_MODE: subset con gold docs garantizados
DEV_MODE=false
DEV_QUERIES=200
DEV_CORPUS_SIZE=4000
```

## Metricas

**Retrieval:** Hit@K, MRR, Recall@K, NDCG@K (K=1,3,5,10,20) sobre top `RETRIEVAL_K` docs pre-rerank.

**Generacion:** F1 (primary), Exact Match (secondary), Faithfulness (LLM-judge, opcional).

**Post-rerank:** `generation_recall`, `generation_hit`, `reranker_rescue_count` (solo con reranker activo).

## Dataset: HotpotQA

| Propiedad | Valor |
|---|---|
| Queries | 7405 |
| Corpus | 66576 docs (Wikipedia) |
| Qrels | 14810 (2.0/query) |
| Tipos | bridge (~80%), comparison (~20%) |

Limitacion: 10 pasajes/query (2 gold + 8 distractores). No comparable con benchmarks publicados (corpus 5.2M). Solo comparaciones relativas entre estrategias.

## Guia de desarrollo

Ver [`CLAUDE.md`](CLAUDE.md) para convenciones, divergencias con el paper, deuda tecnica, y proximos pasos. Ver [`TESTS.md`](TESTS.md) para referencia de la suite de tests.

<details>
<summary>Historial de desarrollo</summary>

**Fases 0-4 (fundacion):** Reproducibilidad, checkpoint/resume, batch adaptativo, evaluator descompuesto de 1225 a 592 LOC.

**Ronda 1 (arquitectura LightRAG):** NetworkX → igraph, batch extraction, RRF fusion, stemming.

**Ronda 2 (robustez):** LRU cache keywords, thread safety, fallbacks, stats.

**Fases A-F (alineacion con original):** Entity VDB, Relationship VDB, description merging, gleaning, contexto estructurado. RRF eliminado, pipeline alineado con el paper. F.5 ejecutado (HotpotQA no discrimina; ver CLAUDE.md).

**Fase H (hardening):** Bare excepts con logging, dead code eliminado, validacion sub-configs.

**Fase I (test coverage):** +42 tests nuevos. Cobertura 86% → 93%.

**Fase G (deuda tecnica):** Stats resilientes, query dedup, keyword cap, entity normalization apostrofes, fingerprint robusto.

**DTm-62/63/72/73/82/83:** Conditional fusion, eviction mejorada, BFS weighted, co-occurrence bridging, mypy zero errors, eliminacion HYBRID_PLUS.

**Audit Fases 1-5:** 8 bugfixes, 48 tests nuevos, DAM-4 LLM synthesis, eviction con score compuesto. 447 tests, 0 fallos.

**Post-refactor (abril 2026):** instrumentacion de tasa de fallback del LLM judge (+19 tests, threshold enforcement via `JUDGE_FALLBACK_THRESHOLD`) + capa de synthesis KG en generacion (divergencia LightRAG #2, +13 tests, prompt query-aware con citas `[ref:N]`, faithfulness contra structured_context original para penalizar hallucinations de la propia synthesis) + resolucion divergencia #8 opcion A (chunks via `source_doc_ids` del KG con fallback a vector search, eliminacion de codigo muerto) + resolucion divergencia #9 (enriquecimiento con vecinos 1-hop y endpoints) + implementacion y resolucion de divergencia #10 (high-level keywords por chunk durante indexacion via piggyback en triplet extraction, tercera VDB `Chunk Keywords VDB`, canal adicional en el path high-level de `_retrieve_via_kg` con scoring simetrico; 29 tests nuevos; serializacion KG v2→v3 con rechazo explicito de caches viejos; `KG_CHUNK_KEYWORDS_ENABLED`/`KG_CHUNK_KEYWORDS_TOP_K` en `.env`) + observabilidad per-query (JSON/CSV exportan `retrieval_metadata` filtrada con `kg_fallback`, `kg_chunk_keyword_matches`, conteos de entidades/relaciones, `kg_synthesis_used/error`; `_synthesize_kg_context_async` retorna `(narrativa, error_code)` para auditoria per-query) + instrumentacion de timing de synthesis (hook `timing_out` en `AsyncLLMService.invoke_async` expone `queue_wait_ms` y `llm_ms` separados; tracker acumula `p50/p95/max_total_ms`, `p50/p95/max_queue_ms`, `p50/p95/max_llm_ms` en `kg_synthesis_stats`, discrimina saturacion de cola vs LLM lento) + cierre de deuda #13 (reranker ya no se instancia al arranque cuando `strategy=LIGHT_RAG`; divergencia #6). **Todas las divergencias arquitectonicas (#2, #4+5, #6, #7, #8, #9, #10) resueltas en codigo y tests**; validacion empirica directa de correctitud en run `20260419_032905` para #2, #6, #8 via observables dedicados; #4+5, #7, #9 ejercitadas implicitamente; **#10 con matiz importante** — canal arquitectonicamente presente (35/35 queries con matches en Chunk Keywords VDB) pero la calidad del output de la extraccion piggyback vs la llamada dedicada del paper no esta verificada empiricamente. Riesgo bloqueante antes de P2 (catalogo especializado) — ver CLAUDE.md fila de divergencia #10 para la advertencia completa y plan de accion pre-P2.

**Pre-P0 cerrado (2026-04-19)**: tres runs diagnosticos sobre HotpotQA (35q/1000docs DEV_MODE/hybrid) cerraron la fase de completitud arquitectural. Config validada final: `NIM_MAX_CONCURRENT_REQUESTS=32`, `KG_SYNTHESIS_MAX_CHARS=50000`, `KG_SYNTHESIS_TIMEOUT_S=180` → `kg_synthesis_stats.fallback_rate = 2.86%` (< 10% umbral), `kg_chunk_keyword_matches > 0` en 35/35 queries (100%, mean 34.8), `judge_fallback default_return_rate = 0%`. Fase actual: P0 (replicacion empirica sobre benchmark de contra-referencia).

80+ issues resueltos. Ver historial git.

</details>

<details>
<summary>Resultados comparativos (pre-Fases C-F, obsoletos)</summary>

> Todos los runs son pre-VDBs y pre-arquitectura-actual. Conclusion estructural vigente documentada en [CLAUDE.md — Resultado F.5](CLAUDE.md#resultado-f5-referencia-historica): sobre HotpotQA el embedding satura el retrieval (Wikipedia en pre-entrenamiento + DEV_MODE con gold docs garantizados + ventana 192K chars), el dataset no discrimina LIGHT_RAG vs SIMPLE_VECTOR. Cualquier medicion comparativa con la arquitectura actual es trabajo de P0 — las tablas numericas pre-refactor/post-refactor se retiraron de CLAUDE.md por obsoletas (git history las preserva).

| Metrica | SIMPLE_VECTOR | LIGHT_RAG (KG activo) | Delta |
|---|---|---|---|
| MRR | 1.000 | 0.518 | -48.2pp |
| Hit@5 | 1.000 | 0.793 | -20.7pp |
| Recall@5 | 0.940 | 0.690 | -25.0pp |
| F1 | 0.754 | 0.776 | +2.2pp |

Causa raiz: divergencias arquitectonicas (DAM-1..8), corregidas en Fases A-F + Audit.

</details>
