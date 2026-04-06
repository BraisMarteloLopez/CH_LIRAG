# CLAUDE.md

## Que es este proyecto

Sistema de evaluacion RAG para benchmarking de pipelines de retrieval y generacion sobre datasets MTEB/BeIR (HotpotQA) con NVIDIA NIM. Dos estrategias: `SIMPLE_VECTOR` (embedding + ChromaDB) y `LIGHT_RAG` (vector + knowledge graph via LLM).

## Estructura clave

```
shared/                        # Libreria core (~3,800 LOC)
  types.py                     # Tipos: NormalizedQuery, LoadedDataset, EvaluationRun, Protocols
  metrics.py                   # F1, EM, Accuracy, Faithfulness (LLM-judge)
  llm.py                       # AsyncLLMService (NIM client, async/sync bridge)
  config_base.py               # InfraConfig, RerankerConfig, _env helpers
  vector_store.py              # ChromaVectorStore (wrapper ChromaDB)
  report.py                    # RunExporter: JSON + CSV summary + CSV detail
  structured_logging.py        # Logging JSONL estructurado
  retrieval/
    core.py                    # RetrievalStrategy enum, RetrievalConfig, SimpleVectorRetriever, RRF
    __init__.py                # Factory get_retriever() — punto de entrada para crear retrievers
    reranker.py                # CrossEncoderReranker (NVIDIARerank)
    lightrag/
      retriever.py             # LightRAGRetriever: vector + KG dual-level (~1,060 LOC)
      knowledge_graph.py       # KnowledgeGraph in-memory (igraph): entidades, relaciones, BFS (~880 LOC)
      triplet_extractor.py     # Extraccion de tripletas y keywords via LLM (~770 LOC)

sandbox_mteb/                  # Pipeline de evaluacion (~2,750 LOC)
  config.py                    # MTEBConfig: .env → dataclass validada (+RerankerConfig.validate)
  evaluator.py                 # Orquestador principal (<600 LOC)
  run.py                       # Entry point CLI (--dry-run, -v, --resume)
  loader.py                    # MinIO/Parquet → LoadedDataset
  retrieval_executor.py        # Loop retrieval + reranking
  generation_executor.py       # Generacion async + metricas
  embedding_service.py         # Pre-embed queries batch + context window detection
  checkpoint.py                # Checkpoint/resume cada N queries (atomic writes)
  result_builder.py            # Construccion EvaluationRun final
  preflight.py                 # Validacion pre-run (deps, NIM, MinIO)
  subset_selection.py          # DEV_MODE: gold docs + distractores

tests/                         # pytest (~7,700 LOC, 205+ tests)
  conftest.py                  # Mocks condicionales de infra (boto3, langchain, chromadb)
  test_*.py                    # 30 unit test files
  integration/                 # 3 files, requieren NIM + MinIO reales
```

## Comandos

```bash
# Tests (unit only, sin infra)
pytest tests/ -m "not integration"

# Run de evaluacion
python -m sandbox_mteb.run
python -m sandbox_mteb.run --dry-run
python -m sandbox_mteb.run -v
python -m sandbox_mteb.run --resume RUN_ID

# Preflight check
python -m sandbox_mteb.preflight
```

## Convenciones

- **Config via .env**: toda la parametrizacion en `sandbox_mteb/.env`, leida por `MTEBConfig.from_env()` una sola vez. Sub-configs delegadas a `InfraConfig`, `RerankerConfig`, `RetrievalConfig` en shared/. `MTEBConfig.validate()` propaga validacion a sub-configs
- **Factory pattern**: `get_retriever(config, embedding_model)` en `shared/retrieval/__init__.py` crea el retriever correcto
- **2 estrategias**: `SIMPLE_VECTOR` y `LIGHT_RAG` — no hay mas. `HYBRID_PLUS` fue eliminada (DTm-83)
- **Enum en core.py**: `RetrievalStrategy` define las estrategias validas. `VALID_STRATEGIES` en `sandbox_mteb/config.py` debe coincidir
- **Tests**: `conftest.py` mockea modulos de infra (boto3, langchain, chromadb) si no estan instalados. Tests de integracion requieren NIM + MinIO reales. Mocks siempre a nivel de funcion, nunca modulos enteros
- **Logging**: JSONL estructurado via `shared/structured_logging.py`. Bare excepts tienen `logger.debug(...)` — no hay excepts silenciosos
- **Idioma**: codigo y comentarios en ingles/espanol mezclado (historico). Docstrings y variables en ingles

## Estrategia LIGHT_RAG — como funciona

Inspirada en [LightRAG (EMNLP 2025)](https://arxiv.org/abs/2410.05779).

**Indexacion**: LLM extrae tripletas (entidad, relacion, entidad) de cada doc → KnowledgeGraph in-memory (igraph) + ChromaDB para vector search. Entity VDB y Relationship VDB para resolucion semantica. Stats se resetean automaticamente al inicio de cada batch (G.5). Gleaning opcional via `KG_GLEANING_ROUNDS`.

**Retrieval**: vector search + query keywords via LLM (dedup automatico de queries identicas, cap 20 keywords/nivel) + graph traversal dual-level (entity VDB low-level + relationship VDB high-level) + fusion RRF.

**Modos** (`LIGHTRAG_MODE`): `hybrid` (default), `graph_primary`, `local`, `global`, `naive`.

**Fallback**: sin igraph o sin LLM → degrada a SimpleVectorRetriever puro.

**Alineacion con original (DAM-1 a DAM-8)**: Entity VDB, Relationship VDB, edge weights, gleaning, BFS 1-hop, graph_primary mode, contexto estructurado — todo implementado. DAM-4 parcial (concatenacion sin LLM synthesis). Pendiente validacion con run comparativo (F.5).

## Deuda tecnica vigente (priorizada)

### Alta — afectan calidad de resultados
(sin items pendientes)

### Media — mejoras funcionales
- **DTm-64**: Normalizacion scores incomparable entre canales vector/graph. RRF mitiga parcialmente. Ubicacion: `retriever.py`
- **DTm-72**: BFS scoring ciego a la relacion (todas las aristas pesan igual). Ubicacion: `knowledge_graph.py`
- **DTm-80**: DAM-4 parcial: merge de descripciones por concatenacion, sin LLM synthesis. [#7](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/7)

### Baja — code quality
- **DTm-82**: 22 errores mypy restantes (`return-value`, `assignment`, `arg-type`). [#9](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/9)
- **DTm-12**: Sesgo LLM-judge en faithfulness para respuestas cortas. Inherente. Aceptado
- **DTm-13**: No-determinismo HNSW: ChromaDB no expone `hnsw:random_seed`. ±0.02. Aceptado

### Resueltos en esta sesion
- **DTm-55** (G.1): Stats snapshot/restore si KG build falla
- **DTm-56** (G.2): Fingerprint incluye `kg_max_entities` en hash
- **DTm-57** (G.7): Entity normalization preserva apostrofes (`O'Brien`)
- **DTm-58** (G.4): Dedup queries identicas en batch keyword extraction
- **DTm-60** (G.5): `reset_stats()` auto al inicio de `extract_batch_async()`
- **DTm-61** (G.6): Keyword size cap 20/nivel
- **DTm-77** (I.5): Tests gleaning — 6 tests en `test_gleaning.py`
- **DTm-78** (I.4): Test E2E LIGHT_RAG en `test_pipeline_e2e.py`
- **DTm-83**: HYBRID_PLUS eliminado (-2,570 LOC, -3 deps)
- **Fase H**: Bare excepts con logging, dead code, validacion sub-configs
- **DTm-62**: Conditional fusion en `_fuse_with_graph()` — overlap gate previene graph ruidoso destruir ranking vectorial. Config: `KG_FUSION_OVERLAP_THRESHOLD`, `KG_FUSION_GRAPH_ONLY_CAP`
- **Tests audit**: assert faltante en test_evaluator.py, ClientError mock roto en test_loader.py. Suite: 340 passed, 0 failed
- **DTm-63**: Eviction por importancia en entity cap — reemplaza FIFO silencioso. Entidades con 1 doc y degree<=1 se evictan para dar paso a nuevas. `_find_eviction_candidate()`, `_evict_entity()` en `knowledge_graph.py`
- **DTm-73**: Co-occurrence bridging reduce fragmentacion del grafo. `build_co_occurrence_edges()` conecta entidades que co-ocurren en un mismo doc. Cap 10 pares/doc. Se ejecuta en post-build

## Bare excepts aceptados (no criticos)

Estos `except Exception as e:` logean el error pero no lo re-lanzan. Aceptable para wrappers de infraestructura:

| Ubicacion | Contexto |
|---|---|
| `reranker.py:147` | Reranking error — retorna fallback sin rerank |
| `vector_store.py:125, 141, 178` | Operaciones ChromaDB — retorna lista vacia |

## Test coverage

| Metrica | Valor |
|---|---|
| Tests unitarios | 340 en 30 archivos |
| Tests integracion | 19 en 3 archivos |
| Ratio test/produccion | 0.86x (~7,700 / ~8,940 LOC) |
| Modulos cubiertos | 29/31 (93%) |
| Tests con assertions | 100% |
| Mocks a nivel funcion | 100% |

**Referencia completa**: ver `TESTS.md` — mapa test→produccion, atributos `object.__new__()`, trampas de mock, gaps de cobertura, reglas de modificacion.

## Que NO tocar sin contexto

- `DatasetType.HYBRID` en `shared/types.py` — es un tipo de dataset (tiene respuesta textual), NO una estrategia de retrieval
- `reciprocal_rank_fusion()` en `core.py` — la usa LIGHT_RAG, no es legacy
- `shared/config_base.py` — la importan todos los modulos, cambios rompen todo
- Tests de integracion (`tests/integration/`) — dependen de NIM + MinIO reales
- `requirements.lock` — es un pin de produccion, no tocar sin razon
- `_PersistentLoop` en `shared/llm.py` — resuelve binding de event loop asyncio (DTm-45). Parece complejo pero es necesario

## Proximos pasos

### Pendiente: Run comparativo F.5 (requiere infra NIM + MinIO)

| Tarea | Descripcion |
|---|---|
| F.5a | Run SIMPLE_VECTOR baseline: 50q, 3500 docs, DEV_MODE, seed=42 |
| F.5b | Run LIGHT_RAG hybrid: misma config |
| F.5c | Run LIGHT_RAG graph_primary |
| F.5d | Analisis comparativo: MRR, Hit@5, Recall |

**Criterio de exito:** LIGHT_RAG MRR > 0.80 (vs 0.52 pre-VDBs). Si no, revisar DTm-62/DTm-64.

### Pendiente: DTm-80 — LLM synthesis para merge de descripciones (DAM-4 completo)

Feature que cambia logica de negocio. Actual: concatenacion con ` | `. Original: LLM map-reduce cuando tokens exceden umbral. [#7](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/7)
