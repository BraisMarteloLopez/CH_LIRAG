# CLAUDE.md

## Que es este proyecto

Sistema de evaluacion RAG para benchmarking de pipelines de retrieval y generacion sobre datasets MTEB/BeIR (HotpotQA) con NVIDIA NIM. Dos estrategias: `SIMPLE_VECTOR` (embedding + ChromaDB) y `LIGHT_RAG` (vector + knowledge graph via LLM).

## Estructura clave

```
shared/                        # Libreria core
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
      retriever.py             # LightRAGRetriever: vector + KG dual-level
      knowledge_graph.py       # KnowledgeGraph in-memory (igraph): entidades, relaciones, BFS
      triplet_extractor.py     # Extraccion de tripletas y keywords via LLM

sandbox_mteb/                  # Pipeline de evaluacion
  config.py                    # MTEBConfig: .env → dataclass validada (+RerankerConfig.validate)
  evaluator.py                 # Orquestador principal
  run.py                       # Entry point CLI (--dry-run, -v, --resume)
  loader.py                    # MinIO/Parquet → LoadedDataset
  retrieval_executor.py        # Loop retrieval + reranking
  generation_executor.py       # Generacion async + metricas
  embedding_service.py         # Pre-embed queries batch + context window detection
  checkpoint.py                # Checkpoint/resume cada N queries (atomic writes)
  result_builder.py            # Construccion EvaluationRun final
  preflight.py                 # Validacion pre-run (deps, NIM, MinIO)
  subset_selection.py          # DEV_MODE: gold docs + distractores

tests/                         # pytest (352 unit tests, 30 files)
  conftest.py                  # Mocks condicionales de infra (boto3, langchain, chromadb)
  test_*.py                    # Unit test files
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

## Divergencias con el paper original — evaluacion de criticidad

Diferencias entre esta implementacion y el [LightRAG original (EMNLP 2025)](https://arxiv.org/abs/2410.05779), evaluadas por impacto en calidad de resultados.

| # | Divergencia | Criticidad | Descripcion | Ubicacion |
|---|---|---|---|---|
| 1 | DAM-4: concat vs LLM synthesis (descripciones) | **7/10** | El paper usa LLM map-reduce para sintetizar descripciones de entidades repetidas. Aqui se concatena con `" \| "`. Las descripciones de entidades frecuentes crecen sin control, meten ruido en el contexto del graph traversal y degradan la calidad de generacion. Es la divergencia que mas distorsiona la representacion del conocimiento en el grafo. | `knowledge_graph.py` / [#7](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/7) |
| 2 | Sin validacion empirica (F.5 pendiente) | **6/10** | No es una divergencia arquitectonica, pero sin datos reales post-fases A-F no se puede saber si las correcciones cerraron la brecha. Los ultimos numeros (-48pp MRR) son pre-fix y obsoletos. Todas las demas correcciones son teoricas sin esto. | N/A — requiere infra NIM + MinIO |
| 3 | Sin LLM synthesis en fusion final de contexto | **5/10** | El paper usa el LLM para sintetizar resultados de vector + graph antes de generar la respuesta. Aqui se concatena directamente. RRF mitiga parcialmente al priorizar chunks relevantes, pero con un graph ruidoso la diferencia puede ser notable. | `retriever.py:_fuse_with_graph()` |
| 4 | Entity cap 100K con sesgo FIFO (DTm-63) | **4/10** | El paper no impone un cap tan agresivo. Con corpus grandes las primeras entidades indexadas dominan y las ultimas se pierden. Para HotpotQA (66K docs) probablemente no se alcanza el cap, pero limita escalabilidad a datasets mayores. | `knowledge_graph.py` |
| 5 | Grafo fragmentado sin bridging (DTm-73) | **3/10** | El paper asume un grafo mas conectado. Componentes desconectados no se enlazan. En HotpotQA (preguntas bridge entre 2 docs) puede impactar, pero el BFS 1-hop ya limita el alcance del traversal. | `knowledge_graph.py` |
| 6 | BFS scoring uniforme (DTm-72) | **3/10** | Todas las aristas pesan igual en el traversal. El paper usa edge weights mas sofisticados. Con 1-hop el efecto es limitado. | `knowledge_graph.py` |

**Nota:** Las divergencias 1 y 3 son las unicas que requieren cambios de logica de negocio (LLM calls adicionales, coste de tokens). Las demas son mejoras algoritmicas internas. La divergencia 2 (F.5) es prerequisito para priorizar cualquier fix — sin datos empiricos, el orden de prioridad es especulativo.

## Deuda tecnica vigente

- **DTm-80**: DAM-4 parcial: merge de descripciones por concatenacion, sin LLM synthesis. Requiere decision sobre coste/latencia. Diferido a post-F.5. [#7](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/7)

## Bare excepts aceptados (no criticos)

Estos `except Exception as e:` logean el error pero no lo re-lanzan. Aceptable para wrappers de infraestructura:

| Ubicacion | Contexto |
|---|---|
| `reranker.py:147` | Reranking error — retorna fallback sin rerank |
| `vector_store.py:125, 141, 178` | Operaciones ChromaDB — retorna lista vacia |

## Test coverage

| Metrica | Valor |
|---|---|
| Tests unitarios | 352 en 30 archivos |
| Tests integracion | 19 en 3 archivos |
| mypy | 0 errores (27 source files) |
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

### Run comparativo F.5 (requiere infra NIM + MinIO)

| Tarea | Descripcion |
|---|---|
| F.5a | Run SIMPLE_VECTOR baseline: 50q, 3500 docs, DEV_MODE, seed=42 |
| F.5b | Run LIGHT_RAG hybrid: misma config |
| F.5c | Run LIGHT_RAG graph_primary |
| F.5d | Analisis comparativo: MRR, Hit@5, Recall |

**Criterio de exito:** LIGHT_RAG MRR > 0.80 (vs 0.52 pre-VDBs). Resultados de F.5 determinan si DTm-80 (LLM synthesis) es necesario.

### Limitaciones conocidas (no accionables)

- Sesgo LLM-judge en faithfulness para respuestas cortas (inherente)
- No-determinismo HNSW ±0.02 (ChromaDB no expone `hnsw:random_seed`)
