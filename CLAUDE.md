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

tests/                         # pytest (447 tests declarados, ver estado real abajo)
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

**Alineacion con original (DAM-1 a DAM-8)**: Entity VDB, Relationship VDB, edge weights (log1p), gleaning, BFS 1-hop weighted, graph_primary mode, contexto estructurado, co-occurrence bridging, LLM description synthesis — todo implementado.

## Divergencias con el paper original — evaluacion de criticidad

Diferencias restantes entre esta implementacion y el [LightRAG original (EMNLP 2025)](https://arxiv.org/abs/2410.05779).

| # | Divergencia | Criticidad | Estado |
|---|---|---|---|
| 1 | Sin validacion empirica (F.5 pendiente) | **8/10** | Requiere infra NIM + MinIO. Prerequisito para validar todo lo demas. Sin un solo run real, todo el codigo es teoria sin confirmar. |
| 2 | Sin LLM synthesis en fusion final de contexto | **7/10** | El paper sintetiza vector + graph antes de generar. Aqui se concatena. **Esta es la diferencia fundamental entre LightRAG y un vector search + graph lookup concatenado.** Sin esto y sin F.5, no se puede afirmar que esto implementa LightRAG. RRF no sustituye la synthesis. |
| 3 | Entity cap 100K | **3/10** | Eviction mejorada con score compuesto, pero cap se mantiene. Para HotpotQA (66K docs) no se alcanza. |

**Resueltas:**
- ~~DAM-4 (7/10)~~: LLM synthesis para descripciones → `KG_DESCRIPTION_SYNTHESIS=true` (A5.1)
- ~~Grafo fragmentado (3/10)~~: Co-occurrence bridging (DTm-73)
- ~~BFS scoring uniforme (3/10)~~: Edge weights via `log(1 + n_docs)` (DTm-72)

## Deuda tecnica vigente

| # | Item | Severidad | Ubicacion | Mitigacion temporal |
|---|---|---|---|---|
| 1 | ChromaDB no se limpia entre runs | **ALTO** | `evaluator.py:_cleanup()` llama `clear_index()` pero solo pone `_is_indexed=False` — no borra la coleccion ChromaDB. Cada run crea `eval_{run_id}`, las viejas se acumulan (~500MB-1GB tras 10 runs) | Borrar manualmente `VECTOR_DB_DIR` antes de cada run |
| 2 | Preflight no valida datos reales | **MEDIO** | `preflight.py` solo verifica bucket MinIO (`head_bucket` + `list_objects MaxKeys=1`). No descarga ni parsea Parquet — dataset corrupto solo falla horas despues. No verifica espacio en disco | `--dry-run` primero y verificar que el dataset carga |
| 3 | HNSW no es determinista | **MEDIO** | ChromaDB no expone `hnsw:random_seed` — dos runs con misma config producen rankings con ~2-5% varianza | Ejecutar 2-3 veces y promediar, o aceptar varianza |
| 4 | LLM Judge puede devolver scores por defecto | **MEDIO-BAJO** | `metrics.py:_extract_score_fallback()` intenta 4 regex patterns; si todos fallan retorna 0.5 — sesga metricas silenciosamente. Se logea a WARNING | Post-run, buscar `"Score extraction fallback"` en logs y contar ocurrencias |
| 5 | Context window fallback silencioso | **BAJO** | `embedding_service.py:resolve_max_context_chars()` — si `GET /v1/models` falla, usa fallback de 4000 chars (~1000 tokens). Puede truncar docs importantes. Se logea WARNING | Configurar `GENERATION_MAX_CONTEXT_CHARS` explicitamente en `.env` |
| 6 | Suite de tests no portable | **CRITICO** | `conftest.py` mockea boto3/langchain/chromadb pero no `python-dotenv` ni `igraph`. 17 archivos fallan en coleccion (dotenv), 65 tests fallan (igraph). Solo 14 de ~447 pasan en un entorno limpio. Ver seccion "Estado real de tests" abajo | Crear `requirements-test.txt` con todas las dependencias de test, o añadir dotenv+igraph a la lista de mocks en conftest.py |
| 7 | Cero validacion empirica | **CRITICO** | 20+ commits, multiples auditorias y refactors, pero F.5 sigue pendiente. El pipeline nunca se ha ejecutado contra datos reales. Metricas, retrieval y generation son teoria sin confirmar — podria haber bugs fundamentales sin detectar | Ejecutar un run minimo (10 queries) antes de cualquier otro desarrollo |
| 8 | Over-engineering para el scope | **MEDIO** | Para 1 dataset (HotpotQA) y 2 estrategias, el proyecto tiene: checkpoint/resume con atomic writes, preflight con validacion MinIO, logging JSONL, export JSON+CSV dual, subset selection, DEV_MODE. Infraestructura prematura para un benchmark que nunca se ha ejecutado | Congelar features de infraestructura hasta completar F.5 |
| 9 | Lock-in a NVIDIA NIM | **MEDIO** | Embeddings, LLM y reranker estan acoplados a NIM sin abstraccion de provider. Para un sistema de evaluacion, esto limita la reproducibilidad — nadie sin acceso a NIM puede ejecutar ni validar resultados | Abstraer detras de interfaces (ya existen Protocols en types.py pero no se usan para desacoplar el provider) |

La divergencia #2 (fusion synthesis) es mas critica de lo documentado previamente — es lo que define a LightRAG como algo mas que vector+graph concatenado. Requiere decision de implementacion, no solo coste/latencia.

## Bare excepts aceptados (no criticos)

Estos `except Exception as e:` logean el error pero no lo re-lanzan. Aceptable para wrappers de infraestructura:

| Ubicacion | Contexto |
|---|---|
| `reranker.py:147` | Reranking error — retorna fallback sin rerank |
| `vector_store.py:125, 141, 178` | Operaciones ChromaDB — retorna lista vacia |

## Test coverage

| Metrica | Valor declarado | Valor real (entorno limpio, abril 2026) |
|---|---|---|
| Tests unitarios | 447 en 38 archivos | **14 pasan**, 65 fallan, 17 archivos con error de coleccion |
| Tests integracion | 19 en 3 archivos | No verificados (requieren NIM + MinIO) |
| mypy | 0 errores (27 source files) | No verificado |

### Estado real de tests

**La suite de tests no es portable.** Ejecutar `pytest tests/ -m "not integration"` en un entorno limpio produce:

- **17 archivos con ERROR de coleccion**: `ModuleNotFoundError: No module named 'dotenv'`. `conftest.py` mockea boto3, langchain y chromadb, pero omite `python-dotenv` (importado por `shared/config_base.py`).
- **65 tests FALLAN** (tras instalar dotenv): todos los tests de `test_knowledge_graph.py` requieren `igraph` real — no hay mock ni skip condicional.
- **14 tests pasan**: solo los que no dependen de config_base ni igraph.

**Causa raiz**: `conftest.py` lineas 33-44 lista los modulos a mockear, pero falta `dotenv` e `igraph`. Los tests de KG asumen igraph instalado sin fallback.

**Implicacion**: la cifra "447 tests passing" solo es valida en el entorno de desarrollo del autor. Nadie mas puede verificarla.

**Fix necesario**: añadir `python-dotenv` e `python-igraph` a un `requirements-test.txt`, o incluirlos en la lista de mocks de `conftest.py`.

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
| F.5b | Run LIGHT_RAG hybrid: misma config (con `KG_DESCRIPTION_SYNTHESIS=true`) |
| F.5c | Run LIGHT_RAG graph_primary |
| F.5d | Analisis comparativo: MRR, Hit@5, Recall |

**Criterio de exito:** LIGHT_RAG MRR > 0.80 (vs 0.52 pre-VDBs). Si no, evaluar divergencia #2 (fusion synthesis).

### Limitaciones conocidas (no accionables)

- Sesgo LLM-judge en faithfulness para respuestas cortas (inherente)
- No-determinismo HNSW ±0.02 (ChromaDB no expone `hnsw:random_seed`)
