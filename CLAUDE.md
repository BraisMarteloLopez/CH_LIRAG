# CLAUDE.md

## Que es este proyecto

Sistema de evaluacion RAG para benchmarking de pipelines de retrieval y generacion sobre datasets MTEB/BeIR (HotpotQA) con NVIDIA NIM. Dos estrategias: `SIMPLE_VECTOR` (embedding + ChromaDB) y `LIGHT_RAG` (vector + knowledge graph via LLM).

## Estructura clave

```
shared/                        # Libreria core (3,850 LOC)
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
      retriever.py             # LightRAGRetriever: vector + KG dual-level (1,056 LOC)
      knowledge_graph.py       # KnowledgeGraph in-memory (igraph): entidades, relaciones, BFS (878 LOC)
      triplet_extractor.py     # Extraccion de tripletas y keywords via LLM (757 LOC)

sandbox_mteb/                  # Pipeline de evaluacion (2,816 LOC)
  config.py                    # MTEBConfig: .env → dataclass validada
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

tests/                         # pytest (6,783 LOC, 199+ tests)
  conftest.py                  # Mocks condicionales de infra (boto3, langchain, chromadb)
  test_*.py                    # 25 unit test files
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

- **Config via .env**: toda la parametrizacion en `sandbox_mteb/.env`, leida por `MTEBConfig.from_env()` una sola vez. Sub-configs delegadas a `InfraConfig`, `RerankerConfig`, `RetrievalConfig` en shared/
- **Factory pattern**: `get_retriever(config, embedding_model)` en `shared/retrieval/__init__.py` crea el retriever correcto
- **2 estrategias**: `SIMPLE_VECTOR` y `LIGHT_RAG` — no hay mas. `HYBRID_PLUS` fue eliminada (DTm-83)
- **Enum en core.py**: `RetrievalStrategy` define las estrategias validas. `VALID_STRATEGIES` en `sandbox_mteb/config.py` debe coincidir
- **Tests**: `conftest.py` mockea modulos de infra (boto3, langchain, chromadb) si no estan instalados. Tests de integracion requieren NIM + MinIO reales. Mocks siempre a nivel de funcion, nunca modulos enteros
- **Logging**: JSONL estructurado via `shared/structured_logging.py`
- **Idioma**: codigo y comentarios en ingles/espanol mezclado (historico). Docstrings y variables en ingles
- **Numeracion de deuda**: serie DT-N (bugs originales, 9 resueltos) + serie DTm-N (refactor/mejoras). Gap DTm-39..44 no asignados

## Estrategia LIGHT_RAG — como funciona

Inspirada en [LightRAG (EMNLP 2025)](https://arxiv.org/abs/2410.05779).

**Indexacion**: LLM extrae tripletas (entidad, relacion, entidad) de cada doc → KnowledgeGraph in-memory (igraph) + ChromaDB para vector search. Entity VDB y Relationship VDB para resolucion semantica.

**Retrieval**: vector search + query keywords via LLM + graph traversal dual-level (entity VDB low-level + relationship VDB high-level) + fusion RRF.

**Modos** (`LIGHTRAG_MODE`): `hybrid` (default), `graph_primary`, `local`, `global`, `naive`.

**Fallback**: sin igraph o sin LLM → degrada a SimpleVectorRetriever puro.

**Alineacion con original (DAM-1 a DAM-8)**: Entity VDB, Relationship VDB, edge weights, gleaning, BFS 1-hop, graph_primary mode, contexto estructurado — todo implementado. DAM-4 parcial (concatenacion sin LLM synthesis). Pendiente validacion con run comparativo (F.5).

## Deuda tecnica vigente (priorizada)

### Alta — afectan calidad de resultados
- **DTm-62**: Fusion KG destruye ranking (MRR -33pp). Pendiente run comparativo F.5. Ubicacion: `retriever.py:_fuse_with_graph()`
- **DTm-63**: Entity cap 100K con sesgo FIFO en orden de indexacion. Ubicacion: `knowledge_graph.py`
- **DTm-73**: Grafo fragmentado impide bridging entre componentes desconectados. Ubicacion: `knowledge_graph.py`

### Media — mejoras funcionales
- **DTm-64**: Normalizacion scores incomparable entre canales vector/graph. RRF mitiga parcialmente. Ubicacion: `retriever.py`
- **DTm-55**: Stats se corrompen si KG build falla a mitad. `_has_graph=False` pero stats parciales persisten. Ubicacion: `retriever.py`
- **DTm-56**: Fingerprint collision edge case con corpus vacio. Ubicacion: `retriever.py`
- **DTm-72**: BFS scoring ciego a la relacion (todas las aristas pesan igual). Ubicacion: `knowledge_graph.py`
- **DTm-77**: Test gap: gleaning sin tests (`glean_from_doc_async`). [#4](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/4)
- **DTm-78**: Test gap: E2E solo cubre SIMPLE_VECTOR, no LIGHT_RAG. [#5](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/5)
- **DTm-80**: DAM-4 parcial: merge de descripciones por concatenacion, sin LLM synthesis. [#7](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/7)

### Baja — code quality
- **DTm-82**: Errores mypy sin resolver (`union-attr`, `dict-item`, imports condicionales). [#9](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/9)
- **DTm-57**: Normalizacion entidades pierde apostrofes. Ubicacion: `knowledge_graph.py:_normalize_name()`
- **DTm-58**: No dedup queries en batch keyword extraction. Ubicacion: `triplet_extractor.py`
- **DTm-60**: `reset_stats()` nunca se llama automaticamente. Ubicacion: `triplet_extractor.py`
- **DTm-61**: Sin validacion de tamano de keywords del LLM. Ubicacion: `triplet_extractor.py`
- **DTm-12**: Sesgo LLM-judge en faithfulness para respuestas cortas. Inherente al LLM-judge. Aceptado
- **DTm-13**: No-determinismo HNSW: ChromaDB no expone `hnsw:random_seed`. ±0.02 entre runs. Aceptado

## Bugs y code smells pendientes (detectados en auditoria)

### Bugs potenciales

| Ubicacion | Severidad | Descripcion |
|---|---|---|
| `knowledge_graph.py:517` | **Alta** | `deque[Tuple[str, int]]` usa sintaxis Python 3.9+. Falla en 3.8. Fix: `from typing import Deque` y usar `Deque[Tuple[str, int]]` |
| `knowledge_graph.py:237-238` | **Media** | `except Exception: pass` en stemmer — fallo silencioso sin log. Degrada tokenizacion sin aviso |
| `evaluator.py:584-585` | **Media** | `except Exception: pass` en `_cleanup()` — silencia errores de ChromaDB, puede dejar file handles abiertos |
| `preflight.py:234` | **Media** | `except Exception:` sin captura del error — usuario no sabe por que fallo la config |

### Dead code

| Ubicacion | Descripcion |
|---|---|
| `loader.py:78-80` | `list_available_datasets()` nunca llamado en toda la codebase |
| `preflight.py:62-72` | Loop sobre lista `optional = []` — dead code post-eliminacion HYBRID_PLUS |
| `config.py:10` | `import sys` sin usar |

### Bare excepts sin logging (6 instancias)

Patrón `except Exception: pass` o `except Exception as e:` con log insuficiente:

| Ubicacion | Contexto |
|---|---|
| `retriever.py:328` | VDB cleanup (entity) |
| `retriever.py:433` | VDB cleanup (relationship) |
| `retriever.py:1044` | Retrieve fallback general |
| `reranker.py:147` | Reranking error — pierde context del error |
| `vector_store.py:125, 141, 178` | Operaciones ChromaDB |

### Validacion de config incompleta

`MTEBConfig.validate()` solo valida storage + infra directamente. Las sub-configs delegadas (`RetrievalConfig`, `RerankerConfig`) no se validan. Si `RERANKER_MODEL_NAME` falta con `RERANKER_ENABLED=true`, el error aparece tarde en runtime, no en validacion.

## Test coverage — gaps conocidos

### Modulos sin tests unitarios (por riesgo)

| Modulo | LOC | Riesgo | Nota |
|---|---|---|---|
| `sandbox_mteb/checkpoint.py` | 160 | **Alto** | Perdida de datos si falla en runs de horas. Atomic writes no testeados |
| `shared/llm.py` | 460 | **Alto** | Core de generacion. Solo testeado via integration tests |
| `sandbox_mteb/embedding_service.py` | 192 | Medio | Auto-deteccion context window afecta calidad |
| `shared/report.py` | 287 | Bajo | Export JSON/CSV, ejercitado indirectamente |
| `shared/structured_logging.py` | 124 | Bajo | Utilidad de logging |
| `shared/vector_store.py` | 275 | Bajo | Wrapper ChromaDB, solo integration |

### Metricas de tests

| Metrica | Valor |
|---|---|
| Tests unitarios | 180+ en 25 archivos |
| Tests integracion | 19 en 3 archivos |
| Ratio test/produccion | 0.76x (6,783 / 8,883 LOC) |
| Modulos cubiertos | 25/29 (86%) |
| Tests con assertions | 100% |
| Mocks a nivel funcion | 100% (ningun mock de modulo entero) |

## Que NO tocar sin contexto

- `DatasetType.HYBRID` en `shared/types.py` — es un tipo de dataset (tiene respuesta textual), NO una estrategia de retrieval
- `reciprocal_rank_fusion()` en `core.py` — la usa LIGHT_RAG, no es legacy
- `shared/config_base.py` — la importan todos los modulos, cambios rompen todo
- Tests de integracion (`tests/integration/`) — dependen de NIM + MinIO reales
- `requirements.lock` — es un pin de produccion, no tocar sin razon
- `_PersistentLoop` en `shared/llm.py` — resuelve binding de event loop asyncio (DTm-45). Parece complejo pero es necesario

## Proximos pasos — plan por fases

### Fase H: Hardening (bugs de auditoria) — sin riesgo funcional

Corrige bugs potenciales y code smells sin alterar logica de negocio.
Cada tarea es independiente, ejecutable en paralelo.

| ID | Tarea | Archivos | Esfuerzo | Descripcion |
|---|---|---|---|---|
| H.1 | Fix `deque[Tuple]` Python 3.9+ | `knowledge_graph.py:517` | Trivial | Cambiar a `Deque[Tuple[str, int]]` de `typing`. Evita `TypeError` en Python 3.8 |
| H.2 | Logging en stemmer silencioso | `knowledge_graph.py:237-238` | Trivial | Reemplazar `except Exception: pass` con `logger.debug(...)` |
| H.3 | Logging en cleanup evaluator | `evaluator.py:584-585` | Trivial | Reemplazar `except Exception: pass` con `logger.debug(...)` |
| H.4 | Logging en VDB cleanup (3 sitios) | `retriever.py:328,433,1044` | Trivial | Reemplazar `except Exception: pass` con `logger.debug(...)` |
| H.5 | Logging en preflight config | `preflight.py:234` | Trivial | Capturar `as e` y mostrar al usuario |
| H.6 | Eliminar dead code | `loader.py:78-80`, `preflight.py:62-72`, `config.py:10` | Trivial | Eliminar `list_available_datasets()`, loop `optional` vacio, `import sys` |
| H.7 | Validacion sub-configs | `config.py:validate()` | Bajo | Propagar validacion a `RerankerConfig` (ej: model_name obligatorio si enabled) |

**Criterio de completitud:** tests pasan sin regresiones, 0 bare excepts sin logging.

### Fase I: Test coverage critico — reducir riesgo de regresion

Añade tests unitarios a los modulos de mayor riesgo sin cobertura.

| ID | Tarea | Archivos | Esfuerzo | Descripcion |
|---|---|---|---|---|
| I.1 | Tests `checkpoint.py` | `tests/test_checkpoint.py` (nuevo) | Medio | `save_checkpoint()` atomicidad, `load_checkpoint()` deserializacion, JSON corrupto, resume parcial |
| I.2 | Tests `llm.py` | `tests/test_llm.py` (nuevo) | Medio | `AsyncLLMService.invoke()` con mock langchain, `_strip_thinking_tags()`, `LLMMetrics.record_request()`, retry logic |
| I.3 | Tests `embedding_service.py` | `tests/test_embedding_service.py` (nuevo) | Bajo | `query_model_context_window()` con mock HTTP, batch embed con retry |
| I.4 | Test E2E LIGHT_RAG (DTm-78) | `tests/test_pipeline_e2e.py` | Medio | Parametrizar con LIGHT_RAG mocked. Verificar KG build + VDBs + fusion + metricas. [#5](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/5) |
| I.5 | Tests gleaning (DTm-77) | `tests/test_triplet_extractor.py` | Bajo | 5 tests para `glean_from_doc_async()`: mock LLM, empty entities, prompt format, rounds=0, integracion KG. [#4](https://github.com/BraisMarteloLopez/CH_LIRAG/issues/4) |

**Criterio de completitud:** cobertura modulos sube de 86% a 95%+. `checkpoint.py` y `llm.py` con tests.

### Fase G: Deuda tecnica menor — ya planificada

10 tareas independientes, ejecutable en paralelo con I.

| ID | Tarea | DTm | Esfuerzo |
|---|---|---|---|
| G.1 | Stats extractor resilientes (snapshot/restore) | DTm-55 | Bajo |
| G.2 | Fingerprint robusto (incluir len + config hash) | DTm-56 | Bajo |
| G.4 | Dedup queries en batch keywords | DTm-58 | Bajo |
| G.5 | `reset_stats()` al inicio de `extract_batch()` | DTm-60 | Trivial |
| G.6 | Keyword size cap (max 20/nivel) | DTm-61 | Trivial |
| G.7 | Entity normalization (preservar guiones internos) | DTm-57 | Bajo |
| G.9 | Tests gleaning | DTm-77 | Bajo |
| G.10 | E2E LIGHT_RAG | DTm-78 | Medio |
| G.11 | LLM synthesis merge (DAM-4 completo) | DTm-80 | Medio |
| G.13 | Mypy cleanup | DTm-82 | Bajo |

**Nota:** G.9 = I.5 y G.10 = I.4 (misma tarea, doble referencia). Ejecutar una vez.

### Fase F.5: Run comparativo — validacion funcional

Requiere infraestructura NIM + MinIO. No es codigo, es ejecucion y analisis.

| ID | Tarea | Dependencia | Descripcion |
|---|---|---|---|
| F.5a | Run SIMPLE_VECTOR baseline | Infra NIM | 50q, 3500 docs, DEV_MODE, seed=42 |
| F.5b | Run LIGHT_RAG hybrid | F.5a (misma config) | Mismo config, RETRIEVAL_STRATEGY=LIGHT_RAG |
| F.5c | Run LIGHT_RAG graph_primary | F.5a | LIGHTRAG_MODE=graph_primary |
| F.5d | Analisis comparativo | F.5a+b+c | Comparar MRR, Hit@5, Recall. Decide si fusion funciona (DTm-62) |

**Criterio de exito:** LIGHT_RAG MRR > 0.80 (vs 0.52 pre-VDBs). Si no, revisar DTm-62/DTm-64.

### Orden de ejecucion recomendado

```
Fase H (hardening)          ← Primero. Trivial, sin riesgo. ~1 hora
     │
     ├── Fase I (tests)     ← Segundo. Reduce riesgo antes de tocar logica. ~4-6 horas
     │
     └── Fase G (deuda)     ← Tercero, en paralelo con I donde no solape. ~6-8 horas
              │
              ▼
         Fase F.5 (runs)    ← Ultimo. Requiere infra. Valida todo lo anterior
```

**Total estimado:** H + I + G (descontando solapamiento G.9/I.5, G.10/I.4) = ~18 tareas unicas.
