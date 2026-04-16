# CLAUDE.md

## Que es este proyecto

Sistema de evaluacion RAG para benchmarking de pipelines de retrieval y generacion sobre datasets MTEB/BeIR (HotpotQA) con NVIDIA NIM. Dos estrategias: `SIMPLE_VECTOR` (embedding + ChromaDB) y `LIGHT_RAG` (vector + knowledge graph via LLM).

## Contexto del producto

Este proyecto es un subsistema de evaluacion RAG, no un producto final. Se integrara dentro de un sistema mas amplio cuya mision es administrar colecciones de datos (corpus documentales) y grafos de conocimiento, orquestando el ciclo de vida de las colecciones, versionado de KGs, consultas multi-tenant y APIs de uso. Los detalles especificos del sistema receptor se definiran mas adelante.

**Implicacion de diseno**: cualquier decision estructural debe favorecer la embedibilidad — configuracion declarativa, interfaces claras entre componentes, ausencia de side-effects globales, capacidad de operar sobre corpus arbitrarios (no solo HotpotQA). El valor de este subsistema no es resolver HotpotQA, es producir metricas fiables sobre cualquier corpus que el sistema administrador le entregue.

**Escenario real de uso esperado**: colecciones pequenas (10-50 PDFs) de dominio especializado, no publico, con idiosincrasia propia (terminologia tecnica, entidades internas, relaciones que no estan en el pre-entrenamiento de los embeddings). En ese regimen se espera que LIGHT_RAG demuestre robustez superior a SIMPLE_VECTOR — hipotesis aun por validar empiricamente (ver "Proximos pasos").

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
    core.py                    # RetrievalStrategy enum, RetrievalConfig, SimpleVectorRetriever
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

**Retrieval**: vector search produce el ranking de chunks. En modos `local`/`global`/`hybrid`, query keywords via LLM (dedup automatico, cap 20 keywords/nivel) resuelven entidades y relaciones del KG que se presentan como secciones separadas al LLM. Sin fusion RRF — cada canal es independiente.

**Modos** (`LIGHTRAG_MODE`): `naive` (solo chunks), `local` (entidades + chunks), `global` (relaciones + chunks), `hybrid` (default, entidades + relaciones + chunks). Todos los modos (excepto naive) presentan secciones separadas al LLM.

**Fallback**: sin igraph o sin LLM → degrada a SimpleVectorRetriever puro.

**Alineacion con original (DAM-1 a DAM-8)**: Entity VDB, Relationship VDB, edge weights (log1p), gleaning, BFS 1-hop weighted, co-occurrence bridging, LLM description synthesis — todo implementado.

**Divergencia arquitectonica critica**: pese a lo anterior, el pipeline de retrieval+generacion **no replica la arquitectura del paper**. La indexacion es fiel, pero la forma en que se consumen los resultados del KG difiere fundamentalmente (ver seccion siguiente).

## Divergencias con el paper original — evaluacion de criticidad

Diferencias entre esta implementacion y el [LightRAG original (HKUDS/LightRAG, EMNLP 2025)](https://arxiv.org/abs/2410.05779). Validadas empiricamente en F.5 (125q, 4000 docs, seed=42): LIGHT_RAG produjo metricas de retrieval **identicas** a SIMPLE_VECTOR, confirmando que las divergencias arquitectonicas anulan la contribucion del KG.

### Divergencias arquitectonicas (descubiertas en F.5)

| # | Divergencia | Criticidad | Detalle |
|---|---|---|---|
| 4+5 | ~~**Pipeline de consumo diverge del paper**~~ | ~~9/10~~ | **Resuelto.** RRF eliminado (`reciprocal_rank_fusion()`, `_full_fusion()`, `_vector_first_fusion()`, `_fuse_with_graph()` eliminados). `_enrich_with_graph()` recopila entidades y relaciones relevantes a la query via VDB y las propaga en `retrieval_metadata` como `kg_entities`/`kg_relations`. `generation_executor.py:89-103` invoca `format_structured_context()` que presenta secciones separadas al LLM. `graph_primary` (DAM-3) eliminado. |
| 6 | ~~**Reranker post-fusion anula senal del grafo**~~ | ~~8/10~~ | **Resuelto.** El `CrossEncoderReranker` ahora se desactiva automaticamente cuando `strategy=LIGHT_RAG` (`retrieval_executor.py:100-105`). Se mantiene activo para `SIMPLE_VECTOR`. |
| 7 | ~~**Sin token budgets separados por tipo**~~ | ~~5/10~~ | **Resuelto.** `format_structured_context()` en `retrieval_executor.py` ahora divide `max_context_chars` en presupuestos proporcionales segun el modo LightRAG: `hybrid` (20%/20%/60%), `local` (30%/0%/70%), `global` (0%/30%/70%). Budget no usado por KG se redistribuye a chunks, evitando que ninguna seccion aplaste a las otras. `generation_executor.py:89-107` propaga `lightrag_mode` desde `retrieval_metadata`. **Pendiente validacion empirica** (ver deuda tecnica #7). |

### Divergencias menores (preexistentes)

| # | Divergencia | Criticidad | Estado |
|---|---|---|---|
| 2 | Sin LLM synthesis en fusion final de contexto | **5/10** | Prerequisitos ya resueltos (#4+5, #7): contexto estructurado con secciones separadas y budgets independientes esta implementado. Synthesis es una mejora incremental: reescribir el contexto multi-seccion via LLM en una narrativa coherente antes de pasar al LLM generador. No bloqueante para validacion empirica; evaluar prioridad post re-run F.5. |
| 3 | Entity cap 100K | **3/10** | Eviction mejorada con score compuesto. Para HotpotQA (66K docs, ~24K entidades) no se alcanza. |

### Resueltas (indexacion)

- ~~DAM-4 (7/10)~~: LLM synthesis para descripciones → `KG_DESCRIPTION_SYNTHESIS=true` (A5.1)
- ~~Grafo fragmentado (3/10)~~: Co-occurrence bridging (DTm-73)
- ~~BFS scoring uniforme (3/10)~~: Edge weights via `log(1 + n_docs)` (DTm-72)

### Resultado F.5 (validacion empirica)

Runs ejecutadas: SIMPLE_VECTOR y LIGHT_RAG hybrid (125q, 4000 docs, DEV_MODE, seed=42, reranker ON).

| Metrica | SIMPLE_VECTOR | LIGHT_RAG | Delta |
|---|---|---|---|
| Hit@5 | 1.000 | 1.000 | 0 |
| MRR | 0.992 | 0.992 | 0 |
| Recall@5 | 0.968 | 0.968 | 0 |
| Recall@20 | 0.988 | 0.988 | 0 |
| Avg gen score | 0.7764 | 0.7877 | +0.011 (ruido LLM) |
| Tiempo total | 194.7s | 9002.1s | ×46 |

**Todas las metricas de retrieval son identicas query por query.** El KG aporto ~49 docs exclusivos por query, pero el reranker colapso el ranking final al mismo top-20 que SIMPLE_VECTOR. Las 10 diferencias en generacion son no-determinismo del LLM (mismo contexto, distinta respuesta).

**Conclusion**: la indexacion KG funciona correctamente (23K entidades, 55K relaciones, 32K co-occurrence edges), pero las divergencias #4+5 y #6 impiden que su senal llegue al LLM.

## Deuda tecnica vigente

| # | Item | Severidad | Ubicacion | Mitigacion temporal |
|---|---|---|---|---|
| 1 | ChromaDB: colecciones huerfanas si el proceso se interrumpe | **BAJO** | `evaluator.py:_cleanup()` ahora elimina la coleccion correctamente via `delete_all_documents()` (que llama `delete_collection()` + recrea). Sin embargo, cada run crea `eval_{run_id}` — si el proceso se interrumpe antes de cleanup, la coleccion queda huerfana. Con `PersistentClient`, se acumulan en disco | Aceptable; borrar manualmente `VECTOR_DB_DIR` si se acumulan |
| 2 | Preflight no valida datos reales | **MEDIO** | `preflight.py` solo verifica bucket MinIO (`head_bucket` + `list_objects MaxKeys=1`). No descarga ni parsea Parquet — dataset corrupto solo falla horas despues. No verifica espacio en disco | `--dry-run` primero y verificar que el dataset carga |
| 3 | HNSW no es determinista | **MEDIO** | ChromaDB no expone `hnsw:random_seed` — dos runs con misma config producen rankings con ~2-5% varianza | Ejecutar 2-3 veces y promediar, o aceptar varianza |
| 4 | LLM Judge puede devolver scores por defecto | **MEDIO-BAJO** | `metrics.py:_extract_score_fallback()` intenta 3 regex patterns (fraccion, decimal, entero con prefijo); si todos fallan retorna 0.5 — sesga metricas silenciosamente. Se logea a WARNING. Deuda a largo plazo: la mitigacion real requiere mas contexto de ventana y/o un modelo judge mas capaz que produzca respuestas estructuradas consistentemente | Post-run, buscar `"Score extraction fallback"` en logs y contar ocurrencias |
| 5 | Context window fallback silencioso | **BAJO** | `embedding_service.py:resolve_max_context_chars()` — si `GET /v1/models` falla, usa fallback de 4000 chars (~1000 tokens). Puede truncar docs importantes. Se logea WARNING | Configurar `GENERATION_MAX_CONTEXT_CHARS` explicitamente en `.env` |
| 6 | ~~Suite de tests no portable~~ | ~~CRITICO~~ | **Resuelto.** `conftest.py` ahora mockea `dotenv` (ademas de boto3/langchain/chromadb). `test_knowledge_graph.py` usa `pytest.importorskip("igraph")` para skip limpio sin igraph. Con `python-igraph` + `snowballstemmer` instalados: 409 pasan, 6 skipped. Sin igraph: 344 pasan, 65 skipped |
| 7 | ~~Validacion empirica pendiente post-refactor~~ | **RESUELTO** | F.5 re-ejecutado post-refactor (abril 2026) con las tres divergencias corregidas. Resultado: delta LIGHT_RAG vs SIMPLE_VECTOR se mantuvo en +1.19pp (vs +1.13pp pre-fix) — dentro del ruido del LLM judge. Los fixes estan bien implementados pero HotpotQA no los discrimina por ser home turf del embedding + DEV_MODE saturado + ventana de contexto amplia. Ver "Proximos pasos" para la siguiente direccion (experimento 3, dataset especializado). | N/A — la siguiente validacion requiere cambiar de dataset, no mas fixes |
| 8 | Infraestructura pesada para el scope | **BAJO** | Para 1 dataset y 2 estrategias, la infraestructura (checkpoint, preflight, JSONL, export dual, subset selection, DEV_MODE) es considerable. Sin embargo, el run F.5 demostro que esta infraestructura funciona y es util en practica | Aceptado — la infraestructura se justifica con uso real |
| 9 | Lock-in a NVIDIA NIM | **MEDIO** | Embeddings, LLM y reranker estan acoplados a NIM sin abstraccion de provider. Para un sistema de evaluacion, esto limita la reproducibilidad — nadie sin acceso a NIM puede ejecutar ni validar resultados | Abstraer detras de interfaces (ya existen Protocols en types.py pero no se usan para desacoplar el provider) |

Las divergencias arquitectonicas #4/#5/#6 son la causa raiz de que LIGHT_RAG no aporte valor. Ver seccion "Divergencias con el paper original".

## Bare excepts aceptados (no criticos)

Estos `except Exception as e:` logean el error pero no lo re-lanzan. Aceptable para wrappers de infraestructura:

| Ubicacion | Contexto |
|---|---|
| `reranker.py:147` | Reranking error — retorna fallback sin rerank |
| `vector_store.py:126, 142, 179, 232, 247` | Operaciones ChromaDB — retorna fallback (lista vacia, dict vacio, o continua cleanup) |

## Test coverage

| Metrica | Valor (abril 2026) |
|---|---|
| Tests unitarios | **409 pasan**, 6 skipped (integracion marker) en entorno con igraph+snowballstemmer. Sin igraph: 344 pasan, 65 skipped |
| Tests integracion | 19 en 3 archivos, requieren NIM + MinIO reales |
| mypy | 0 errores (27 source files) — no verificado en entorno limpio |

### Portabilidad de tests

`conftest.py` mockea modulos de infra (dotenv, boto3, langchain, chromadb) si no estan instalados. `test_knowledge_graph.py` usa `pytest.importorskip("igraph")` para skip limpio sin igraph. Dependencias opcionales para suite completa: `python-igraph`, `snowballstemmer`.

**Referencia completa**: ver `TESTS.md` — mapa test→produccion, atributos `object.__new__()`, trampas de mock, gaps de cobertura, reglas de modificacion.

## Que NO tocar sin contexto

- `DatasetType.HYBRID` en `shared/types.py` — es un tipo de dataset (tiene respuesta textual), NO una estrategia de retrieval
- `shared/config_base.py` — la importan todos los modulos, cambios rompen todo
- Tests de integracion (`tests/integration/`) — dependen de NIM + MinIO reales
- `requirements.lock` — es un pin de produccion, no tocar sin razon
- `_PersistentLoop` en `shared/llm.py` — resuelve binding de event loop asyncio (DTm-45). Parece complejo pero es necesario

## Proximos pasos

### Resultado F.5 post-refactor (abril 2026)

Tras resolver las divergencias arquitectonicas #4+5, #6 y #7, se re-ejecuto F.5 con config identica para ambas estrategias (125q, 2500 docs, DEV_MODE, seed=42, nemotron-3-nano).

| Metrica | SIMPLE_VECTOR | LIGHT_RAG hybrid | Delta |
|---|---|---|---|
| Hit@5 / MRR | 1.000 / 1.000 | 1.000 / 1.000 | 0 (saturado) |
| Avg gen score | 0.8038 | 0.8157 | **+0.0119** |
| Tiempo | 144s | 4589s | ×31.8 |

Delta pre-refactor era +0.0113. Delta post-refactor es +0.0119. **Los tres fixes arquitectonicos movieron la aguja 0.6 decimas de porcentaje** — dentro del ruido del LLM judge.

**Interpretacion**: HotpotQA es el *home turf* del embedding. Wikipedia esta en el pre-entrenamiento de `llama-embed-nemotron-8b`, DEV_MODE garantiza gold docs en el corpus (satura retrieval a 1.0), y la ventana de 192K chars permite al LLM leer los gold docs completos sin necesitar ayuda estructural del KG. Este dataset **no discrimina** entre estrategias.

**Los fixes estan correctamente implementados**. El KG se construye, las secciones estructuradas llegan al LLM con budgets proporcionales, el reranker no colapsa el ranking. Pero todo eso es invisible en un benchmark donde el embedding ya resuelve el problema por si solo.

### P0 — Experimento 3: evaluar subsistema sobre catalogo de PDFs especializados

La hipotesis del proyecto, alineada con su escenario real de uso (ver "Contexto del producto"), es que **LIGHT_RAG mantiene su rendimiento cuando el embedding se degrada por domain shift**, mientras que SIMPLE_VECTOR colapsa. HotpotQA no puede validar esto. El dataset especializado no sera un MTEB/BeIR publico — sera un **catalogo de PDFs de dominio especializado** gestionado por el futuro sistema administrador.

**Separacion de responsabilidades**:
- **Sistema administrador (futuro)**: ingesta de PDFs, chunking, extraccion de queries y qrels, almacenamiento en MinIO como Parquet. Define el catalogo.
- **Este subsistema (repo actual)**: consume el catalogo desde MinIO con el loader actual (`sandbox_mteb/loader.py`), ejecuta SIMPLE_VECTOR y LIGHT_RAG, produce metricas comparativas.

La interfaz entre ambos es el formato Parquet de queries/corpus/qrels que ya consume el loader para HotpotQA. **No hace falta cambiar la forma de consumir datos** — solo apuntar a un prefijo MinIO distinto con el nuevo catalogo.

**Trabajo tecnico necesario** (cuando el administrador este listo):
1. Confirmar que el administrador produce Parquet con el mismo schema que HotpotQA (columnas, tipos, ids). Si diverge, adaptar `_populate_from_dataframes()` en `loader.py` o pedir que el administrador se alinee con el schema existente.
2. Anadir entrada en `shared/types.py:DATASET_CONFIG` para el catalogo nuevo, con `primary_metric` y `dataset_type` apropiados segun como el administrador estructure el ground truth.
3. Parametrizar `S3_DATASETS_PREFIX` en `.env` para apuntar al catalogo nuevo.
4. Ejecutar F.6 comparativo (SIMPLE_VECTOR vs LIGHT_RAG hybrid) con metodologia estandar: seed fijo, subset reproducible, DEV_MODE para iteracion rapida + run completo para validacion final.

**Hipotesis a validar**: sobre un catalogo de 10-50 PDFs especializados (no publicos, terminologia propia, entidades internas), el delta LIGHT_RAG > SIMPLE_VECTOR crece significativamente (>3-5pp en gen score, >5-10pp en Recall@K) porque el embedding no tiene el dominio aprendido mientras que el KG se construye a partir del corpus mismo. Si se valida, este subsistema queda justificado como pieza del producto final y LIGHT_RAG pasa a ser estrategia default. Si no se valida, hay que reconsiderar el rol de LIGHT_RAG en el producto.

**Bloqueado por**: disponibilidad del catalogo en el administrador. Mientras tanto, el experimento 1 (P1) puede dar senal intermedia sobre robustez del KG en condiciones de retrieval dificil, sobre el mismo dataset HotpotQA.

### P1 — Experimento 1 (control intermedio, no bloqueante)

Control experimental gratis antes de invertir en adaptacion de dataset nuevo: desactivar DEV_MODE sobre HotpotQA y usar corpus completo (66K docs). No reemplaza el experimento 3, pero da senal rapida sobre si el KG aporta valor cuando el retrieval se vuelve dificil (sin gold docs garantizados en el corpus). Coste: 0 codigo, ~2-3h de infraestructura.

### P2 — Embedibilidad del subsistema (preparacion para integracion)

Auditar interfaces pensando en el sistema administrador que integrara este subsistema:
- Configuracion via diccionario inyectado, no solo via `.env` global
- Operacion sobre corpus pasados en memoria (no solo MinIO/Parquet)
- Sin asunciones sobre el sistema de ficheros excepto `EVALUATION_RESULTS_DIR` explicito
- Separacion limpia entre "cargar/indexar" y "evaluar" para permitir reuso de indices entre runs

No bloqueante para el experimento 3, pero necesario antes de la integracion real.

### Limitaciones conocidas (no accionables)

- Sesgo LLM-judge en faithfulness para respuestas cortas (inherente)
- No-determinismo HNSW ±0.02 (ChromaDB no expone `hnsw:random_seed`)
- Lock-in a NVIDIA NIM (deuda #9) — solo reproducible con acceso a NIM
