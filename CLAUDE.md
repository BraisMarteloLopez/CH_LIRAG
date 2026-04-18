# CLAUDE.md

## Que es este proyecto

Sistema de evaluacion RAG para benchmarking de pipelines de retrieval y generacion sobre datasets MTEB/BeIR (HotpotQA) con NVIDIA NIM. Dos estrategias: `SIMPLE_VECTOR` (embedding + ChromaDB) y `LIGHT_RAG` (vector + knowledge graph via LLM).

## Contexto del producto

Este proyecto es un subsistema de evaluacion RAG, no un producto final. Su proposito se descompone en dos fases secuenciales:

1. **Fase actual — Pre-P0: completitud arquitectural de LIGHT_RAG**. Cerrar todas las divergencias con el paper ([HKUDS/LightRAG, EMNLP 2025](https://arxiv.org/abs/2410.05779)) y demostrar que la arquitectura descrita corre end-to-end sin degradaciones sistematicas. **Nada puede estar parcialmente implementado**: si una pieza del paper falta (p.ej. un canal del path high-level) o se ejecuta pero degrada al fallback en una fraccion no despreciable de queries (p.ej. synthesis con `fallback_rate > 10%`), la implementacion no esta lista. Las funcionalidades extra propias del entorno (caches, fallbacks defensivos, instrumentacion) son adaptaciones operativas, no sustitutos de piezas del paper. Ver "Proximos pasos · Pre-P0".

2. **Fase siguiente — P0: replicacion empirica**. Una vez cerrada la fase anterior, demostrar sobre un benchmark donde el paper muestre ventaja que esta implementacion reproduce la direccion del delta (`LIGHT_RAG > SIMPLE_VECTOR`). Ver "Proximos pasos · P0".

Estado a fecha: las divergencias arquitectonicas #2, #4+5, #6, #7, #8, #9 estan resueltas (implementadas y ejercitadas en runs end-to-end previos). La divergencia #10 (keywords high-level por chunk durante indexacion) esta **implementada en codigo y cubierta por tests unitarios, pero pendiente de validacion end-to-end con NIM + MinIO reales** antes de promoverse a "Resuelta" — ver su fila en la tabla de divergencias para los criterios observables. Queda ademas la validacion de **ejecucion estable** — ningun run end-to-end cumple todavia los tres criterios de Pre-P0 simultaneamente (ver "Proximos pasos · Pre-P0"); en particular `kg_synthesis fallback_rate` supero el umbral del 10% en el ultimo smoke-test (deuda #16). Hasta cerrar Pre-P0, P0 sobre cualquier benchmark compararia una implementacion cuya arquitectura no esta validada end-to-end — el resultado no replicaria ni invalidaria el paper. Sin esa demostracion no se puede promover LIGHT_RAG a produccion ni proponerlo al sistema administrador.

**Vision a largo plazo — sistema administrador**: eventualmente este subsistema se integrara dentro de un sistema mas amplio cuya mision es administrar colecciones de datos, orquestar el ciclo de vida de corpus, versionado de KGs, consultas multi-tenant y APIs de uso. El administrador compartira infraestructura con este subsistema (MinIO + Parquet como contrato), asi que la integracion no implica cambiar como consumimos datos, solo apuntar a un prefijo MinIO distinto. **La integracion esta condicionada a que P0 (replicacion del paper) cierre con exito**; si no replicamos, lo unico integrable es SIMPLE_VECTOR y el trabajo sobre KG se vuelve inutil. Trabajo concreto en "Proximos pasos · P3".

**Implicacion de diseno**: las decisiones estructurales favorecen la embedibilidad futura — configuracion declarativa, interfaces claras, sin side-effects globales, capacidad de operar sobre corpus arbitrarios. **El valor de este subsistema no es resolver HotpotQA, es producir metricas fiables sobre cualquier corpus que el administrador le entregue.** Pero mientras P0 no este verde, la embedibilidad es solo un objetivo de diseno, no trabajo activo.

**Escenario de uso esperado (no confundir con el experimento 3)**: colecciones pequenas (10-50 PDFs) de dominio especializado, no publico, con idiosincrasia propia — terminologia tecnica, entidades internas, relaciones que no estan en el pre-entrenamiento de los embeddings. Este es el escenario tipico del producto a largo plazo. El experimento 3 (P2) es la prueba empirica concreta de ese escenario; el marco general queda aqui.

**Export de KG — proposito del futuro serializador**: cuando LIGHT_RAG sea estrategia de produccion (post-P0), los KGs deberan persistirse **para tres usos concretos del administrador**: versionado (distintas revisiones del mismo corpus), reuso entre runs (no re-indexar cada vez que entra una query nueva) y consultas multi-tenant (un mismo KG servido a varios consumidores). Detalle tecnico del serializador en "Proximos pasos · P3".

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

tests/                         # pytest (contador exacto: ver seccion "Test coverage")
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

Implementa la arquitectura de [LightRAG (EMNLP 2025)](https://arxiv.org/abs/2410.05779), con adaptaciones operativas propias del entorno (cache de KG a disco, fallbacks ante errores del LLM/igraph, instrumentacion de observabilidad).

**Indexacion**: LLM extrae tripletas (entidad, relacion, entidad) de cada doc + **high-level keywords tematicas del chunk** (divergencia #10 — piggyback en la misma llamada LLM, pendiente validacion end-to-end) → KnowledgeGraph in-memory (igraph) + ChromaDB para vector search. Entity VDB, Relationship VDB y Chunk Keywords VDB (tercera VDB, divergencia #10) para resolucion semantica. Stats se resetean automaticamente al inicio de cada batch (G.5). Gleaning opcional via `KG_GLEANING_ROUNDS` (no re-extrae keywords; solo entidades/relaciones perdidas).

**Retrieval (paper-aligned, divergencias #8 y #10 cerradas en codigo)**: en modos `local`/`global`/`hybrid`, los chunks se obtienen **a traves del KG** con tres canales:
— **Canal de entidades** (local/hybrid): query low-level keywords via LLM → Entity VDB → entidades resueltas → `source_doc_ids`.
— **Canal de relaciones** (global/hybrid): query high-level keywords → Relationship VDB → relaciones resueltas → `source_doc_ids` de los endpoints.
— **Canal de chunk keywords** (global/hybrid, divergencia #10): query high-level keywords → Chunk Keywords VDB → doc_ids directos.

Los tres canales suman al mismo `doc_scores` con formula simetrica `1/(1+rank) × similarity [× edge_weight]`. Contenido real se fetcha desde el vector store via `get_documents_by_ids`. Fallback a vector search directo cuando el KG no produce doc_ids (logueado en `metadata["kg_fallback"]`). Entidades y relaciones resueltas van en `retrieval_metadata.kg_entities`/`kg_relations` para la capa de synthesis; el conteo de matches del canal de chunk keywords va en `retrieval_metadata.kg_chunk_keyword_matches`. **Nota sobre scoring**: la formula usa `1/(1+rank)` como decay por posicion (inverse-rank), mientras el paper usa un contador descendiente lineal; la intencion es la misma pero la curva difiere. Las entidades y chunks no ponderan por weight (no hay peso equivalente almacenado); `len(source_doc_ids)` seria proxy viable para entidades pero el paper no lo hace explicitamente. Ver docstring de `_retrieve_via_kg` para detalle.

**Enriquecimiento KG (paper-aligned, divergencia #9 resuelta)**: cada entidad resuelta incluye sus vecinos 1-hop del grafo, rankeados por `edge_weight + degree_centrality` (`get_neighbors_ranked` en `knowledge_graph.py`), anidados como lista `"neighbors"` dentro del dict de entidad. Configurable via `KG_MAX_NEIGHBORS_PER_ENTITY` (default 5, 0 = desactivado). Cada relacion resuelta incluye las descripciones y tipos de sus entidades endpoint (`source_description`, `source_type`, `target_description`, `target_type`) obtenidos del KG. Degradacion graceful: si el lookup de vecinos o endpoints falla, la entidad/relacion aparece sin los campos extra.

**Modos** (`LIGHTRAG_MODE`): `naive` (solo chunks), `local` (entidades + chunks), `global` (relaciones + chunks), `hybrid` (default, entidades + relaciones + chunks). Todos los modos (excepto naive) presentan secciones separadas al LLM.

**Synthesis del contexto**: para queries con datos KG, `GenerationExecutor._synthesize_kg_context_async()` reescribe las 3 secciones como narrativa coherente via LLM *antes* de la generacion final. Prompt query-aware en `sandbox_mteb/config.py:KG_SYNTHESIS_SYSTEM_PROMPT` con reglas anti-fabricacion y citas `[ref:N]` inline. Se activa automaticamente para LIGHT_RAG (`KG_SYNTHESIS_ENABLED=true` default). **Faithfulness se evalua contra el contexto estructurado original**, no contra la narrativa, para penalizar cualquier alucinacion introducida por la propia capa de synthesis. Degradacion graceful: error LLM / vacio / timeout → fallback al contexto estructurado; stats por evento en `config_snapshot._runtime.kg_synthesis_stats` (ver "Observabilidad de runs").

**Fallback**: sin igraph o sin LLM → degrada a SimpleVectorRetriever puro. Fallos en la capa de synthesis → degrada al contexto estructurado (el run nunca se rompe).

## Divergencias con el paper original — evaluacion de criticidad

Diferencias entre esta implementacion y el [LightRAG original (HKUDS/LightRAG, EMNLP 2025)](https://arxiv.org/abs/2410.05779). Las divergencias 2, 4+5, 6, 7, 8, 9 estan resueltas (implementadas y ejercitadas en runs end-to-end). **La divergencia 10 esta implementada en codigo y cubierta por tests unitarios (29 tests), pero pendiente de validacion end-to-end con NIM + MinIO**: verificar que el LLM produce `high_level_keywords` consistentemente sobre un corpus real, que la Chunk Keywords VDB se construye sin errores, y que el canal contribuye al scoring en queries reales. Solo despues de ese run end-to-end se podra promover #10 a "Resuelta". La validacion de ejecucion estable y la validacion empirica sobre un benchmark donde el paper muestra ventaja son el gate Pre-P0 + P0 ("Proximos pasos").

### Divergencias arquitectonicas

| # | Divergencia | Criticidad | Estado |
|---|---|---|---|
| 8 | **Chunks via KG (opcion A)** | **Resuelta** | `_retrieve_via_kg()` en `retriever.py` obtiene chunks a traves de `source_doc_ids` de entidades/relaciones resueltas via Entity/Relationship VDB, con reference-count scoring. Fallback a vector search cuando KG devuelve vacio. Codigo muerto eliminado: `query_entities`, `query_by_keywords`, `get_entities_for_docs`, `get_relations_for_docs`, keyword index infrastructure, `_resolve_entity_names`, `_resolve_relationships_via_vdb`. |

### Divergencias de indexacion

| # | Divergencia | Criticidad | Estado |
|---|---|---|---|
| 10 | **High-level keywords por chunk durante indexacion** | **Implementada, pendiente validacion end-to-end** | Piggyback en la extraccion de tripletas (misma llamada LLM). El LLM emite ademas de `entities` y `relations` un campo `high_level_keywords` con los temas abstractos del chunk. Pipeline: (1) `TRIPLET_EXTRACTION_PROMPT`/`TRIPLET_EXTRACTION_BATCH_PROMPT` en `triplet_extractor.py` instruyen al LLM a emitir el campo; `_build_entities_relations` aplica dedup case-insensitive, cap de 10 por doc y longitud in [2, 80]. (2) `KnowledgeGraph` persiste `_doc_to_keywords: Dict[doc_id, List[str]]` via `add_doc_keywords()` con serializacion v3 (v2 invalidado explicitamente con `ValueError` en `from_dict`). (3) `LightRAGRetriever._build_chunk_keywords_vdb()` construye una tercera VDB `{base}_chunk_keywords` (cosine, una entrada por doc_id, `page_content` = keywords joined por coma). (4) `_retrieve_via_kg` en modos `global`/`hybrid` llama `_resolve_chunks_via_keywords_vdb(high_level, top_k=KG_CHUNK_KEYWORDS_TOP_K)` y acumula `1/(1+rank) * similarity` por doc matched al mismo `doc_scores` de entidades/relaciones. Toggled por `KG_CHUNK_KEYWORDS_ENABLED` (default true). Metadata por query: `kg_chunk_keyword_matches` (conteo). Tests: `test_chunk_keywords_retrieval.py` (12 casos) + extensiones en `test_triplet_extractor.py` (7 casos) y `test_knowledge_graph.py` (10 casos, incluye rechazo de caches v2/v1). **Pendiente antes de marcar "Resuelta"**: ejecutar un run end-to-end con NIM + MinIO reales y verificar en los logs de `TripletExtractor` que `stats.docs_with_keywords / stats.docs_success > 0` (el LLM emite el campo sobre texto real), en logs de `_build_chunk_keywords_vdb` que la tercera VDB se construye sin errores, y en `retrieval_metadata.kg_chunk_keyword_matches` por-query que al menos una fraccion no despreciable de queries con mode=`hybrid`/`global` obtiene chunks via este canal. Si en el run se observa `docs_with_keywords=0`, el prompt/modelo no coopera y #10 no esta cerrada. |

### Divergencias menores (cosmeticas / no funcionales)

| # | Divergencia | Criticidad | Estado |
|---|---|---|---|
| 3 | Entity cap 100K | **3/10** | Eviction mejorada con score compuesto. Para HotpotQA (66K docs, ~24K entidades) no se alcanza. |
| 11 | **Chunks en coleccion principal de ChromaDB, no en `text_chunks_vdb` dedicado** | **2/10** | El paper mantiene 3 VDBs: `entities_vdb`, `relationships_vdb` y `text_chunks_vdb` (coleccion separada para chunks con su id y metadata). Aqui los chunks se indexan en la coleccion principal de `ChromaVectorStore` (heredada de `SimpleVectorRetriever` via `retriever.py:87-90`), no en un VDB dedicado. Las Entity VDB y Relationship VDB si son colecciones separadas (`_build_entities_vdb` en `retriever.py:380`, `_build_relationships_vdb` en `retriever.py:487`). **Consecuencia practica post-#8 resuelta**: con #8 cerrada (opcion A), los chunks se obtienen via `source_doc_ids` del KG en el path principal; la coleccion principal de `ChromaVectorStore` actua de facto como `text_chunks_vdb` (contiene los chunks con su id y metadata) y ademas sirve al fallback de vector search directo cuando el KG no produce doc_ids. Separar formalmente en colecciones dedicadas seria un cambio de naming/organizacion sin efecto funcional. **Criterio para re-evaluar**: si en P3 el schema de export hacia el administrador requiere distinguir VDBs por rol (p.ej. versionado independiente de chunks vs. VDBs del KG), o si aparece un segundo consumidor del `text_chunks_vdb` que no quiera la dependencia con `ChromaVectorStore`. |
| 12 | **Formato de contexto JSON-lines, no CSV con headers del paper** | **2/10** | El paper original presenta el contexto al LLM como tablas CSV con headers `Entities \| Relationships \| Sources`. Aqui `format_structured_context()` en `retrieval_executor.py:246-340` usa bloques JSON-lines etiquetados como `"Knowledge Graph Data (Entity)"`, `"Knowledge Graph Data (Relationship)"` y `"Document Chunks"` con `reference_id` numerico (usado por la capa de synthesis para citas `[ref:N]`). **Consecuencia practica**: ambos son formatos estructurados que el LLM parsea sin problema. El JSON-lines tiene la ventaja de que la capa de synthesis (divergencia #2 resuelta) usa los `reference_id` para anclar citas. No requiere accion — cambiar a CSV romperia el esquema de citas sin beneficio funcional. |

Runs F.5 (resultados empiricos pre-refactor y post-refactor) en "Proximos pasos · Resultado F.5".

## Deuda tecnica vigente

| # | Item | Severidad | Ubicacion | Mitigacion temporal |
|---|---|---|---|---|
| 1 | ChromaDB: colecciones huerfanas si el proceso se interrumpe | **BAJO** | `evaluator.py:_cleanup()` ahora elimina la coleccion correctamente via `delete_all_documents()` (que llama `delete_collection()` + recrea). Sin embargo, cada run crea `eval_{run_id}` — si el proceso se interrumpe antes de cleanup, la coleccion queda huerfana. Con `PersistentClient`, se acumulan en disco | Criterio de accion: auditar `VECTOR_DB_DIR` entre campanas (p.ej. pre/post P0) y purgar colecciones `eval_*` que no correspondan a runs vivos. Si el tamano del directorio supera el presupuesto de disco del host evaluador, automatizar la purga en `preflight.py` |
| 2 | Preflight no valida datos reales | **MEDIO** | `preflight.py` solo verifica bucket MinIO (`head_bucket` + `list_objects MaxKeys=1`). No descarga ni parsea Parquet, no valida schema contra `DATASET_CONFIG`, no verifica espacio en disco. El riesgo principal no es infra (MinIO ya es compartido con el administrador) sino **schema drift del contrato upstream**: cuando el administrador produzca un catalogo nuevo con columnas/tipos/ids diferentes, el fallo ocurre horas despues del start en `_populate_from_dataframes()`, quemando compute | `--dry-run` primero y verificar que el dataset carga |
| 3 | HNSW no es determinista | **MEDIO** | ChromaDB no expone `hnsw:random_seed` — dos runs con misma config producen rankings con ~2-5% varianza | Ejecutar 2-3 veces y promediar, o aceptar varianza |
| 5 | Context window fallback silencioso | **BAJO** | `embedding_service.py:resolve_max_context_chars()` — si `GET /v1/models` falla, usa fallback de 4000 chars (~1000 tokens). **Aclaracion importante**: este valor es el **presupuesto total de contexto** pasado al LLM, que `format_context()`/`format_structured_context()` usan para seleccionar cuantos fragmentos (chunks) caben. No es "el LLM recibe un documento truncado a 4000 chars". Con chunks tipicos de 500-1000 chars, 4000 = ~4-8 chunks relevantes. El riesgo es dejar senal en la mesa cuando el modelo soporta mas (p.ej. 192K chars): no se rompe nada, simplemente no se aprovecha toda la ventana. Se logea WARNING pero el run continua sin detectar la degradacion. Criterio de accion: para los runs de P0 el valor efectivo debe verificarse en `config_snapshot._runtime.max_context_chars` y ser consistente con la capacidad declarada del modelo LLM usado | Configurar `GENERATION_MAX_CONTEXT_CHARS` explicitamente en `.env`. Antes de cada run P0, auditar `_runtime.max_context_chars` en el JSON — si muestra 4000 sin intencion explicita, descartar el run |
| 8 | Infraestructura pesada para el scope | **BAJO** | Para 1 dataset y 2 estrategias, la infraestructura (checkpoint, preflight, JSONL, export dual, subset selection, DEV_MODE) es considerable. El run F.5 ejercito cada componente sin incidentes. Criterio de accion: revisar al final de P0 que componentes tuvieron al menos una invocacion efectiva en los runs reales (checkpoint usado en `--resume`, preflight bloqueando runs mal configurados, etc.); si alguno queda sin uso despues de tres runs P0 consecutivos, candidato a eliminacion | Pendiente de revision post-P0 con el criterio anterior. No actuar antes de tener los datos |
| 9 | Lock-in a NVIDIA NIM | **MEDIO** | Embeddings, LLM y reranker estan acoplados a NIM sin abstraccion de provider. Para un sistema de evaluacion, esto limita la reproducibilidad — nadie sin acceso a NIM puede ejecutar ni validar resultados | Abstraer detras de interfaces (ya existen Protocols en types.py pero no se usan para desacoplar el provider) |
| 10 | Sin indexacion incremental del KG | **MEDIO** | `LightRAGRetriever.index_documents()` (`retriever.py:127-195`) siempre hace build completo del KG o carga desde cache. No hay `append_documents()` que anada docs nuevos a un KG existente sin reconstruir desde cero. `KnowledgeGraph` tiene `add_triplets()` y `add_entity_metadata()` como primitivas, pero falta el path de alto nivel que: (1) extraiga tripletas solo de docs nuevos, (2) las integre en el KG existente, (3) reconstruya solo los VDBs afectados. El paper (HKUDS/LightRAG) soporta ingestion incremental via `insert()`. Para el escenario del sistema administrador (PDFs que llegan en tandas), la falta de append obliga a re-indexar todo el corpus ante cada doc nuevo. Workaround: usar cache de disco y reconstruir | Cache de disco mitiga la re-extraccion LLM pero no el rebuild del grafo ni de VDBs. P3 (embedibilidad) requiere resolver esto |
| 11 | Duplicacion parcial de logica de iteracion de vecinos en KG | **BAJO** | `_get_neighbors_weighted()` (`knowledge_graph.py:307`) itera `self._graph.incident(vid)` para obtener `(neighbor_name, edge_weight)`. `get_neighbors_ranked()` (divergencia #9) necesita ademas `degree_centrality` y la etiqueta de relacion del edge, asi que hace su propia iteracion sobre los mismos edges en vez de reutilizar `_get_neighbors_weighted`. La duplicacion es deliberada: `get_neighbors_ranked` accede a campos distintos del edge (`relations[0]`, `graph.degree()`) que `_get_neighbors_weighted` no expone, y extender la firma privada para cubrir ambos casos complicaria la API interna sin beneficio claro | Aceptable mientras solo haya dos consumidores. Si aparece un tercero, refactorizar ambos en un iterador comun sobre edges |
| 12 | Tests acoplados a mensajes de error de `AsyncLLMService` | **BAJO** | `tests/test_llm.py:145` (empty-response path) y `tests/test_llm.py:252` (retry exhaustion) usan regex laxa sobre el mensaje de `RuntimeError` porque hoy no hay excepciones custom que discriminen los casos. Mejor solucion: definir excepciones especializadas en `shared/llm.py` (`EmptyResponseError`, `RetriesExhaustedError`) y asertar por tipo + `call_count` del mock. **Coste real**: si el texto de `raise RuntimeError(...)` en `shared/llm.py` cambia, los tests fallan por regex, no por cambio de comportamiento — falso positivo del que solo se detecta editando el fichero | Criterio de accion: la proxima modificacion funcional a `shared/llm.py` incluye el refactor de excepciones como parte del PR. Mientras tanto, cualquier cambio a los mensajes de `RuntimeError` requiere sincronizar los regex de los dos tests |
| 13 | Reranker se inicializa innecesariamente para LIGHT_RAG | **BAJO** | `evaluator.py:342-353` crea `CrossEncoderReranker` (conecta a NIM) siempre que `RERANKER_ENABLED=true`, sin verificar la estrategia. `retrieval_executor.py:102-104` lo auto-desactiva para LIGHT_RAG (`use_reranker = self._reranker and configured_strategy != RetrievalStrategy.LIGHT_RAG`), asi que nunca se usa. Resultado: conexion HTTP desperdiciada al start. No afecta metricas ni resultados | Mover el guard a `_init_components()` en `evaluator.py`: skip init si `strategy == LIGHT_RAG`. Alternativa: log WARNING al detectar la combinacion |
| 14 | Acceso a `_vector_store.collection_name` desde `LightRAGRetriever` | **BAJO** | `retriever.py:_build_entities_vdb()` y `_build_relationships_vdb()` acceden a `self._vector_retriever._vector_store.collection_name` para derivar nombres de colecciones Entity/Relationship VDB. Es un acceso a atributo privado de otro objeto, pero solo para naming (no para datos). **Coste real**: si alguien renombra `collection_name` en `SimpleVectorRetriever._vector_store` sin buscar referencias externas, `LightRAGRetriever` rompe en runtime al indexar. Sin test que lo detecte aisladamente | Criterio de accion: exponer property `collection_name` en `SimpleVectorRetriever` en el primer cambio que toque esa clase, o cuando aparezca un segundo consumidor externo del nombre de coleccion |
| 15 | **`retrieval_metadata` por query — exportacion JSON/CSV** | **RESUELTA** | Fix unificado sobre ambos exports. (1) **JSON**: `QueryEvaluationResult.to_dict()` en `shared/types.py` ahora emite un bloque `retrieval_metadata` per-query con el subset relevante — `lightrag_mode`, `kg_fallback` (`no_keywords`/`no_doc_ids`/`docs_not_in_store`/ausente), `kg_chunk_keyword_matches`, `kg_entities_count`, `kg_relations_count`, `kg_synthesis_used`, `kg_synthesis_error` (`timeout`/`error`/`empty`/ausente). Listas completas de entidades/relaciones se colapsan a conteos para no inflar el JSON; queries SIMPLE_VECTOR no emiten el bloque (omitted si subset vacio). (2) **CSV**: `shared/report.py:to_detail_csv()` tenia un guard muerto que buscaba `graph_candidates` (clave eliminada con la resolucion de divergencia #8), por lo que el bloque `if has_lightrag:` nunca ejecutaba y el detail.csv no tenia columnas KG en runs reales (auditado en `mteb_hotpotqa_20260418_223530_detail.csv`, cabecera sin `graph_*`). El guard y las columnas se reescribieron usando `extract_retrieval_metadata_subset()` como unica fuente de verdad compartida con el JSON: columnas `lightrag_mode`, `kg_fallback`, `kg_entities_count`, `kg_relations_count`, `kg_chunk_keyword_matches`, `kg_synthesis_used`, `kg_synthesis_error`. (3) **Synthesis outcome per-query**: `_synthesize_kg_context_async` ahora retorna `Tuple[str, Optional[str]]` con el `error_code`, y `_process_single_async` escribe `kg_synthesis_used`/`kg_synthesis_error` en `retrieval_detail.retrieval_metadata` para que ambos exports lo consuman. Tests: `test_report.py` (4 casos nuevos JSON + CSV, + regresion guard SIMPLE_VECTOR), `test_kg_synthesis.py::TestPerQuerySynthesisMetadata` (5 casos cubriendo success/timeout/empty/error/no-kg-data). Listo para usar en el run diagnostico del paso 2 del HANDOFF. | — |
| 16 | **Synthesis timeouts en corpus pequeno** | **MEDIO (observacional)** | Smoke-test `mteb_hotpotqa_20260418_183018` (800 docs, 25q, hybrid, `KG_SYNTHESIS_TIMEOUT_S=90`, `KG_SYNTHESIS_MAX_CHARS=0`) reporto `kg_synthesis_stats.fallback_rate=0.24` (6/25 timeouts, 0 errores, 0 empty). El umbral de CLAUDE.md es 10% — el run no refleja la arquitectura completa. En un corpus de 800 docs no deberia haber saturacion de entidades/relaciones; las hipotesis a descartar son: (a) prompt de synthesis demasiado largo cuando `KG_SYNTHESIS_MAX_CHARS=0` deja pasar contexto de 192K chars, (b) latencia del LLM bajo carga concurrente, (c) timeout de 90s insuficiente para el NIM configurado. Sin resolver, cualquier run LIGHT_RAG con synthesis ON es inconclusivo porque una fraccion no despreciable de queries se evalua contra el contexto estructurado — no contra la narrativa que es la tesis del paper | Instrumentar el tiempo por invocacion de synthesis (p50/p95) para discriminar entre (a)(b)(c). Mientras tanto, ningun run con `fallback_rate > 0.1` vale como evidencia de P0 — re-ejecutar o ajustar `KG_SYNTHESIS_TIMEOUT_S` / `KG_SYNTHESIS_MAX_CHARS` |
| 17 | **Parametros fijos del canal de chunk keywords (divergencia #10)** | **BAJO (observacion)** | La resolucion de #10 expone solo `KG_CHUNK_KEYWORDS_ENABLED` y `KG_CHUNK_KEYWORDS_TOP_K`. Hay tres parametros fijados en `triplet_extractor.py`/`retriever.py` que podrian necesitar exposicion si el comportamiento empirico lo exige: (a) `MAX_CHUNK_KEYWORDS_PER_DOC=10` (cap de keywords por doc en el parseo), (b) `MIN_CHUNK_KEYWORD_LEN=2` / `MAX_CHUNK_KEYWORD_LEN=80` (validacion de longitud), (c) `_CHUNK_KEYWORDS_VDB_MAX_DISTANCE=0.8` (threshold de cosine distance, simetrico con Entity/Relationship VDB). Criterio de accion: exponer al `.env` solo si en un run end-to-end se observa que un caso real (p.ej. corpus con terminologia muy larga) cae fuera de estos defaults. No actuar antes de tener esa evidencia — exponer parametros sin uso concreto es over-engineering | Sin accion hasta que un run real demuestre necesidad |

## Observabilidad de runs

Los `EvaluationRun` exportados a JSON incluyen en `config_snapshot._runtime` dos bloques de stats para auditoria post-run:

**`judge_fallback_stats`**: solo aparecen las `MetricType` del judge **efectivamente invocadas en el run**. Para datasets `HYBRID` (p.ej. HotpotQA) el judge corre `faithfulness` como secundaria → solo esa clave aparece. `answer_relevance` aparece cuando el dataset usa `MetricType.ANSWER_RELEVANCE` como primary o secondary (hoy, datasets tipo `ADAPTED` sin expected answer o con `secondary_metrics=[ANSWER_RELEVANCE]`). Por metrica se reporta `invocations`, `parse_failures` (JSON no parseable), `default_returns` (0.5 por defecto), `parse_failure_rate`, `default_return_rate`. Si `default_return_rate > JUDGE_FALLBACK_THRESHOLD` (default 2%) en cualquier metrica invocada, el run falla con `RuntimeError`. Que una metrica no aparezca no es un bug — significa que no se invoco; auditar via `primary_metric_type` / `secondary_metrics` en `query_results`.

```bash
jq '.config_snapshot._runtime.judge_fallback_stats' data/results/<run_id>.json
```

**`kg_synthesis_stats`**: cuando `KG_SYNTHESIS_ENABLED=true` y la estrategia es `LIGHT_RAG`, reporta `invocations`, `successes`, `errors`, `empty_returns`, `truncations`, `timeouts`, `fallback_rate`. `fallback_rate > 10%` indica degradacion frecuente — el run no refleja la arquitectura completa. **Observacion empirica (abril 2026)**: smoke-test `mteb_hotpotqa_20260418_183018` (800 docs, 25q, hybrid, timeout 90s) reporto `fallback_rate=0.24` con 6 timeouts / 0 errores — inusual en corpus pequeno, ver deuda #16.

```bash
jq '.config_snapshot._runtime.kg_synthesis_stats' data/results/<run_id>.json
```

Ambos tambien se emiten en el evento estructurado `run_complete` del JSONL y en logs INFO al final de cada run.

## Bare excepts tolerados (con criterio)

Estos `except Exception as e:` logean el error y devuelven un fallback en vez de re-lanzar. El criterio para tolerarlos es que esten en wrappers de infraestructura donde el run debe continuar ante errores operacionales puntuales (ChromaDB transitorio, NIM latencia), cada uno contabiliza el evento en stats (`kg_synthesis_stats`, `rerank_failures`) o loguea a `logger.warning`/`debug` para trazabilidad post-mortem, y el fallback es observable desde el JSON del run. Si un bare except no cumple las tres condiciones, es candidato a reclasificar:

| Ubicacion | Contexto |
|---|---|
| `reranker.py:147` | Reranking error — retorna fallback sin rerank |
| `vector_store.py:126, 142, 179, 232, 247` | Operaciones ChromaDB — retorna fallback (lista vacia, dict vacio, o continua cleanup) |
| `generation_executor.py` (`_synthesize_kg_context_async`) | `asyncio.TimeoutError` + `Exception` genericos durante synthesis KG — fallback al contexto estructurado. Todos los eventos contabilizados en `kg_synthesis_stats` (errors/timeouts) |

## Test coverage

| Metrica | Valor (abril 2026) |
|---|---|
| Tests unitarios | **~465 pasan**, 6 skipped con `python-igraph` + `snowballstemmer` instalados; **~392 pasan**, 7 skipped sin igraph (los tests que lo requieren se saltan limpiamente). Cifras orientativas — drift con cada PR; para el valor exacto del entorno: `pytest --collect-only -q tests/ \| tail -1` y `pytest -m "not integration" tests/`. Ultimas adiciones reseñables: `test_judge_fallback_tracker.py` (deuda #4), `test_kg_synthesis.py` (divergencia #2) |
| Tests integracion | **~15** en 3 archivos (`tests/integration/`), requieren NIM + MinIO reales. Cifra orientativa — verificar via `pytest --collect-only -q tests/integration/` (solo si `.env` esta configurado) |
| mypy | 0 errores nuevos en ficheros modificados; 3 errores preexistentes no relacionados (dotenv/numpy sin stubs, `retrieval_executor.py:124` union-attr) |

### Portabilidad de tests

`conftest.py` mockea modulos de infra (dotenv, boto3, langchain, chromadb) si no estan instalados. `test_knowledge_graph.py` usa `pytest.importorskip("igraph")` para skip limpio sin igraph. Dependencias opcionales para suite completa: `python-igraph`, `snowballstemmer`.

**Referencia completa**: ver `TESTS.md` — mapa test→produccion, atributos `object.__new__()`, trampas de mock, gaps de cobertura, reglas de modificacion.

## Que NO tocar sin contexto

- `DatasetType.HYBRID` en `shared/types.py` — es un tipo de dataset (tiene respuesta textual), NO una estrategia de retrieval
- `shared/config_base.py` — la importan todos los modulos, cambios rompen todo
- Tests de integracion (`tests/integration/`) — dependen de NIM + MinIO reales
- `requirements.lock` — es un pin de produccion, no tocar sin razon
- `_PersistentLoop` en `shared/llm.py` — resuelve binding de event loop asyncio (DTm-45). Parece complejo pero es necesario

## Limitaciones de claude_code sobre este proyecto

Patron de fallo observado empiricamente durante el desarrollo: **claude_code tiende a presentar trabajo como completo antes de tiempo, y a categorizar problemas detectados como "menores" o "aceptables" cuando arreglarlos implicaria mas trabajo**. Es un sesgo hacia cerrar la tarea en vez de dejarla verdaderamente terminada.

**Por que importa aqui**: el ciclo de validacion de este proyecto es caro. Un run LIGHT_RAG sobre HotpotQA (125 queries, DEV_MODE) tarda ~1h30min y consume presupuesto de NIM. Un parametro mal capeado, un timeout hardcoded sin exponer, un test que no estresa el edge case real, o una alineacion parcial con el paper (secciones si, total no) no se detectan en `pytest`; se descubren cuando el run ya se ejecuto y las metricas no son interpretables, o cuando la comparacion con el paper se vuelve inverificable. Cada iteracion fallida son horas perdidas y datos no publicables.

**Manifestaciones concretas a vigilar**:
- Auto-evaluacion reactiva (solo tras peticion explicita) en vez de integrada al entregable.
- Adjetivos como "aceptable", "pendiente menor", "por ahora" aplicados a deuda tecnica sin medir su coste real.
- Tests que pasan pero no estresan el caso que motivo el cambio (p.ej. capping proporcional probado solo con budgets holgados).
- Alineacion con el paper descrita como completa cuando solo cubre un subconjunto de parametros (p.ej. budgets por seccion sin el budget total).
- Parametros operacionales hardcoded en `constants.py` en vez de expuestos en `.env` con la excusa de que "el default es razonable".

**Contramedidas para claude_code (al trabajar en este repo)**:
1. Antes de decir "listo para validar", simular mentalmente el run con los parametros puestos: con los valores tipicos, ¿el contexto real se parece al del paper? ¿el timeout da margen? ¿el test cubre el caso que motivo el cambio, o solo un caso feliz?
2. Enumerar las limitaciones de la solucion **antes** de entregarla, no despues. La auto-evaluacion debe ser parte del entregable.
3. Distinguir "menor" de "conveniente de no arreglar": si la razon para clasificarlo como menor es que arreglarlo requiere mas trabajo, es deuda real, no severidad baja.
4. Todo parametro que dependa del LLM usado (timeouts, tamaños de contexto, concurrencia) debe ir al `.env`, no a `constants.py`. Si se pone en `constants.py` es porque nunca deberia tocarse (p.ej. `CHARS_PER_TOKEN`).
5. Cualquier cambio que afecte el contexto que ve el LLM debe tener test que estrese el budget, no solo el caso con budget holgado.

**Validacion a ciegas del pipeline completo**: claude_code no ejecuta en el entorno con NIM+MinIO (ver deuda #9 para el contexto infra; TESTS.md F23 para el e2e actual mockeado). Implicaciones operativas:
- Cambios que afectan al flujo retrieval→synthesis→generation solo se validan cuando el usuario lanza el run.
- Antes de declarar un cambio como completo, claude_code debe enumerar los **criterios observables en el run**: que variable de `config_snapshot._runtime` esperar, que valor en `kg_synthesis_stats` o `judge_fallback_stats`, que rango en las metricas agregadas.
- "Falta test e2e de X" NO es deuda pendiente que claude_code deba resolver — es consecuencia estructural del entorno. Listarla como deuda induce iteraciones futuras intentando construir algo imposible de validar en sesion.

Esta seccion se actualiza con nuevas manifestaciones del patron segun se detecten. No borrar sin consenso.

## Proximos pasos

### Orden de prioridades

```
Pre-P0 (completitud arquitectural + ejecucion estable)  <-- GATE activo
  |
P0 (replicacion empirica del paper)                     <-- contingente a Pre-P0
  |
  +-- P1 (sanity on/off synthesis sobre HotpotQA)       <-- barato, en paralelo a P0
  |
  +-- P2 (experimento 3: catalogo especializado)        <-- SOLO si P0 pasa
  |
  +-- P3 (embedibilidad + export KG + integracion)      <-- SOLO si P2 pasa
```

### Pre-P0 — Completitud arquitectural de LIGHT_RAG · **gate activo**

Antes de cualquier medicion comparativa (P0), la implementacion de LIGHT_RAG debe cumplir tres condiciones verificables:

1. **Arquitectonicamente completa frente al paper**: la tabla de "Divergencias con el paper original" no tiene ningun item marcado como abierto. Nada puede estar parcialmente implementado (p.ej. un canal del path high-level sin su contrapartida de indexacion; una capa de synthesis presente pero inestable; un VDB sin los campos que el paper usa para scoring).
2. **Funcionando en ejecucion**: el pipeline produce al menos un run end-to-end donde los stats de observabilidad reflejan que la arquitectura descrita corre efectivamente, sin degradaciones sistematicas a paths de fallback. Criterios concretos: `kg_synthesis_stats.fallback_rate < 0.10`, `judge_fallback_stats.default_return_rate` bajo el umbral configurado, `retrieval_metadata` por query poblada (no todas en `kg_fallback=null` por falta de datos KG, y no todas en fallback al vector search).
3. **Funcionalidades extra alineadas con el entorno**: cualquier adicion propia (cache de KG, fallbacks de LLM, capping defensivo) debe estar documentada como adaptacion operativa, no como sustituto de una pieza del paper.

**Criterio de paso al gate P0**: las tres condiciones anteriores cumplidas simultaneamente sobre un mismo run (puede ser el smoke-test actual sobre HotpotQA 800 docs; el objetivo es la salud de la implementacion, no la calidad del resultado).

**Criterio de fallo**: cualquier condicion incumplida bloquea el avance a P0. Lanzar P0 sin Pre-P0 cerrado mide una variante degradada de LIGHT_RAG — el resultado no es informativo ni para replicar ni para invalidar el paper.

### Resultado F.5 (referencia historica)

F.5 se ejecuto dos veces sobre HotpotQA (125q, DEV_MODE, seed=42, reranker ON).

**F.5 pre-refactor** (divergencias #4+5/#6/#7 abiertas, 4000 docs):

| Metrica | SIMPLE_VECTOR | LIGHT_RAG | Delta |
|---|---|---|---|
| Hit@5 | 1.000 | 1.000 | 0 |
| MRR | 0.992 | 0.992 | 0 |
| Recall@5 | 0.968 | 0.968 | 0 |
| Recall@20 | 0.988 | 0.988 | 0 |
| Avg gen score | 0.7764 | 0.7877 | +0.0113 |
| Tiempo total | 194.7s | 9002.1s | ×46 |

Todas las metricas de retrieval identicas query por query. El KG aporto ~49 docs exclusivos por query, pero el reranker colapso el ranking final al mismo top-20 que SIMPLE_VECTOR. KG indexado correctamente (23K entidades, 55K relaciones, 32K co-occurrence edges), pero las divergencias #4+5 y #6 impedian que su senal llegara al LLM. Las 10 diferencias en generacion son no-determinismo del LLM con mismo contexto.

**F.5 post-refactor** (#4+5/#6/#7 resueltos, 2500 docs):

| Metrica | SIMPLE_VECTOR | LIGHT_RAG hybrid | Delta |
|---|---|---|---|
| Hit@5 / MRR | 1.000 / 1.000 | 1.000 / 1.000 | 0 (saturado) |
| Avg gen score | 0.8038 | 0.8157 | +0.0119 |
| Tiempo | 144s | 4589s | ×31.8 |

Delta pre → post se movio 0.6 decimas de porcentaje — dentro del ruido del LLM judge. **HotpotQA no discrimina** en generacion: Wikipedia en el pre-entrenamiento del embedding + DEV_MODE saturando gold docs + ventana 192K chars → el embedding resuelve por si solo sin necesitar el KG. Los fixes estan correctamente implementados (el KG se construye, las secciones llegan al LLM con budgets proporcionales, el reranker no colapsa el ranking); simplemente este dataset no es util para validar la arquitectura por el lado de la generacion.

**Nota estructural sobre metricas de retrieval identicas en F.5**: las metricas identicas en F.5 eran consecuencia directa de la divergencia #8 (ahora resuelta). Con #8 opcion A, los chunks de LIGHT_RAG vienen del KG via `source_doc_ids` — las metricas de retrieval pueden diverger en cualquier dataset. **F.5 debe re-ejecutarse post-fix para verificar.**

### P0 — Replicacion empirica del paper · **bloqueado por Pre-P0**

**Objetivo**: demostrar que sobre al menos un benchmark donde el paper reporta `LIGHT_RAG > baseline vector`, nuestra implementacion reproduce la **direccion** del delta (magnitudes exactas son secundarias; el signo y su significancia sobre el ruido es lo que importa).

**Estado**: bloqueado hasta cerrar Pre-P0 (completitud arquitectural + ejecucion estable). Lanzar P0 antes mediria una variante degradada de LIGHT_RAG.

**Prerequisitos arquitectonicos**: divergencia #8 cerrada (opcion A; chunks via `source_doc_ids` del KG). **Pendiente #10** (keywords high-level por chunk durante indexacion) — bloquea Pre-P0.

**Candidatos de dataset**:
- UltraDomain subsets (Legal-QA, Agriculture, CS, Mix) — los del paper original
- Cualquier QA especializado con domain shift real al embedding

Ninguno esta en formato MTEB/BeIR nativo; todos requieren ETL propio al contrato MinIO/Parquet de `loader.py`.

**Trabajo necesario (varias sesiones adicionales de claude_code)**:
1. **Seleccion de benchmark** publico donde exista contra-referencia publicada (paper u otra fuente revisada) a favor de LightRAG/GraphRAG. Sin contra-referencia no hay "replicacion" que validar.
2. **ETL al contrato Parquet**: mapear queries, corpus, qrels; extender `DATASET_CONFIG` en `shared/types.py`.
3. **Protocolo experimental**: seed fijo, N>=3 runs por estrategia (mitiga deuda #3), reranker segun config del paper, metricas alineadas con lo que reporta el paper.
4. **Comparativa SIMPLE_VECTOR vs LIGHT_RAG hybrid** con synthesis on. Opcionalmente ablacion off para aislar la aportacion de la synthesis.
5. **Analisis**: validar `judge_fallback_stats` y `kg_synthesis_stats` antes de interpretar deltas (ver "Observabilidad de runs"). Si alguno degrada, los resultados no son interpretables.

**Criterio de exito**: delta `LIGHT_RAG > SIMPLE_VECTOR` en la metrica principal del benchmark, distinguible del ruido (seed×LLM), con signo consistente con el paper.

**Criterio de fallo**: deltas dentro del ruido o invertidos → debug (¿synthesis llega al generador? ¿KG se construye? ¿indexacion falla silenciosamente?), no avance a P2/P3.

### P1 — Sanity de synthesis sobre HotpotQA · barato, paralelo a P0

Dos controles gratuitos que no sustituyen P0 (por la razon de arriba):

1. **Ablacion synthesis on/off** (F.7): repetir F.5 con `KG_SYNTHESIS_ENABLED=true` y `false`. Objetivo: detectar regresion introducida por la capa de synthesis antes de invertir en P0.
2. **Full corpus sin DEV_MODE** (66K docs): senal intermedia sobre robustez del KG cuando el retrieval deja de saturar.

Coste: F.7 ~2-3h, full corpus ~4-6h.

### P2 — Experimento 3: catalogo especializado · **futuro, contingente a P0**

**No empezar sin P0 verde.** Sobre un catalogo privado de 10-50 PDFs especializados (upstream del sistema administrador), LIGHT_RAG vs SIMPLE_VECTOR se evalua en dos ejes:

- **Eje 1 — Precision/calidad**: se espera delta LIGHT_RAG > SIMPLE_VECTOR >3-5pp en gen score, >5-10pp en Recall@K, porque el embedding no ha visto el dominio y el KG se construye del propio corpus.
- **Eje 2 — Resistencia a alucinacion**: el KG aporta grounding explicito que deberia reducir la tasa de fabricacion cuando el retrieval devuelve chunks poco relevantes. Se mide via `faithfulness` (LLM-judge); la fiabilidad depende de que `judge_fallback_stats` reporte tasas sanas (ver "Observabilidad de runs").

**Criterio de decision**: validar cualquiera de los dos ejes (idealmente ambos) → LIGHT_RAG pasa a estrategia default. Ninguno → reconsiderar el rol de LIGHT_RAG en el producto.

**Bloqueado por**: P0 verde + disponibilidad del catalogo upstream.

### P3 — Embedibilidad + export de KG + integracion · **futuro lejano, contingente a P2**

**No empezar sin P0 y P2 verdes.** Trabajo de producto, no de investigacion:

- **Embedibilidad**: configuracion via dict inyectado (no solo `.env` global), corpus en memoria, separar "cargar/indexar" de "evaluar" para reusar indices entre runs, sin asunciones sobre el filesystem excepto `EVALUATION_RESULTS_DIR` explicito.
- **Export de KG a MinIO/Parquet**: serializador `KnowledgeGraph` → Parquet (nodos + aristas + pesos + metadatos de co-ocurrencia) + VDBs. Schema a acordar con el administrador. Hoy el KG es efimero (igraph + ChromaDB en memoria, descartado en `_cleanup()`). Sin export, multi-tenant y versionado son imposibles.
- **Contract testing con el administrador**: validar el schema Parquet upstream contra `DATASET_CONFIG` desde preflight (cierra deuda #2).

### Limitaciones conocidas (no accionables)

- Sesgo LLM-judge en faithfulness para respuestas cortas (inherente)
- No-determinismo HNSW ±0.02 (ChromaDB no expone `hnsw:random_seed`; deuda #3)
- Lock-in a NVIDIA NIM (deuda #9) — solo reproducible con acceso a NIM
