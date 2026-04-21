# Auditoría R7/R9/R10/R11 — reporte

> Reporte de solo lectura. No se modificó código durante esta auditoría.
> Rama: `claude/read-documentation-QDpby` · HEAD: `27e81d2e` · Baseline tests: `499 passed, 6 skipped`.

## Resumen ejecutivo

| Regla | Violaciones claras | Borderline / incierto |
|---|---|---|
| R7 | 16 | 10 |
| R9 | 3 | 0 |
| R10 | ~12 candidatos | 1 (from_dict) |
| R11 | 3 | 0 |
| Apéndice R1 (`Dict[str, Any]` sin TypedDict) | ~5 sitios | — |

Alcance barrido: `sandbox_mteb/**/*.py`, `shared/**/*.py`, `tests/**/*.py` (excluyendo `tests/integration/`).
Patrones R7 aplicados en inglés y español.

---

## R7 — Comentarios que traducen literalmente el código siguiente

Criterio: el comentario traducido a código produce exactamente la siguiente línea; no aporta WHY, contexto histórico, referencia cruzada ni etiqueta un bloque multi-línea.

### Violaciones claras (16)

| # | Fichero:línea | Comentario | Línea siguiente |
|---|---|---|---|
| 1 | `sandbox_mteb/evaluator.py:610` | `# Update evaluated IDs and save checkpoint` | `for qr in chunk_results: evaluated_ids.add(...)` + `save_checkpoint(...)` |
| 2 | `sandbox_mteb/evaluator.py:517` | `# Filtrar queries pendientes` | `pending_queries = [q for q in queries if q.query_id not in evaluated_ids]` |
| 3 | `sandbox_mteb/loader.py:135` | `# Guardar en cache` | `self._save_to_cache(result)` |
| 4 | `sandbox_mteb/generation_executor.py:453` | `# Convertir secondary a dict` | `secondary_dict = {}` + `for sr in secondary_results: secondary_dict[sr.metric_type.value] = sr.value` |
| 5 | `sandbox_mteb/preflight.py:193` | `# Check bucket exists` | `client.head_bucket(Bucket=config.storage.minio_bucket)` |
| 6 | `sandbox_mteb/preflight.py:195` | `# Check dataset prefix exists` | `prefix = f"{...}/"` + `client.list_objects_v2(...)` |
| 7 | `sandbox_mteb/preflight.py:272` | `# Count pinned packages` | `pinned = [l for l in content.splitlines() if ... "==" in l]` |
| 8 | `shared/vector_store.py:281` | `# Eliminar coleccion via cliente nativo` | `self._client.delete_collection(self.collection_name)` (el docstring previo ya dice "Ahora se usa el cliente nativo para delete + recreate" → también R9 ligero) |
| 9 | `shared/retrieval/reranker.py:106` | `# Mapear vector_scores originales por doc_id` | `orig_vector_scores: Dict[str, float] = {}` + for loop que llena el dict |
| 10 | `shared/retrieval/lightrag/retriever.py:265` | `# Anadir tripletas` | `added = self._kg.add_triplets(doc_id, relations)` |
| 11 | `shared/retrieval/lightrag/retriever.py:269` | `# Actualizar metadata de entidades` | `for entity in entities: self._kg.add_entity_metadata(...)` |
| 12 | `shared/retrieval/lightrag/retriever.py:388` | `# Formatear descripciones como lista numerada` | `desc_list = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(...))` |
| 13 | `shared/retrieval/lightrag/knowledge_graph.py:431` | `# Eliminar todas las aristas conectadas a este nodo` | `edge_ids = self._graph.incident(vid)` + `self._graph.delete_edges(edge_ids)` |
| 14 | `shared/retrieval/lightrag/knowledge_graph.py:519` | `# Anadir nodos y arista al grafo` | `src_vid = self._ensure_node(...)` + `tgt_vid = self._ensure_node(...)` |
| 15 | `shared/retrieval/lightrag/triplet_extractor.py:523` | `# Registrar vacios` | `for d in empty: ... self._stats["docs_empty_input"] += 1` |
| 16 | `shared/retrieval/lightrag/triplet_extractor.py:560` | `# Update stats for each doc that came back` | `for doc in non_empty: ... self._stats[...] += 1` |

### Borderline / section-header (10 — requiere criterio del usuario)

Comentarios que literalmente traducen el código pero funcionan como marcador de bloque o fase multi-línea. Criterio R7 estricto los flaggea; criterio laxo los acepta como ayuda de navegación. Depende de si el usuario quiere suprimirlos todos o mantener marcadores de fase.

| # | Fichero:línea | Comentario | Observación |
|---|---|---|---|
| 1 | `sandbox_mteb/evaluator.py:128,131,134,137,150` | `# 1. Inicializar componentes`, `# 2. Cargar y preparar datos`, `# 3. Indexar documentos`, `# 3b. Crear retrieval executor`, `# 5. Construir EvaluationRun` | Marcadores de fase en `run()`. Cada uno precede una sola línea que los traduce literalmente. La línea `# 6. Validar tasa de fallback del judge...` SÍ aporta WHY ("Se hace DESPUES de construir el run para que las stats queden persistidas...") y no es R7 |
| 2 | `sandbox_mteb/evaluator.py:617` | `# Log summary` | Etiqueta el bloque de logger.error/warning (10+ líneas) |
| 3 | `sandbox_mteb/evaluator.py:716` | `# BUILD RUN (delegado a result_builder.py)` | Más sección header que traducción; "delegado a result_builder.py" aporta contexto externo |
| 4 | `sandbox_mteb/preflight.py:82` | `# Verificar que la estrategia es LIGHT_RAG` | Precede `if strategy != RetrievalStrategy.LIGHT_RAG:` |
| 5 | `sandbox_mteb/result_builder.py:96` | `# Agregar metricas de retrieval` | Etiqueta un bloque de ~30 líneas de agregación |
| 6 | `shared/retrieval/lightrag/retriever.py:262` | `# Construir grafo` | Etiqueta for-loop de 15 líneas que construye el grafo |
| 7 | `shared/retrieval/lightrag/knowledge_graph.py:428` | `# Remover aristas del grafo` | Etiqueta bloque 428-434 |
| 8 | `shared/retrieval/lightrag/knowledge_graph.py:443` | `# Remover relaciones asociadas del indice _doc_to_relations` | Etiqueta bloque de filtrado; menciona índice concreto |
| 9 | `shared/retrieval/lightrag/knowledge_graph.py:625` | `# Serialize graph as edge list with attributes` | Etiqueta bloque de ~10 líneas de serialización |
| 10 | `shared/retrieval/lightrag/triplet_extractor.py:517` | `# Filtrar docs vacios antes del batch` | Precede las dos list comprehensions (empty/non_empty); "antes del batch" es contexto menor |

**Recomendación del auditor**: mantener los marcadores de fase numerada en orquestadores (`evaluator.py run()`, `retriever.py _retrieve_via_kg`) si se dejan como únicos marcadores (sin duplicar invariantes en docstring — ver R9). Suprimir los demás borderline.

### Falsos positivos descartados tras revisión (muestra)

- `sandbox_mteb/evaluator.py:117-125` — los tres `reset_*` triples tienen un bloque de WHY compartido (lineas 117-119) que aplica al grupo; la línea 121 usa "equivalente" para referirse al bloque, y 123-124 aporta "contadores de degradaciones silenciosas en 7 puntos". No R7.
- `sandbox_mteb/embedding_service.py:156` — `# Ordenar por index (la API puede devolver desordenado)` tiene WHY explícito.
- `shared/retrieval/reranker.py:93-94` — `# Ordenar explicitamente por relevance_score descendente: compress_documents() no garantiza orden formalmente en su contrato.` WHY claro.
- `shared/retrieval/lightrag/knowledge_graph.py:453` — `# Remover de _entities (libera el slot para el cap)` tiene WHY parentético.
- `shared/retrieval/lightrag/retriever.py:275` — `# Divergencia #10: persistir chunk keywords en el KG (si hay)` referencia cruzada + condicional.
- `shared/retrieval/lightrag/retriever.py:1171` — `# Filtrar queries ya cacheadas (acceso al cache bajo lock).` WHY parentético.
- `shared/metrics.py:83` — `# Asegurar que el valor esta en rango valido` aporta semántica del rango [0,1].
- `shared/operational_tracker.py:85-87, 93-94` — bloques con WHY explícito (tolerancia mypy, intencionalidad de construcción explícita).
- `shared/vector_store.py:113` — `# Crear cliente Chroma nativo para control directo de colecciones` tiene WHY ("para control directo").
- `shared/retrieval/lightrag/triplet_extractor.py:621, 301, 779` — todos con WHY parentético o contexto de dedup específico.
- `shared/retrieval/lightrag/knowledge_graph.py:586, 838` — `# Acumular en vez de sobrescribir`, `# Dedup por contenido (case-insensitive, stripped)` aportan estrategia no visible en código.

---

## R9 — Invariante duplicado en docstring + inline adyacente

Criterio: la misma razón aparece en docstring y en comentario inline dentro del body. Regla: si aplica a toda la función, va al docstring; si aplica a un bloque concreto, va inline y el docstring no la repite.

### Violaciones claras (3)

#### 1. `shared/retrieval/lightrag/retriever.py:847` — `_retrieve_via_kg`

Docstring enumera 6 pasos del "Flujo":
```
1. Extraer query keywords (low-level + high-level)
2. Resolver keywords contra Entity VDB, Relationship VDB y Chunk Keywords VDB
3. Obtener source_doc_ids de entidades/relaciones + doc_ids directos del canal de chunk keywords
4. Scoring: cada doc_id acumula score de los tres canales
5. Fetch contenido de chunks desde el vector store
6. Fallback a vector search si el KG no produce resultados
```

Inline (mismo método) usa 4 marcadores `--- Paso N ---`:
- Línea 911 `# --- Paso 1: Resolver entidades y relaciones via VDBs ---` → mapea a pasos 2-3 del docstring
- Línea 963 `# --- Paso 2: Scoring sobre source_doc_ids (order × similarity × weight) ---` → mapea a paso 4
- Línea 1002 `# --- Paso 3: Fallback si el KG no produjo doc_ids ---` → mapea a paso 6
- Línea 1024 `# --- Paso 4: Fetch contenido desde el vector store ---` → mapea a paso 5

**Problema doble**: además de la duplicación, la numeración está **mis-alineada** (inline `Paso 3` = docstring paso 6, inline `Paso 4` = docstring paso 5). Un lector que use la numeración del docstring como mapa del cuerpo se pierde.

**Recomendación**: dejar la enumeración solo en el docstring (es el punto de mayor precedencia para un flujo completo), eliminar los 4 marcadores `--- Paso N ---` inline. Los bloques siguen siendo navegables porque el código tiene structure natural (if-branches, loops).

#### 2. `shared/retrieval/lightrag/retriever.py:154` — `index_documents`

Docstring enumera 3 pasos:
```
1. Indexa en ChromaDB (contenido original, embeddings limpios)
2. Extrae tripletas via LLM (batch async)
3. Construye knowledge graph
```

Inline (mismo método):
- Línea 174 `# Paso 1: Vector index (siempre)`
- Línea 182 `# Paso 2: Knowledge graph (si disponible)`

Los 2 marcadores inline añaden condicionalidad en paréntesis ("siempre" / "si disponible") que el docstring no tiene. Esa condicionalidad podría moverse al docstring para eliminar la duplicación, o los marcadores pueden quedarse como únicas referencias (eliminando la enumeración del docstring).

**Recomendación**: mover la condicionalidad al docstring (`1. Indexa ChromaDB (siempre) / 2. Extrae tripletas via LLM y construye KG (si disponible)`) y suprimir los marcadores inline.

#### 3. `shared/metrics.py:703` — `_extract_score_fallback_with_status`

Docstring enumera 3 patrones de fallback:
```
1. Fraccion N/M (mas especifico)
2. Decimal 0.X o 1.0 (rango 0-1 directo)
3. Entero 1-10 con prefijo "score:" (normalizado a 0-1)
```

Inline (mismo método):
- Línea 721 `# 1. Fracciones como 8/10`
- Línea 734 `# 2. Decimal en rango 0-1`
- Línea 746 `# 3. Entero 1-10 con prefijo "score:"`

Duplicación clara; los 3 marcadores inline traducen el docstring casi palabra por palabra.

**Recomendación**: suprimir los 3 marcadores inline; el docstring ya documenta el orden de intento.

### Criterios de parada aplicados

- Se ignoraron casos donde el docstring es genérico ("Initialize X.") y el comentario inline es específico de bloque.
- Se ignoraron casos donde el comentario inline describe una rama/condición (ej. `# Legacy mode: one LLM call per doc` en `triplet_extractor.py:627`) que el docstring no menciona.

---

## R10 — Funciones no triviales sin ejemplo ejecutable

Criterio: regex, parsing, transformación con edge cases o dispatch. Formato aceptado: doctest `>>>` o prosa `Input: X → Output: Y` (confirmado por el usuario).

### Candidatos (12)

| # | Fichero:símbolo | Disparador | Ejemplo sugerido (prosa, orientativo) |
|---|---|---|---|
| 1 | `shared/citation_parser.py::parse_citation_refs` | Regex + edge cases (vacío, malformed, out-of-range) | `Input: "foo [ref:2] bar [ref:99]", n_valid_chunks=5 → Output: valid=2, in_range=1, out_of_range=1, malformed=0` |
| 2 | `shared/metrics.py::_extract_score_fallback_with_status` | 3 regex patterns con fallback | `Input: "score: 8/10" → (0.8, False)`, `Input: "I think 0.7" → (0.7, False)`, `Input: "great answer" → (0.5, True)` |
| 3 | `shared/metrics.py::_parse_judge_response` | JSON parsing con fallback | `Input: '{"score": 0.8}' → (0.8, False, None)`, `Input: 'foo score: 8/10' → (0.8, False, None)`, `Input: 'no score' → (0.5, True, "no_parse")` |
| 4 | `shared/metrics.py::normalize_text` | Regex + transformación con edge cases (articles, puntuación, whitespace) | `Input: "The Apple, Inc." → "apple inc"` |
| 5 | `shared/retrieval/lightrag/knowledge_graph.py::_normalize_name` | Regex + lowercase + article removal | `Input: "The United States" → "united states"`, `Input: "  Foo!! " → "foo"` |
| 6 | `shared/retrieval/lightrag/triplet_extractor.py::_parse_keywords_json` | JSON parsing con fallback ante malformed LLM output | `Input: '{"low_level": ["a"], "high_level": ["b"]}' → (["a"], ["b"])`, `Input: "garbage" → ([], [])` |
| 7 | `shared/retrieval/lightrag/triplet_extractor.py::_parse_extraction_json` | JSON parsing de tripletas LLM | `Input: '{"entities": [...], "relations": [...]}' → (entities, relations, keywords)` |
| 8 | `shared/retrieval/lightrag/triplet_extractor.py::_parse_batch_extraction_json` | JSON parsing batch multi-doc | `Input: '{"doc1": {...}, "doc2": {...}}' → {"doc1": (ents, rels, kws), "doc2": (...)}` |
| 9 | `sandbox_mteb/retrieval_executor.py::format_context` | Transformación con truncation edge cases | `Input: contents=["a"*10_000, "b"*10_000], max_chars=15_000 → contexto truncado con [N chunks omitted]` |
| 10 | `sandbox_mteb/retrieval_executor.py::_build_kg_section` | Formato condicional según mode/estructura | `Input: mode="hybrid", entities=[...], relations=[...] → "=== ENTIDADES ===\n..."` |
| 11 | `shared/types.py::parse_answer_type` | Dispatch sobre enum | `Input: "yes/no" → AnswerType.YES_NO`, `Input: "entity" → AnswerType.ENTITY` |
| 12 | `shared/config_base.py::_parse_embedding_model_type` + `shared/retrieval/core.py::_parse_lightrag_mode` | Validación con ValueError | `Input: "hybrid" → LightRAGMode.HYBRID`, `Input: "xxx" → raises ValueError("Modo invalido...")` |

### Casos incierto/borderline (1)

- **`shared/retrieval/lightrag/knowledge_graph.py::from_dict`** `[incierto: recomendar al usuario]`. Es deserialización v3 pero el shape de entrada/salida es demasiado grande para un ejemplo en prosa breve (~5 campos top-level + listas anidadas). El docstring ya describe bien el schema. Alternativa al doctest: referencia a `tests/test_knowledge_graph.py` que ejercita `to_dict` + `from_dict` round-trip. **Decisión del usuario**: ¿exigir ejemplo o aceptar referencia a test?

### Falsos positivos descartados

- Getters/setters triviales de `shared/types.py` (ej. `get_dataset_config`): dispatch trivial sobre nombre de dataset, los 2 casos existen y el `KeyError` es obvio.
- `shared/metrics.py::f1_score`, `exact_match`, `accuracy`: el algoritmo es estándar y bien conocido. El docstring ya menciona cómo se calcula. Añadir ejemplo sería ruido. (Si el usuario prefiere, son candidatos de "valor añadido bajo".)
- Resolvers en `generation_executor.py` (`_resolve_primary_*`): cada uno es 1-2 líneas de composición; no hay edge cases ocultos.

---

## R11 — Docstring duplica shape expresado por el tipo de retorno

Criterio: si la función retorna un `TypedDict` o `@dataclass`, el docstring no debe re-enumerar los campos. Puede decir "Ver `FooResult`" y, si hay semántica que un lector del tipo no deduciría, moverla al docstring del propio `TypedDict`.

### Violaciones claras (3)

#### 1. `shared/citation_parser.py::parse_citation_refs` (línea 51)

Firma: `-> CitationStats`. El docstring del `TypedDict` `CitationStats` ya existe (`shared/citation_parser.py:25-39`) y documenta bien los 7 campos más la invariante `valid + malformed` disjuntos y `in_range + out_of_range == valid`.

Pero el docstring de la función re-enumera todos los campos:
```
Returns:
    Dict con las 7 metricas documentadas:
      - total: valid + malformed
      - valid: matches del formato estricto `[ref:N]`
      - malformed: candidatos reconocibles fuera de formato estricto
      - in_range: valid con N en [1, n_valid_chunks]
      - out_of_range: valid con N fuera de rango (senal roja de alucinacion)
      - distinct: unique N en in_range
      - coverage_ratio: distinct / in_range (diversidad de fuentes citadas)
```

**Semántica a trasladar al `TypedDict`**: `"out_of_range ... senal roja de alucinacion"` es interpretación que pertenece al tipo (ya mencionada en CLAUDE.md pero no en el docstring de `CitationStats`). `"coverage_ratio ... diversidad de fuentes citadas"` idem.

**Acción propuesta**: reemplazar el bloque `Returns:` por `Returns: ver CitationStats.`, y añadir las dos líneas semánticas al docstring del `TypedDict`.

#### 2. `shared/metrics.py::get_judge_fallback_stats` (línea 449)

Firma: `-> Dict[str, JudgeMetricStats]`. El docstring enumera los 5 campos del `JudgeMetricStats`:
```
  - invocations: llamadas totales al judge
  - parse_failures: respuestas no parseables como JSON
  - default_returns: casos donde se devolvio 0.5 por defecto
  - parse_failure_rate, default_return_rate: ratios contra invocations
```

**Semántica a trasladar al `TypedDict`**: el bloque `"default_return_rate es la senal clave — valores altos indican que el judge esta fallando..."` es interpretación que pertenece a `JudgeMetricStats` (igual que en CitationStats).

**Acción propuesta**: reemplazar por `Returns: dict metric_name -> JudgeMetricStats. Ver JudgeMetricStats para semántica.`, y mover la nota interpretativa al docstring del `TypedDict`.

#### 3. `sandbox_mteb/generation_executor.py::get_kg_synthesis_stats` (línea 227)

Firma: `-> KGSynthesisStats`. El docstring enumera las claves:
```
Claves devueltas:
  - invocations: queries donde se intento la synthesis
  - successes, errors, empty_returns, truncations, timeouts
  - fallback_rate: (errors + empty_returns + timeouts) / invocations
  - timing per-categoria (total/queue/llm): p50/p95/max_*_ms + n_*_samples
```

La fórmula de `fallback_rate` y la descomposición del timing per-categoría SÍ son semántica que pertenece al `TypedDict` (no se deducen de `fallback_rate: float`).

**Acción propuesta**: reemplazar por `Snapshot del tracker de KG synthesis. Ver KGSynthesisStats.` y añadir la fórmula de `fallback_rate` + descomposición del timing al docstring del `TypedDict`.

### Falsos positivos descartados

- `shared/operational_tracker.py::get_operational_stats` / `snapshot`: docstring corto ("Retorna contadores operacionales."), no enumera campos. OK.
- `sandbox_mteb/checkpoint.py::serialize_query_result` / `deserialize_query_result`: docstring corto, no enumera campos del `TypedDict`. OK.
- `shared/retrieval/lightrag/knowledge_graph.py::get_stats`: docstring corto. OK.

---

## Apéndice · Violaciones R1 (Dict[str, Any] sin TypedDict)

Detectadas de paso; **no son parte del alcance R7/R9/R10/R11** pero el briefing pide reportarlas. Son candidatas a ampliación de PR-4 (TypedDicts cross-module, `7350438`).

| Fichero:símbolo | Retorno actual | Tipo sugerido |
|---|---|---|
| `shared/retrieval/lightrag/knowledge_graph.py::to_dict` | `Dict[str, Any]` | `KGSerialized` (ya existe) — trivial de cerrar |
| `shared/metrics.py::MetricResult.to_dict` | `Dict[str, Any]` | `MetricResultDict` (nuevo) |
| `shared/types.py` — varios `to_dict` en dataclasses (`QueryEvaluationResult`, `GenerationResult`, `QueryRetrievalDetail`, `EvaluationRun`) | `Dict[str, Any]` | `TypedDict` per dataclass |

`KnowledgeGraph.to_dict` es el más inmediato: `KGSerialized` ya está definido en el mismo fichero y solo hay que cambiar la anotación de la firma. Cerrarlo elimina también una asimetría con `from_dict` (que sí consume un shape estructurado).

**Recomendación**: incluir en un PR separado o como sub-tarea de una futura ampliación de PR-4, **no** mezclar con el trabajo de R7/R9/R10/R11.

---

## Decisión propuesta sobre el siguiente paso

Considerando:
- **R11**: 3 violaciones claras pero concentradas. Trabajo mínimo: 3 funciones + añadir semántica a 3 `TypedDict`. Unidad coherente.
- **R9**: 3 violaciones claras, también concentradas y de alcance pequeño (suprimir enumeraciones duplicadas o marcadores inline).
- **R7**: 16 violaciones claras + 10 borderline. Volumen moderado, todas suprimir-un-comentario; sin riesgo semántico si el criterio está acordado.
- **R10**: 12 candidatos que requieren redactar prosa de ejemplo per-función. Es el trabajo más lento. Ninguno bloquea P0.
- **Apéndice R1**: ortogonal; PR independiente cuando se decida.

**Recomendación**: **PR dedicado (PR-7)** que cubra R7 + R9 + R11 simultáneamente. Son cambios pequeños y homogéneos (supresión o reubicación de comentarios, ningún cambio de comportamiento), y agruparlos en un PR mantiene el contexto cohesivo para el reviewer.

**R10 como PR separado (PR-8)** o como sub-tarea de PR-6 (descomposición de funciones largas): la redacción de ejemplos de prosa es trabajo de escritura técnica, no hay solapamiento semántico con R7/R9/R11, y PR-6 tocará varias de las funciones identificadas de todos modos (parsers, dispatch, formatters). Integrar R10 en PR-6 aprovecha que el autor ya tiene la función abierta y en contexto.

**Apéndice R1**: tercer PR o ampliación de PR-4; trivial de abrir cuando haya ancho de banda.

### Riesgo esperado

- PR-7 (R7+R9+R11): bajo. Cero cambios de lógica. Tests no deberían moverse. Revisar que ningún test assert'ea sobre docstring/comment (búsqueda rápida: `assert ... __doc__` / `inspect.getsource`).
- PR-8 (R10): bajo si los ejemplos se redactan como comentarios (prosa); medio si se optan por doctests ejecutables (requiere `pytest --doctest-modules` en CI, ahora mismo no configurado).

---

## Sanity confirmación

- Baseline re-reproducido: `499 passed, 6 skipped, 2 warnings in 6.94s` (con `/root/.local/bin/pytest tests/ -q --ignore=tests/integration`).
- `git status` esperado: limpio salvo `AUDIT_R7_R9_R10_R11_REPORT.md` nuevo.
- `git diff` contra HEAD: solo el reporte.

## Limitaciones del análisis

- **R7 heurística**: el patrón grep-based EN+ES cubre verbos comunes pero puede haber omitido comentarios idiomáticos que no empiezan con verbo imperativo (ej. `# Cleanup del caché`). Revisión manual complementaria recomendada al autor al abrir PR-7.
- **R9 detección**: se priorizó detección por marcadores numéricos explícitos (`# Paso N`, `# N.`) porque son los casos inequívocos. Pueden existir duplicaciones docstring↔inline más sutiles (parafraseo) no detectadas por esta pasada.
- **R10 "no trivial"**: el juicio sobre qué función requiere ejemplo es subjetivo. La lista es conservadora — preferible añadir ejemplo en una función "simple" que omitirlo en una que los reviewers pregunten.
- **R11 ámbito**: solo se cruzó contra los 11 `TypedDict` de PR-4 + los @dataclass más obvios. Puede haber funciones que devuelven diccionarios parciales de un `TypedDict` (proyecciones), que R11 no cubre literalmente — ver si la documentación los distingue.

