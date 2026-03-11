# Problemas no documentados en README

## P-01 — BFS con `list.pop(0)` en lugar de `deque.popleft()`

**Archivo:** `shared/retrieval/knowledge_graph.py` → `query_entities()`

```python
queue: List[Tuple[str, int]] = [(norm, 0)]
current, depth = queue.pop(0)  # O(n) por operacion
```

`list.pop(0)` desplaza todos los elementos en cada llamada: O(n) por operacion, O(n²) en total para el traversal. Con grafos de hasta 50K entidades esto es el cuello de botella mas directo de LIGHT_RAG en retrieval. La correccion es trivial: sustituir por `collections.deque` con `popleft()`.

---

## P-02 — `query_by_keywords` sin indice: O(entidades × keywords) por query

**Archivo:** `shared/retrieval/knowledge_graph.py` → `query_by_keywords()`

El metodo recorre la totalidad de `self._entities` y la totalidad de las aristas del grafo en cada llamada. Con el cap por defecto de 50K entidades y corpus completo (~66K docs), cada query LIGHT_RAG ejecuta un scan lineal completo. No existe indice invertido sobre nombres de entidad ni sobre descripciones de relaciones. El impacto se acumula multiplicado por el numero de queries evaluadas.

---

## P-03 — Corpus duplicado en memoria (HYBRID_PLUS)

**Archivos:** `shared/retrieval/hybrid_retriever.py` (`_doc_map`) y `shared/retrieval/hybrid_plus_retriever.py` (`_original_contents`)

`HybridPlusRetriever` contiene un `HybridRetriever` interno. Ambos mantienen una copia independiente del contenido de todos los documentos indexados. El corpus completo de HotpotQA (~66K pasajes Wikipedia) queda duplicado en memoria. Esto no esta registrado como deuda tecnica aunque es del mismo orden de magnitud que DTm-14 (ya documentado para `retrieved_contents` + `generation_contents`).

---

## P-04 — `random.seed()` global en el flujo estandar

**Archivo:** `sandbox_mteb/evaluator.py` → `run()`

```python
random.seed(corpus_seed)
random.shuffle(corpus_ids)
```

Muta el estado global del modulo `random`. El DEV_MODE lo resuelve correctamente con `random.Random(seed)` (instancia aislada), pero el flujo estandar no. Cualquier componente del proceso que use `random` despues de este punto —incluyendo librerias de terceros— recibe un estado RNG contaminado. La reproducibilidad declarada del shuffle no es completa.

---

## P-05 — Fallos de extraccion de tripletas por documento no visibles en resultados

**Archivo:** `shared/retrieval/triplet_extractor.py` → `extract_from_doc_async()`

```python
except Exception as e:
    logger.warning(f"Error extrayendo tripletas de {doc_id}: {e}")
    return [], []
```

Un fallo LLM en la extraccion de un documento produce `([], [])` silenciosamente. `get_stats()` reporta el total de entidades y relaciones pero no cuantos documentos fallaron la extraccion. En un corpus de 66K docs con timeouts o errores intermitentes de NIM, es posible que una fraccion significativa del corpus quede sin representacion en el KG sin que el `config_snapshot` del `EvaluationRun` lo refleje.

---

## P-06 — Ausencia de persistencia del Knowledge Graph entre runs

**Archivos:** `shared/retrieval/lightrag_retriever.py`, `shared/retrieval/knowledge_graph.py`

El KG se construye desde cero en cada ejecucion via ~N llamadas LLM (una por documento). Para el corpus completo de HotpotQA (66K docs, concurrencia=32, ~2s/llamada) la estimacion del propio codigo es ~69 minutos solo de indexacion. No existe mecanismo de serialización ni cache del grafo construido. Iterar sobre parametros de fusion (`kg_graph_weight`, `kg_vector_weight`, `kg_max_hops`) requiere repetir la indexacion completa cada vez, lo que hace inviable cualquier busqueda de hiperparametros sobre LIGHT_RAG.

---

## P-07 — `pre_fusion_k` usado con semantica incorrecta cuando el reranker esta activo

**Archivo:** `sandbox_mteb/evaluator.py` → `_execute_retrieval()`

```python
if self._reranker:
    fetch_k = self.config.retrieval.pre_fusion_k
    full_result = _do_retrieve(fetch_k)
```

`RETRIEVAL_PRE_FUSION_K` esta definido como el numero de candidatos que cada canal (BM25, vector) aporta antes del RRF. Se reutiliza aqui como el numero de candidatos a recuperar para el reranker, que es una semantica distinta. El README documenta DTm-24 (naming ambiguo entre `RETRIEVAL_VECTOR_WEIGHT` y `KG_VECTOR_WEIGHT`), pero este problema es operacionalmente mas grave: afecta cuantos documentos ve el reranker, no solo como se nombran las variables.
