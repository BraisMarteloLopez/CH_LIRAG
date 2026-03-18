# Plan de Trabajo: Resolver Hallazgos de la Evaluacion

Basado en `EVALUATION.md`, este plan ataca los 5 hallazgos criticos en orden de prioridad e impacto.

---

## Fase A: Corregir tests fallidos (5 tests)

**Objetivo:** Suite de tests 100% verde (0 failures).

### A1. Corregir 4 tests con `asyncio.get_event_loop()` deprecado

**Fichero:** `tests/test_triplet_extractor.py`

**Tests afectados:**
- `test_truncation` (L182)
- `test_extract_from_doc_llm_error` (L204)
- `test_extract_from_doc_empty_text` (L221)
- `test_extract_from_doc_whitespace_only` (L235)

**Cambio:** Reemplazar `asyncio.get_event_loop().run_until_complete(coro)` por `asyncio.run(coro)` en cada test. Patron identico al ya usado en los tests TE18-TE21 (L355-401) del mismo fichero, que pasan correctamente.

**Ejemplo:**
```python
# Antes (falla en Python 3.11+):
asyncio.get_event_loop().run_until_complete(
    ext.extract_from_doc_async("doc1", long_text)
)

# Despues:
asyncio.run(ext.extract_from_doc_async("doc1", long_text))
```

### A2. Corregir test inconsistente con DTm-27

**Fichero:** `tests/test_triplet_extractor.py`

**Test afectado:** `test_entity_name_too_short_rejected` (L281-287)

**Problema:** DTm-27 cambio `MIN_ENTITY_NAME_LEN` de 2 a 1. El test aun espera que entidades de 1 caracter ("A") sean rechazadas, pero el codigo ahora las acepta.

**Cambio:** Actualizar el test para reflejar el nuevo comportamiento. La entidad "A" (1 char) es valida, pero entidades vacias siguen rechazadas:

```python
def test_entity_name_min_length_accepted():
    """Entity con nombre de 1 char se acepta (DTm-27: MIN_ENTITY_NAME_LEN=1)."""
    ext = _make_extractor()
    raw = '{"entities": [{"name": "A", "type": "PERSON"}, {"name": "Bob", "type": "PERSON"}], "relations": []}'
    entities, _ = ext._parse_extraction_json(raw, "doc1")
    assert len(entities) == 2  # ambas aceptadas
    assert entities[0].name == "A"
    assert entities[1].name == "Bob"
```

Tambien actualizar el docstring del fichero (L15: `TE11. Entity name < 2 chars rechazada` -> `TE11. Entity name de 1 char aceptada (DTm-27)`).

**Verificacion:** `pytest tests/test_triplet_extractor.py -v` → 0 failures.

---

## Fase B: Descomposicion de `evaluator.py` (1225 → <600 LOC)

**Objetivo:** Cumplir el criterio de salida de Fase 4: `evaluator.py < 600 LOC`.

La estrategia es extraer bloques de logica autocontenidos a modulos nuevos dentro de `sandbox_mteb/`, manteniendo `MTEBEvaluator` como orquestador delgado que delega.

### B1. Extraer `sandbox_mteb/subset_selection.py` (~90 LOC)

**Metodos a mover:**
- `_select_subset_dev()` (L413-474, 62 LOC)
- La logica de seleccion de subset estandar que esta inline en `run()` (L243-271, ~29 LOC), encapsulada en una funcion `select_subset_standard()`

**API del nuevo modulo:**
```python
def select_subset_dev(dataset, config) -> Tuple[List[NormalizedQuery], Dict[str, Any]]
def select_subset_standard(dataset, config) -> Tuple[List[NormalizedQuery], Dict[str, Any]]
```

**Impacto en evaluator.py:** `run()` llama a `select_subset_dev()` o `select_subset_standard()` importando del nuevo modulo. Elimina ~90 LOC.

### B2. Extraer `sandbox_mteb/retrieval_executor.py` (~160 LOC)

**Metodos a mover:**
- `_execute_retrieval()` (L897-1019, 123 LOC) con las closures `_do_retrieve` y `_check_strategy`
- `_format_context()` (L1049-1072, 24 LOC)

**API del nuevo modulo:**
```python
class RetrievalExecutor:
    def __init__(self, retriever, reranker, config): ...
    def execute(self, query_text, expected_doc_ids, query_vector=None) -> Tuple[QueryRetrievalDetail, Optional[bool]]
    def format_context(self, contents, max_length) -> str

    @property
    def rerank_failures(self) -> int
    @property
    def strategy_mismatches(self) -> int
```

**Impacto en evaluator.py:** `MTEBEvaluator` instancia `RetrievalExecutor` en `_init_components()`, reemplaza `self._execute_retrieval()` y `self._format_context()`. Elimina ~150 LOC.

### B3. Extraer `sandbox_mteb/generation_executor.py` (~140 LOC)

**Metodos a mover:**
- `_execute_generation_async()` (L1025-1047, 23 LOC)
- `_calculate_metrics_async()` (L1078-1183, 106 LOC)
- `_GenMetricsResult` (L61-75, 15 LOC)
- `_batch_generate_and_evaluate()` (L841-853, 13 LOC)
- `_process_single_async()` (L855-891, 37 LOC)

**API del nuevo modulo:**
```python
class GenMetricsResult: ...

class GenerationExecutor:
    def __init__(self, llm_service, metrics_calculator, config, max_context_chars): ...
    async def batch_generate_and_evaluate(self, queries, retrievals, ds_config, dataset_name) -> list
```

**Impacto en evaluator.py:** Reemplaza 5 metodos. `_evaluate_queries()` llama a `generation_executor.batch_generate_and_evaluate()`. Elimina ~140 LOC.

### B4. Extraer `sandbox_mteb/embedding_service.py` (~100 LOC)

**Metodos a mover:**
- `_batch_embed_queries()` (L539-625, 87 LOC)
- `_query_model_context_window()` (L109-146, 38 LOC)
- `_resolve_max_context_chars()` (L148-187, 40 LOC)

**API del nuevo modulo:**
```python
def batch_embed_queries(query_texts, config) -> List[List[float]]
def resolve_max_context_chars(config) -> int
```

**Impacto en evaluator.py:** Elimina ~140 LOC. (Nota: hay overlap con los 40 LOC de `_resolve_max_context_chars` que tambien llama a `_query_model_context_window`.)

### Resultado esperado

| Componente | LOC estimado |
|---|---|
| `evaluator.py` (orquestador) | ~500 LOC |
| `subset_selection.py` | ~90 LOC |
| `retrieval_executor.py` | ~160 LOC |
| `generation_executor.py` | ~140 LOC |
| `embedding_service.py` | ~100 LOC |
| `checkpoint.py` (ya existe) | ~160 LOC |
| `result_builder.py` (ya existe) | ~180 LOC |

Total: ~500 LOC en evaluator.py (cumple criterio <600).

### Tests

Cada modulo nuevo necesita tests unitarios minimos:
- `tests/test_subset_selection.py`: DEV_MODE con gold garantizado, standard con shuffle
- `tests/test_retrieval_executor.py`: con/sin reranker, strategy mismatch detection
- `tests/test_generation_executor.py`: metricas HYBRID/ADAPTED, error handling
- `tests/test_embedding_service.py`: batch embed retry, context window detection

**Nota:** Los tests existentes (`test_format_context.py`, `test_dt5_pre_rerank_traceability.py`, etc.) importan directamente de `evaluator.py`. Despues de la extraccion, actualizar los imports o mantener re-exports en evaluator.py para backwards compatibility temporal.

---

## Fase C: Completar `requirements.lock`

**Objetivo:** Cumplir criterio de salida de Fase 0.

**Cambio:** Ejecutar `pip freeze > requirements.lock` en el entorno actual para capturar todas las versiones exactas. Esto reemplaza el lock parcial existente (3 paquetes).

**Verificacion:** `diff <(pip freeze | sort) <(sort requirements.lock)` muestra 0 diferencias.

---

## Fase D: Resolver DTm-15 y DTm-20 (deuda tecnica menor)

### D1. DTm-15: `answer_type="label"` para comparison queries

**Estado:** Ya implementado en `loader.py:165-166`. El codigo detecta `question_type == "comparison"` y asigna `answer_type = "label"`. Este issue esta **ya resuelto** — solo falta actualizar el README para marcarlo como hecho.

### D2. DTm-20: Metadata passthrough generico

**Estado:** Ya implementado en `evaluator.py:791-794`. El loop `{k: v for k, v in query.metadata.items() if v}` pasa todo el metadata dict. Este issue esta **ya resuelto** — solo falta actualizar el README.

### D3. DTm-24: Renaming variables de configuracion

**Estado actual:** El README dice "abierto" pero `RRF_VECTOR_WEIGHT` y `KG_VECTOR_WEIGHT` ya existen como nombres separados en la config. Verificar y actualizar README si ya esta resuelto.

**Cambio:** Actualizar tabla de deuda tecnica en README.md para reflejar el estado real de DTm-15, DTm-20, y DTm-24.

---

## Fase E: Actualizar README con estado real de las fases

**Objetivo:** Eliminar la "falsa sensacion de completitud" (hallazgo #5).

**Cambios:**
1. Fase 0: Marcar lock como "parcial" con instruccion de regenerar. Marcar DTm-28 como "parcial".
2. Fase 4: Actualizar estado de la descomposicion (tras completar Fase B) a "Hecho — evaluator.py ~500 LOC".
3. Tabla de deuda tecnica: Marcar DTm-15, DTm-20, DTm-24 como "Resuelto" con referencia al commit.
4. Nota sobre Fase 0 baseline: Sin entorno NIM, el baseline LIGHT_RAG no se puede ejecutar. Documentar como pendiente de infra.

---

## Orden de ejecucion

```
Fase A (tests)       ─── rapido, desbloquea confianza en la suite
    │
Fase B (decompose)   ─── cambio mas grande, requiere tests verdes primero
    │
Fase C (lock)        ─── independiente, puede hacerse en paralelo con B
    │
Fase D+E (docs)      ─── ultima, refleja el estado final
```

**Dependencias:**
- A desbloquea B (no refactorizar con tests rotos)
- B antes de E (el README refleja el estado final post-descomposicion)
- C es independiente
