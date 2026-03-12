# Plan de trabajo — Integridad de benchmarking LIGHT_RAG

## Contexto

El run etiquetado como LIGHT_RAG ejecutó SimpleVector+Reranker.
Causa raíz: `networkx` no estaba instalado → `HAS_NETWORKX=False` → fallback silencioso.
El evaluator no detectó la discrepancia. Los resultados comparativos son inválidos.

**Ya resuelto en commits anteriores:**
- `networkx` añadido a `requirements.txt`
- `fetch_k = max(fetch_k, retrieval_k)` (DTm-35, fix del 15 vs 20 doc_ids)
- Prompt de generación HotpotQA endurecido (problemas 3+4, verbosidad/yes pattern)
- DTm-36, DTm-37, DTm-38 documentados en tabla de deuda técnica

---

## Fase 1 — Guardrail de estrategia (DTm-38, prioridad Alta)

Objetivo: hacer imposible que un run se etiquete como LIGHT_RAG sin ejecutar LIGHT_RAG.

### 1.1 — `lightrag_retriever.py`: reportar estrategia real

**Líneas 257-260 y 282-285** — Actualmente `strategy_used = LIGHT_RAG` incondicional.

```python
# ANTES (ambos métodos retrieve y retrieve_by_vector):
result.strategy_used = RetrievalStrategy.LIGHT_RAG

# DESPUÉS:
if self._has_graph:
    result.strategy_used = RetrievalStrategy.LIGHT_RAG
else:
    result.strategy_used = RetrievalStrategy.SIMPLE_VECTOR
```

### 1.2 — `evaluator.py:_execute_retrieval()`: validar estrategia post-retrieval

Después de obtener el resultado del retriever (~línea 919-928), añadir validación:

```python
# Tras obtener result (con o sin reranker):
actual_strategy = result.strategy_used  # o full_result.strategy_used en rama reranker
configured_strategy = self.config.retrieval.strategy
if actual_strategy.name != configured_strategy.name:
    logger.error(
        f"STRATEGY MISMATCH: configurado={configured_strategy.name}, "
        f"ejecutado={actual_strategy.name}. Resultados no representan "
        f"la estrategia configurada."
    )
```

### 1.3 — `evaluator.py:_build_run()`: registrar `strategy_actual`

En `config_snapshot` (~línea 1197), añadir campo que refleje la estrategia real ejecutada.
Determinarla a partir de los `strategy_used` de las queries individuales (si todas coinciden,
es esa; si hay mezcla, registrar "MIXED").

### 1.4 — Añadir `graph_active` al metadata del retriever

En `lightrag_retriever.py`, incluir `metadata["graph_active"] = self._has_graph` en todos
los `RetrievalResult` devueltos. Esto permite diagnóstico post-hoc sin depender solo del
strategy_used.

---

## Fase 2 — Robustez del evaluator (DTm-35 residual + DTm-32)

### 2.1 — Logging de fetch_k efectivo

En `_execute_retrieval()`, loggear el `fetch_k` real después del `max()`:

```python
logger.debug(f"  fetch_k={fetch_k} (retrieval_k={retrieval_k}, reranker.top_n={self.config.reranker.top_n})")
```

### 2.2 — Fix `random.seed()` global (DTm-32)

En `evaluator.py:run()`, reemplazar `random.seed(seed)` por instancia aislada
`self._rng = random.Random(seed)` y usar `self._rng` en todas las operaciones
(subset selection, corpus shuffle). Evita contaminar RNG de terceros.

---

## Fase 3 — Tests de los cambios

### 3.1 — Test unitario: strategy_used refleja fallback

Test que crea `LightRAGRetriever` con `HAS_NETWORKX=False`, ejecuta retrieve,
y verifica que `result.strategy_used == SIMPLE_VECTOR`.

### 3.2 — Test unitario: fetch_k >= retrieval_k

Test que configura `reranker.top_n=5` y `retrieval_k=20`, verifica que
el retrieval devuelve >= 20 doc_ids en la rama con reranker.

### 3.3 — Test unitario: strategy mismatch logging

Test que verifica que el evaluator loggea ERROR cuando estrategia configurada
≠ ejecutada (mock del retriever devolviendo SIMPLE_VECTOR con config LIGHT_RAG).

---

## Fuera de scope (deuda técnica de prioridad Baja, sin impacto en integridad)

Los siguientes items quedan documentados pero no se abordan en este plan:

| ID | Razón de exclusión |
|----|--------------------|
| DTm-12 | Sesgo faithfulness — informativo, sin impacto en métricas primarias |
| DTm-13 | No-determinismo HNSW — variación ±0.02, aceptable |
| DTm-14, DTm-31 | Duplicación memoria — optimización, no correctness |
| DTm-15 | ETL answer_type — sin impacto numérico |
| DTm-18 | Entity normalization — mejora de recall, no bug |
| DTm-20 | question_type propagación — conveniencia |
| DTm-24 | Naming ambiguo — claridad, no funcional |
| DTm-25 | Batch size vs semáforo — eficiencia |
| DTm-26 | KG entity cap silencioso — visibilidad |
| DTm-27 | Filtro name < 2 — edge case |
| DTm-28 | Dependencies no pinneadas — reproducibilidad (importante pero ortogonal) |
| DTm-36 | God Object evaluator — refactor estructural, requiere plan propio |
| DTm-37 | Tantivy sanitization — recall BM25, fix aislado |

---

## Orden de ejecución

```
Fase 1 (1.1 → 1.2 → 1.3 → 1.4)  ← prioridad máxima, fixes de correctness
Fase 2 (2.1 → 2.2)                ← robustez, bajo riesgo
Fase 3 (3.1 → 3.2 → 3.3)         ← validación de los cambios
```

Estimación: Fases 1-3 son cambios quirúrgicos en 3 ficheros (`lightrag_retriever.py`,
`evaluator.py`, `tests/`). No requieren refactor estructural.
