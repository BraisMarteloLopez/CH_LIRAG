# Plan de implementacion: Fase F — Grafo como primary retriever

## Contexto

El objetivo es alinear CH_LIRAG con la arquitectura del original HKUDS/LightRAG,
donde el knowledge graph es el mecanismo PRIMARIO de retrieval (no un suplemento
de vector search).

**Flujo actual (CH_LIRAG):**
```
Query → vector search sobre chunks (primary, 70%)
      → graph traversal → doc_ids (supplement, 30%)
      → RRF fusion → top_k docs
      → format_context: "[Doc 1]\n<texto plano>..."
```

**Flujo objetivo (alineado con original):**
```
Query → keyword extraction (low-level + high-level)
      → [local] entity VDB → KG nodes → edges → source_doc_ids
      → [global] relationship VDB → KG edges → entities → source_doc_ids
      → chunk selection (WEIGHT o VECTOR) sobre doc_ids del grafo
      → structured context: JSON {entities, relations, chunks}
      → [optional] vector fallback si grafo no produce suficientes resultados
```

## Archivos afectados

| Archivo | Cambios |
|---------|---------|
| `shared/retrieval/lightrag/retriever.py` | Nuevo `_retrieve_via_graph()`, refactor `retrieve()`, nuevo `_select_chunks_from_graph()` |
| `shared/retrieval/lightrag/knowledge_graph.py` | Nuevo `get_entities_for_docs()` para obtener entidades asociadas a los docs seleccionados |
| `shared/retrieval/core.py` | Nuevo campo `lightrag_mode` en `RetrievalConfig`, nuevo campo `kg_context` en `RetrievalResult` |
| `sandbox_mteb/retrieval_executor.py` | Nuevo `format_structured_context()` (DAM-8) |
| `sandbox_mteb/config.py` | Leer `LIGHTRAG_MODE` de env |
| `sandbox_mteb/env.example` | Documentar `LIGHTRAG_MODE` |
| `tests/test_lightrag_fusion.py` | Tests para el nuevo flujo |
| `tests/test_pipeline_e2e.py` | E2E con LIGHT_RAG |

## Tareas detalladas

---

### F.1 — Chunk selection desde grafo (DTm-76)

**Objetivo:** Dado un conjunto de entidades/relaciones encontradas por el grafo,
recopilar los `source_doc_ids` y seleccionar los chunks mas relevantes.

**Archivo:** `shared/retrieval/lightrag/retriever.py`

**Nuevo metodo `_select_chunks_from_graph()`:**

```python
def _select_chunks_from_graph(
    self,
    entity_results: List[Tuple[str, float]],     # (doc_id, score) de query_entities
    relationship_results: List[Tuple[str, float]], # (doc_id, score) de relationship VDB
    query: str,
    top_k: int,
) -> Tuple[List[str], List[float]]:
    """Selecciona y rankea chunks desde los doc_ids del grafo.

    Combina doc_ids de ambos canales (entity + relationship), deduplica,
    y aplica weighted polling basado en frecuencia de aparicion.

    Returns:
        (doc_ids, scores) ordenados por relevancia
    """
```

**Logica:**
1. Acumular scores por doc_id desde entity_results y relationship_results
2. Deduplicar: si un doc_id aparece en ambos canales, sumar scores
3. Ordenar por score acumulado descendente
4. Devolver top_k doc_ids con scores

**Razon:** En el original, `_find_related_text_unit_from_entities()` hace
weighted polling usando frecuencia de chunk. Nuestro equivalente es mas simple
porque CH_LIRAG no chunka documentos — cada `source_doc_id` ES un documento
completo. No necesitamos weighted polling sobre sub-chunks.

**Tests:** 3-4 tests en `test_lightrag_fusion.py`:
- `test_select_chunks_combines_entity_and_relationship_sources`
- `test_select_chunks_deduplicates_across_channels`
- `test_select_chunks_respects_top_k`
- `test_select_chunks_empty_inputs`

---

### F.2 — Modo graph_primary en retrieve() (DAM-3)

**Objetivo:** Nuevo flujo de retrieval donde el grafo es primario y vector search
es fallback.

**Archivo:** `shared/retrieval/lightrag/retriever.py`

**Cambio 1: Nuevo campo en `__init__()`:**
```python
self._lightrag_mode = config.lightrag_mode  # "hybrid" (default), "local", "global", "graph_primary", "naive"
```

**Cambio 2: Refactor de `retrieve()`:**

```python
def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
    k = top_k or self.config.retrieval_k
    start_time = time.perf_counter()

    if self._lightrag_mode == "naive" or not self._has_graph:
        # Modo naive: solo vector search (equivale a SIMPLE_VECTOR)
        result = self._vector_retriever.retrieve(query, top_k=k)
    elif self._lightrag_mode == "graph_primary":
        # Modo graph_primary: grafo traza chunks, vector como fallback
        result = self._retrieve_via_graph(query, k)
    else:
        # Modos local/global/hybrid: flujo actual (vector + graph fusion)
        vector_result = self._vector_retriever.retrieve(query, top_k=k)
        result = self._fuse_with_graph(query, vector_result, k)

    result.retrieval_time_ms = (time.perf_counter() - start_time) * 1000
    result.strategy_used = (
        RetrievalStrategy.LIGHT_RAG if self._has_graph
        else RetrievalStrategy.SIMPLE_VECTOR
    )
    result.metadata["graph_active"] = self._has_graph
    result.metadata["lightrag_mode"] = self._lightrag_mode
    return result
```

**Cambio 3: Nuevo metodo `_retrieve_via_graph()`:**

```python
def _retrieve_via_graph(self, query: str, top_k: int) -> RetrievalResult:
    """Retrieval con grafo como mecanismo primario (DAM-3).

    Flujo:
    1. Extraer keywords (low-level + high-level)
    2. Entity VDB → query_entities() → doc_ids (low-level)
    3. Relationship VDB → doc_ids (high-level)
    4. _select_chunks_from_graph() → top_k doc_ids
    5. Recuperar contenido de ChromaDB
    6. Si insuficientes resultados, complementar con vector search
    """
    low_level, high_level = self._get_query_keywords(query)

    if not low_level and not high_level:
        # Sin keywords → fallback a vector
        return self._vector_retriever.retrieve(query, top_k=top_k)

    # Low-level: entidades
    entity_results = []
    if low_level:
        pre_resolved = None
        if self._entities_vdb is not None:
            pre_resolved = self._resolve_entities_via_vdb(low_level)
        entity_results = self._kg.query_entities(
            low_level,
            max_hops=self._kg_max_hops,
            max_docs=top_k * self._GRAPH_OVERFETCH_FACTOR,
            pre_resolved=pre_resolved,
        )

    # High-level: relaciones
    relationship_results = []
    if high_level:
        if self._relationships_vdb is not None:
            relationship_results = self._resolve_relationships_via_vdb(high_level)
        else:
            relationship_results = self._kg.query_by_keywords(
                high_level, max_docs=top_k * self._GRAPH_OVERFETCH_FACTOR,
            )

    # Seleccionar chunks desde el grafo
    graph_doc_ids, graph_scores = self._select_chunks_from_graph(
        entity_results, relationship_results, query, top_k,
    )

    # Recuperar contenido desde ChromaDB
    contents_map = self._vector_retriever.get_documents_by_ids(graph_doc_ids)

    final_ids = []
    final_contents = []
    final_scores = []
    for doc_id, score in zip(graph_doc_ids, graph_scores):
        if doc_id in contents_map:
            final_ids.append(doc_id)
            final_contents.append(contents_map[doc_id])
            final_scores.append(score)

    # Fallback: si el grafo produjo menos de top_k/2, complementar con vector
    if len(final_ids) < top_k // 2:
        vector_result = self._vector_retriever.retrieve(query, top_k=top_k)
        seen = set(final_ids)
        for doc_id, content, score in zip(
            vector_result.doc_ids, vector_result.contents, vector_result.scores
        ):
            if doc_id not in seen:
                final_ids.append(doc_id)
                final_contents.append(content)
                final_scores.append(score * 0.5)  # Penalizar fallback
                seen.add(doc_id)
            if len(final_ids) >= top_k:
                break

    return RetrievalResult(
        doc_ids=final_ids[:top_k],
        contents=final_contents[:top_k],
        scores=final_scores[:top_k],
        vector_scores=[0.0] * min(len(final_ids), top_k),
        metadata={
            "graph_primary": True,
            "graph_docs": len(graph_doc_ids),
            "graph_resolved": len([d for d in graph_doc_ids if d in contents_map]),
            "vector_fallback_used": len(final_ids) > len(graph_doc_ids),
            "query_keywords": {"low": low_level, "high": high_level},
        },
    )
```

**Tests:** 5-6 tests:
- `test_retrieve_graph_primary_uses_graph_as_main_source`
- `test_retrieve_graph_primary_falls_back_to_vector_when_insufficient`
- `test_retrieve_graph_primary_no_keywords_falls_back`
- `test_retrieve_naive_mode_skips_graph`
- `test_retrieve_hybrid_mode_preserves_current_behavior`

---

### F.3 — Contexto estructurado para generacion (DAM-8)

**Objetivo:** Pasar al LLM un contexto JSON con 3 secciones (entidades, relaciones,
chunks) en lugar de texto plano concatenado.

**Archivo principal:** `sandbox_mteb/retrieval_executor.py`

**Cambio 1: Nuevo `kg_context` en RetrievalResult:**

En `shared/retrieval/core.py`, anadir campo opcional:
```python
@dataclass
class RetrievalResult:
    # ... campos existentes ...
    kg_context: Optional[Dict[str, Any]] = None  # DAM-8: entidades/relaciones para contexto estructurado
```

**Cambio 2: Poblar `kg_context` en `_retrieve_via_graph()`:**

Al final de `_retrieve_via_graph()`, antes de retornar, recopilar las entidades y
relaciones que participaron en el retrieval:

```python
# Recopilar entidades y relaciones para contexto estructurado
if self._kg is not None:
    kg_entities = []
    for doc_id in final_ids:
        for entity in self._kg.get_entities_for_doc(doc_id):
            kg_entities.append({
                "entity": entity.name,
                "type": entity.entity_type,
                "description": entity.description,
            })
    # Dedup por nombre
    seen_entities = set()
    unique_entities = []
    for e in kg_entities:
        if e["entity"] not in seen_entities:
            unique_entities.append(e)
            seen_entities.add(e["entity"])

    kg_relations = []
    for rel in self._kg.get_relations_for_docs(final_ids):
        kg_relations.append({
            "source": rel["source"],
            "target": rel["target"],
            "relation": rel["relation"],
            "description": rel.get("description", ""),
        })

    result.kg_context = {
        "entities": unique_entities[:30],   # Top 30 entidades
        "relations": kg_relations[:30],     # Top 30 relaciones
    }
```

**Cambio 3: Nuevos metodos en knowledge_graph.py:**

```python
def get_entities_for_doc(self, doc_id: str) -> List[KGEntity]:
    """Retorna entidades asociadas a un doc_id."""
    entity_names = self._doc_to_entities.get(doc_id, set())
    return [self._entities[name] for name in entity_names if name in self._entities]

def get_relations_for_docs(self, doc_ids: List[str]) -> List[Dict]:
    """Retorna relaciones cuyos source_doc_id esta en doc_ids."""
    doc_set = set(doc_ids)
    results = []
    for doc_id in doc_ids:
        for rel in self._doc_to_relations.get(doc_id, []):
            results.append(rel)
    return results
```

**Cambio 4: Nuevo `format_structured_context()` en retrieval_executor.py:**

```python
def format_structured_context(
    contents: List[str],
    kg_context: Optional[Dict[str, Any]],
    max_length: int,
) -> str:
    """Formatea contexto con secciones KG (DAM-8)."""
    if kg_context is None:
        # Fallback a formato plano para SIMPLE_VECTOR / HYBRID_PLUS
        return format_context(contents, max_length)

    import json
    parts = []

    # Seccion 1: Entidades
    entities = kg_context.get("entities", [])
    if entities:
        entities_str = "\n".join(
            json.dumps(e, ensure_ascii=False) for e in entities
        )
        parts.append(f"Knowledge Graph Data (Entity):\n\n```json\n{entities_str}\n```")

    # Seccion 2: Relaciones
    relations = kg_context.get("relations", [])
    if relations:
        relations_str = "\n".join(
            json.dumps(r, ensure_ascii=False) for r in relations
        )
        parts.append(f"Knowledge Graph Data (Relationship):\n\n```json\n{relations_str}\n```")

    # Seccion 3: Document Chunks (con budget de tokens restante)
    header_len = sum(len(p) for p in parts) + 100  # buffer
    chunk_budget = max_length - header_len
    if chunk_budget > 0:
        chunks_parts = []
        for i, content in enumerate(contents, 1):
            chunk_json = json.dumps({"reference_id": i, "content": content}, ensure_ascii=False)
            if sum(len(c) for c in chunks_parts) + len(chunk_json) > chunk_budget:
                break
            chunks_parts.append(chunk_json)
        if chunks_parts:
            chunks_str = "\n".join(chunks_parts)
            parts.append(f"Document Chunks:\n\n```json\n{chunks_str}\n```")

    return "\n\n".join(parts)
```

**Cambio 5: Integrar en generation_executor.py:**

En la linea donde se llama `format_context()`, condicionar:
```python
if retrieval_detail.kg_context is not None:
    context = format_structured_context(
        retrieval_detail.get_generation_contents(),
        retrieval_detail.kg_context,
        self._max_context_chars,
    )
else:
    context = format_context(
        retrieval_detail.get_generation_contents(),
        self._max_context_chars,
    )
```

**Tests:** 3-4 tests:
- `test_format_structured_context_with_entities_and_relations`
- `test_format_structured_context_without_kg_falls_back`
- `test_format_structured_context_respects_max_length`

---

### F.4 — Modos de query explicitos (DTm-79)

**Archivo:** `shared/retrieval/core.py`

**Cambio: Nuevo campo en RetrievalConfig:**
```python
lightrag_mode: str = "hybrid"  # "local", "global", "hybrid", "graph_primary", "naive"
```

**Archivo:** `shared/retrieval/core.py` en `from_env()`:
```python
lightrag_mode=_env("LIGHTRAG_MODE", "hybrid"),
```

**Archivo:** `sandbox_mteb/env.example`:
```
# Modo de retrieval LightRAG:
#   hybrid        — entity VDB + relationship VDB + vector, fusion RRF (default actual)
#   graph_primary — grafo traza source docs, vector como fallback (DAM-3)
#   local         — solo entity VDB + BFS (low-level)
#   global        — solo relationship VDB (high-level)
#   naive         — solo vector search, sin KG (= SIMPLE_VECTOR)
LIGHTRAG_MODE=hybrid
```

**Archivo:** `shared/retrieval/lightrag/retriever.py` en `_fuse_with_graph()`:

Condicionar que canales se activan segun el modo:
```python
# Dentro de _fuse_with_graph, despues de keyword extraction:
use_local = self._lightrag_mode in ("local", "hybrid")
use_global = self._lightrag_mode in ("global", "hybrid")

if use_local and low_level:
    # ... entity VDB + query_entities (codigo existente)

if use_global and high_level:
    # ... relationship VDB (codigo existente)
```

**Tests:** 3 tests:
- `test_local_mode_only_uses_entity_vdb`
- `test_global_mode_only_uses_relationship_vdb`
- `test_hybrid_mode_uses_both`

---

### F.5 — Evaluacion comparativa

**No requiere codigo.** Ejecutar 3 runs con la misma config:
1. `LIGHTRAG_MODE=naive` (baseline, equivale a SIMPLE_VECTOR)
2. `LIGHTRAG_MODE=hybrid` (flujo actual)
3. `LIGHTRAG_MODE=graph_primary` (nuevo flujo DAM-3)

Comparar MRR, Hit@5, Recall@5. Decidir default.

---

## Orden de implementacion

```
F.4 (modos) ← mas simple, desbloquea testing granular
    │
    ▼
F.1 (chunk selection) ← logica nueva, core del cambio
    │
    ▼
F.2 (graph_primary) ← usa F.1, nuevo flujo completo
    │
    ▼
F.3 (contexto estructurado) ← mejora generacion, independiente
    │
    ▼
F.5 (evaluacion) ← runs comparativos
```

## Invariantes que se preservan

1. **RetrievalResult interface no cambia** (solo se anade campo opcional `kg_context`)
2. **`LIGHTRAG_MODE=hybrid` es el default** → comportamiento actual intacto
3. **Fallback a SIMPLE_VECTOR sigue funcionando** si no hay LLM/igraph
4. **Tests existentes no se rompen** — el modo `hybrid` ejecuta el flujo actual
5. **format_context() sigue existiendo** para SIMPLE_VECTOR y HYBRID_PLUS

## Riesgos

| Riesgo | Mitigacion |
|--------|-----------|
| graph_primary produce peores resultados que hybrid | F.5 compara ambos; default sigue siendo hybrid |
| KG fragmentado (DTm-73) limita cobertura | Fallback a vector cuando graph < top_k/2 |
| Contexto estructurado confunde al LLM | A/B test: structured vs plain en F.5 |
| get_documents_by_ids lento para muchos IDs | Batch lookup ya implementado en ChromaVectorStore |
