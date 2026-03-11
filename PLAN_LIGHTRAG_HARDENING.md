# Plan: Hardening LIGHT_RAG para produccion

## Objetivo

Resolver los riesgos pendientes antes de runs en entorno real (NIM + MinIO).
Cada fase es independiente y aporta valor incremental.

---

## Fase 1: Tests unitarios (DTm-23) — prerequisito

Sin tests no podemos tocar el codigo con confianza. Prioridad maxima.

### 1.1 Tests KnowledgeGraph

| Test | Que valida |
|------|-----------|
| `test_add_triplets_basic` | Entidades y relaciones se anaden al grafo |
| `test_add_triplets_dedup` | Misma entidad desde 2 docs -> 1 nodo, 2 source_doc_ids |
| `test_add_triplets_empty_name` | Entidad con nombre vacio se ignora |
| `test_query_entities_bfs` | BFS devuelve docs conectados con scoring `1/(1+depth)` |
| `test_query_entities_max_hops` | Docs a distancia > max_hops no aparecen |
| `test_query_entities_unknown` | Entidad inexistente -> lista vacia |
| `test_query_by_keywords` | Substring match en nombres de entidad |
| `test_query_by_keywords_empty` | Sin keywords -> lista vacia |
| `test_empty_graph` | Todas las queries sobre grafo vacio -> listas vacias |
| `test_get_stats` | Estadisticas correctas (nodos, aristas, componentes) |

**Archivo:** `tests/test_knowledge_graph.py`
**Dependencias:** Solo `networkx` (ya en requirements).

### 1.2 Tests TripletExtractor

| Test | Que valida |
|------|-----------|
| `test_parse_valid_json` | JSON bien formado -> entidades + relaciones |
| `test_parse_json_with_markdown` | `` ```json ... ``` `` -> strip y parse |
| `test_parse_malformed_json` | JSON roto -> `([], [])`, no exception |
| `test_parse_missing_fields` | JSON sin "relations" -> entities ok, relations=[] |
| `test_parse_invalid_entity_type` | Entidad con type no-enum -> se acepta (no hay constraint) |
| `test_truncation` | Doc > max_text_chars se trunca antes de enviar al LLM |
| `test_extract_from_doc_llm_error` | LLM lanza exception -> `([], [])` |

**Archivo:** `tests/test_triplet_extractor.py`
**Dependencias:** Mock de AsyncLLMService.

### 1.3 Tests _fuse_with_graph

| Test | Que valida |
|------|-----------|
| `test_fuse_no_graph_docs` | Sin graph docs -> retorna vector_result sin cambios |
| `test_fuse_graph_only_docs` | Docs solo del grafo se recuperan via lookup |
| `test_fuse_normalization` | Scores se normalizan a [0,1] correctamente |
| `test_fuse_all_scores_zero` | graph_scores todos 0.0 -> no division by zero |
| `test_fuse_weights` | Fusion respeta vector_weight/graph_weight |
| `test_fuse_unresolved_docs` | Docs del grafo sin contenido -> excluidos del resultado |

**Archivo:** `tests/test_lightrag_fusion.py`
**Dependencias:** Mocks de KnowledgeGraph y SimpleVectorRetriever.

---

## Fase 2: Batching de coroutines (DTm-22)

### Problema

`extract_batch_async()` crea 66K coroutines de golpe:
```python
tasks = [_extract_one(doc) for doc in documents]  # 66K objetos
await asyncio.gather(*tasks)
```

El semaforo en `AsyncLLMService` solo limita HTTP (32 concurrentes),
pero los 66K coroutines ya existen en memoria con sus doc references.

### Solucion

Partir en chunks y procesar secuencialmente por chunk:

```python
_EXTRACTION_BATCH_SIZE = 500  # coroutines por batch

for start in range(0, len(documents), _EXTRACTION_BATCH_SIZE):
    chunk = documents[start:start + _EXTRACTION_BATCH_SIZE]
    tasks = [_extract_one(doc) for doc in chunk]
    await asyncio.gather(*tasks, return_exceptions=True)
```

**Archivos:** `shared/retrieval/triplet_extractor.py`
**Impacto:** Solo cambia `extract_batch_async()`. API publica sin cambios.

---

## Fase 3: Cap de memoria del KG (DTm-21)

### Problema

Con 66K docs * ~5 tripletas/doc = ~330K entidades+relaciones.
NetworkX + 4 indices invertidos crecen sin limite.
Relaciones duplicadas se acumulan en listas de aristas.

### Solucion

Tres cambios aditivos:

**3.1 Dedup de relaciones en aristas**

En `add_triplets()`, antes de append a `relations` list,
verificar si la misma relacion (source, target, relation_type) ya existe:

```python
existing = edge_data.get("relations", [])
if not any(r.source_doc_id == rel.source_doc_id
           and r.relation == rel.relation for r in existing):
    existing.append(rel)
```

**3.2 Config `max_entities` en RetrievalConfig**

Nuevo parametro `kg_max_entities: int = 50000` (default conservador).
En `add_triplets()`, si `len(self._entities) >= max_entities`, skip
entidades nuevas y log warning. Entidades existentes se actualizan.

**3.3 Logging de memoria**

En `get_stats()`, anadir estimacion de memoria:
```python
import sys
approx_bytes = sys.getsizeof(self._graph) + ...
```

Log al final de indexacion para visibilidad.

**Archivos:** `shared/retrieval/knowledge_graph.py`, `shared/retrieval/core.py`

---

## Fase 4: Validacion de output LLM (DTm-16)

### Problema

`_parse_extraction_json()` acepta cualquier estructura sin validar:
- Entity type puede ser cualquier string (no se valida contra enum)
- No hay limite de longitud en descriptions
- Entidades con nombre vacio pasan (se filtran mas tarde en KG, pero no aqui)

### Solucion

Validacion ligera post-parse (sin pydantic, sin dependencia nueva):

```python
VALID_ENTITY_TYPES = {"PERSON", "ORG", "PLACE", "CONCEPT", "EVENT", "OTHER"}
MAX_DESCRIPTION_CHARS = 200

def _validate_entity(e: Dict) -> Optional[Dict]:
    name = e.get("name", "").strip()
    if not name or len(name) < 2:
        return None
    etype = e.get("type", "OTHER").upper()
    if etype not in VALID_ENTITY_TYPES:
        etype = "OTHER"
    desc = e.get("description", "")[:MAX_DESCRIPTION_CHARS]
    return {"name": name, "type": etype, "description": desc}
```

**Archivos:** `shared/retrieval/triplet_extractor.py`
**Impacto:** Filtra ruido sin cambiar API. Log de entidades rechazadas.

---

## Orden de ejecucion

```
Fase 1 (tests)         -> Confianza para tocar codigo
Fase 2 (batching)      -> Safe para corpus grande
Fase 3 (cap memoria)   -> Safe para RAM limitada
Fase 4 (validacion)    -> Calidad del KG
```

Cada fase se puede commitear y testear independientemente.
Tras fase 1+2, el sistema es safe para un run con DEV_MODE=true.
Tras fase 3+4, es safe para corpus completo (66K).

---

## Fuera de scope (no bloquean runs)

| Item | Razon |
|------|-------|
| DTm-12 (sesgo LLM-judge) | Faithfulness es informativa, no primaria |
| DTm-13 (HNSW non-determinism) | Variacion +/-0.02, aceptable |
| DTm-14 (duplicacion RAM) | Solo con 7K queries, no afecta a DEV_MODE |
| DTm-15 (answer_type) | Sin impacto numerico |
| DTm-18 (entity aliases) | Mejora de calidad, no bloquea |
| DTm-20 (question_type CSV) | Cosmetic |
| DTm-24 (naming ambiguo) | Cosmetic |
