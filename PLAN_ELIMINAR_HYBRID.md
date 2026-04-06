# Plan: Eliminar estrategia HYBRID_PLUS

## Justificacion

- LIGHT_RAG cubre el mismo caso de uso (multi-hop bridging) con mayor profundidad semantica
- SIMPLE_VECTOR cubre el caso "barato y determinista"
- HYBRID_PLUS no existe en el original HKUDS/LightRAG
- Elimina dependencias: spaCy, tantivy, rank-bm25
- Reduce 1,680 LOC + 3 dependencias pesadas

## Archivos a ELIMINAR (completos)

| Archivo | LOC | Tipo |
|---------|-----|------|
| `shared/retrieval/hybrid/entity_linker.py` | 414 | Implementacion |
| `shared/retrieval/hybrid/plus_retriever.py` | 311 | Implementacion |
| `shared/retrieval/hybrid/retriever.py` | 356 | Implementacion |
| `shared/retrieval/hybrid/tantivy_index.py` | ~100 | Implementacion |
| `shared/retrieval/hybrid/__init__.py` | 23 | Exports |
| `tests/test_entity_linker.py` | 386 | Tests |
| `tests/test_hybrid_plus_retriever.py` | 527 | Tests |
| **Total** | **~2,117** | |

Tambien eliminar el directorio `shared/retrieval/hybrid/` completo.

## Archivos a MODIFICAR

### 1. `shared/retrieval/core.py`

- Eliminar `HYBRID_PLUS = auto()` del enum `RetrievalStrategy` (linea 31)
- Eliminar campos de config:
  - `rrf_bm25_weight` (linea 49)
  - `pre_fusion_k` (linea 50)
  - `rrf_k` (linea 51) — NOTA: kg_rrf_k existe aparte, este es el de HYBRID
  - `bm25_language` (linea 55)
  - `entity_max_cross_refs` (linea 63)
  - `entity_min_shared` (linea 64)
  - `entity_max_doc_fraction` (linea 65)
- Eliminar correspondientes env reads en `from_env()`
- Eliminar `bm25_scores` de `RetrievalResult` (linea 130)
- Actualizar comentarios que referencian HYBRID_PLUS

### 2. `shared/retrieval/__init__.py`

- Eliminar imports de `hybrid` subpackage (linea 22)
- Eliminar caso `HYBRID_PLUS` en `get_retriever()` (lineas 57-70)
- Eliminar exports en `__all__` (lineas 92-93)

### 3. `sandbox_mteb/env.example`

- Eliminar seccion completa de RRF weights (lineas ~47-57)
- Eliminar seccion de entity cross-linking (lineas ~148-165)
- Eliminar variables: RRF_BM25_WEIGHT, RRF_VECTOR_WEIGHT, RETRIEVAL_PRE_FUSION_K,
  RETRIEVAL_RRF_K, RETRIEVAL_BM25_LANGUAGE, ENTITY_MAX_CROSS_REFS,
  ENTITY_MIN_SHARED, ENTITY_MAX_DOC_FRACTION

### 4. `sandbox_mteb/preflight.py`

- Eliminar checks de spacy y tantivy (lineas 53-54)
- Actualizar mensajes de dependencias opcionales

### 5. `sandbox_mteb/config.py`

- Eliminar validacion de HYBRID_PLUS (si existe)
- Actualizar comentarios

### 6. `requirements.txt`

- Eliminar: `spacy>=3.7.0,<4.0`
- Eliminar: `tantivy>=0.22.0`
- Eliminar: `rank-bm25>=0.2.2,<0.3`
- Eliminar comentarios de NER/BM25

### 7. `mypy.ini`

- Eliminar secciones: `[mypy-tantivy.*]`, `[mypy-rank_bm25.*]`
- Considerar eliminar `[mypy-spacy.*]` si no se usa en otro lugar

### 8. `README.md`

- Eliminar descripcion de HYBRID_PLUS en tabla de estrategias
- Actualizar "3 estrategias" → "2 estrategias"
- Eliminar instrucciones de instalacion de spaCy
- Eliminar configuracion de entity cross-linking
- Actualizar DTm-81 (import fantasma) como "Resuelto (eliminado)"
- Eliminar DTm-24 (naming ambiguo RRF weights) — ya no aplica

### 9. `tests/conftest.py`

- Verificar si hay mocks de spacy/tantivy que se puedan limpiar

## Orden de ejecucion

```
1. Eliminar archivos completos (hybrid/, tests)
2. Modificar core.py (enum, config, RetrievalResult)
3. Modificar __init__.py (factory, imports)
4. Modificar env.example, preflight.py, config.py
5. Modificar requirements.txt, mypy.ini
6. Actualizar README.md
7. Ejecutar tests → verificar 0 fallos
8. Commit y push
```

## Invariantes a preservar

1. `SIMPLE_VECTOR` y `LIGHT_RAG` siguen funcionando identico
2. `RetrievalResult` mantiene doc_ids, contents, scores, vector_scores, metadata
3. `RetrievalConfig.from_env()` sigue funcionando (sin campos eliminados)
4. `get_retriever()` solo acepta SIMPLE_VECTOR y LIGHT_RAG
5. Tests existentes de SIMPLE_VECTOR y LIGHT_RAG no se rompen

## Riesgo

**Bajo.** La eliminacion es aditiva-inversa: solo se quita codigo, no se modifica
logica de estrategias existentes. Si alguien tiene `.env` con `RETRIEVAL_STRATEGY=HYBRID_PLUS`,
obtendra un KeyError claro al arrancar.
