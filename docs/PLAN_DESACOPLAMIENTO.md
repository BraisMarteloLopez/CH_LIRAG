# Plan de Desacoplamiento — CH_LIRAG

> **Autor**: Análisis automatizado
> **Fecha**: 2026-03-31
> **Estado**: Propuesta
> **Branch**: `claude/evaluate-project-complexity-JqXKc`

---

## 1. El problema en lenguaje llano

Hoy el directorio `shared/retrieval/` es una carpeta plana con **10 archivos Python**
que mezclan tres mundos muy distintos:

- **Búsqueda por vectores** (lo básico, ChromaDB)
- **Búsqueda híbrida** (BM25 + vectores + NER con spaCy)
- **Grafos de conocimiento** (igraph + extracción de tripletas con LLM)

Cualquier desarrollador que abra esa carpeta ve 10 ficheros al mismo nivel
sin saber cuáles van juntos, cuáles son opcionales o cuáles puede ignorar.
Además, las dependencias pesadas (spaCy, igraph, tantivy) se cargan
aunque no se usen, porque todo vive en el mismo espacio.

**No es un problema de corrección** — el código funciona bien y no tiene
dependencias circulares. Es un problema de **claridad y mantenibilidad**
a medida que el proyecto crece.

---

## 2. Qué NO vamos a tocar (y por qué)

Estos módulos son el "esqueleto" del proyecto. Moverlos rompería más de lo
que arreglaría:

| Módulo | LOC | Razón para dejarlo quieto |
|--------|-----|---------------------------|
| `shared/types.py` | 612 | 12 módulos dependen de él. Es el contrato central. |
| `shared/llm.py` | 459 | Singleton del event loop async. Moverlo crearía imports circulares. |
| `shared/config_base.py` | 176 | Parsing de .env sin estado. Pequeño y estable. |
| `shared/retrieval/core.py` | 369 | ABC + enums + `SimpleVectorRetriever`. Es la base de todo. |
| `shared/metrics.py` | 808 | Aunque es grande, solo depende de `types.py`. Ya está limpio. |

**Regla de oro**: si un módulo tiene más dependencias entrantes que salientes,
no se mueve — se deja como ancla y se reorganiza alrededor de él.

---

## 3. Qué SÍ vamos a reorganizar

### Estructura actual (plana)

```
shared/retrieval/
├── __init__.py            (factory)
├── core.py                (ABC + SimpleVector)
├── vector_store.py        (ChromaDB wrapper)
├── hybrid_retriever.py    (BM25 + Vector + RRF)
├── hybrid_plus_retriever.py (↑ + entity linking)
├── entity_linker.py       (spaCy NER)
├── tantivy_index.py       (BM25 backend)
├── lightrag_retriever.py  (Vector + KG)
├── knowledge_graph.py     (igraph graph)
├── triplet_extractor.py   (LLM extraction)
└── reranker.py            (post-procesador)
```

### Estructura propuesta (por estrategia)

```
shared/retrieval/
├── __init__.py              (factory — misma interfaz pública)
├── core.py                  (ABC, enums, RetrievalResult — sin cambios)
├── vector_store.py          (ChromaDB wrapper — sin cambios)
├── reranker.py              (post-procesador — sin cambios)
│
├── hybrid/                  ← NUEVO subpaquete
│   ├── __init__.py          (exporta HybridRetriever, HybridPlusRetriever)
│   ├── retriever.py         (antes hybrid_retriever.py)
│   ├── plus_retriever.py    (antes hybrid_plus_retriever.py)
│   ├── entity_linker.py     (antes entity_linker.py — sin cambios)
│   └── tantivy_index.py     (antes tantivy_index.py — sin cambios)
│
└── lightrag/                ← NUEVO subpaquete
    ├── __init__.py          (exporta LightRAGRetriever)
    ├── retriever.py         (antes lightrag_retriever.py)
    ├── knowledge_graph.py   (antes knowledge_graph.py — sin cambios)
    └── triplet_extractor.py (antes triplet_extractor.py — sin cambios)
```

### Qué cambia para el usuario del código

**Nada.** La factory `get_retriever()` sigue siendo el único punto de entrada.
Los imports públicos en `shared/retrieval/__init__.py` se mantienen idénticos.
Es un cambio 100% interno.

---

## 4. Plan de implementación por fases

### Fase 0 — Preparación (riesgo: nulo)

**Objetivo**: Asegurar que tenemos red de seguridad antes de mover nada.

| Paso | Acción | Verificación |
|------|--------|--------------|
| 0.1 | Ejecutar `pytest` completo y confirmar que pasa | ✅ Suite verde |
| 0.2 | Anotar la lista exacta de imports públicos de `shared/retrieval/__init__.py` | Snapshot de referencia |
| 0.3 | Crear rama de trabajo desde la rama actual | Branch limpio |

**Criterio de éxito**: Suite de tests verde, snapshot de la API pública guardado.

---

### Fase 1 — Subpaquete `hybrid/` (riesgo: bajo)

**Objetivo**: Agrupar los 4 ficheros de búsqueda híbrida.

| Paso | Acción | Ficheros afectados |
|------|--------|--------------------|
| 1.1 | Crear directorio `shared/retrieval/hybrid/` | — |
| 1.2 | Mover `hybrid_retriever.py` → `hybrid/retriever.py` | 1 fichero |
| 1.3 | Mover `hybrid_plus_retriever.py` → `hybrid/plus_retriever.py` | 1 fichero |
| 1.4 | Mover `entity_linker.py` → `hybrid/entity_linker.py` | 1 fichero |
| 1.5 | Mover `tantivy_index.py` → `hybrid/tantivy_index.py` | 1 fichero |
| 1.6 | Crear `hybrid/__init__.py` que re-exporte las clases públicas | 1 fichero nuevo |
| 1.7 | Actualizar imports internos dentro del subpaquete (imports relativos) | 2-3 ficheros |
| 1.8 | Actualizar `shared/retrieval/__init__.py` para importar desde `hybrid/` | 1 fichero |
| 1.9 | Actualizar imports en tests que referencien paths antiguos | ~5 ficheros test |
| 1.10 | Ejecutar `pytest` — debe pasar sin cambios de lógica | ✅ Suite verde |

**LOC movidas**: ~1,363 (hybrid_retriever: 398 + hybrid_plus: 311 + entity_linker: 414 + tantivy: 240)
**Criterio de éxito**: Todos los tests pasan. `get_retriever(HYBRID_PLUS, ...)` funciona igual.

---

### Fase 2 — Subpaquete `lightrag/` (riesgo: bajo)

**Objetivo**: Agrupar los 3 ficheros del grafo de conocimiento.

| Paso | Acción | Ficheros afectados |
|------|--------|--------------------|
| 2.1 | Crear directorio `shared/retrieval/lightrag/` | — |
| 2.2 | Mover `lightrag_retriever.py` → `lightrag/retriever.py` | 1 fichero |
| 2.3 | Mover `knowledge_graph.py` → `lightrag/knowledge_graph.py` | 1 fichero |
| 2.4 | Mover `triplet_extractor.py` → `lightrag/triplet_extractor.py` | 1 fichero |
| 2.5 | Crear `lightrag/__init__.py` que re-exporte `LightRAGRetriever` | 1 fichero nuevo |
| 2.6 | Actualizar imports internos (retriever → knowledge_graph, triplet_extractor) | 1-2 ficheros |
| 2.7 | Actualizar `shared/retrieval/__init__.py` | 1 fichero |
| 2.8 | Actualizar imports en tests | ~4 ficheros test |
| 2.9 | Ejecutar `pytest` | ✅ Suite verde |

**LOC movidas**: ~1,997 (lightrag: 568 + knowledge_graph: 725 + triplet_extractor: 704)
**Criterio de éxito**: Todos los tests pasan. `get_retriever(LIGHT_RAG, ...)` funciona igual.

---

### Fase 3 — Limpieza y validación (riesgo: nulo)

**Objetivo**: Confirmar que la API pública no cambió y limpiar residuos.

| Paso | Acción |
|------|--------|
| 3.1 | Comparar `__all__` de `shared/retrieval/__init__.py` con el snapshot de Fase 0 — debe ser idéntico |
| 3.2 | Ejecutar `mypy` sobre módulos core (si está configurado) |
| 3.3 | Verificar que `HAS_SPACY`, `HAS_IGRAPH`, `HAS_TANTIVY` se siguen exportando correctamente |
| 3.4 | Eliminar ficheros viejos que ya se movieron (git los trackea como rename) |
| 3.5 | Ejecutar suite completa de tests (unitarios + integración) |
| 3.6 | Revisar que `sandbox_mteb/evaluator.py` no necesitó cambios (solo importa via factory) |

**Criterio de éxito**: 0 cambios en la interfaz pública. Tests verdes. mypy limpio.

---

## 5. Lo que ganamos con cada fase

### Después de Fase 1 (hybrid/)

```
Antes:  Un desarrollador abre retrieval/ y ve 10 ficheros.
        "¿entity_linker va con hybrid o con lightrag? ¿tantivy_index es general?"

Después: Abre retrieval/hybrid/ y ve 4 ficheros que claramente van juntos.
         El nombre del directorio ya te dice qué estrategia usan.
```

- **Testeo aislado**: `pytest tests/test_hybrid_* tests/test_entity_linker.py` cubre
  exactamente el subpaquete.
- **Dependencias claras**: spaCy + tantivy son dependencias de `hybrid/`, no del proyecto.

### Después de Fase 2 (lightrag/)

```
Antes:  knowledge_graph.py y triplet_extractor.py flotan sueltos.
        No es obvio que solo los usa lightrag_retriever.py.

Después: Los 3 están en lightrag/ — la relación es explícita.
```

- **Testeo aislado**: `pytest tests/test_knowledge_graph.py tests/test_triplet_extractor.py
  tests/test_lightrag_fusion.py` cubre el subpaquete completo.
- **Dependencias claras**: igraph + snowballstemmer son de `lightrag/`, no del proyecto.

### Después de Fase 3

La carpeta `shared/retrieval/` pasa de **10 ficheros planos** a:

```
shared/retrieval/
├── __init__.py       ← factory (punto de entrada único)
├── core.py           ← contratos base
├── vector_store.py   ← almacenamiento vectorial
├── reranker.py       ← post-procesamiento
├── hybrid/           ← 4 ficheros, 1,363 LOC
└── lightrag/         ← 3 ficheros, 1,997 LOC
```

**4 ficheros en raíz + 2 subpaquetes** vs **10 ficheros planos**.
La carga cognitiva baja drásticamente.

---

## 6. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|:------------:|:-------:|------------|
| Import roto por path incorrecto | Media | Bajo | Tests en cada paso; git blame preserva historial |
| Test que importa directamente `shared.retrieval.hybrid_retriever` | Media | Bajo | Grep de imports antes de mover; actualizar en la misma fase |
| `mypy` falla por cambio de paths | Baja | Bajo | Ejecutar mypy en Fase 3; paths en `mypy.ini` son por módulo |
| Conflicto con ramas activas | Baja | Medio | Comunicar el refactor al equipo antes de empezar |
| Performance regression | Nula | — | No cambiamos lógica, solo ubicación de ficheros |

---

## 7. Lo que NO hacemos (y por qué)

| Tentación | Por qué no |
|-----------|-----------|
| Separar en repos/paquetes pip | `types.py` como hub hace inviable mantener repos separados sin duplicar tipos |
| Mover `metrics.py` a un paquete aparte | Solo tiene 1 dependencia. Moverlo añade complejidad de packaging sin beneficio real |
| Refactorizar `types.py` en múltiples ficheros | 612 LOC es manejable. Partirlo crearía 5+ ficheros pequeños con imports cruzados |
| Abstraer `llm.py` tras una interfaz | Ya usa protocols. Añadir otra capa de abstracción es overengineering |
| Crear un sistema de plugins | Solo hay 3 estrategias. Un plugin system para 3 implementaciones es matar moscas a cañonazos |

---

## 8. Estimación de impacto

| Métrica | Antes | Después |
|---------|:-----:|:-------:|
| Ficheros en `retrieval/` raíz | 10 | 4 + 2 dirs |
| Claridad de pertenencia | Implícita (hay que leer imports) | Explícita (por directorio) |
| Ficheros a tocar para añadir estrategia nueva | Investigar los 10 | Solo mirar `core.py` + crear nuevo subpaquete |
| Interfaz pública (`__init__.py`) | Sin cambios | Sin cambios |
| Tests existentes | Sin cambios de lógica | Solo actualización de imports |
| Dependencias opcionales | Mezcladas | Aisladas por subpaquete |

---

## 9. Orden de ejecución recomendado

```
Fase 0 ──► Fase 1 ──► Fase 2 ──► Fase 3
 (prep)    (hybrid)   (lightrag)  (limpieza)
  │          │           │           │
  │ 30 min   │ 1-2h      │ 1-2h      │ 30 min
  │          │           │           │
  └── commit ┘── commit ──┘── commit ─┘── commit + tag
```

Cada fase termina con un **commit atómico** y tests verdes.
Si algo sale mal en Fase 2, Fase 1 ya está consolidada y estable.

---

## 10. Decisión pendiente

Este documento propone la reorganización. **No se ejecuta ningún cambio de
código hasta aprobación explícita.** Las opciones son:

- **A) Implementar las 3 fases** — reorganización completa
- **B) Solo Fase 1 (hybrid/)** — el subpaquete con más ficheros sueltos
- **C) Solo Fase 2 (lightrag/)** — el subpaquete con más LOC y complejidad
- **D) No hacer nada** — el código funciona bien tal como está

La recomendación técnica es **opción A**, ejecutando fase a fase con validación
entre cada una.
