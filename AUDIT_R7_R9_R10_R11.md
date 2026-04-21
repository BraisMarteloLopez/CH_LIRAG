# Auditoría R7/R9/R10/R11 — briefing de continuación

> Documento autocontenido para una sesión nueva de claude_code. No depende de historial conversacional. Léelo entero antes de actuar.

---

## 0 · Qué hacer en esta sesión (TL;DR)

1. **Verificar entorno**: reproducir baseline de tests y confirmar rama/HEAD (sección 2).
2. **Ejecutar auditoría de solo lectura** de las reglas R7, R9, R10, R11 sobre el código del repo (metodología en sección 6).
3. **Producir un único reporte**: `AUDIT_R7_R9_R10_R11_REPORT.md` en la raíz (formato en sección 7).
4. **No modificar código**. El reporte clasifica violaciones y propone si merecen PR dedicado (PR-7) o sub-tarea de PR-6. El usuario decide.
5. **Si hay bloqueos** (tests no reproducen, heurística da ruido masivo, dudas de criterio): **parar y preguntar**, no adivinar. Ver sección 8 para casos límite concretos.

**Orden de ejecución sugerido**: R11 (alcance más acotado) → R10 → R7 → R9.

**Tiempo estimado**: medio día de trabajo automatizado + revisión manual de falsos positivos. Si se extiende más, probablemente hay ruido en la heurística que debería reportarse al usuario antes de continuar.

---

## 1 · Estado al cierre de la sesión anterior (2026-04-21)

**Rama**: `claude/read-documentation-BtSiS`
**PR de la sesión anterior**: [#54](https://github.com/BraisMarteloLopez/CH_LIRAG/pull/54) (merged a `main`)
**HEAD al cierre**: `471eac5` (último commit: adición de sección 3.1 al briefing)
**Baseline de tests**: `499 passed, 6 skipped` con `/root/.local/bin/pytest tests/ -q --ignore=tests/integration`

**Commits introducidos en la sesión anterior** (verificables con `git log --oneline`):

| SHA | Descripción | Regla |
|---|---|---|
| `20a9746` | PR-2 — higiene documental (anchors, pins, refs navegables) | R12/R13 |
| `da2c81f` | PR-1 — asserts en lugar de ifs defensivos, eliminar código muerto | R8/R6 |
| `0899b5c` | PR-3 — `Literal[...]` en dominios cerrados | R2 |
| `b398f13` | PR-5 — `OperationalTracker` singleton, 7 eventos silenciosos contabilizados | R14 |
| `7350438` | PR-4 — 11 `TypedDict`s cross-module | R5 |

**Importante**: las reglas R2, R5, R8, R12, R13, R14 **ya fueron tratadas**. Esta sesión se ocupa de las **cuatro reglas restantes de comentarios/docstrings**: R7, R9, R10, R11.

---

## 2 · Verificación de entorno antes de empezar

Ejecutar estos tres comandos en orden. Si alguno falla, **parar y reportar al usuario**:

```bash
# 1. Confirmar rama
git rev-parse --abbrev-ref HEAD
# Esperado: claude/read-documentation-BtSiS (o main si ya se mergeó)

# 2. Confirmar HEAD
git log -1 --format="%H %s"
# Esperado: 471eac5... o posterior. Si es anterior, el repo está desactualizado.

# 3. Confirmar baseline de tests
/root/.local/bin/pytest tests/ -q --ignore=tests/integration
# Esperado: 499 passed, 6 skipped
```

Si el baseline no reproduce, el problema está en el entorno (deps, versión de Python, path), no en el código. Reportar al usuario con el output exacto antes de intentar ninguna otra cosa.

---

## 4 · Las cuatro reglas (literales)

### R7 — Un comentario explica por qué, no qué
> Criterio de violación: el comentario traducido a código produce exactamente la siguiente línea.

### R9 — Un invariante se explica una vez, en el sitio de mayor precedencia
> Criterio de violación: la misma razón aparece en docstring + comentario inline adyacente.
>
> Regla de elección: si la explicación vale para todo el método, va al docstring; si explica solo un bloque específico, va inline y el docstring no la repite.

### R10 — Función con lógica no trivial incluye al menos un ejemplo ejecutable
> Aplica a: regex, parsing, transformación con edge cases, dispatch.
>
> Formato: bloque `>>> f(input) -> output` compatible con doctest.
>
> Criterio de "no trivial": si el comportamiento del edge case requiere prosa para explicarse, requiere ejemplo.

### R11 — El shape de retorno estructurado no se duplica en el docstring si el tipo ya lo expresa
> Si devuelves `TypedDict` o `@dataclass`, el docstring dice "ver `FooResult`", no re-lista las claves. Si devuelves `Dict[str, Any]`, estás violando R1 primero — arregla eso antes.

---

## 5 · Intuición rápida de cada regla

Antes de entrar en la metodología detallada, la formulación llana:

- **R7**: si al borrar el comentario un lector nuevo entiende la línea con igual velocidad, el comentario sobra.
- **R9**: si al borrar el comentario inline el docstring sigue respondiendo lo mismo, el inline sobra.
- **R10**: si alguien en code-review razonablemente preguntaría "¿qué hace con input X?", la función necesita un ejemplo en su docstring.
- **R11**: si el docstring lista campos que ya están en el `TypedDict`/`@dataclass`, o mueves la descripción al tipo, o la borras del docstring.

---

## 6 · Metodología por regla

**Alcance de búsqueda**: todos los `.py` bajo estos directorios (excluye `tests/integration/`):

```
sandbox_mteb/
shared/
tests/ (excepto tests/integration/)
```

### 6.1 · R7 — Comentarios que traducen literalmente el código siguiente

**Heurística primaria**: comentario seguido de una línea de código donde las palabras clave del comentario aparecen en la línea.

**Patrones detectables con grep** (primer filtro, revisar manualmente):

- `# increment` seguido de `+= 1`
- `# reset ` / `# initialize ` seguido de asignación
- `# check if` seguido de `if`
- `# loop` / `# iterate` seguido de `for`
- `# return` seguido de `return`
- `# call ` seguido de `.xxx(` literal
- `# append` seguido de `.append(`
- `# log` seguido de `logger.` / `log.`

**Patrón AST más robusto** (recomendado para muestra representativa):

1. Walk por todos los `.py`.
2. Para cada módulo, extraer `ast.get_source_segment` de cada nodo statement-level.
3. Capturar el comentario inmediatamente anterior vía `tokenize.generate_tokens` o lectura lineal del fichero mapeando `lineno`.
4. Calcular similitud léxica (`difflib.SequenceMatcher.ratio` sobre lemas básicos) entre comentario (stripeando `#`) y source del siguiente statement.
5. Marcar como candidato si `ratio > 0.55` y comentario `< 120 chars`. **Verificar cada uno manualmente** — la heurística tiene falsos positivos.

**Falsos positivos conocidos** (NO marcar):

- Comentarios de sección ASCII art: `# --- seccion X ---`.
- Comentarios que apuntan a deuda técnica: `# Deuda #14`, `# divergencia #10`, `# Ver CLAUDE.md`.
- Comentarios en dispatchers con muchos casos donde el "qué" no es obvio al lector no familiar.
- Marcadores `# type: ignore[...]`, `# noqa`, `# pragma:`.
- Referencias a código externo: `# HKUDS/LightRAG operate.py:L417`.

### 6.2 · R9 — Invariante duplicado en docstring + inline adyacente

**Heurística**:

1. Walk AST, encontrar nodos `FunctionDef`/`AsyncFunctionDef`/`Method` con docstring no vacío.
2. Tokenizar docstring en "razones" (frases separadas por `.`, filtrando las que describen signatura).
3. Recorrer el body; por cada comentario inline, verificar si el texto (lowercase, sin puntuación, **sin stopwords en español**) comparte `>= 60%` de tokens significativos con alguna razón del docstring.
4. Si el comentario está en los primeros 5 statements del body, marcar como R9.

**Filtro de stopwords en español** (crítico para evitar ruido): excluir al menos `de la el los las un una y o por para con sin que se es`.

**Falsos positivos conocidos**:

- Docstring genérico ("Initialize X.") + comentario específico al bloque: NO es R9.
- Comentario en medio de la función que parafrasea un caso del docstring pero aplica a una rama concreta: NO es R9.
- Aplica SOLO cuando la misma razón aparece dos veces; repetir un hecho distinto en sitios distintos no es R9.

### 6.3 · R10 — Funciones no triviales sin ejemplo `>>>`

**Identificación de "no trivial"** (aplicar los cuatro criterios con `or`):

1. **Regex**: la función usa `re.` y tiene ramas sobre match/no-match.
2. **Parsing**: nombre contiene `parse_`, `deserialize`, `decode`, `from_str`, `from_dict`, o retorna objeto construido a partir de un string/dict de entrada.
3. **Transformación con edge cases**: body contiene `>= 2` ramas `if` que devuelven cosas distintas del happy-path (explicitamente tratando vacío/None/malformed).
4. **Dispatch**: usa dict literal como tabla de despacho, o cadena `if/elif` sobre valor enumerable.

**Verificación**: para cada función identificada, comprobar si su docstring contiene línea que empiece con `>>>`.

**Sitios probables de arranque** (muestreo inicial, NO lista cerrada — generar inventario propio):

- `shared/citation_parser.py::parse_citation_refs` (regex + edge cases)
- `shared/retrieval/lightrag/triplet_extractor.py` (parse JSON LLM output)
- `sandbox_mteb/checkpoint.py::deserialize_query_result` (JSON → dataclass)
- `shared/retrieval/lightrag/knowledge_graph.py::from_dict` (deserialización v3)
- `sandbox_mteb/config.py` — dispatchers por dataset type
- **Los 6 resolvers introducidos en PR-1 y PR-3** (ver `da2c81f` y `0899b5c`): `_METRIC_DISPATCH`, `_PRIMARY_METRIC_RESOLVERS`, `parse_answer_type`, etc. Son dispatch/parsing fresco y candidato natural.

**Falsos positivos conocidos**:

- Funciones privadas `_helper_foo` de uso interno obvio: NO requiere doctest.
- Getters/setters triviales: NO.
- Factories 1-línea: NO.

### 6.4 · R11 — Docstring duplica shape de retorno ya expresado por el tipo

**Heurística primaria** (barrido dirigido, dado que PR-4 añadió 11 `TypedDict`s):

1. Localizar funciones que devuelven uno de estos tipos (firma `-> XxxStats`, `-> XxxSnapshot`, `-> XxxRecord`):
   - `CitationStats` (`shared/citation_parser.py`)
   - `JudgeMetricStats` (`shared/metrics.py`)
   - `OperationalStats` (`shared/operational_tracker.py`)
   - `KGSynthesisStats` (`sandbox_mteb/generation_executor.py`)
   - `RuntimeSnapshot` (`sandbox_mteb/result_builder.py`)
   - `InvokeTiming` (`shared/llm.py`)
   - `DatasetSpec` (`shared/types.py`)
   - `ChromaStoreConfig` (`shared/vector_store.py`)
   - `KGStats`, `KGSerialized` (`shared/retrieval/lightrag/knowledge_graph.py`)
   - `CheckpointQueryRecord`, `CheckpointRetrievalRecord`, `CheckpointGenerationRecord` (`sandbox_mteb/checkpoint.py`)
   - Cualquier `@dataclass` del proyecto: muestreo en `QueryEvaluationResult`, `GenerationResult`, `QueryRetrievalDetail`, `EvaluationRun` — ver `shared/types.py`.

2. Por cada función identificada, leer su docstring. **Marcar violación** si el docstring:
   - Enumera claves del `TypedDict`/dataclass (líneas tipo `- invocations: int, numero de llamadas`).
   - Describe el shape palabra por palabra cuando el tipo ya lo hace.

3. **NO marcar violación** si el docstring:
   - Dice `"Ver CitationStats"`, `"ver dataclass X"`, `"retorna snapshot según schema v3"`.
   - Describe semántica del retorno que **no está clara solo por el nombre** del campo o del tipo. Pregunta clave: ¿la semántica pertenece al docstring del `TypedDict` mismo o a la función? Si pertenece al `TypedDict`, es violación de R11 en la función — **anotar como "mover descripción al TypedDict"** en vez de "eliminar". Si es específica de esta función (ej. "este snapshot excluye categorías no invocadas en este run"), NO es violación.

4. **Complementario**: grepear `Dict[str, Any]` como tipo de retorno. Si existe con docstring listando claves, hay **violación de R1 primero** (marcarlo explicitamente — no es R11 hasta que el tipo se cierre). Anotar como candidato a PR separado o ampliación de PR-4.

**Falsos positivos conocidos**:

- Docstring que menciona 1-2 campos clave para orientar al lector sin enumerarlos todos: OK si es ilustrativo.
- Docstring de `__init__` de dataclass describiendo cada campo: OK si el dataclass no tiene docstrings de campo.

---


## 8 · Casos límite y criterios de parada

**Parar y preguntar** en cualquiera de estas situaciones — no adivinar:

1. **Baseline de tests no reproduce**: reportar output exacto del pytest, no intentar arreglarlo.
2. **HEAD encontrado anterior a `471eac5`**: el repo está desactualizado, pedir al usuario que sincronice.
3. **Heurística R7/R9 produce >100 candidatos brutos**: antes de revisar manualmente 100 sitios, reportar número y pedir confirmación de si seguir o ajustar umbrales.
4. **Duda de criterio "no trivial" en R10**: anotar en reporte como `[incierto: recomendar al usuario]` en vez de decidir. No descartar ni marcar unilateralmente.
5. **Violación R11 donde el docstring describe semántica no reflejada en el `TypedDict`**: anotar como "mover descripción al TypedDict" (no como eliminación). Esto implica trabajo en `TypedDict` que puede merecer PR propio.
6. **Aparición de violaciones R1 nuevas** (`Dict[str, Any]` sin TypedDict): reportar separadamente en el apéndice. Son candidatos a ampliación de PR-4, no a este PR.

---

## 9 · Verificación de sanity antes de cerrar

- `/root/.local/bin/pytest tests/ -q --ignore=tests/integration` → `499 passed, 6 skipped` (confirmar que no se modificó código por accidente).
- `git status` → limpio salvo el archivo de reporte nuevo.
- `git diff` contra HEAD → solo el reporte.
- El reporte contiene todas las secciones de la plantilla (sección 7), aunque alguna tenga 0 violaciones (en ese caso, indicar "0 violaciones encontradas — ver apéndice para criterio aplicado").

---

## 10 · Limitaciones conocidas del entorno

- **Sin acceso a infra NIM/MinIO**: cualquier tentación de validar código llamando al pipeline real es imposible; todo es análisis estático.
- **Heurísticas imperfectas**: la detección automática de R7 por similitud léxica tiene ruido. Verificar manualmente las 3-5 líneas alrededor de cada candidato.
- **Tokenización naive en R9**: la comparación docstring↔comentario funciona mal en español con preposiciones. Filtrar stopwords antes del overlap.
- **R11 ambiguo en frontera**: cuando el docstring describe *semántica* y no solo shape, la frontera es borrosa. Aplicar estrictamente la regla "¿pertenece al TypedDict o a la función?".

---

## 11 · Decisión sobre siguiente paso tras esta auditoría

El PR-6 (Tier C — descomposición de funciones largas) está planificado pero pendiente de aprobación del usuario. **Esta auditoría es ortogonal a PR-6**: puede ejecutarse antes, después o en paralelo. La recomendación del reporte (sección `Decisión propuesta`) debe considerar:

- Si las violaciones se concentran en funciones que PR-6 va a tocar de todos modos → integrar como sub-tarea de PR-6.
- Si las violaciones están dispersas en código estable → PR-7 dedicado.
- Si son muy pocas (<10 total) → incluir directamente en el PR más oportuno sin crear nuevo.

La decisión final es del usuario, no del agente.
