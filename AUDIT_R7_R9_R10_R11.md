# Auditoria: comentarios y docstrings (R7, R9, R10, R11)

> **Proposito**: este archivo es briefing autocontenido para una sesion nueva de claude_code. No depende de historial de sesiones previas. Leelo entero antes de actuar.

## 1 · Contexto minimo

- **Proyecto**: motor de ejecucion para construccion y consulta de grafos de conocimiento (SIMPLE_VECTOR baseline, LIGHT_RAG target). Fase actual P0 — replicacion empirica del paper. Detalles en `CLAUDE.md`.
- **Rama de trabajo**: `claude/read-documentation-BtSiS`. Todos los commits nuevos van ahi. No abrir PR contra `main` sin permiso explicito del usuario.
- **HEAD al arranque**: `7350438` (PR-4: TypedDicts).
- **Idioma de codigo**: espanol/ingles mezclado (historico). Preserva el idioma del sitio que edites.
- **Tests baseline**: `499 passed, 6 skipped` con `/root/.local/bin/pytest tests/ -q --ignore=tests/integration`. Hay que mantenerlo.

## 2 · Que se ha hecho y que NO tocar

Los seis PRs del plan de remediacion derivado de la auditoria previa estan en estos estados:

| PR | Regla | Commit | Estado |
|---|---|---|---|
| PR-1 | R8/R6 — ifs defensivos → assert, eliminar codigo muerto | `da2c81f` | **Cerrado**. **No re-abrir sin evidencia.** Si encuentras violaciones R8 residuales, anota en el reporte (seccion 6) en vez de arreglarlas en este PR. |
| PR-2 | R12/R13 — higiene documental (anchors, pins, referencias navegables) | `20a9746` | Cerrado |
| PR-3 | R2 — `Literal[...]` en dominios cerrados | `0899b5c` | Cerrado |
| PR-4 | R5 — `TypedDict`s para dicts cross-module | `7350438` | Cerrado. **Relevante para R11** (ver seccion 5.4). |
| PR-5 | R14 — trackers para eventos silenciosos | `b398f13` | Cerrado |
| PR-6 | Tier C — descomposicion de funciones largas | **pendiente** | No empezar hasta despues de esta auditoria |

**Reglas duras**:
- NO tocar tests de integracion (`tests/integration/`).
- NO tocar `shared/config_base.py`, `DatasetType` en `shared/types.py`, `_PersistentLoop` en `shared/llm.py`, `requirements.lock`.
- NO modificar el flujo retrieval→synthesis→generation sin que el usuario lo valide — no hay infraestructura NIM/MinIO en esta sesion (ver `CLAUDE.md` seccion "Limitaciones de claude_code sobre este proyecto").
- NO abrir pull request. El usuario lo abre cuando quiere.

## 3 · Objetivo de esta auditoria

Producir un **reporte** (formato en seccion 6) que enumere las violaciones a las cuatro reglas de comentarios y docstrings. La auditoria es **solo lectura**: no modificar codigo. Al final del reporte, clasificar cada violacion por coste estimado de arreglo (trivial / pequeno / medio) y proponer si merece PR dedicado (PR-7) o si encaja como sub-tarea del PR-6 (descomposicion de funciones largas, donde ya se van a tocar esos sitios).

**Tiempo estimado**: 90-120 minutos de exploracion automatizada + revision manual de falsos positivos.

## 3.1 · Intuicion rapida de cada regla

Antes de las reglas literales (seccion 4) y la metodologia detallada (seccion 5), el objetivo de cada una en lenguaje llano:

- **R7** — contar comentarios que traducen literalmente la siguiente linea ("incrementa contador", "abre archivo", "retorna True si X"). Suelen detectarse por proximidad semantica comentario↔codigo.
- **R9** — detectar docstring + comentario inline adyacente repitiendo la misma razon en el mismo bloque.
- **R10** — identificar funciones con regex, parsing, dispatch o edge cases que no tienen bloque `>>>` ejecutable en su docstring.
- **R11** — detectar docstrings que re-listan las claves de un `TypedDict` o los campos de un `@dataclass` que acabamos de tipar en PR-4.

## 4 · Las cuatro reglas, literales

### R7 — Un comentario explica por que, no que
> Criterio de violacion: el comentario traducido a codigo produce exactamente la siguiente linea.

### R9 — Un invariante se explica una vez, en el sitio de mayor precedencia
> Criterio de violacion: la misma razon aparece en docstring + comentario inline adyacente.
>
> Regla de eleccion: si la explicacion vale para todo el metodo, va al docstring; si explica solo un bloque especifico, va inline y el docstring no la repite.

### R10 — Funcion con logica no trivial incluye al menos un ejemplo ejecutable
> Aplica a: regex, parsing, transformacion con edge cases, dispatch.
>
> Formato: bloque `>>> f(input) -> output` compatible con doctest.
>
> Criterio de "no trivial": si el comportamiento del edge case requiere prosa para explicarse, requiere ejemplo.

### R11 — El shape de retorno estructurado no se duplica en el docstring si el tipo ya lo expresa
> Si devuelves `TypedDict` o `@dataclass`, el docstring dice "ver `FooResult`", no re-lista las claves. Si devuelves `Dict[str, Any]`, estas violando R1 primero — arregla eso antes.

## 5 · Metodologia por regla

**Alcance de busqueda**: todo `.py` bajo estos directorios (excluye `tests/integration/`):

```
sandbox_mteb/
shared/
tests/ (excepto tests/integration/)
```

Usa `Grep` / `Glob` del agente, no `rg` directo. Para recorrer el AST de forma sistematica, `python3 -c "import ast; ..."` es aceptable.

### 5.1 · R7 — Comentarios que traducen literalmente el codigo siguiente

**Heuristica primaria**: comentario seguido de una linea de codigo donde las palabras clave del comentario aparecen en la linea.

**Patrones detectables con grep** (usar como primer filtro, luego revisar manualmente):

- `# increment` seguido de `+= 1`
- `# reset ` / `# initialize ` seguido de asignacion
- `# check if` seguido de `if`
- `# loop` / `# iterate` seguido de `for`
- `# return` seguido de `return`
- `# call ` seguido de `.xxx(` literal
- `# append` seguido de `.append(`
- `# log` seguido de `logger.` / `log.`

**Patron AST mas robusto** (recomendado para muestra representativa):
1. Walk por todos los `.py`.
2. Para cada modulo, extraer `ast.get_source_segment` de cada nodo statement-level.
3. Capturar el comentario inmediatamente anterior via `tokenize.generate_tokens` (o lectura lineal del fichero mapeando `lineno`).
4. Calcular similitud lexica (`difflib.SequenceMatcher.ratio` sobre lemas basicos) entre comentario (stripeando `#`) y source del siguiente statement.
5. Marcar como candidato a violacion si ratio > 0.55 y comentario < 120 chars. Manualmente verificar cada uno — la heuristica tiene falsos positivos.

**Falsos positivos conocidos** (NO marcar):
- Comentarios de seccion ASCII art tipo `# --- seccion X ---`.
- Comentarios que apuntan a deuda tecnica: `# Deuda #14`, `# divergencia #10`, `# Ver CLAUDE.md`.
- Comentarios en bloques de tabla / datos estaticos donde el "que" no es obvio al lector no familiar con el dominio (por ejemplo, en dispatchers con muchos casos).
- Marcadores `# type: ignore[...]`, `# noqa`, `# pragma:`.
- Referencias a codigo externo: `# HKUDS/LightRAG operate.py:L417`.

**Heuristica de decision**: si al borrar el comentario un lector nuevo del codebase entiende la linea con igual velocidad, el comentario es R7.

### 5.2 · R9 — Invariante duplicado en docstring + inline adyacente

**Heuristica**:
1. Walk AST, encontrar nodos `FunctionDef`/`AsyncFunctionDef`/`Method` con docstring no-vacio.
2. Por cada funcion, tokenizar docstring en "razones" (frases separadas por `.`, tras filtrar las que describen la signatura).
3. Recorrer el body de la funcion; por cada comentario inline, verificar si el texto (normalizado a lowercase, sin puntuacion) comparte >= 60% de tokens significativos con alguna razon del docstring.
4. Si ademas el comentario inline esta en los primeros 5 statements del body, marcar como R9.

**Falsos positivos conocidos**:
- Docstring generico ("Initialize X.") + comentario especifico al bloque posterior: NO es R9.
- Comentario en medio de la funcion que parafrasea un caso del docstring pero aplica a una rama concreta: NO es R9 (el docstring describe el invariante global, el inline describe por que esta rama lo cumple).
- Aplica SOLO cuando la misma razon aparece dos veces. Repetir un hecho distinto en sitios distintos no es R9.

**Heuristica de decision**: si al borrar el comentario inline el docstring sigue respondiendo la misma pregunta con la misma calidad, es R9.

### 5.3 · R10 — Funciones no triviales sin ejemplo `>>>`

**Identificacion de funcion "no trivial"** (aplicar los cuatro criterios con `or`):

1. **Regex**: la funcion usa `re.` (import del modulo en el fichero y uso dentro del body) y tiene ramas sobre match/no-match.
2. **Parsing**: nombre contiene `parse_`, `deserialize`, `decode`, `from_str`, `from_dict`, o retorna un dict/objeto construido a partir de un string/dict de entrada.
3. **Transformacion con edge cases**: body contiene >= 2 ramas `if` que devuelven cosas distintas del caso happy-path (explicitamente tratando vacio/None/malformed).
4. **Dispatch**: usa un dict literal como tabla de despacho, o una cadena de `if/elif` sobre un valor enumerable.

**Verificacion**: para cada funcion identificada como no trivial, comprobar si su docstring contiene una linea que empiece con `>>>` (compatible con doctest).

**Sitios probables** (para muestreo, NO lista cerrada — el agente debe generar su propio inventario):
- `shared/citation_parser.py::parse_citation_refs` (regex + edge cases obvios)
- `shared/retrieval/lightrag/triplet_extractor.py` (parse LLM output JSON)
- `sandbox_mteb/checkpoint.py::deserialize_query_result` (parse JSON → dataclass)
- `shared/retrieval/lightrag/knowledge_graph.py::from_dict` (deserializacion v3)
- `sandbox_mteb/config.py` — dispatchers por dataset type

**Falsos positivos conocidos**:
- Funciones privadas `_helper_foo` de uso interno obvio cuyo comportamiento queda claro por tipo y nombre: NO requiere doctest.
- Getters/setters triviales: NO.
- Factories 1-linea: NO.

**Heuristica de decision**: si la funcion aparece en una revision de code-review y alguien razonablemente preguntaria "¿que hace con input X?", requiere ejemplo.

### 5.4 · R11 — Docstring duplica shape de retorno ya expresado por el tipo

**Heuristica primaria (barrido dirigido, dado que PR-4 acaba de anadir 11 TypedDicts)**:

1. Localizar todas las funciones que devuelven uno de estos tipos (firmas `-> XxxStats`, `-> XxxSnapshot`, `-> XxxRecord`):
   - `CitationStats` (shared/citation_parser.py)
   - `JudgeMetricStats` (shared/metrics.py)
   - `OperationalStats` (shared/operational_tracker.py)
   - `KGSynthesisStats` (sandbox_mteb/generation_executor.py)
   - `RuntimeSnapshot` (sandbox_mteb/result_builder.py)
   - `InvokeTiming` (shared/llm.py)
   - `DatasetSpec` (shared/types.py)
   - `ChromaStoreConfig` (shared/vector_store.py)
   - `KGStats`, `KGSerialized` (shared/retrieval/lightrag/knowledge_graph.py)
   - `CheckpointQueryRecord`, `CheckpointRetrievalRecord`, `CheckpointGenerationRecord` (sandbox_mteb/checkpoint.py)
   - Cualquier `@dataclass` del proyecto retornado por alguna funcion (muestreo: `QueryEvaluationResult`, `GenerationResult`, `QueryRetrievalDetail`, `EvaluationRun` — ver `shared/types.py`).

2. Por cada funcion identificada, leer su docstring. Marcar violacion si el docstring:
   - Enumera claves del TypedDict/dataclass (por ejemplo: lineas tipo `- invocations: int, numero de llamadas`).
   - Describe el "shape" del retorno palabra por palabra cuando el tipo ya lo hace.

3. NO marcar violacion si el docstring:
   - Dice cosas tipo `"Ver CitationStats"`, `"ver dataclass X"`, `"retorna snapshot segun schema v3"`.
   - Describe semantica del retorno (que significa cada campo semanticamente) **que no esta clara solo por el nombre del campo o del tipo**. En ese caso preguntate: ¿la semantica pertenece al docstring del TypedDict mismo o a la funcion? Si pertenece al TypedDict, es violacion de R11 en la funcion (anotar como "mover descripcion al TypedDict"). Si es especifica de esta funcion (por ejemplo, "este snapshot excluye categorias no invocadas en este run"), NO es violacion.

4. Complementario: grepear `Dict[str, Any]` como tipo de retorno. Si existe y ademas hay docstring listando claves, hay **violacion de R1 primero** (marcarlo explicitamente — no es R11 hasta que el tipo se cierre). Anotar como candidato a PR separado o a ampliacion de PR-4.

**Falsos positivos conocidos**:
- Docstring que menciona un par de campos clave para orientar al lector, sin enumerarlos todos: frontera borrosa — usa criterio estricto. Si son 1-2 campos mencionados como ejemplo ilustrativo, OK.
- Docstring de `__init__` de dataclass mencionando que hace cada campo: OK si el dataclass no tiene docstrings de campo.

## 6 · Formato del reporte entregable

Crear un archivo `AUDIT_R7_R9_R10_R11_REPORT.md` en la raiz con esta estructura exacta:

```markdown
# Reporte auditoria R7 / R9 / R10 / R11

- HEAD analizado: <sha>
- Fecha: <YYYY-MM-DD>
- Alcance: sandbox_mteb/, shared/, tests/ (excluye tests/integration/)

## Resumen ejecutivo

- R7: N violaciones, M falsos positivos descartados
- R9: ...
- R10: ...
- R11: ...

Recomendacion general: [PR-7 dedicado | integrar en PR-6 | dejar sin arreglar]

## R7 — Comentarios "que" en lugar de "por que"

### Violacion 1 — `path/archivo.py:LINEA`
**Comentario actual**:
```python
# increment counter
self.counter += 1
```
**Veredicto**: viola R7 (comentario parafrasea linea).
**Coste**: trivial (borrar).
**Accion**: eliminar comentario.

### Violacion 2 — ...

## R9 — ...

## R10 — ...

## R11 — ...

## Apendice: falsos positivos descartados

Cada uno con 1-2 lineas de justificacion — esto es importante para que el usuario pueda auditar el criterio.

## Decision propuesta

[Justificacion de si merece PR-7 o integrar en PR-6]
```

**Reglas del reporte**:
- Cada violacion con path absoluto `file.py:LINE` para navegacion.
- Coste en categorias cerradas: `trivial` (1 linea), `pequeno` (<10 lineas), `medio` (>10 lineas o cambio de firma).
- NO proponer correcciones hipoteticas — describe el problema, no lo resuelvas. El usuario decide.
- Si encuentras un falso positivo recurrente (p.ej. marcadores de seccion detectados como R7), anadir regla al `Apendice` y deduplicar.

## 7 · Verificacion sanity antes de cerrar

- `/root/.local/bin/pytest tests/ -q --ignore=tests/integration` → `499 passed, 6 skipped`. (Confirma que no se modifico codigo por accidente.)
- `git status` → limpio salvo el archivo de reporte nuevo.
- `git diff` contra HEAD → solo el reporte.

## 8 · Limitaciones conocidas

- **Sin acceso a infra**: los criterios de "no trivial" en R10 dependen del dominio. Si dudas si algo merece doctest, anotalo en el reporte como `[incierto: recomendar al usuario]` en vez de decidir tu.
- **Heuristicas imperfectas**: la deteccion automatica de R7 por similitud lexica tiene ruido. Cuando marques algo como violacion, verificalo leyendo el contexto — al menos las 3-5 lineas alrededor.
- **Tokenizacion naive en R9**: la comparacion docstring↔comentario funciona mal en espanol con preposiciones ("de", "el", "la"). Filtra stopwords antes de calcular overlap.
- **R11 ambiguo en frontera**: cuando el docstring describe *semantica* de campos y no solo shape, la frontera con R11 es borrosa. Aplica estrictamente la regla "¿la descripcion pertenece al TypedDict o a la funcion?" — si pertenece al TypedDict, anotar como "mover al TypedDict" en vez de "eliminar".

## 9 · Primer paso sugerido

1. Leer este archivo completo.
2. Leer `CLAUDE.md` (lectura rapida, ~10 min — orienta sobre producto, fase, deuda tecnica).
3. Confirmar baseline de tests (`499 passed, 6 skipped`).
4. Empezar por R11 (alcance mas acotado, lista fija de tipos del PR-4).
5. Luego R10 (lista semi-cerrada por heuristica de "no trivial").
6. R7 y R9 al final (barridos mas amplios, mayor ruido).
7. Producir reporte unico.
