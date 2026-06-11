# Contrato de export de KG — CH_LIRAG (productor del grafo) y LI_AD (consumidor)

**Estado**: BORRADOR v0 — propuesta de CH_LIRAG. Reciproco de `INGESTION_CONTRACT.md` (alli LI_AD -> motor entrega chunks; aqui motor -> LI_AD entrega el grafo construido a partir de esos chunks).

**IMPORTANTE — estado de implementacion**: hoy el KG del motor es **efimero** (igraph + ChromaDB en memoria, descartado en `_cleanup()` al final del run). Este documento define el **schema de export** para que LI_AD pueda planificar su Fase 2 (UI que pinta el grafo), pero la **implementacion del export es trabajo P3 del motor, aun no construido**. No es un entregable inmediato ni se compromete fecha.

## 1. Layout MinIO (propuesto)

El motor escribiria el grafo junto a la coleccion de origen, versionado por la `generation` de los chunks de la que se construyo:

```
{prefix}/collections/{collection_id}/kg/
  kg.json            # manifest del export (§4)
  nodes.parquet      # entidades (§2)
  edges.parquet      # relaciones (§3)
```

## 2. `nodes.parquet` (un row = una entidad/nodo)

| Columna | Tipo Arrow | Semantica |
|---|---|---|
| `node_id` | string | ID estable del nodo dentro del grafo. |
| `name` | string | Nombre de la entidad. |
| `entity_type` | string | Tipo (p.ej. PERSON, ORG, CONCEPT...). |
| `description` | string | Descripcion sintetizada de la entidad. |
| `degree` | int32 | Grado (nº de aristas); util para layout/ranking en la UI. |
| `source_chunk_ids` | list<string> | `chunk_id`s de la coleccion donde aparece (trazabilidad a procedencia). |

## 3. `edges.parquet` (un row = una relacion/arista)

| Columna | Tipo Arrow | Semantica |
|---|---|---|
| `src_id` | string | `node_id` origen. |
| `dst_id` | string | `node_id` destino. |
| `relation` | string | Descripcion/tipo de la relacion. |
| `weight` | float | Peso (co-ocurrencia / fuerza). |
| `source_chunk_ids` | list<string> | `chunk_id`s que soportan la relacion. |

(El grafo del motor es no dirigido con pesos por co-ocurrencia; `src`/`dst` son convencion de almacenamiento, no direccionalidad semantica.)

## 4. Manifest `kg.json`

```json
{
  "contract_version": "0",
  "collection_id": "col_20260429082254_3c22096a",
  "built_from_generation": 3,
  "engine": "CH_LIRAG",
  "num_nodes": 1234,
  "num_edges": 5678,
  "created_at": "2026-05-20T12:00:00Z"
}
```

- `built_from_generation`: la `generation` de los chunks (de `INGESTION_CONTRACT.md`) a partir de la cual se construyo este grafo. Permite a LI_AD saber si el grafo esta al dia respecto a la coleccion (si la coleccion avanzo de generacion, el KG esta obsoleto).

## 5. Preguntas abiertas para LI_AD

1. ¿Que necesita mostrar la UI: solo nodos/aristas, o tambien `description` sintetizada y la procedencia (`source_chunk_ids`)?
2. ¿Os vale el grafo **no dirigido con pesos**, o necesitais direccionalidad / tipado fuerte de relaciones?
3. ¿Donde quereis el export: junto a la coleccion (`{collection_id}/kg/`) o en un prefijo propio?
4. ¿Necesitais el export en cada `generation`, o solo bajo demanda?

## 6. Estado de implementacion (honesto)

El export **no existe** aun. `KnowledgeGraph` (igraph) y las VDBs viven en memoria y se descartan en `_cleanup()`. Implementarlo (serializador `KnowledgeGraph` -> Parquet + escritura a MinIO) es trabajo **P3** del motor, posterior a cerrar la ingesta. Este schema se publica ahora para que LI_AD planifique; no comprometemos fecha.

---

> Borrador v0 del lado motor. Reciproco de `INGESTION_CONTRACT.md`. Implementacion P3, no inmediata.
