"""
Contador agregado para degradaciones silenciosas del pipeline (R14).

Siete bare-excepts del pipeline (retrieval KG, extraccion de tripletas,
generation) tragan la excepcion, logean `.debug`/`.warning` y devuelven un
fallback para que el run continue. Historicamente esos eventos solo existian
en logs dispersos — un 30% de fallos de parseo JSON de keywords o de
`neighbor_lookup` no era visible en el export del run.

Este tracker agrega cada evento a un contador observable, expuesto en
`config_snapshot._runtime.operational_stats` y en el evento `run_complete`
del JSONL estructurado. El comportamiento del fallback no cambia — solo
se añade la senal.

Eventos tipificados (OperationalEventType):
    - neighbor_lookup_failure:     KG.get_neighbors_ranked fallo (enrichment 1-hop).
    - chunk_keywords_vdb_error:    consulta a Chunk Keywords VDB fallo (div #10).
    - description_synthesis_error: LLM merge de descripciones de entidad fallo.
    - gleaning_error:              ronda extra de extraccion de tripletas fallo.
    - keywords_parse_failure:      parseo del JSON de keywords fallo.
    - retrieval_error:             build del KG durante index_documents fallo,
                                   fallback a vector puro.
    - generation_error:            llamada LLM de generacion final fallo.

Uso:
    from shared.operational_tracker import record_operational_event

    try:
        neighbors = kg.get_neighbors_ranked(name, ...)
    except Exception:
        logger.debug("Neighbor lookup failed for %s", name)
        record_operational_event("neighbor_lookup_failure")

El run nunca falla por un evento operacional. Valores altos son senal de
degradacion del canal (p.ej. keywords_parse_failure > 10% sugiere revisar
el prompt de keywords o el modelo LLM).
"""

from __future__ import annotations

import threading
from typing import Dict, Literal, TypedDict, get_args


OperationalEventType = Literal[
    "neighbor_lookup_failure",
    "chunk_keywords_vdb_error",
    "description_synthesis_error",
    "gleaning_error",
    "keywords_parse_failure",
    "retrieval_error",
    "generation_error",
]


class OperationalStats(TypedDict):
    """Contadores por tipo de evento operacional (R14).

    Siempre presente con los 7 tipos, valor 0 si no ocurrio. Valores altos
    indican degradacion del canal aunque el run termine sin error fatal.
    """

    neighbor_lookup_failure: int
    chunk_keywords_vdb_error: int
    description_synthesis_error: int
    gleaning_error: int
    keywords_parse_failure: int
    retrieval_error: int
    generation_error: int


class _OperationalTracker:
    """Contador thread-safe por tipo de evento. Inicializa a 0 todos los tipos."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[OperationalEventType, int] = {
            name: 0 for name in get_args(OperationalEventType)
        }

    def record(self, event: OperationalEventType) -> None:
        # Si llega un string fuera del Literal (codigo nuevo que olvido
        # anadir el tipo), lo aceptamos para no romper el run pero queda
        # visible en mypy.
        with self._lock:
            self._counters[event] = self._counters.get(event, 0) + 1

    def snapshot(self) -> OperationalStats:
        with self._lock:
            # Construccion explicita para no depender de **kwargs sobre un
            # dict[Literal, int] (mypy no lo estrecha a OperationalStats).
            return OperationalStats(
                neighbor_lookup_failure=self._counters["neighbor_lookup_failure"],
                chunk_keywords_vdb_error=self._counters["chunk_keywords_vdb_error"],
                description_synthesis_error=self._counters["description_synthesis_error"],
                gleaning_error=self._counters["gleaning_error"],
                keywords_parse_failure=self._counters["keywords_parse_failure"],
                retrieval_error=self._counters["retrieval_error"],
                generation_error=self._counters["generation_error"],
            )

    def reset(self) -> None:
        with self._lock:
            for key in list(self._counters.keys()):
                self._counters[key] = 0


_operational_tracker = _OperationalTracker()


def record_operational_event(event: OperationalEventType) -> None:
    """Incrementa el contador del evento operacional dado."""
    _operational_tracker.record(event)


def get_operational_stats() -> OperationalStats:
    """Snapshot del tracker. Incluye siempre los 7 tipos (0 si no ocurrio)."""
    return _operational_tracker.snapshot()


def reset_operational_stats() -> None:
    """Resetea contadores. Llamar al inicio de cada run de evaluacion."""
    _operational_tracker.reset()


__all__ = [
    "OperationalEventType",
    "OperationalStats",
    "record_operational_event",
    "get_operational_stats",
    "reset_operational_stats",
]
