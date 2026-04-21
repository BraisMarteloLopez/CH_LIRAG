"""
Modulo: Citation Reference Parser
Descripcion: Parseo de citas `[ref:N]` emitidas por el LLM, observable para
             divergencia #7 (contexto estructurado JSON-lines con reference_id).

El parser produce 7 contadores por cada texto analizado, discriminando formato
(valid vs malformed), rango (in_range vs out_of_range respecto a los chunks
realmente emitidos) y diversidad (distinct + coverage_ratio). Se invoca dos
veces por query: una sobre la narrativa post-synthesis y otra sobre la
respuesta final del LLM generador, con prefijos `synth_` y `gen_`
respectivamente en la serializacion a `retrieval_metadata`.

Acoplamiento al prompt: el formato estricto `[ref:N]` proviene del contrato
expuesto en `KG_SYNTHESIS_SYSTEM_PROMPT` (`sandbox_mteb/config.py`). Cualquier
cambio a ese prompt exige revisar este parser y la interpretacion de sus
metricas — ver deuda tecnica asociada en CLAUDE.md.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# Formato estricto aceptado: [ref:N] con N entero sin espacios ni mayusculas.
_VALID_RE = re.compile(r"\[ref:(\d+)\]")

# Superset de candidatos reconocibles (incluye variantes que NO cumplen el
# formato estricto). Se cuenta como malformed todo lo que matchea este regex
# pero NO matchea _VALID_RE, para que la union cubra el texto una sola vez y
# las categorias valid/malformed sean disjuntas.
_CANDIDATE_RE = re.compile(r"\[\s*[Rr][Ee][Ff]\s*:?\s*[^\]]*\]")


def parse_citation_refs(
    text: Optional[str], n_valid_chunks: int,
) -> Dict[str, Any]:
    """Parse citas `[ref:N]` y variantes reconocibles.

    Args:
        text: Texto emitido por el LLM (narrativa o respuesta). None o vacio
              retornan contadores en 0.
        n_valid_chunks: Numero de chunks realmente emitidos en el contexto
              estructurado. El rango valido es `[1, n_valid_chunks]` inclusive.
              `0` implica que cualquier referencia valida es out_of_range.

    Returns:
        Dict con las 7 metricas documentadas:
          - total: valid + malformed
          - valid: matches del formato estricto `[ref:N]`
          - malformed: candidatos reconocibles fuera de formato estricto
          - in_range: valid con N en [1, n_valid_chunks]
          - out_of_range: valid con N fuera de rango (senal roja de alucinacion)
          - distinct: unique N en in_range
          - coverage_ratio: distinct / in_range (diversidad de fuentes citadas)
    """
    if not text:
        return {
            "total": 0,
            "valid": 0,
            "malformed": 0,
            "in_range": 0,
            "out_of_range": 0,
            "distinct": 0,
            "coverage_ratio": 0.0,
        }

    valid_matches = _VALID_RE.findall(text)
    valid_ns = [int(m) for m in valid_matches]
    valid_count = len(valid_ns)

    candidate_matches = _CANDIDATE_RE.findall(text)
    # malformed = candidates - valid. El regex de candidates captura tambien
    # los valid; restamos para que las categorias sean disjuntas. Invariante:
    # _CANDIDATE_RE es superset estricto de _VALID_RE, por tanto diff >= 0.
    malformed_count = len(candidate_matches) - valid_count
    assert malformed_count >= 0, (
        "_CANDIDATE_RE debe ser superset de _VALID_RE"
    )

    in_range_ns = [n for n in valid_ns if 1 <= n <= n_valid_chunks]
    out_of_range_count = valid_count - len(in_range_ns)

    distinct_count = len(set(in_range_ns))
    in_range_count = len(in_range_ns)
    coverage_ratio = (
        distinct_count / in_range_count if in_range_count > 0 else 0.0
    )

    return {
        "total": valid_count + malformed_count,
        "valid": valid_count,
        "malformed": malformed_count,
        "in_range": in_range_count,
        "out_of_range": out_of_range_count,
        "distinct": distinct_count,
        "coverage_ratio": round(coverage_ratio, 3),
    }


__all__ = ["parse_citation_refs"]
