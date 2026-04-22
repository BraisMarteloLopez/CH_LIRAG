"""
Exportador plano para resultados de evaluacion.

Un run = un JSON con todo (run metadata + per-query details). El JSON es la
unica fuente de verdad; derivaciones tabulares (summary, detail) se obtienen
con jq / pandas sobre el JSON.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from shared.types import EvaluationRun

logger = logging.getLogger(__name__)


class RunExporter:
    """
    Exporta un EvaluationRun a JSON.

    Uso:
        exporter = RunExporter(output_dir=Path("./results"))
        exporter.export(run)
        # Genera: results/<run_id>.json
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, run: EvaluationRun) -> dict:
        """
        Exporta el run a JSON.

        Returns:
            Dict con path del archivo generado: {"json": Path}
        """
        paths = {"json": self.to_json(run)}
        logger.info(
            f"Run {run.run_id} exportado a {self.output_dir}: "
            f"{paths['json'].name}"
        )
        return paths

    def to_json(self, run: EvaluationRun, filename: Optional[str] = None) -> Path:
        """Exporta run completo (con query_results) a JSON."""
        fname = filename or f"{run.run_id}.json"
        path = self.output_dir / fname

        data = run.to_dict_full()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f"JSON exportado: {path}")
        return path


__all__ = ["RunExporter"]
