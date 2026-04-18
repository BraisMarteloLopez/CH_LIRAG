#!/usr/bin/env python3
"""
Entry point para sandbox MTEB.

Uso:
    python -m sandbox_mteb.run              # Run con config del .env
    python -m sandbox_mteb.run --dry-run    # Solo valida config
    python -m sandbox_mteb.run --env /path  # .env alternativo
    python -m sandbox_mteb.run -v           # Logging verbose (DEBUG)

La parametrizacion del run (queries, corpus, DEV_MODE, estrategia, etc.)
se controla exclusivamente via .env. Ver env.example.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Asegurar que el proyecto raiz esta en sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sandbox_mteb.config import MTEBConfig
from sandbox_mteb.evaluator import MTEBEvaluator
from shared.report import RunExporter
from shared.structured_logging import configure_logging


class _Tee:
    """Duplica writes a multiples streams (stream original + archivo log).

    Se usa para capturar stdout/stderr a un archivo manteniendo la salida
    en consola. Preserva `isatty()` del stream original para libs que
    cambian su comportamiento si detectan terminal.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return self._streams[0].isatty()


def _setup_console_capture(results_dir: Path, strategy: str) -> Path:
    """Captura stdout/stderr a archivo en results_dir (ademas de consola).

    Debe llamarse ANTES de setup_logging() para que el StreamHandler use
    el sys.stderr ya redirigido.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"console_log_{strategy.lower()}_{timestamp}.txt"
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    # DT-3: configure_logging lee LOG_FORMAT del entorno ("text" o "jsonl")
    configure_logging(level=level)
    # Reducir ruido de libs externas
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MTEB RAG Evaluation Sandbox"
    )
    # Default .env: buscar primero en el directorio del sandbox
    sandbox_dir = Path(__file__).resolve().parent
    default_env = str(sandbox_dir / ".env")

    parser.add_argument(
        "--env",
        default=default_env,
        help=f"Ruta al archivo .env (default: {default_env})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo validar config y mostrar resumen",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Logging verbose (DEBUG)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_ID",
        help="Reanudar un run previo desde su checkpoint (DTm-36)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Construir config (sin logging aun, errores tempranos via print)
    env_path = Path(args.env)
    if not env_path.exists():
        print(
            f"[ERROR] Archivo .env no encontrado: {env_path.resolve()}",
            file=sys.stderr,
        )
        print("[ERROR] Copiar .env.example a .env y completar valores.", file=sys.stderr)
        return 1

    try:
        config = MTEBConfig.from_env(str(env_path))
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    config.ensure_directories()

    # 2. Dry run? (no genera log file, solo muestra config en consola)
    if args.dry_run:
        setup_logging(args.verbose)
        print(config.summary())
        print("\n[DRY RUN] Config valida. No se ejecuta evaluacion.")
        return 0

    # 3. Captura de consola a archivo ANTES de setup_logging.
    # El StreamHandler del logging toma sys.stderr en __init__, asi que
    # debe redirigirse antes para que el log tambien quede en el archivo.
    log_path = _setup_console_capture(
        Path(config.storage.evaluation_results_dir),
        config.retrieval.strategy.name,
    )
    setup_logging(args.verbose)

    # 4. Mostrar resumen (ya capturado en el log file)
    print(config.summary())
    print(f"Console log: {log_path}")

    # 5. Ejecutar
    evaluator = MTEBEvaluator(config)
    run_result = evaluator.run(resume_run_id=args.resume)

    # 6. Exportar
    exporter = RunExporter(output_dir=config.storage.evaluation_results_dir)
    paths = exporter.export(run_result)

    print(f"\nResultados exportados:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
