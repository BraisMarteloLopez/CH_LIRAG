"""
Tests unitarios para sandbox_mteb/run.py (Audit Fase 3 — A3.1).

Cobertura:
  R1. parse_args defaults
  R2. parse_args con --dry-run, -v, --resume
  R3. main() con .env inexistente retorna 1
  R4. main() con config invalida retorna 1
  R5. main() dry-run retorna 0
  R6. setup_logging sets correct level
"""

import argparse
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sandbox_mteb.run import parse_args, setup_logging, main


class TestParseArgs:
    """Tests para parse_args()."""

    def test_defaults(self):
        with patch("sys.argv", ["run.py"]):
            args = parse_args()
        assert not args.dry_run
        assert not args.verbose
        assert args.resume is None
        assert args.env.endswith(".env")

    def test_dry_run(self):
        with patch("sys.argv", ["run.py", "--dry-run"]):
            args = parse_args()
        assert args.dry_run is True

    def test_verbose_short(self):
        with patch("sys.argv", ["run.py", "-v"]):
            args = parse_args()
        assert args.verbose is True

    def test_resume(self):
        with patch("sys.argv", ["run.py", "--resume", "run_12345"]):
            args = parse_args()
        assert args.resume == "run_12345"

    def test_custom_env(self):
        with patch("sys.argv", ["run.py", "--env", "/tmp/custom.env"]):
            args = parse_args()
        assert args.env == "/tmp/custom.env"


class TestSetupLogging:
    """Tests para setup_logging()."""

    def test_default_info(self):
        with patch("sandbox_mteb.run.configure_logging") as mock_conf:
            setup_logging(verbose=False)
            mock_conf.assert_called_once_with(level=logging.INFO)

    def test_verbose_debug(self):
        with patch("sandbox_mteb.run.configure_logging") as mock_conf:
            setup_logging(verbose=True)
            mock_conf.assert_called_once_with(level=logging.DEBUG)


class TestMain:
    """Tests para main()."""

    def test_missing_env_returns_1(self, tmp_path):
        fake_env = str(tmp_path / "nonexistent.env")
        with patch("sys.argv", ["run.py", "--env", fake_env]):
            result = main()
        assert result == 1

    def test_invalid_config_returns_1(self, tmp_path):
        # .env existe pero config es invalida (faltan campos obligatorios)
        env_file = tmp_path / ".env"
        env_file.write_text("RETRIEVAL_STRATEGY=INVALID_STRATEGY\n")
        with patch("sys.argv", ["run.py", "--env", str(env_file)]):
            with patch("sandbox_mteb.run.MTEBConfig.from_env") as mock_from_env:
                mock_config = MagicMock()
                mock_config.validate.return_value = ["campo obligatorio falta"]
                mock_from_env.return_value = mock_config
                result = main()
        assert result == 1

    def test_dry_run_returns_0(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# minimal\n")
        with patch("sys.argv", ["run.py", "--env", str(env_file), "--dry-run"]):
            with patch("sandbox_mteb.run.MTEBConfig.from_env") as mock_from_env:
                mock_config = MagicMock()
                mock_config.validate.return_value = []
                mock_config.summary.return_value = "Config summary"
                mock_from_env.return_value = mock_config
                result = main()
        assert result == 0
        mock_config.ensure_directories.assert_called_once()

    def test_full_run_exports(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# minimal\n")
        with patch("sys.argv", ["run.py", "--env", str(env_file)]):
            with patch("sandbox_mteb.run.MTEBConfig.from_env") as mock_from_env, \
                 patch("sandbox_mteb.run.MTEBEvaluator") as mock_eval_cls, \
                 patch("sandbox_mteb.run.RunExporter") as mock_exp_cls:
                mock_config = MagicMock()
                mock_config.validate.return_value = []
                mock_config.summary.return_value = "Config"
                mock_from_env.return_value = mock_config

                mock_evaluator = MagicMock()
                mock_evaluator.run.return_value = MagicMock()
                mock_eval_cls.return_value = mock_evaluator

                mock_exporter = MagicMock()
                mock_exporter.export.return_value = {"json": Path("/tmp/r.json")}
                mock_exp_cls.return_value = mock_exporter

                result = main()

        assert result == 0
        mock_evaluator.run.assert_called_once()
        mock_exporter.export.assert_called_once()
