"""
Tests: semantic_similarity() en shared/metrics.py (lineas 299-354)

Verifica calculo coseno con normalizacion [-1,1] -> [0,1], manejo de
vectores cero, inputs vacios, y errores del embedding model.

Coverage IDs:
  SS1: Vectores identicos -> cosine=1.0, normalized=1.0
  SS2: Vectores opuestos -> cosine=-1.0, normalized=0.0
  SS3: Vectores ortogonales -> cosine=0.0, normalized=0.5
  SS4: Vector cero -> retorna normalized=0.5 sin NaN (linea 330-331)
  SS5: Input vacio -> early return con reason=empty_input
  SS6: Excepcion del embedding model -> MetricResult con error
  SS7: numpy no disponible -> error descriptivo
"""

from unittest.mock import MagicMock, patch

import pytest

from shared.types import MetricType

# numpy puede no estar instalado en el entorno de test.
try:
    import numpy as _np
    _has_real_numpy = True
except ImportError:
    _has_real_numpy = False
    _np = None

from shared.metrics import semantic_similarity

needs_numpy = pytest.mark.skipif(
    not _has_real_numpy,
    reason="numpy requerido para calculo coseno",
)


def _mock_embedding(*vectors):
    """Crea mock de embedding_model que retorna vectores en secuencia."""
    model = MagicMock()
    model.embed_query = MagicMock(side_effect=list(vectors))
    return model


# ---------------------------------------------------------------------------
# SS1: Vectores identicos
# ---------------------------------------------------------------------------
@needs_numpy
def test_identical_vectors():
    """SS1: identical embeddings -> cosine=1.0, normalized=1.0."""
    model = _mock_embedding([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    result = semantic_similarity("text a", "text b", model)
    assert result.metric_type == MetricType.SEMANTIC_SIMILARITY
    assert result.value == pytest.approx(1.0)
    assert result.details["raw_cosine_similarity"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SS2: Vectores opuestos
# ---------------------------------------------------------------------------
@needs_numpy
def test_opposite_vectors():
    """SS2: opposite embeddings -> cosine=-1.0, normalized=0.0."""
    model = _mock_embedding([1.0, 0.0], [-1.0, 0.0])
    result = semantic_similarity("a", "b", model)
    assert result.value == pytest.approx(0.0)
    assert result.details["raw_cosine_similarity"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# SS3: Vectores ortogonales
# ---------------------------------------------------------------------------
@needs_numpy
def test_orthogonal_vectors():
    """SS3: orthogonal embeddings -> cosine=0.0, normalized=0.5."""
    model = _mock_embedding([1.0, 0.0], [0.0, 1.0])
    result = semantic_similarity("a", "b", model)
    assert result.value == pytest.approx(0.5)
    assert result.details["raw_cosine_similarity"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SS4: Vector cero
# ---------------------------------------------------------------------------
@needs_numpy
def test_zero_vector_no_nan():
    """SS4: zero vector -> cosine_sim=0.0, normalized=0.5 (no NaN)."""
    model = _mock_embedding([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    result = semantic_similarity("a", "b", model)
    # Line 330-331: norm_gen == 0 -> cosine_sim = 0.0 -> normalized = 0.5
    assert result.value == pytest.approx(0.5)
    assert result.error is None


# ---------------------------------------------------------------------------
# SS5: Input vacio
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gen,exp", [
    ("", "some text"),
    ("some text", ""),
    ("", ""),
])
def test_empty_input_early_return(gen, exp):
    """SS5: empty generated or expected -> value=0.0, reason=empty_input."""
    model = MagicMock()
    result = semantic_similarity(gen, exp, model)
    assert result.value == 0.0
    assert result.details["reason"] == "empty_input"
    model.embed_query.assert_not_called()


# ---------------------------------------------------------------------------
# SS6: Excepcion del embedding model
# ---------------------------------------------------------------------------
@needs_numpy
def test_embedding_error_returns_metric_with_error():
    """SS6: embedding raises -> MetricResult with error string, value=0.0."""
    model = MagicMock()
    model.embed_query.side_effect = RuntimeError("NIM timeout")
    result = semantic_similarity("text a", "text b", model)
    assert result.value == 0.0
    assert "NIM timeout" in result.error
    assert result.metric_type == MetricType.SEMANTIC_SIMILARITY


# ---------------------------------------------------------------------------
# SS7: numpy no disponible
# ---------------------------------------------------------------------------
def test_no_numpy_returns_error():
    """SS7: HAS_NUMPY=False -> error message about numpy."""
    model = MagicMock()
    with patch("shared.metrics.HAS_NUMPY", False):
        result = semantic_similarity("text a", "text b", model)
    assert result.value == 0.0
    assert "numpy" in result.error
    model.embed_query.assert_not_called()
