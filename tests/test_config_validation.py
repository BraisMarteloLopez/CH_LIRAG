"""
Tests: InfraConfig.validate() y RerankerConfig.validate()

Produccion: shared/config_base.py (lineas 110-161)

Verifica que configuraciones invalidas se rechazan con mensajes claros.
Tests puros, sin mocks — construccion directa de dataclasses.

Coverage IDs:
  CV1: InfraConfig valida completa -> sin errores
  CV2: llm_base_url vacia -> error
  CV3: llm_model_name vacia -> error
  CV4: embedding_base_url vacia -> error
  CV5: embedding_model_type invalido -> error
  CV6: nim_max_concurrent=0 (bajo rango) -> error
  CV7: nim_max_concurrent=200 (sobre rango) -> error
  CV8: Multiples errores acumulados -> todos reportados
  CV9: RerankerConfig enabled sin base_url -> error
  CV10: RerankerConfig enabled sin model_name -> error
  CV11: RerankerConfig disabled con campos vacios -> sin errores
"""

from shared.config_base import InfraConfig, RerankerConfig


def _valid_infra(**overrides):
    """InfraConfig valida base. Overrides para inyectar campos invalidos."""
    defaults = dict(
        llm_base_url="http://llm:8000",
        llm_model_name="meta/llama3",
        embedding_base_url="http://embed:8000",
        embedding_model_name="nvidia/nv-embedqa-e5-v5",
        embedding_model_type="symmetric",
        nim_max_concurrent=32,
    )
    defaults.update(overrides)
    return InfraConfig(**defaults)


# ---------------------------------------------------------------------------
# CV1: Config valida -> sin errores
# ---------------------------------------------------------------------------
def test_infra_valid_no_errors():
    """CV1: fully valid InfraConfig -> empty error list."""
    errors = _valid_infra().validate()
    assert errors == []


# ---------------------------------------------------------------------------
# CV2: llm_base_url vacia
# ---------------------------------------------------------------------------
def test_infra_missing_llm_base_url():
    """CV2: empty llm_base_url -> error."""
    errors = _valid_infra(llm_base_url="").validate()
    assert len(errors) >= 1
    assert any("LLM_BASE_URL" in e for e in errors)


# ---------------------------------------------------------------------------
# CV3: llm_model_name vacia
# ---------------------------------------------------------------------------
def test_infra_missing_llm_model_name():
    """CV3: empty llm_model_name -> error."""
    errors = _valid_infra(llm_model_name="").validate()
    assert any("LLM_MODEL_NAME" in e for e in errors)


# ---------------------------------------------------------------------------
# CV4: embedding_base_url vacia
# ---------------------------------------------------------------------------
def test_infra_missing_embedding_url():
    """CV4: empty embedding_base_url -> error about embedding."""
    errors = _valid_infra(embedding_base_url="").validate()
    assert any("EMBEDDING" in e for e in errors)


# ---------------------------------------------------------------------------
# CV5: embedding_model_type invalido
# ---------------------------------------------------------------------------
def test_infra_invalid_embedding_model_type():
    """CV5: invalid embedding_model_type -> error."""
    errors = _valid_infra(embedding_model_type="invalid_type").validate()
    assert any("EMBEDDING_MODEL_TYPE" in e for e in errors)
    assert any("invalid_type" in e for e in errors)


# ---------------------------------------------------------------------------
# CV6: nim_max_concurrent bajo rango
# ---------------------------------------------------------------------------
def test_infra_concurrent_below_range():
    """CV6: nim_max_concurrent=0 -> error (min is 1)."""
    errors = _valid_infra(nim_max_concurrent=0).validate()
    assert any("NIM_MAX_CONCURRENT" in e for e in errors)


# ---------------------------------------------------------------------------
# CV7: nim_max_concurrent sobre rango
# ---------------------------------------------------------------------------
def test_infra_concurrent_above_range():
    """CV7: nim_max_concurrent=200 -> error (max is 128)."""
    errors = _valid_infra(nim_max_concurrent=200).validate()
    assert any("NIM_MAX_CONCURRENT" in e for e in errors)


# ---------------------------------------------------------------------------
# CV8: Multiples errores acumulados
# ---------------------------------------------------------------------------
def test_infra_multiple_errors_all_reported():
    """CV8: multiple invalid fields -> all errors reported."""
    config = InfraConfig(
        llm_base_url="",
        llm_model_name="",
        embedding_base_url="",
        embedding_model_name="",
        embedding_model_type="wrong",
        nim_max_concurrent=0,
    )
    errors = config.validate()
    # Expect at least 4 errors: llm_url, llm_model, embedding, model_type, concurrent
    assert len(errors) >= 4


# ---------------------------------------------------------------------------
# CV9: RerankerConfig enabled sin base_url
# ---------------------------------------------------------------------------
def test_reranker_enabled_missing_base_url():
    """CV9: reranker enabled without base_url -> error."""
    config = RerankerConfig(enabled=True, base_url="", model_name="nvidia/rerank")
    errors = config.validate()
    assert any("base_url" in e or "model_name" in e for e in errors)


# ---------------------------------------------------------------------------
# CV10: RerankerConfig enabled sin model_name
# ---------------------------------------------------------------------------
def test_reranker_enabled_missing_model_name():
    """CV10: reranker enabled without model_name -> error."""
    config = RerankerConfig(enabled=True, base_url="http://rerank:8000", model_name="")
    errors = config.validate()
    assert any("base_url" in e or "model_name" in e for e in errors)


# ---------------------------------------------------------------------------
# CV11: RerankerConfig disabled -> no valida campos
# ---------------------------------------------------------------------------
def test_reranker_disabled_no_validation():
    """CV11: reranker disabled with empty fields -> no errors."""
    config = RerankerConfig(enabled=False, base_url="", model_name="")
    errors = config.validate()
    assert errors == []
