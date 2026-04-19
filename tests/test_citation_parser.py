"""
Tests para shared/citation_parser.py::parse_citation_refs.

Cobertura:
  CP1. Texto vacio / None -> todos los contadores a 0
  CP2. Un [ref:N] valido en rango
  CP3. [ref:N] fuera de rango (N > n_chunks)
  CP4. Multiples bien formados distintos
  CP5. Duplicados: mismo N varias veces -> distinct discrimina
  CP6. Mezcla valid + duplicados + out_of_range
  CP7. Solo malformed (Ref mayuscula, espacios, sin dos puntos, etc.)
  CP8. Mezcla valid + malformed
  CP9. n_valid_chunks=0 -> todas las refs validas van a out_of_range
  CP10. Texto sin nada parecido a ref
  CP11. [ref:0] (borde inferior fuera de rango)
"""

from shared.citation_parser import parse_citation_refs


# =============================================================================
# CP1: empty / None input
# =============================================================================


def test_empty_text_returns_zeroed_counters():
    result = parse_citation_refs("", n_valid_chunks=5)
    assert result == {
        "total": 0,
        "valid": 0,
        "malformed": 0,
        "in_range": 0,
        "out_of_range": 0,
        "distinct": 0,
        "coverage_ratio": 0.0,
    }


def test_none_text_returns_zeroed_counters():
    result = parse_citation_refs(None, n_valid_chunks=5)
    assert result["total"] == 0
    assert result["coverage_ratio"] == 0.0


# =============================================================================
# CP2: single valid in-range reference
# =============================================================================


def test_single_valid_in_range_reference():
    result = parse_citation_refs(
        "The answer is X [ref:1] end.", n_valid_chunks=5,
    )
    assert result["valid"] == 1
    assert result["in_range"] == 1
    assert result["out_of_range"] == 0
    assert result["distinct"] == 1
    assert result["coverage_ratio"] == 1.0
    assert result["total"] == 1
    assert result["malformed"] == 0


# =============================================================================
# CP3: out of range (N > n_chunks)
# =============================================================================


def test_out_of_range_reference_above_upper_bound():
    result = parse_citation_refs("Claim [ref:99]", n_valid_chunks=5)
    assert result["valid"] == 1
    assert result["in_range"] == 0
    assert result["out_of_range"] == 1
    assert result["distinct"] == 0
    assert result["coverage_ratio"] == 0.0


# =============================================================================
# CP4: multiple distinct valid
# =============================================================================


def test_multiple_distinct_valid_references():
    result = parse_citation_refs(
        "A [ref:1] B [ref:2] C [ref:3]", n_valid_chunks=5,
    )
    assert result["valid"] == 3
    assert result["in_range"] == 3
    assert result["distinct"] == 3
    assert result["coverage_ratio"] == 1.0


# =============================================================================
# CP5: duplicates on the same N
# =============================================================================


def test_duplicates_detected_by_distinct():
    result = parse_citation_refs(
        "[ref:1] [ref:1] [ref:1]", n_valid_chunks=5,
    )
    assert result["valid"] == 3
    assert result["in_range"] == 3
    assert result["distinct"] == 1
    assert result["coverage_ratio"] == round(1 / 3, 3)


# =============================================================================
# CP6: mezcla valid + duplicados + out_of_range
# =============================================================================


def test_mixed_valid_duplicates_and_out_of_range():
    result = parse_citation_refs(
        "[ref:1] [ref:2] [ref:1] [ref:99]", n_valid_chunks=5,
    )
    assert result["valid"] == 4
    assert result["in_range"] == 3  # 1, 2, 1
    assert result["out_of_range"] == 1  # 99
    assert result["distinct"] == 2  # {1, 2}
    assert result["coverage_ratio"] == round(2 / 3, 3)


# =============================================================================
# CP7: solo malformed
# =============================================================================


def test_malformed_variants():
    # Capitalizacion, espacios, sin "ref:", parentesis (parentesis NO matchea)
    text = "[Ref:3] [REF:4] [ref: 5] [ref:abc]"
    result = parse_citation_refs(text, n_valid_chunks=10)
    assert result["valid"] == 0
    # Los 4 matchean el superset pero no el formato estricto
    assert result["malformed"] == 4
    assert result["in_range"] == 0
    assert result["out_of_range"] == 0


def test_malformed_variants_parentheses_do_not_count():
    # Los parentesis (ref:3) no son reconocibles como variante de [ref:N]
    result = parse_citation_refs("(ref:3)", n_valid_chunks=5)
    assert result["total"] == 0
    assert result["malformed"] == 0


# =============================================================================
# CP8: mezcla valid + malformed
# =============================================================================


def test_mixed_valid_and_malformed():
    # 2 validos + 2 malformed = 4 total
    text = "Bien [ref:1] [ref:2] mal [Ref:3] [ref: 4]"
    result = parse_citation_refs(text, n_valid_chunks=10)
    assert result["valid"] == 2
    assert result["malformed"] == 2
    assert result["total"] == 4
    assert result["in_range"] == 2
    assert result["distinct"] == 2


# =============================================================================
# CP9: n_valid_chunks=0
# =============================================================================


def test_zero_chunks_all_valid_refs_become_out_of_range():
    result = parse_citation_refs("[ref:1] [ref:2]", n_valid_chunks=0)
    assert result["valid"] == 2
    assert result["in_range"] == 0
    assert result["out_of_range"] == 2
    assert result["distinct"] == 0
    assert result["coverage_ratio"] == 0.0


# =============================================================================
# CP10: texto sin ninguna cita
# =============================================================================


def test_text_without_any_citation():
    result = parse_citation_refs(
        "This is plain prose without any reference.", n_valid_chunks=5,
    )
    assert result["total"] == 0
    assert result["malformed"] == 0


# =============================================================================
# CP11: [ref:0] borde inferior fuera de rango
# =============================================================================


def test_ref_zero_is_out_of_range():
    # El rango es [1, n_valid_chunks], asi que 0 queda fuera.
    result = parse_citation_refs("[ref:0]", n_valid_chunks=5)
    assert result["valid"] == 1
    assert result["in_range"] == 0
    assert result["out_of_range"] == 1
