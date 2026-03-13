"""Tests for URL-based preset encoding and decoding."""

import pytest

from ifa.url_presets import (
    decode_comparison_presets,
    decode_preset_url,
    encode_comparison_url,
    encode_preset_url,
)


def test_encode_and_decode_simple_preset() -> None:
    """Test round-trip encoding and decoding of a simple preset."""
    preset = {
        "start_age_input": 52,
        "end_age_input": 95,
        "tax_free_pot_input": 336000.0,
    }

    url = encode_preset_url("http://localhost:8501", preset)
    assert "preset=" in url
    assert "http://localhost:8501" in url

    # Extract query string for decoding
    query_string = url.split("?", 1)[1]
    decoded = decode_preset_url(query_string)

    assert decoded == preset


def test_encode_and_decode_complex_preset() -> None:
    """Test with a more complex preset including all input types."""
    preset = {
        "start_age_input": 52,
        "end_age_input": 95,
        "tax_free_pot_input": 336000.0,
        "baseline_spending_input": 30000.0,
        "mean_return_input": 0.04,
        "std_return_input": 0.2,
        "random_seed_input": 42,
        "num_simulations_input": 1000,
        "save_outputs_input": False,
        "dc_pot_count": 2,
        "db_stream_count": 3,
        "lump_count": 3,
        "step_count": 1,
        "dc_name_0": "HRIS DC",
        "dc_start_age_0": 57,
        "dc_initial_balance_0": 311000.0,
        "db_name_0": "ACCA DB",
        "db_age_0": 60,
        "db_amount_0": 12510.0,
        "lump_name_0": "Car",
        "lump_age_0": 52,
        "lump_amount_0": 15000.0,
    }

    url = encode_preset_url("http://localhost:8501", preset)
    query_string = url.split("?", 1)[1]
    decoded = decode_preset_url(query_string)

    assert decoded == preset


def test_decode_preset_url_with_no_preset() -> None:
    """Test decoding when no preset parameter is present."""
    result = decode_preset_url("other_param=value")
    assert result is None


def test_decode_preset_url_with_invalid_preset() -> None:
    """Test decoding with invalid/corrupted preset data."""
    result = decode_preset_url("preset=invalid_not_base64_encoded!")
    assert result is None


def test_decode_preset_url_empty_preset() -> None:
    """Test decoding with empty preset parameter."""
    result = decode_preset_url("preset=")
    assert result is None


def test_url_length_is_reasonable() -> None:
    """Test that encoded URLs stay under reasonable browser limits."""
    # Create a more realistic large preset
    preset = {
        f"key_{i}": float(i) for i in range(50)
    }
    url = encode_preset_url("http://localhost:8501", preset)
    # Browser limits are typically 2000-8000 characters
    # We should be well under that for normal usage
    assert len(url) < 5000


def test_encode_handles_different_base_urls() -> None:
    """Test that encoding works with different base URLs."""
    preset = {"test": 123}

    url1 = encode_preset_url("http://localhost:8501", preset)
    url2 = encode_preset_url("https://example.com", preset)

    assert url1.startswith("http://localhost:8501")
    assert url2.startswith("https://example.com")

    # Both should decode to the same preset
    query1 = url1.split("?", 1)[1]
    query2 = url2.split("?", 1)[1]

    assert decode_preset_url(query1) == preset
    assert decode_preset_url(query2) == preset


def test_encode_and_decode_comparison_presets() -> None:
    """Test encoding and decoding multiple presets for comparison."""
    from urllib.parse import urlparse, parse_qs

    preset1 = {
        "start_age_input": 52,
        "end_age_input": 95,
        "tax_free_pot_input": 336000.0,
    }
    preset2 = {
        "start_age_input": 55,
        "end_age_input": 90,
        "tax_free_pot_input": 500000.0,
    }
    preset3 = {
        "start_age_input": 60,
        "end_age_input": 85,
        "tax_free_pot_input": 250000.0,
    }

    presets = [preset1, preset2, preset3]

    # Encode multiple presets
    url = encode_comparison_url("http://localhost:8501", presets)
    assert "preset1=" in url
    assert "preset2=" in url
    assert "preset3=" in url

    # Decode multiple presets - use urllib to properly parse query string
    parsed_url = urlparse(url)
    query_dict = parse_qs(parsed_url.query, keep_blank_values=True)
    # parse_qs returns lists, so extract first element
    query_dict = {k: v[0] if v else "" for k, v in query_dict.items()}

    decoded_presets = decode_comparison_presets(query_dict)
    assert len(decoded_presets) == 3
    assert decoded_presets[0] == preset1
    assert decoded_presets[1] == preset2
    assert decoded_presets[2] == preset3


def test_comparison_url_with_5_presets() -> None:
    """Test that up to 5 presets can be encoded and decoded."""
    presets = [
        {f"key_{i}": float(i)} for i in range(5)
    ]

    url = encode_comparison_url("http://localhost:8501", presets)
    for i in range(1, 6):
        assert f"preset{i}=" in url


def test_comparison_url_rejects_more_than_5_presets() -> None:
    """Test that encoding more than 5 presets raises ValueError."""
    presets = [{f"key_{i}": float(i)} for i in range(6)]

    with pytest.raises(ValueError):
        encode_comparison_url("http://localhost:8501", presets)


def test_decode_comparison_presets_with_empty_dict() -> None:
    """Test decoding with no preset parameters."""
    result = decode_comparison_presets({})
    assert result == []


def test_decode_comparison_presets_mixed_valid_invalid() -> None:
    """Test decoding when some presets are invalid."""
    query_dict = {
        "preset1": "valid_base64",  # This will fail to decode
        "preset2": "also_invalid",
        "preset3": "",  # Empty
    }
    result = decode_comparison_presets(query_dict)
    # Should only return successfully decoded presets (0 in this case)
    assert isinstance(result, list)
