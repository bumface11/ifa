"""Tests for URL-based preset encoding and decoding."""

import pytest

from ifa.url_presets import decode_preset_url, encode_preset_url


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
