"""URL-based preset system using compressed query parameters."""

from __future__ import annotations

import base64
import zlib
from typing import TypeAlias
from urllib.parse import urlencode, parse_qs, urlparse, urlunparse

import orjson

JsonScalar: TypeAlias = str | int | float | bool | None
JsonMap: TypeAlias = dict[str, JsonScalar]

_PRESET_PARAM = "preset"


def encode_preset_url(
    base_url: str, preset_state: JsonMap
) -> str:
    """Encode preset state into a shareable URL with compressed query parameter.

    Args:
        base_url: Base URL (e.g., "http://localhost:8501")
        preset_state: Serializable map of tracked sidebar widget values.

    Returns:
        Full URL with encoded preset as query parameter.
    """
    # Serialize to JSON and compress
    json_bytes = orjson.dumps(preset_state)
    compressed = zlib.compress(json_bytes, level=9)
    # Base64 encode for URL safety
    encoded = base64.urlsafe_b64encode(compressed).decode('ascii').rstrip('=')

    # Parse base URL and add query parameter
    parsed = urlparse(base_url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    params[_PRESET_PARAM] = [encoded]

    # Rebuild query string (urlencode handles the dict of lists)
    query = urlencode(params, doseq=True)
    new_parsed = parsed._replace(query=query)
    return urlunparse(new_parsed)


def decode_preset_url(url_query_string: str) -> JsonMap | None:
    """Decode preset state from URL query parameter.

    Args:
        url_query_string: The query string from the URL (e.g., "preset=abc123...")

    Returns:
        Decoded preset state dict, or None if not present/invalid.
    """
    if not url_query_string:
        return None

    try:
        params = parse_qs(url_query_string, keep_blank_values=True)
        encoded_list = params.get(_PRESET_PARAM, [])
        if not encoded_list:
            return None

        encoded = encoded_list[0]
        if not encoded:
            return None

        # Restore padding if needed
        padding = (4 - len(encoded) % 4) % 4
        encoded_padded = encoded + '=' * padding

        compressed = base64.urlsafe_b64decode(encoded_padded)
        json_bytes = zlib.decompress(compressed)
        payload = orjson.loads(json_bytes)

        if not isinstance(payload, dict):
            return None

        # Validate and normalize state
        normalized_state: JsonMap = {}
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(
                value, (str, int, float, bool)
            ):
                normalized_state[key] = value
            elif isinstance(key, str) and value is None:
                normalized_state[key] = None

        return normalized_state if normalized_state else None

    except (ValueError, TypeError, zlib.error, orjson.JSONDecodeError):
        return None


def get_current_page_url() -> str | None:
    """Get the current Streamlit page URL for sharing.

    Note: In Streamlit, this uses st.query_params to reconstruct the URL.
    Returns the full URL as users would see it in their browser.

    Returns:
        Full URL string, or None if unable to determine.
    """
    try:
        import streamlit as st

        # Get the base URL from Streamlit config
        base_url = st.query_params.get_all("") or ""
        if not base_url:
            # Fallback: reconstruct from window location (handled client-side)
            return None

        return base_url
    except ImportError:
        return None


def decode_comparison_presets(query_params: dict[str, str]) -> list[JsonMap]:
    """Decode multiple preset parameters from query params for comparison mode.

    Args:
        query_params: Dictionary from st.query_params (e.g. from st.query_params.to_dict())

    Returns:
        List of decoded preset states (1-5 presets), empty if none present.
    """
    presets: list[JsonMap] = []

    # Check for preset1, preset2, ..., preset5
    for i in range(1, 6):
        param_name = f"preset{i}"
        param_value = query_params.get(param_name, "")

        if not param_value:
            continue

        # Decode the preset value directly (it's already base64-encoded)
        try:
            padding = (4 - len(param_value) % 4) % 4
            param_value_padded = param_value + '=' * padding

            compressed = base64.urlsafe_b64decode(param_value_padded)
            json_bytes = zlib.decompress(compressed)
            payload = orjson.loads(json_bytes)

            if not isinstance(payload, dict):
                continue

            # Validate and normalize state
            normalized_state: JsonMap = {}
            for key, value in payload.items():
                if isinstance(key, str) and isinstance(
                    value, (str, int, float, bool)
                ):
                    normalized_state[key] = value
                elif isinstance(key, str) and value is None:
                    normalized_state[key] = None

            if normalized_state:
                presets.append(normalized_state)

        except (ValueError, TypeError, zlib.error, orjson.JSONDecodeError):
            continue

    return presets


def encode_comparison_url(
    base_url: str, presets: list[JsonMap]
) -> str:
    """Encode multiple presets into a single shareable URL.

    Args:
        base_url: Base URL (e.g., "http://localhost:8501")
        presets: List of preset states to encode (1-5 presets)

    Returns:
        Full URL with multiple preset parameters.
    """
    if not presets or len(presets) > 5:
        raise ValueError("Must provide 1-5 presets")

    parsed = urlparse(base_url)
    params = parse_qs(parsed.query, keep_blank_values=True)

    # Encode each preset
    for i, preset_state in enumerate(presets, start=1):
        param_name = f"preset{i}"
        json_bytes = orjson.dumps(preset_state)
        compressed = zlib.compress(json_bytes, level=9)
        encoded = base64.urlsafe_b64encode(compressed).decode('ascii').rstrip('=')
        params[param_name] = [encoded]

    query = urlencode(params, doseq=True)
    new_parsed = parsed._replace(query=query)
    return urlunparse(new_parsed)
