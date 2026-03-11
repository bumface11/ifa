"""Local parameter preset persistence helpers for Streamlit."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

import orjson

JsonScalar: TypeAlias = str | int | float | bool | None
JsonMap: TypeAlias = dict[str, JsonScalar]

_PRESET_VERSION = 1
UTC_TZ = getattr(datetime, "UTC", timezone.utc)  # noqa: UP017


def _write_payload_atomic(target_path: Path, payload: dict[str, object]) -> None:
    """Write preset payload atomically to avoid partial-file corruption."""
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    temp_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    temp_path.replace(target_path)


def build_default_preset_name(now: datetime | None = None) -> str:
    """Build a friendly default preset name.

    Args:
        now: Optional timestamp override.

    Returns:
        Default preset display name.
    """
    current = datetime.now(UTC_TZ) if now is None else now
    return current.astimezone().strftime("Preset %Y-%m-%d %H-%M")


def sanitize_preset_filename(name: str) -> str:
    """Convert a display name into a safe filename stem.

    Args:
        name: User-provided preset display name.

    Returns:
        Sanitized filename stem.
    """
    stripped = name.strip()
    cleaned = "".join(
        character
        for character in stripped
        if character.isalnum() or character in {"-", "_", " "}
    )
    normalized = "_".join(cleaned.split())
    return normalized or "preset"


def list_preset_files(preset_dir: Path) -> list[Path]:
    """List saved preset files sorted by filename.

    Args:
        preset_dir: Directory where preset JSON files are stored.

    Returns:
        Sorted list of preset files.
    """
    if not preset_dir.exists():
        return []
    return sorted(preset_dir.glob("*.json"), key=lambda path: path.name.lower())


def save_preset(
    preset_dir: Path,
    preset_name: str,
    sidebar_state: JsonMap,
) -> Path:
    """Save sidebar parameters to a local JSON preset file.

    Args:
        preset_dir: Target directory for presets.
        preset_name: Display name selected by the user.
        sidebar_state: Serializable map of tracked sidebar widget values.

    Returns:
        Path to the written preset file.
    """
    preset_dir.mkdir(parents=True, exist_ok=True)
    file_stem = sanitize_preset_filename(preset_name)
    target_path = preset_dir / f"{file_stem}.json"

    payload = {
        "version": _PRESET_VERSION,
        "name": preset_name.strip() or "preset",
        "saved_at": datetime.now(UTC_TZ).isoformat(),
        "sidebar_state": sidebar_state,
    }
    _write_payload_atomic(target_path, payload)
    return target_path


def load_preset(preset_path: Path) -> tuple[str, JsonMap]:
    """Load preset display name and sidebar state from a JSON file.

    Args:
        preset_path: Path to a saved preset JSON file.

    Returns:
        Tuple of display name and sidebar state map.

    Raises:
        ValueError: If the file is missing required keys.
    """
    payload = orjson.loads(preset_path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError("Preset payload must be an object")

    if "name" not in payload or "sidebar_state" not in payload:
        raise ValueError("Preset file missing required keys: name/sidebar_state")

    name = str(payload["name"])
    state = payload["sidebar_state"]
    if not isinstance(state, dict):
        raise ValueError("Preset sidebar_state must be an object")

    normalized_state: JsonMap = {}
    for key, value in state.items():
        if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
            normalized_state[key] = value
        elif isinstance(key, str) and value is None:
            normalized_state[key] = None

    return name, normalized_state


def get_preset_saved_at(preset_path: Path) -> str | None:
    """Return ISO saved timestamp from preset payload, if present."""
    try:
        payload = orjson.loads(preset_path.read_bytes())
    except OSError:
        return None
    except orjson.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    saved_at = payload.get("saved_at")
    if isinstance(saved_at, str):
        return saved_at
    return None


def rename_preset(preset_path: Path, new_name: str) -> Path:
    """Rename a preset file and update its display name in the payload.

    Args:
        preset_path: Existing preset path.
        new_name: New display name.

    Returns:
        New file path after rename.
    """
    name, state = load_preset(preset_path)
    _ = name
    new_path = preset_path.with_name(f"{sanitize_preset_filename(new_name)}.json")
    if new_path != preset_path:
        preset_path.rename(new_path)
    save_preset(new_path.parent, new_name, state)
    return new_path


def delete_preset(preset_path: Path) -> None:
    """Delete a preset file if it exists."""
    if preset_path.exists():
        preset_path.unlink()
