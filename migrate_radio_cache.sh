#!/usr/bin/env bash
# migrate_radio_cache.sh
#
# Extracts the radio cache code from the ifa repo git history and pushes it to
# the bumface11/radiocache repository.
#
# Usage:
#   cd <ifa-repo-directory>
#   bash migrate_radio_cache.sh
#
# Prerequisites:
#   - Git must be configured with credentials that can push to
#     https://github.com/bumface11/radiocache
#   - Run from inside the ifa repository root
set -euo pipefail

SOURCE_COMMIT="cf83c68"   # latest radio-cache state (with code-review fixes)
FALLBACK_COMMIT="b734f91" # original feat commit (files not touched by fix)
DEST_DIR="$(mktemp -d)"
DEST_REPO="https://github.com/bumface11/radiocache.git"

echo "Extracting radio cache files to $DEST_DIR ..."

FILES=(
  "radio_cache/__init__.py"
  "radio_cache/bbc_feed_parser.py"
  "radio_cache/cache_db.py"
  "radio_cache/models.py"
  "radio_cache/refresh.py"
  "radio_cache/search.py"
  "radio_cache_api.py"
  "static/radio_cache/style.css"
  "templates/radio_cache/base.html"
  "templates/radio_cache/brand_detail.html"
  "templates/radio_cache/brand_list.html"
  "templates/radio_cache/index.html"
  "templates/radio_cache/search_results.html"
  "templates/radio_cache/series_detail.html"
  "templates/radio_cache/series_list.html"
  "tests/test_radio_cache_db.py"
  "tests/test_radio_feed_parser.py"
  "tests/test_radio_models.py"
  "tests/test_radio_refresh.py"
  "tests/test_radio_search.py"
  ".github/workflows/refresh-radio-cache.yml"
)

for f in "${FILES[@]}"; do
  mkdir -p "$DEST_DIR/$(dirname "$f")"
  if ! git show "$SOURCE_COMMIT":"$f" > "$DEST_DIR/$f" 2>/dev/null; then
    git show "$FALLBACK_COMMIT":"$f" > "$DEST_DIR/$f"
  fi
done

# ── Create standalone project files ──────────────────────────────────────────

cat > "$DEST_DIR/.gitignore" << 'GITIGNORE'
.venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/
__pycache__/
*.egg-info/
radio_cache.db
*.db-journal
GITIGNORE

cat > "$DEST_DIR/pyproject.toml" << 'PYPROJECT'
[project]
name = "radiocache"
version = "0.1.0"
description = "Cloud-hostable cache of BBC Radio drama programme metadata with search"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "jinja2>=3.1",
    "uvicorn[standard]>=0.34",
]

[tool.setuptools.packages.find]
include = ["radio_cache*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
warn_unused_configs = true
check_untyped_defs = true
disallow_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
show_error_codes = true
files = ["radio_cache", "radio_cache_api.py", "tests"]
PYPROJECT

cat > "$DEST_DIR/README.md" << 'README'
# BBC Radio Drama Cache

A cloud-hostable cache of BBC Radio drama programme metadata with a modern
search interface.

## Features

- **Daily refresh** from BBC Sounds feeds via GitHub Actions.
- **Full-text search** across titles, synopses, series, and categories.
- **Series grouping** -- episodes are grouped by parent series, with episode
  numbering for serialisations.
- **Brand hierarchy** -- series are grouped under their parent brands.
- **REST API** (FastAPI) for programmatic access at `/api/search`,
  `/api/series`, `/api/programme/{pid}`, and `/api/stats`.
- **Web UI** for searching and browsing, with one-click `get_iplayer` command
  copying for local downloads.
- **Static JSON export** (`radio_cache_export.json`) for cheap static hosting
  on GitHub Pages or similar.

## Quick Start

```bash
# Install in editable mode
pip install -e .

# Refresh the cache (fetches from BBC feeds)
python -m radio_cache.refresh --verbose

# Or import from a JSON export
python -m radio_cache.refresh --import-json radio_cache_export.json

# Start the web search UI
uvicorn radio_cache_api:app --reload
```

Open `http://localhost:8000` to search and browse programmes.  Each programme
shows a copyable `get_iplayer` command for local download.

## Hosting Options (Cheap/Free)

| Option | Cost | Notes |
|---|---|---|
| **Render free tier** | Free | Deploy `radio_cache_api.py`; spins down on idle |
| **Fly.io** | Free tier | 3 shared VMs free |
| **Railway** | Free trial | Simple Docker deploy |
| **GitHub Pages** | Free | Host `radio_cache_export.json` as static file |
| **GitHub Actions** | Free | Daily cache refresh via cron workflow |

## Downloading Programmes

Find a programme in the web UI or JSON export, then use:

```bash
get_iplayer --pid=<PID> --type=radio
```

Requires [get_iplayer](https://github.com/get-iplayer/get_iplayer) installed
locally.

## Running Tests

```bash
pip install -e .
pip install pytest
python -m pytest
```

## Project Structure

- `radio_cache/` -- core Python package (models, database, parser, search)
- `radio_cache_api.py` -- FastAPI web application
- `templates/radio_cache/` -- Jinja2 HTML templates
- `static/radio_cache/` -- CSS styles
- `tests/` -- unit tests (54 tests)
- `.github/workflows/refresh-radio-cache.yml` -- daily cache refresh cron job
README

# ── Initialize git and push ─────────────────────────────────────────────────

cd "$DEST_DIR"
git init -b main
git add -A
git commit -m "feat: BBC Radio Drama cloud cache with search, API, and daily refresh

Standalone radio cache package extracted from bumface11/ifa.

- SQLite-backed programme cache with FTS5 full-text search
- BBC Sounds feed parser for radio drama metadata
- Search with series grouping and brand hierarchy
- FastAPI web API and HTML search UI
- Cache refresh CLI with JSON export/import
- GitHub Actions daily cron workflow
- 54 unit tests covering all modules"

git remote add origin "$DEST_REPO"

echo ""
echo "Ready to push. Running: git push -u origin main"
git push -u origin main

echo ""
echo "Done! Radio cache code is now at: $DEST_REPO"
echo "Temp directory: $DEST_DIR"
