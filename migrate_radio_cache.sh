#!/usr/bin/env bash
# migrate_radio_cache.sh
#
# Extracts the radio cache code from the ifa repo git history and pushes
# it to the bumface11/radiocache repository.
#
# Usage (run from the ifa repo root):
#
#   git fetch --unshallow origin   # only needed if you have a shallow clone
#   bash migrate_radio_cache.sh
#
# The script uses your normal git credentials (SSH key or credential
# manager).  Make sure you can push to bumface11/radiocache before running.
set -euo pipefail

SOURCE_COMMIT="cf83c68"   # radio-cache state with code-review fixes
FALLBACK_COMMIT="b734f91" # original feat commit (files not touched by fix)
DEST_DIR="$(mktemp -d)"
DEST_REPO="https://github.com/bumface11/radiocache.git"
IFA_DIR="$(pwd)"

# ── Preflight checks ────────────────────────────────────────────────────────

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: Not inside a git repository.  Run this from the ifa repo root."
  exit 1
fi

for sha in "$SOURCE_COMMIT" "$FALLBACK_COMMIT"; do
  if ! git cat-file -e "$sha" 2>/dev/null; then
    echo "ERROR: Commit $sha not found in history."
    echo "If this is a shallow clone, run:  git fetch --unshallow origin"
    exit 1
  fi
done

# ── Extract source files from git history ────────────────────────────────────
# Files that exist in the working tree are preferred (they may contain fixes
# applied after the original commits).  Fall back to git history otherwise.

echo "Extracting radio cache files to $DEST_DIR ..."

FILES=(
  radio_cache/__init__.py
  radio_cache/bbc_feed_parser.py
  radio_cache/cache_db.py
  radio_cache/models.py
  radio_cache/refresh.py
  radio_cache/search.py
  radio_cache_api.py
  static/radio_cache/style.css
  templates/radio_cache/base.html
  templates/radio_cache/brand_detail.html
  templates/radio_cache/brand_list.html
  templates/radio_cache/index.html
  templates/radio_cache/search_results.html
  templates/radio_cache/series_detail.html
  templates/radio_cache/series_list.html
  tests/test_radio_cache_db.py
  tests/test_radio_feed_parser.py
  tests/test_radio_models.py
  tests/test_radio_refresh.py
  tests/test_radio_search.py
  .github/workflows/refresh-radio-cache.yml
)

for f in "${FILES[@]}"; do
  mkdir -p "$DEST_DIR/$(dirname "$f")"
  # Prefer the working-tree copy (may contain fixes)
  if [ -f "$IFA_DIR/$f" ]; then
    cp "$IFA_DIR/$f" "$DEST_DIR/$f"
  elif ! git show "$SOURCE_COMMIT:$f" > "$DEST_DIR/$f" 2>/dev/null; then
    if ! git show "$FALLBACK_COMMIT:$f" > "$DEST_DIR/$f" 2>/dev/null; then
      echo "ERROR: Could not find $f in working tree or source commits."
      exit 1
    fi
  fi
done

echo "  Extracted ${#FILES[@]} files."

# ── Create standalone project files ──────────────────────────────────────────

cat > "$DEST_DIR/.gitignore" << 'EOF'
.venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/
__pycache__/
*.egg-info/
radio_cache.db
*.db-journal
EOF

cat > "$DEST_DIR/pyproject.toml" << 'EOF'
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
EOF

cat > "$DEST_DIR/README.md" << 'EOF'
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
pip install -e .

# Refresh the cache (fetches from BBC feeds)
python -m radio_cache.refresh --verbose

# Or import from a JSON export
python -m radio_cache.refresh --import-json radio_cache_export.json

# Start the web search UI
uvicorn radio_cache_api:app --reload
```

Open <http://localhost:8000> to search and browse programmes.  Each programme
shows a copyable `get_iplayer` command for local download.

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
EOF

# ── Commit and push ─────────────────────────────────────────────────────────

cd "$DEST_DIR"
git init -b main

# Use the caller's global git identity; fall back to a sensible default
git config user.name  "$(git config --global user.name  2>/dev/null || echo 'radiocache-migrator')"
git config user.email "$(git config --global user.email 2>/dev/null || echo 'noreply@github.com')"

git add -A
git commit -q -m "feat: BBC Radio Drama cloud cache with search, API, and daily refresh

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
echo "Pushing to $DEST_REPO ..."
if git push -u origin main 2>&1; then
  echo ""
  echo "Done!  Radio cache code is now at: $DEST_REPO"
  echo "You can delete the temp directory: rm -rf $DEST_DIR"
else
  echo ""
  echo "Push failed.  The prepared repo is ready at:"
  echo "  $DEST_DIR"
  echo ""
  echo "You can push manually:"
  echo "  cd $DEST_DIR && git push -u origin main"
  echo ""
  echo "Common fixes:"
  echo "  - Make sure https://github.com/bumface11/radiocache exists (create it on GitHub first)"
  echo "  - Check your git credentials (try: git ls-remote $DEST_REPO)"
  exit 1
fi
