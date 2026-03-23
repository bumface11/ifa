# BBC Feed Parser: Fix broken API endpoint (HTTP 404)

## Problem

The cache refresh fails to fetch any programmes — every BBC API request
returns `HTTP Error 404: Not Found`. The cache ends up empty (0 programmes).

**Error log:**

```
ERROR radio_cache.bbc_feed_parser: Failed to fetch
  https://rms.api.bbc.co.uk/v2/experience/inline/categories/classic-drama/programmes?sort=date&page=1:
  HTTP Error 404: Not Found
INFO  radio_cache.bbc_feed_parser: Fetched 0 unique programmes
INFO  __main__: Cache contains 0 programmes
```

## Root Cause

`radio_cache/bbc_feed_parser.py` constructs URLs using the endpoint:

```
https://rms.api.bbc.co.uk/v2/experience/inline/categories/{slug}/programmes?sort=date&page={page}
```

**This endpoint does not exist in the BBC Sounds API.** It returns 404 for
every category slug.

## Correct Endpoint

The working endpoint (confirmed against the
[BBC Sounds OpenAPI spec](https://github.com/kieranhogg/auntie-sounds)
and the actively-maintained
[LMS BBC Sounds Plugin](https://github.com/expectingtofly/LMS_BBC_Sounds_Plugin/blob/master/BBCSounds/BBCSoundsFeeder.pm#L528))
is:

```
https://rms.api.bbc.co.uk/v2/programmes/playable?category={slug}&sort=date&tleoDistinct=true&offset={n}&limit={n}
```

Key differences:
- **Path**: `/v2/programmes/playable` (not `/v2/experience/inline/categories/...`)
- **Category**: passed as `?category={slug}` query parameter
- **Pagination**: uses `offset` + `limit` (not `page`)
- **De-duplication**: `tleoDistinct=true` filters to unique top-level editorial objects

The response format (`PlayableItemsPaginatedResponse`) returns `{ "data": [...], "total": N, "offset": N, "limit": N }` — the `data` array contains the same `PlayableItem` objects that `_parse_programme_item()` already handles, so no parser changes are needed.

## Required Changes

### 1. `radio_cache/bbc_feed_parser.py`

Replace the API base URL constant:

```python
# BEFORE (broken)
_BBC_SOUNDS_API: Final[str] = (
    "https://rms.api.bbc.co.uk/v2/experience/inline/categories"
)

# AFTER (working)
_BBC_PLAYABLE_API: Final[str] = (
    "https://rms.api.bbc.co.uk/v2/programmes/playable"
)
```

Add a page-size constant (the BBC API default and maximum is 30):

```python
_PAGE_LIMIT: Final[int] = 30
```

Remove the unused constants `_BBC_SOUNDS_PLAY` and `_DRAMA_CATEGORY`.

Update `_USER_AGENT` to reference `bumface11/radiocache` instead of
`bumface11/ifa`.

Rewrite `fetch_drama_programmes()` to use offset-based pagination:

```python
def fetch_drama_programmes(
    category_slugs: list[str] | None = None,
    max_pages: int = 10,
    delay: float = _REQUEST_DELAY_SECS,
) -> list[Programme]:
    slugs = category_slugs or _CATEGORY_SLUGS
    programmes: list[Programme] = []
    seen_pids: set[str] = set()

    for slug in slugs:
        logger.info("Fetching category: %s", slug)
        for page in range(max_pages):
            offset = page * _PAGE_LIMIT
            url = (
                f"{_BBC_PLAYABLE_API}?category={slug}"
                f"&sort=date&tleoDistinct=true"
                f"&offset={offset}&limit={_PAGE_LIMIT}"
            )
            data = _fetch_json(url)
            if data is None:
                break

            items = data.get("data") or []
            if not items:
                break

            for item in items:
                prog = _parse_programme_item(item)
                if prog is not None and prog.pid not in seen_pids:
                    seen_pids.add(prog.pid)
                    programmes.append(prog)

            logger.info(
                "  page %d: %d items (total %d)",
                page + 1, len(items), len(programmes),
            )

            total = data.get("total", 0)
            if offset + len(items) >= total or len(items) < _PAGE_LIMIT:
                break

            time.sleep(delay)
        time.sleep(delay)

    logger.info("Fetched %d unique programmes", len(programmes))
    return programmes
```

### 2. `tests/test_radio_feed_parser.py`

Add tests that validate the corrected URL construction (`_PAGE_LIMIT = 30`
is the constant defined in the main module):

```python
from unittest.mock import patch
from radio_cache.bbc_feed_parser import _PAGE_LIMIT, fetch_drama_programmes

class TestFetchDramaProgrammes:
    @patch("radio_cache.bbc_feed_parser._fetch_json")
    def test_uses_playable_endpoint(self, mock_fetch):
        mock_fetch.return_value = {"data": [], "total": 0}
        fetch_drama_programmes(category_slugs=["drama"], max_pages=1, delay=0)
        url = mock_fetch.call_args[0][0]
        assert "/v2/programmes/playable?" in url
        assert "category=drama" in url
        assert "inline/categories" not in url

    @patch("radio_cache.bbc_feed_parser._fetch_json")
    def test_pagination_uses_offset(self, mock_fetch):
        items = [{"id": f"p{i:04d}", "title": f"P{i}"} for i in range(_PAGE_LIMIT)]
        mock_fetch.side_effect = [
            {"data": items, "total": _PAGE_LIMIT + 5},
            {"data": [{"id": "p9999", "title": "Last"}], "total": _PAGE_LIMIT + 5},
        ]
        fetch_drama_programmes(category_slugs=["drama"], max_pages=5, delay=0)
        urls = [c[0][0] for c in mock_fetch.call_args_list]
        assert "offset=0" in urls[0]
        assert f"offset={_PAGE_LIMIT}" in urls[1]

    @patch("radio_cache.bbc_feed_parser._fetch_json")
    def test_includes_tleo_distinct(self, mock_fetch):
        mock_fetch.return_value = {"data": [], "total": 0}
        fetch_drama_programmes(category_slugs=["drama"], max_pages=1, delay=0)
        url = mock_fetch.call_args[0][0]
        assert "tleoDistinct=true" in url

    @patch("radio_cache.bbc_feed_parser._fetch_json")
    def test_deduplicates_across_categories(self, mock_fetch):
        item = {"id": "b09dup", "title": "Shared Drama"}
        mock_fetch.return_value = {"data": [item], "total": 1}
        result = fetch_drama_programmes(
            category_slugs=["drama", "thriller"], max_pages=1, delay=0
        )
        assert len(result) == 1
        assert result[0].pid == "b09dup"
```

## References

- [BBC Sounds OpenAPI types](https://github.com/kieranhogg/auntie-sounds/blob/main/src/sounds/sounds_types.py) — `PlayableItemsPaginatedResponse` schema
- [LMS BBC Sounds Plugin](https://github.com/expectingtofly/LMS_BBC_Sounds_Plugin/blob/master/BBCSounds/BBCSoundsFeeder.pm#L528) — working usage of `/v2/programmes/playable?category=`
