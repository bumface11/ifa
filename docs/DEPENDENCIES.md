# Dependencies

This page shows how modules depend on each other.

## Runtime Dependencies

- `numpy`: numeric arrays and random returns.
- `matplotlib`: charts.
- `streamlit`: web app UI.
- `orjson`: fast local JSON serialization for saved sidebar parameter presets.
- `pandas`: currently installed as a project dependency.

### Radio Cache (optional extras: `pip install -e ".[radio]"`)

- `fastapi`: lightweight async web framework for the radio cache API.
- `jinja2`: HTML template rendering for the web search UI.
- `uvicorn`: ASGI server to run the FastAPI application.

## Internal Import Diagram

```mermaid
flowchart TD
    Web[ifa_web.py] --> Config[ifa.config]
    Web --> Models[ifa.models]
    Web --> Events[ifa.events]
    Web --> Engine[ifa.engine]
    Web --> Metrics[ifa.metrics]
    Web --> Explain[ifa.explain]
    Web --> Plotting[ifa.plotting]
    Web --> Presets[ifa.presets]

    CLI[pension_drawdown_simulator.py] --> Config
    CLI --> Models
    CLI --> Events
    CLI --> Engine
    CLI --> Metrics
    CLI --> Plotting

    Events --> Models
    Engine --> Models
    Explain --> Models
    Explain --> Metrics
    Plotting --> Engine
    Plotting --> Market[ifa.market]
    Plotting --> Models
```

## Package-Level Dependency Diagram

```mermaid
graph LR
    Numpy[numpy] --> Engine[ifa.engine]
    Numpy --> Events[ifa.events]
    Numpy --> Metrics[ifa.metrics]
    Numpy --> Market[ifa.market]
    Numpy --> Web[ifa_web.py]

    Matplotlib[matplotlib] --> Plotting[ifa.plotting]
    Plotting --> Web
    Plotting --> CLI[pension_drawdown_simulator.py]

    Streamlit[streamlit] --> Web
    Orjson[orjson] --> Presets[ifa.presets]
    Presets --> Web
    Pandas[pandas] --> Project[ifa project]
```

## Notes

- Core simulation logic does not require Streamlit.
- Streamlit is only needed for `ifa_web.py`.
- CLI and Streamlit both reuse the same engine, events, and plotting modules.
- Both frontends now pass multi-DC-pot inputs into `ifa.engine` using per-pot
    drawdown start ages.
- Preset file persistence is isolated in `ifa.presets`; preset UI behavior
    (button actions, auto-load on selection, unsaved-change prompts) is handled
    in `ifa_web.py`.
