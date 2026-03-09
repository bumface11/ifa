# Dependencies

This page shows how modules depend on each other.

## Runtime Dependencies

- `numpy`: numeric arrays and random returns.
- `matplotlib`: charts.
- `streamlit`: web app UI.
- `pandas`: currently installed as a project dependency.

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
    Pandas[pandas] --> Project[ifa project]
```

## Notes

- Core simulation logic does not require Streamlit.
- Streamlit is only needed for `ifa_web.py`.
- CLI and Streamlit both reuse the same engine, events, and plotting modules.
- Both frontends now pass multi-DC-pot inputs into `ifa.engine` using per-pot
    drawdown start ages.
