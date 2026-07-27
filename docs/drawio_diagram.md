# IFA Simulation Flow Diagram

This diagram was generated using draw.io (via Mermaid input) and stored in Markdown for easy editing.

![IFA Simulation Flow](./ifa_simulation_flow.drawio.svg)

```mermaid
flowchart TD
    A[Load Saved Parameters] --> B[Validate Inputs]
    B --> C[Build Simulation Config]
    C --> D[Run Engine]
    D --> E[Apply Strategy Rules]
    E --> F[Compute Tax and Metrics]
    F --> G[Generate Charts and Explanations]
    G --> H[Export Results]
```

## Edit Notes

- Open this file in VS Code to update the Mermaid diagram.
- You can regenerate or refine the same diagram with draw.io tooling when needed.
