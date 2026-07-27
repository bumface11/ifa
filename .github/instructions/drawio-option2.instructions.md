---
applyTo: ""
description: Use Option 2 draw.io workflow (local @drawio/mcp MCP Tool Server) for diagram requests in VS Code. Trigger on terms like drawio, draw.io, diagram, ERD, flowchart, architecture diagram.
---

When a user asks for a diagram in this repository:

1. Prefer Option 2 workflow:
   - local MCP Tool Server via `npx @drawio/mcp`
   - AI-generated first draft
   - user/manual refinement in draw.io editor
2. Prefer saving editable artifacts as `.drawio.svg` for GitHub-friendly rendering.
3. If the tool server is not connected, ask the user to start it and then continue.
4. Do not silently fall back to static-only diagrams unless the user asks for that fallback.
