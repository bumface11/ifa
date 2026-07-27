---
name: drawio-option2
description: Generate an initial draw.io diagram draft using the MCP Tool Server option (local @drawio/mcp), then hand-edit in draw.io and commit to GitHub. Use when users ask for a draft-first diagram workflow in VS Code.
---

# Draw.io Option 2 Workflow

Use this skill to produce a first-pass diagram from chat, then refine it manually.

## What This Skill Optimizes For

- Quick AI-generated draft
- Immediate manual editing in draw.io
- Git-friendly output (`.drawio.svg` preferred)

## Prerequisites

- Node.js and npm are installed
- The MCP Tool Server is running locally: `npx @drawio/mcp`
- VS Code has the draw.io extension available for direct diagram editing

## Execution Steps

1. Confirm the user wants Option 2 behavior: AI draft first, manual edits second.
2. If the draw.io MCP tool server is not connected, instruct the user to run:
   - `npx @drawio/mcp`
3. Build a concise diagram spec from the user request:
   - diagram type
   - entities/nodes
   - relationships/edges
   - key labels
4. Call the draw.io MCP diagram creation tool to generate the initial draft.
5. Ask the user to manually adjust layout/style/labels in the draw.io editor.
6. Save in repo as `.drawio.svg` when possible for GitHub rendering + editability.
7. If requested, also produce a PNG export for documents.

## Output Conventions

- Primary editable artifact: `docs/<name>.drawio.svg`
- Optional docs embed path: `docs/<name>.drawio.svg`

## Fallback

If the MCP Tool Server is unavailable, provide setup guidance and pause before generating. Do not silently switch to a different implementation unless the user explicitly approves.
