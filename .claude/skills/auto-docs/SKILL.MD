---
name: auto-docs
description: When asked to summarize changes, read CLAUDE.md, README.md, CHANGELOG.md and similar doc files first, then produce a targeted update. Use this before writing any summary or changelog entry.
---

# Auto-Docs

When the user asks you to summarize changes, document what changed, or update project docs, follow these steps.

## Step 1 — Read existing docs

Before writing anything, read whichever of these files exist in the project root:

- `CLAUDE.md` — architecture notes, commands, package layout
- `README.md` — user-facing overview
- `CHANGELOG.md` — version history
- `HISTORY.md` — alternative changelog name

Use `git diff main...HEAD` (or `git diff HEAD~1` for a single commit) to see what actually changed in code.

## Step 2 — Identify what needs updating

For each doc file you found, decide:

- **CLAUDE.md**: Does the package layout, dependency flow, command list, or architecture notes need updating? Only touch sections that are now wrong or incomplete.
- **README.md**: Does the feature list, usage examples, or install instructions reflect the new behavior?
- **CHANGELOG.md**: Add an entry under `## Unreleased` (or the appropriate version). Follow the existing format exactly — don't invent a new style.

If a doc file doesn't exist and the changes are significant enough to warrant it, say so but don't create it unless the user asks.

## Step 3 — Write the update

- Be surgical: only change what's stale. Don't rewrite sections that are still accurate.
- Match the existing tone and format of each file.
- For CHANGELOG entries: use `Added`, `Changed`, `Fixed`, `Removed` categories. One line per change, past tense.
- For CLAUDE.md: keep it as a reference card, not prose. Bullet points and code blocks preferred.

## Step 4 — Report what you did

End with a short summary: which files you read, which you updated, and what you left alone (and why).
