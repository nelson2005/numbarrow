# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

numbarrow — Numba adapters for PyArrow and PySpark. Enables working with Arrow arrays and PySpark data inside numba `@njit` code.

## Build & Dev

- Venv: `python3.12 -m venv venv && venv/bin/pip install -e . flake8 pytest`
- Install: `pip install -e .`
- Test: `pytest`
- Lint: `flake8`
- Supported Python and every dependency range: see `requires-python` and `dependencies` in [`pyproject.toml`](pyproject.toml). Those declarations are authoritative and machine-enforced, so they are deliberately not restated here — a copy in this file goes stale on an upstream sync without anything failing.
- CI deliberately tests *below* the declared floor. The matrix covers Python 3.10, 3.11 and 3.12 and installs with `--ignore-requires-python`, because the package still builds, imports and passes the full suite on 3.10 and 3.11. Treat those jobs as regression signal, not as a supported configuration — pip refuses the install for a real user below the floor.

## Key Paths

- `numbarrow/core/` — core adapter implementations
- `numbarrow/utils/` — utility functions
- `test/` — tests

## Preferences

- Never include "Co-Authored-By" in git commit messages
- Avoid shell variable substitution in bash — inline actual values directly into commands
- Prefer simpler approaches
- Always git pull before making edits
- Commit messages must not mention AI, Claude, Anthropic, or any AI tooling — only attribute to the user
- Keep all memories in both MEMORY.md and the project CLAUDE.md (CLAUDE.md is in git and survives OS reinstalls)
- Environment details go in MEMORY.md only (may change between OS installs)
- Always exclude CLAUDE.md from upstream PRs (use a dedicated branch based on upstream/main)
- Always use a feature branch — never commit directly to main
- Never merge to main locally — only merge via PR on GitHub after all Actions pass
- Never merge local feature branches into main — main must always match upstream/main (exception: CLAUDE.md)
- Feature branches: base off origin/main (has CLAUDE.md); upstream PR branches: base off upstream/main (no CLAUDE.md)
- Always enable GitHub Actions on forked repos
- Never assume a reviewer's comment is wrong — always verify claims against actual runtime before responding
- Before posting PR comments, check for pending reviews with existing comments — never silently delete a pending review
- Preface all AI-authored GitHub comments with "From the fake Slim Shady:"
- Never guess about things that can be verified — check the source of truth before making claims
- Always clean `__pycache__` and numba cache (`~/.cache/numba`) before every pytest run — stale JIT artifacts cause false failures
- Never put implementation planning details (task numbers, phase references, internal tracking) into code comments — comments must be context-independent
- Never create PRs against upstream without explicit command — always default to the fork
- Use Glob instead of `find` for file searches. Bash `find` is only for operations with side effects (e.g., `-exec rm`)

## Venv
- Python 3.12 (use `python3.12 -m venv venv` not `python3`)
