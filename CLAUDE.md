# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

numbarrow — Numba adapters for PyArrow and PySpark. Enables working with Arrow arrays and PySpark data inside numba `@njit` code.

## Build & Dev

- Venv: `python3 -m venv venv && venv/bin/pip install -e . flake8 pytest`
- Install: `pip install -e .`
- Test: `pytest`
- Lint: `flake8`
- Python: >=3.10
- Key dependencies: `numba>=0.60.0`, `pyarrow<=15.0.0`
- Optional: `pyspark>=3.3.0` (test), `pandas>=1.5.0` (mapinarrow)

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
