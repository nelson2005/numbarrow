# Documentation Improvement Design

## Goal

Bring numbarrow documentation up to standard for both end users and contributors. Cover installation, usage, conceptual understanding, and API reference.

## Audience

- End users installing and using numbarrow in their projects
- Contributors (including upstream maintainer) understanding and extending the code

## 1. README.md

Expand from 3 lines to include:
- One-paragraph description of what numbarrow does and why
- Installation instructions (pip + optional deps for pyspark/pandas)
- Quick start example showing Arrow array adapter in `@njit`
- Features list (supported Arrow types, PySpark integration, null handling)
- Compatibility table (Python 3.10-3.12, numba 0.60-0.63, pyarrow 14-18)
- Link to full Sphinx docs

## 2. Sphinx Documentation

Match numbox's pattern (`docs/` with Sphinx, RTD theme, autodoc).

### Structure

```
docs/
  conf.py              # autodoc + viewcode + sphinx_sitemap, sphinx_rtd_theme
  Makefile
  make.bat
  index.rst            # Brief description + GitHub link + toctree to modules
  modules.rst          # toctree listing all module pages
  numbarrow.core.adapters.rst
  numbarrow.core.is_null.rst
  numbarrow.core.mapinarrow_factory.rst
  numbarrow.core.configurations.rst
  numbarrow.utils.rst
```

Each `.rst` file has an Overview section explaining the module's purpose and concepts, followed by `automodule` directives pulling API reference from docstrings.

### Sphinx Config

- Extensions: `sphinx.ext.autodoc`, `sphinx.ext.viewcode`, `sphinx_sitemap`
- Theme: `sphinx_rtd_theme`
- `sys.path.insert(0, os.path.abspath('..'))`

## 3. Module Docstrings

Add module-level docstrings to all 6 source modules:
- `core/adapters.py` — singledispatch adapter pattern, Arrow-to-NumPy conversion
- `core/is_null.py` — bitmap-based null detection for Arrow validity buffers
- `core/mapinarrow_factory.py` — factory for PySpark mapInArrow UDFs
- `core/configurations.py` — default Numba JIT options
- `utils/utils.py` — low-level pointer/memory utilities for Arrow buffer access
- `utils/arrow_array_utils.py` — Arrow array buffer extraction and type adaptation

## 4. Function Docstrings

Fill gaps in:
- `is_null.py`: `is_null()` — bitmap bit layout, validity semantics
- `utils.py`: `_ptr_as_int_to_voidptr()`, `numpy_array_from_ptr_factory()` — pointer arithmetic
- `configurations.py`: explain `default_jit_options`

Leave already-good docstrings in adapters.py, mapinarrow_factory.py, arrow_array_utils.py unchanged.

## 5. Inline Comments

Add comments on tricky logic:
- `is_null.py`: bitmap byte/bit indexing (`i // 8`, `i % 8`)
- `utils.py`: `carray` usage for viewing memory as typed arrays
- `arrow_array_utils.py`: `create_str_array` offset/buffer iteration logic

## Not In Scope

- Changing existing docstrings that are already good
- Adding type hints
- Test docstrings
- GitHub Pages deployment (future work)
