# Documentation Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Bring numbarrow documentation up to standard for both end users and contributors — README, Sphinx docs, docstrings, and inline comments.

**Architecture:** Add module/function docstrings to source code, set up Sphinx autodoc matching numbox's pattern, expand README with usage examples. Docstrings feed the auto-generated API reference.

**Tech Stack:** Sphinx, sphinx_rtd_theme, sphinx_sitemap, reStructuredText

---

### Task 1: Add module docstrings and inline comments to source files

This task adds all missing docstrings and comments to the Python source. These must be in place before Sphinx autodoc can generate useful API docs.

**Files:**
- Modify: `numbarrow/core/configurations.py`
- Modify: `numbarrow/core/is_null.py`
- Modify: `numbarrow/core/adapters.py`
- Modify: `numbarrow/core/mapinarrow_factory.py`
- Modify: `numbarrow/utils/utils.py`
- Modify: `numbarrow/utils/arrow_array_utils.py`

**Step 1: Add module docstring and comment to `configurations.py`**

Add at top of file:

```python
"""
Default configuration options for Numba JIT compilation used throughout numbarrow.
"""
```

Add inline comment above `default_jit_options`:

```python
# Passed as **kwargs to @njit decorators. "cache=True" persists compiled
# functions to disk so subsequent imports skip recompilation.
default_jit_options = {
```

**Step 2: Add module docstring, function docstring, and inline comments to `is_null.py`**

Add at top of file (before imports):

```python
"""
Null detection for Apache Arrow validity bitmaps.

Arrow uses a packed bitmap to track which elements in an array are valid (non-null).
Each bit corresponds to one element: bit=1 means valid, bit=0 means null.
Bits are packed LSB-first into uint8 bytes — element *i* lives at byte ``i // 8``,
bit position ``i % 8`` within that byte.
"""
```

Add docstring to `is_null` function:

```python
@njit(boolean(int64, Array(uint8, 1, "C")), **default_jit_options)
def is_null(index_: int, bitmap: np.ndarray) -> bool:
    """Check whether element *index_* is null according to *bitmap*.

    Arrow validity bitmaps store one bit per element, packed LSB-first into
    uint8 bytes. A set bit (1) means valid; a cleared bit (0) means null.

    :param index_: zero-based element index
    :param bitmap: uint8 array containing the packed validity bitmap
    :returns: True if the element is null (bit is 0), False if valid (bit is 1)
    """
    # Locate the byte containing the bit for this element
    byte_for_index = bitmap[index_ // 8]
    # Isolate the specific bit within that byte (LSB-first order)
    bit_position_in_byte = index_ % 8
    return not (byte_for_index >> bit_position_in_byte) % 2
```

**Step 3: Add module docstring to `adapters.py`**

Add at top of file (before imports):

```python
"""
Type-dispatched adapters that convert PyArrow arrays into NumPy arrays for use
in Numba ``@njit`` compiled functions.

Uses :func:`functools.singledispatch` to route each PyArrow array type
(BooleanArray, Int32Array, Date32Array, etc.) to a handler that extracts the
underlying data buffer as a NumPy array and the validity bitmap as a uint8 array.
Where possible, data is viewed without copying; types that require layout changes
(e.g. Date32 → datetime64[D]) produce a copy.
"""
```

**Step 4: Add module docstring to `mapinarrow_factory.py`**

Add at top of file (before imports):

```python
"""
Factory for PySpark ``mapInArrow`` UDF functions.

Bridges PySpark's Arrow-based batch processing with Numba JIT-compiled functions
by converting each :class:`pyarrow.RecordBatch` column through
:func:`~numbarrow.core.adapters.arrow_array_adapter` before passing the data
to a user-supplied computation function.
"""
```

**Step 5: Add module docstring, function docstrings, and inline comments to `utils/utils.py`**

Add at top of file (before imports):

```python
"""
Low-level pointer utilities for zero-copy access to Arrow memory buffers.

Provides Numba-compatible functions that reinterpret a raw memory address
(obtained from :attr:`pyarrow.Buffer.address`) as a typed NumPy array, enabling
``@njit`` code to read Arrow buffer data directly without copying.
"""
```

Add docstring to `_ptr_as_int_to_voidptr`:

```python
@intrinsic
def _ptr_as_int_to_voidptr(typingctx, arg_type):
    """Convert an integer memory address to a Numba ``voidptr``.

    This is a Numba intrinsic (compiler-level function) that emits an
    LLVM ``inttoptr`` instruction, converting a Python int holding a
    memory address into a void pointer that :func:`numba.carray` can
    dereference.
    """
    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], context.get_value_type(voidptr))
    return voidptr(arg_type), codegen
```

Add docstring to `numpy_array_from_ptr_factory`:

```python
def numpy_array_from_ptr_factory(dtype_):
    """Create a JIT-compiled function that views memory at a given address as a NumPy array.

    Returns an ``@njit`` function with signature ``(ptr_as_int, sz) -> ndarray``
    that uses :func:`numba.carray` to reinterpret *sz* elements starting at
    address *ptr_as_int* as a contiguous C-order NumPy array of *dtype_*.
    No data is copied — the returned array is a view over the original memory.

    :param dtype_: NumPy dtype for the resulting array (e.g. ``np.int32``)
    :returns: JIT-compiled function ``(int, int) -> np.ndarray``
    """
    @njit(Array(from_dtype(dtype_), 1, "C")(intp, int64), **default_jit_options)
    def _(ptr_as_int: int, sz: int):
        # carray interprets raw memory at ptr as a typed NumPy array (zero-copy view)
        return carray(_ptr_as_int_to_voidptr(ptr_as_int), shape=(sz,), dtype=dtype_)
    return _
```

Add inline comment above `arrays_viewers`:

```python
# Pre-built viewers for common NumPy types. Each entry maps a dtype to a
# JIT-compiled function that views a memory address as an array of that type.
arrays_viewers = {
```

**Step 6: Add module docstring and inline comments to `utils/arrow_array_utils.py`**

Add at top of file (before imports):

```python
"""
Utilities for extracting data from PyArrow array buffers as NumPy arrays.

Handles uniform arrays (fixed-width elements), string arrays (variable-length
with offset buffers), struct arrays, and list-of-struct arrays. Validity
bitmaps are extracted as uint8 arrays for use with :func:`~numbarrow.core.is_null.is_null`.
"""
```

Add inline comments to `create_str_array` (inside the function, around the loop):

```python
    # Arrow StringArray layout: [validity_bitmap, offsets (int32), char_data (uint8)]
    # offsets[i] and offsets[i+1] delimit the byte range of string i in char_data
    bitmap_buf, offsets_buf, data_buf = pa_str_array.buffers()
```

And before the for-loop:

```python
    # Copy each variable-length string from the packed Arrow buffer into a
    # fixed-width NumPy Unicode array (padded to the longest string's length)
    for i in range(len(offsets_array) - 1):
```

**Step 7: Run tests to verify docstrings didn't break anything**

Run: `find /home/erik/projects/numbarrow -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba; cd /home/erik/projects/numbarrow && pytest -v`

Expected: All tests pass (docstrings and comments don't affect behavior).

**Step 8: Commit**

```bash
git add numbarrow/
git commit -m "Add module docstrings, function docstrings, and inline comments"
```

---

### Task 2: Set up Sphinx documentation scaffold

**Files:**
- Create: `docs/conf.py`
- Create: `docs/Makefile`
- Create: `docs/make.bat`
- Create: `docs/index.rst`
- Create: `docs/modules.rst`

**Step 1: Create `docs/conf.py`**

Match numbox's configuration exactly:

```python
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys


sys.path.insert(0, os.path.abspath('..'))


project = "numbarrow"
copyright = "2025, Mikhail Goykhman"
author = "Mikhail Goykhman"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_sitemap",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_baseurl = "https://goykhman.github.io/numbarrow"
```

**Step 2: Create `docs/Makefile`**

```makefile
# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
```

**Step 3: Create `docs/make.bat`**

```bat
@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
```

**Step 4: Create `docs/index.rst`**

```rst
.. numbarrow documentation master file, created by sphinx-quickstart


numbarrow
=========


Numba adapters for `PyArrow <https://arrow.apache.org/docs/python/>`_ and `PySpark <https://spark.apache.org/docs/latest/api/python/>`_.

Source code on `GitHub <https://github.com/goykhman/numbarrow>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
```

**Step 5: Create `docs/modules.rst`**

```rst
numbarrow
=========

.. toctree::
   :maxdepth: 4

   numbarrow.core.adapters
   numbarrow.core.is_null
   numbarrow.core.mapinarrow_factory
   numbarrow.core.configurations
   numbarrow.utils
```

**Step 6: Commit**

```bash
git add docs/conf.py docs/Makefile docs/make.bat docs/index.rst docs/modules.rst
git commit -m "Add Sphinx documentation scaffold"
```

---

### Task 3: Write module .rst files with overviews

**Files:**
- Create: `docs/numbarrow.core.adapters.rst`
- Create: `docs/numbarrow.core.is_null.rst`
- Create: `docs/numbarrow.core.mapinarrow_factory.rst`
- Create: `docs/numbarrow.core.configurations.rst`
- Create: `docs/numbarrow.utils.rst`

**Step 1: Create `docs/numbarrow.core.adapters.rst`**

```rst
numbarrow.core.adapters
=======================

Overview
++++++++

Type-dispatched adapters that convert PyArrow arrays into NumPy arrays
for use in Numba ``@njit`` compiled functions.

Uses ``functools.singledispatch`` to route each PyArrow array type to a
handler that extracts the underlying data buffer as a NumPy view (where
possible) and the validity bitmap as a uint8 array.

Supported types:

- ``BooleanArray`` (requires copy due to bit-packed layout)
- ``Int32Array``, ``Int64Array``, ``DoubleArray`` (zero-copy view)
- ``Date32Array`` (copy: int32 days → datetime64[D])
- ``Date64Array`` (zero-copy view as datetime64[ms])
- ``TimestampArray`` (zero-copy view as datetime64[unit])
- ``StringArray`` (copy into fixed-width NumPy Unicode array)
- ``StructArray`` (returns dicts of field name → array)
- ``ListArray`` (delegates to StructArray adapter for list-of-struct)

Module
++++++

.. automodule:: numbarrow.core.adapters
   :members:
   :show-inheritance:
   :undoc-members:
```

**Step 2: Create `docs/numbarrow.core.is_null.rst`**

```rst
numbarrow.core.is_null
======================

Overview
++++++++

Arrow uses a packed validity bitmap to track which elements in an array
are non-null. Each bit corresponds to one element: bit=1 means valid,
bit=0 means null. Bits are packed LSB-first into uint8 bytes.

This module provides a single Numba ``@njit`` compiled function that
reads the bitmap and returns whether a given element index is null.

Module
++++++

.. automodule:: numbarrow.core.is_null
   :members:
   :show-inheritance:
   :undoc-members:
```

**Step 3: Create `docs/numbarrow.core.mapinarrow_factory.rst`**

```rst
numbarrow.core.mapinarrow_factory
=================================

Overview
++++++++

Factory for PySpark ``mapInArrow`` UDF functions. Bridges PySpark's
Arrow-based batch processing with Numba JIT-compiled functions by
converting each ``pyarrow.RecordBatch`` column through the adapter
layer before passing data to a user-supplied computation function.

Usage::

    from numbarrow.core.mapinarrow_factory import make_mapinarrow_func

    def my_func(data_dict, bitmap_dict, broadcasts):
        # data_dict:   {col_name: np.ndarray}
        # bitmap_dict: {col_name: np.ndarray (uint8 bitmap)}
        # broadcasts:  {key: value}
        result = ...
        return {"output_col": result}

    udf = make_mapinarrow_func(my_func, broadcasts={"scale": 1.5})
    df_out = df_in.mapInArrow(udf, output_schema)

Module
++++++

.. automodule:: numbarrow.core.mapinarrow_factory
   :members:
   :show-inheritance:
   :undoc-members:
```

**Step 4: Create `docs/numbarrow.core.configurations.rst`**

```rst
numbarrow.core.configurations
=============================

Overview
++++++++

Default Numba JIT compilation options shared across all ``@njit``
decorated functions in numbarrow.

Module
++++++

.. automodule:: numbarrow.core.configurations
   :members:
   :show-inheritance:
   :undoc-members:
```

**Step 5: Create `docs/numbarrow.utils.rst`**

```rst
numbarrow.utils
===============

numbarrow.utils.utils
---------------------

Overview
''''''''

Low-level pointer utilities for zero-copy access to Arrow memory buffers.
Provides Numba-compatible functions that reinterpret a raw memory address
(from ``pyarrow.Buffer.address``) as a typed NumPy array, enabling ``@njit``
code to read Arrow buffer data directly without copying.

The key abstraction is ``arrays_viewers`` — a dictionary mapping NumPy dtypes
to pre-compiled viewer functions. Each viewer takes ``(address, length)`` and
returns a NumPy array backed by the memory at that address.

.. automodule:: numbarrow.utils.utils
   :members:
   :show-inheritance:
   :undoc-members:

numbarrow.utils.arrow_array_utils
---------------------------------

Overview
''''''''

Higher-level utilities for extracting data from PyArrow array buffers as
NumPy arrays. Handles uniform arrays (fixed-width elements), string arrays
(variable-length with offset buffers), struct arrays, and list-of-struct
arrays.

.. automodule:: numbarrow.utils.arrow_array_utils
   :members:
   :show-inheritance:
   :undoc-members:
```

**Step 6: Commit**

```bash
git add docs/*.rst
git commit -m "Add module .rst files with overviews for Sphinx docs"
```

---

### Task 4: Build and verify Sphinx docs

**Step 1: Install Sphinx dependencies**

Run: `cd /home/erik/projects/numbarrow && pip install sphinx sphinx_rtd_theme sphinx_sitemap`

**Step 2: Build the docs**

Run: `cd /home/erik/projects/numbarrow/docs && make html`

Expected: Build completes with no errors. Warnings about missing `_static` or `_templates` dirs are acceptable.

**Step 3: Verify the output**

Run: `ls /home/erik/projects/numbarrow/docs/_build/html/index.html`

Expected: File exists. Optionally open in browser to spot-check.

**Step 4: Add `_static` placeholder and `.gitignore` for build output**

Create `docs/_static/.gitkeep` (empty file, needed by Sphinx config).

Add to `docs/.gitignore`:

```
_build/
_templates/
```

**Step 5: Commit**

```bash
git add docs/_static/.gitkeep docs/.gitignore
git commit -m "Add Sphinx build ignore and static placeholder"
```

---

### Task 5: Expand README.md

**Files:**
- Modify: `README.md`

**Step 1: Rewrite README.md**

```markdown
# numbarrow

Numba adapters for [PyArrow](https://arrow.apache.org/docs/python/) and [PySpark](https://spark.apache.org/docs/latest/api/python/).

numbarrow lets you work with Apache Arrow arrays directly inside Numba `@njit` compiled functions. It converts PyArrow arrays into NumPy views (zero-copy where possible) and extracts validity bitmaps for null handling — bridging PySpark's Arrow-based batch processing with high-performance JIT-compiled code.

## Installation

```bash
pip install numbarrow
```

Optional dependencies for PySpark and pandas support:

```bash
pip install numbarrow[test]       # adds pyspark
pip install numbarrow[mapinarrow] # adds pandas
```

## Quick Start

```python
import pyarrow as pa
from numba import njit
from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.core.is_null import is_null

# Convert a PyArrow array to NumPy for use in @njit
arrow_array = pa.array([10, None, 30, 40], type=pa.int32())
bitmap, data = arrow_array_adapter(arrow_array)

@njit
def sum_non_null(data, bitmap):
    total = 0
    for i in range(len(data)):
        if bitmap is not None and not is_null(i, bitmap):
            total += data[i]
    return total

result = sum_non_null(data, bitmap)  # 80
```

## Supported Types

| PyArrow Type | NumPy Result | Copy? |
|---|---|---|
| `Int32Array`, `Int64Array`, `DoubleArray` | Matching dtype | No (view) |
| `BooleanArray` | `bool_` | Yes (bit-unpacking) |
| `Date32Array` | `datetime64[D]` | Yes (int32 → int64) |
| `Date64Array` | `datetime64[ms]` | No (view) |
| `TimestampArray` | `datetime64[unit]` | No (view) |
| `StringArray` | Fixed-width Unicode | Yes (repacking) |
| `StructArray` | Dict of field arrays | Per-field |
| `ListArray` (of structs) | Dict of field arrays | Per-field |

## PySpark Integration

Use `make_mapinarrow_func` to create functions compatible with PySpark's `mapInArrow`:

```python
from numbarrow.core.mapinarrow_factory import make_mapinarrow_func

def compute(data_dict, bitmap_dict, broadcasts):
    # data_dict: {col_name: np.ndarray}
    # bitmap_dict: {col_name: uint8 bitmap array}
    result = data_dict["value"] * broadcasts["scale"]
    return {"output": result}

udf = make_mapinarrow_func(compute, broadcasts={"scale": 2.0})
df_out = df_in.mapInArrow(udf, output_schema)
```

## Compatibility

| Dependency | Versions |
|---|---|
| Python | 3.10 – 3.12 |
| numba | 0.60 – 0.63 |
| pyarrow | 14 – 18 |
| pyspark | 3.3 – 3.x (optional) |
| pandas | 1.5+ (optional) |

## Documentation

Full API documentation: [numbarrow docs](https://goykhman.github.io/numbarrow)

## License

See [LICENSE](LICENSE).
```

**Step 2: Run tests to verify nothing broke**

Run: `find /home/erik/projects/numbarrow -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba; cd /home/erik/projects/numbarrow && pytest -v`

Expected: All tests pass.

**Step 3: Commit**

```bash
git add README.md
git commit -m "Expand README with install, quickstart, supported types, and compatibility"
```

---
