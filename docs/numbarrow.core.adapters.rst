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
- ``Int32Array``, ``Int64Array``, ``DoubleArray``, ``UInt8Array`` (zero-copy view)
- ``Date32Array`` (copy: int32 days → datetime64[D])
- ``Date64Array`` (zero-copy view as datetime64[ms])
- ``TimestampArray`` (zero-copy view as datetime64[unit])
- ``StringArray``, ``LargeStringArray`` (tuple of bitmap and a copy of data into a
  fixed-width NumPy Unicode array, whose width counts characters)
- ``StructArray`` (returns a 3-tuple: the struct-level validity bitmap, then bitmaps
  and data as two dicts keyed by field name)
- ``ListArray`` of structs (delegates to the StructArray adapter, honouring the list
  array's own offset; any other element type raises ``NotImplementedError``)

A row that is null as a whole carries no validity bits in its fields, so the
struct-level bitmap is the only record of it. Pass both layers to
``numbarrow.core.is_null.is_null_struct``.

Module
++++++

.. automodule:: numbarrow.core.adapters
   :members:
   :show-inheritance:
   :undoc-members:
