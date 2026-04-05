# Struct-Level Bitmap Support for StructArray Adapter

## Problem

`structured_array_adapter` returns `(field_bitmaps, field_datas)` but ignores the
struct-level validity bitmap. When an entire struct row is `None`, Arrow tracks this
in a separate bitmap from per-field nulls. The current adapter uses
`struct_array.field(name)` which returns raw child arrays without incorporating the
struct's validity bitmap, making struct-level nulls invisible to consumers.

This was surfaced by Goykhman's review comment on upstream PR #3, asking whether
null struct elements are handled correctly in offset tests.

## Design

### Philosophy: Two-Layer Nullability

Inspired by [Awkward Array's `BitMaskedArray(RecordArray)` design](https://awkward-array.org/doc/main/reference/generated/ak.contents.BitMaskedArray.html):
struct-level and field-level nullability are semantically distinct and must be stored
independently. "This row doesn't exist" is different from "this field within an
existing row is missing."

Awkward Array defers composition to field access time, collapsing nested option layers
into an `IndexedOptionArray` (expanding bitmaps to int64 index arrays with -1 sentinels).
That machinery is powerful but too heavy for Numba `@njit` hot paths. numbarrow instead
provides a lightweight Numba-compatible helper that checks both layers in a single call
with no index expansion and no allocations.

### API Changes

#### `structured_array_adapter` — new return signature

```python
def structured_array_adapter(struct_array: pa.StructArray) -> Tuple[
    Optional[np.ndarray],            # struct-level bitmap (None if no struct nulls)
    Dict[str, Optional[np.ndarray]], # per-field bitmaps (unchanged)
    Dict[str, np.ndarray]            # per-field data (unchanged)
]:
```

Struct bitmap extraction:

```python
struct_bitmap_buf = struct_array.buffers()[0]
struct_bitmap = create_bitmap(struct_bitmap_buf, struct_array.offset, len(struct_array))
```

Fields continue using `struct_array.field(name)` (zero-copy raw child access, no eager
AND-ing of bitmaps).

#### `structured_list_array_adapter` — propagates the 3-tuple

#### `arrow_array_adapter` dispatch for `pa.StructArray` — returns the 3-tuple

#### `is_null_struct` — new Numba-compatible helper in `is_null.py`

```python
@njit
def is_null_struct(index, struct_bitmap, field_bitmap):
    if struct_bitmap is not None and is_null(index, struct_bitmap):
        return True
    if field_bitmap is not None and is_null(index, field_bitmap):
        return True
    return False
```

Checks both layers in a single call. The existing `is_null` remains unchanged for
non-struct use cases.

### Attribution

Credit comments referencing Awkward Array's `BitMaskedArray(RecordArray)` design will
be added in `is_null.py` and `arrow_array_utils.py`.

## Files Changed

| File | Change |
|------|--------|
| `numbarrow/utils/arrow_array_utils.py` | `structured_array_adapter` returns 3-tuple; Awkward Array credit |
| `numbarrow/core/adapters.py` | `StructArray` dispatch passes through 3-tuple |
| `numbarrow/core/is_null.py` | Add `is_null_struct`; Awkward Array credit |
| `test/test_arrow_array_utils.py` | Unpack 3-tuple in existing test |
| `test/test_offset.py` | Unpack 3-tuple in existing struct tests; add 3 new tests |

No new files created. No signature changes to any other adapter type.

## Test Plan

### Updated tests (3-tuple unpacking)

- `test_structured_array_adapter` — expect `struct_bitmap=None` (no struct-level nulls)
- `TestStructOffset.test_struct_sliced` — expect `struct_bitmap=None`
- `TestStructOffset.test_struct_sliced_with_nulls` — expect `struct_bitmap=None` (field-level nulls only)

### New tests

1. **`test_struct_null_rows_sliced`** — Goykhman's example: `[{"a":1,"b":10}, None, {"a":3,"b":30}, None, {"a":5,"b":50}]` sliced at offset 2. Verify struct bitmap marks null rows, field bitmaps are None, data correct for valid rows.

2. **`test_struct_both_null_layers_sliced`** — Both struct-level and field-level nulls: `[{"a":1,"b":None}, None, {"a":None,"b":30}, None, {"a":5,"b":50}]` sliced at offset 1. Verify struct bitmap identifies null struct rows, field bitmaps identify null fields within valid rows, `is_null_struct` catches both layers.

3. **`test_is_null_struct_helper`** — Unit test for `is_null_struct` in a `@njit` function exercising all four combinations: both valid, struct null only, field null only, both null.
