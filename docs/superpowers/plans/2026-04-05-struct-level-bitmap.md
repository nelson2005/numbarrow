# Struct-Level Bitmap Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add struct-level validity bitmap support to `structured_array_adapter` so that null struct rows are visible to consumers, and provide a Numba-compatible helper for checking both null layers.

**Architecture:** Extract the struct-level validity bitmap alongside existing per-field bitmaps, returning a 3-tuple `(struct_bitmap, field_bitmaps, field_datas)`. Add `is_null_struct` helper in `is_null.py` that checks both layers in a single `@njit` call. Inspired by Awkward Array's `BitMaskedArray(RecordArray)` two-layer design, adapted for Numba's constraints.

**Tech Stack:** Python 3.10, PyArrow <=15.0.0, Numba >=0.60.0, NumPy, pytest

---

### Task 1: Add `is_null_struct` helper to `is_null.py`

**Goal:** Add a Numba-compiled helper that checks both struct-level and field-level bitmaps in one call.

**Files:**
- Modify: `numbarrow/core/is_null.py`
- Test: `test/test_offset.py` (new test class `TestIsNullStruct`)

**Acceptance Criteria:**
- [ ] `is_null_struct` is `@njit`-compiled and callable from other `@njit` functions
- [ ] Returns `True` when struct bitmap marks index as null
- [ ] Returns `True` when field bitmap marks index as null
- [ ] Returns `True` when both mark index as null
- [ ] Returns `False` when both mark index as valid
- [ ] Handles `None` for either or both bitmaps (meaning "all valid")
- [ ] Awkward Array credit comment present

**Verify:** `source venv/bin/activate && find . -path ./venv -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba; pytest test/test_offset.py::TestIsNullStruct -v` -> all pass

**Steps:**

- [ ] **Step 1: Write the failing test**

Add to the end of `test/test_offset.py`:

```python
class TestIsNullStruct:
    def test_both_valid(self):
        valid_bm = np.array([0b11111111], dtype=np.uint8)
        assert not is_null_struct(0, valid_bm, valid_bm)

    def test_struct_null_only(self):
        null_bm = np.array([0b11111110], dtype=np.uint8)
        valid_bm = np.array([0b11111111], dtype=np.uint8)
        assert is_null_struct(0, null_bm, valid_bm)

    def test_field_null_only(self):
        valid_bm = np.array([0b11111111], dtype=np.uint8)
        null_bm = np.array([0b11111110], dtype=np.uint8)
        assert is_null_struct(0, valid_bm, null_bm)

    def test_both_null(self):
        null_bm = np.array([0b11111110], dtype=np.uint8)
        assert is_null_struct(0, null_bm, null_bm)

    def test_struct_bitmap_none(self):
        valid_bm = np.array([0b11111111], dtype=np.uint8)
        assert not is_null_struct(0, None, valid_bm)

    def test_field_bitmap_none(self):
        valid_bm = np.array([0b11111111], dtype=np.uint8)
        assert not is_null_struct(0, valid_bm, None)

    def test_both_bitmap_none(self):
        assert not is_null_struct(0, None, None)

    def test_in_njit_context(self):
        from numba import njit as njit_local
        @njit_local
        def check(idx, sb, fb):
            return is_null_struct(idx, sb, fb)
        null_bm = np.array([0b11111110], dtype=np.uint8)
        valid_bm = np.array([0b11111111], dtype=np.uint8)
        assert check(0, null_bm, valid_bm)
        assert not check(0, valid_bm, valid_bm)
```

Also add `is_null_struct` to the imports at the top of `test/test_offset.py`:

```python
from numbarrow.core.is_null import is_null, is_null_struct
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && find . -path ./venv -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba; pytest test/test_offset.py::TestIsNullStruct -v`
Expected: ImportError — `is_null_struct` does not exist yet

- [ ] **Step 3: Implement `is_null_struct`**

Add to `numbarrow/core/is_null.py` after the `unpack_booleans` function:

```python
# Two-layer struct nullability inspired by Awkward Array's BitMaskedArray(RecordArray)
# design. See: https://awkward-array.org/doc/main/reference/generated/ak.contents.BitMaskedArray.html


@njit(**default_jit_options)
def is_null_struct(index_, struct_bitmap, field_bitmap):
    """Check whether a struct field value is null at either the struct or field layer.

    Arrow StructArrays carry a validity bitmap for the struct itself (is this
    entire row null?) independent of each child field's bitmap (is this
    particular field null within a non-null row?).  A value is null if either
    layer marks it as null.

    :param index_: zero-based element index
    :param struct_bitmap: uint8 packed bitmap for struct-level validity, or None
    :param field_bitmap: uint8 packed bitmap for field-level validity, or None
    :returns: True if null at either layer
    """
    if struct_bitmap is not None and is_null(index_, struct_bitmap):
        return True
    if field_bitmap is not None and is_null(index_, field_bitmap):
        return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && find . -path ./venv -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba; pytest test/test_offset.py::TestIsNullStruct -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add numbarrow/core/is_null.py test/test_offset.py
git commit -m "Add is_null_struct helper for two-layer struct nullability"
```

---

### Task 2: Return struct-level bitmap from `structured_array_adapter`

**Goal:** Extract the struct-level validity bitmap and return it as the first element of a 3-tuple.

**Files:**
- Modify: `numbarrow/utils/arrow_array_utils.py:75-95`
- Modify: `numbarrow/core/adapters.py:103-105`
- Test: `test/test_arrow_array_utils.py:17-31`
- Test: `test/test_offset.py:128-161`

**Acceptance Criteria:**
- [ ] `structured_array_adapter` returns `(struct_bitmap, field_bitmaps, field_datas)`
- [ ] `struct_bitmap` is `None` when struct has no null rows
- [ ] `struct_bitmap` is a uint8 packed bitmap when struct has null rows
- [ ] `structured_list_array_adapter` propagates the 3-tuple
- [ ] `arrow_array_adapter` for `StructArray` returns the 3-tuple
- [ ] All existing struct tests updated and passing
- [ ] Awkward Array credit comment present

**Verify:** `source venv/bin/activate && find . -path ./venv -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba; pytest test/test_arrow_array_utils.py::test_structured_array_adapter test/test_offset.py::TestStructOffset -v` -> all pass

**Steps:**

- [ ] **Step 1: Update existing test to expect 3-tuple**

In `test/test_arrow_array_utils.py`, change line 21:

```python
# Before:
bitmap, data = structured_array_adapter(struct_array)
# After:
struct_bitmap, bitmap, data = structured_array_adapter(struct_array)
```

Add an assertion after line 21:

```python
assert struct_bitmap is None  # no struct-level nulls in this test
```

In `test/test_offset.py`, update `TestStructOffset.test_struct_sliced` (line 134):

```python
# Before:
bitmaps, datas = arrow_array_adapter(s)
# After:
struct_bitmap, bitmaps, datas = arrow_array_adapter(s)
```

Add assertion:

```python
assert struct_bitmap is None
```

In `test/test_offset.py`, update `TestStructOffset.test_struct_sliced_with_nulls` (line 148):

```python
# Before:
bitmaps, datas = arrow_array_adapter(s)
# After:
struct_bitmap, bitmaps, datas = arrow_array_adapter(s)
```

Add assertion:

```python
assert struct_bitmap is None  # only field-level nulls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && find . -path ./venv -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba; pytest test/test_arrow_array_utils.py::test_structured_array_adapter test/test_offset.py::TestStructOffset -v`
Expected: FAIL — cannot unpack 2-tuple into 3 values

- [ ] **Step 3: Update `structured_array_adapter`**

In `numbarrow/utils/arrow_array_utils.py`, replace the `structured_array_adapter` function (lines 75-95):

```python
# Two-layer struct nullability inspired by Awkward Array's BitMaskedArray(RecordArray)
# design. See: https://awkward-array.org/doc/main/reference/generated/ak.contents.BitMaskedArray.html


def structured_array_adapter(struct_array: pa.StructArray) -> Tuple[
    Optional[np.ndarray], Dict[str, Optional[np.ndarray]], Dict[str, np.ndarray]
]:
    """
    NumPy adapter of PyArrow `StructArray`.

    Returns a 3-tuple:
    - struct-level validity bitmap (None if all rows valid)
    - dict mapping field names to per-field validity bitmaps
    - dict mapping field names to per-field value arrays
    """
    assert isinstance(struct_array, pa.StructArray)
    data_type: pa.StructType = struct_array.type
    assert isinstance(data_type, pa.StructType)
    struct_bitmap_buf = struct_array.buffers()[0]
    struct_bitmap = create_bitmap(
        struct_bitmap_buf, struct_array.offset, len(struct_array)
    )
    bitmaps = {}
    datas = {}
    for field_ind in range(len(data_type)):
        field: pa.Field = data_type[field_ind]
        field_name = field.name
        pa_array = struct_array.field(field_name)
        bitmap, data = uniform_arrow_array_adapter(pa_array)
        bitmaps[field_name] = bitmap
        datas[field_name] = data
    return struct_bitmap, bitmaps, datas
```

- [ ] **Step 4: Update `structured_list_array_adapter`**

In `numbarrow/utils/arrow_array_utils.py`, the function at lines 98-116 already calls `structured_array_adapter` and returns its result. No change needed — it already propagates the return value.

- [ ] **Step 5: Update type hint in import**

In `numbarrow/utils/arrow_array_utils.py` line 13, the existing `Tuple` import already covers 3-tuples. No change needed.

- [ ] **Step 6: Run tests to verify they pass**

Run: `source venv/bin/activate && find . -path ./venv -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba; pytest test/test_arrow_array_utils.py::test_structured_array_adapter test/test_offset.py::TestStructOffset -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add numbarrow/utils/arrow_array_utils.py numbarrow/core/adapters.py test/test_arrow_array_utils.py test/test_offset.py
git commit -m "Return struct-level bitmap from structured_array_adapter"
```

---

### Task 3: Add struct null row tests

**Goal:** Add tests for Goykhman's example (null struct rows with offset) and for combined struct+field nulls.

**Files:**
- Modify: `test/test_offset.py`

**Acceptance Criteria:**
- [ ] `test_struct_null_rows_sliced` verifies struct bitmap for null rows with offset
- [ ] `test_struct_both_null_layers_sliced` verifies both bitmap layers with offset
- [ ] `is_null_struct` used in the combined test to verify two-layer checking

**Verify:** `source venv/bin/activate && find . -path ./venv -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba; pytest test/test_offset.py::TestStructOffset -v` -> all pass

**Steps:**

- [ ] **Step 1: Write `test_struct_null_rows_sliced`**

Add to `TestStructOffset` class in `test/test_offset.py`:

```python
    def test_struct_null_rows_sliced(self):
        """Goykhman's example: struct array with null rows, sliced."""
        arr = pa.array([
            {"a": 1, "b": 10},
            None,
            {"a": 3, "b": 30},
            None,
            {"a": 5, "b": 50},
        ])
        s = arr[2:]  # [{"a": 3, "b": 30}, None, {"a": 5, "b": 50}], offset=2
        struct_bitmap, bitmaps, datas = arrow_array_adapter(s)
        assert struct_bitmap is not None
        assert not is_null(0, struct_bitmap)  # {"a": 3, "b": 30} valid
        assert is_null(1, struct_bitmap)      # None
        assert not is_null(2, struct_bitmap)  # {"a": 5, "b": 50} valid
        assert bitmaps["a"] is None  # no field-level nulls
        assert bitmaps["b"] is None
        assert datas["a"][0] == 3
        assert datas["a"][2] == 5
        assert datas["b"][0] == 30
        assert datas["b"][2] == 50
```

- [ ] **Step 2: Write `test_struct_both_null_layers_sliced`**

Add to `TestStructOffset` class:

```python
    def test_struct_both_null_layers_sliced(self):
        """Both struct-level and field-level nulls with offset."""
        arr = pa.array([
            {"a": 1, "b": None},
            None,
            {"a": None, "b": 30},
            None,
            {"a": 5, "b": 50},
        ])
        s = arr[1:]  # [None, {"a": None, "b": 30}, None, {"a": 5, "b": 50}], offset=1
        struct_bitmap, bitmaps, datas = arrow_array_adapter(s)
        # Struct-level: rows 0 and 2 are null structs
        assert struct_bitmap is not None
        assert is_null(0, struct_bitmap)
        assert not is_null(1, struct_bitmap)
        assert is_null(2, struct_bitmap)
        assert not is_null(3, struct_bitmap)
        # Field-level: field "a" has a null at index 1 (the {"a": None, "b": 30} row)
        assert bitmaps["a"] is not None
        assert is_null(1, bitmaps["a"])
        assert not is_null(3, bitmaps["a"])
        # Two-layer check with is_null_struct
        assert is_null_struct(0, struct_bitmap, bitmaps["a"])      # struct null
        assert is_null_struct(1, struct_bitmap, bitmaps["a"])      # field null
        assert is_null_struct(2, struct_bitmap, bitmaps["a"])      # struct null
        assert not is_null_struct(3, struct_bitmap, bitmaps["a"])  # both valid
```

- [ ] **Step 3: Run new tests**

Run: `source venv/bin/activate && find . -path ./venv -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba; pytest test/test_offset.py::TestStructOffset -v`
Expected: all pass (5 tests: 3 existing + 2 new)

- [ ] **Step 4: Run full test suite**

Run: `source venv/bin/activate && find . -path ./venv -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba; pytest -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add test/test_offset.py
git commit -m "Add tests for struct-level null rows with offset"
```
