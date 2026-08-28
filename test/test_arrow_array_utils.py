import gc

import numpy as np
import pyarrow as pa
from numbarrow.core.adapters import arrow_array_adapter
from numbarrow.utils.arrow_array_utils import (
    create_str_array, structured_array_adapter,
    uniform_arrow_array_adapter
)
from numbarrow.core.is_null import is_null


def owner_of(array):
    """Walk an array's base chain and return the pyarrow object backing it."""
    node = array
    for _ in range(8):
        node = node.obj if isinstance(node, memoryview) else getattr(node, "base", None)
        if node is None:
            return None
        if isinstance(node, (pa.Buffer, pa.Array)):
            return node
    return None


def test_create_str_array():
    pa_a = pa.array(["first", "second", None, "third", "fourth element", "f"], type=pa.string())
    bitmap, np_a = create_str_array(pa_a)
    ref = ["first", "second", "", "third", "fourth element", "f"]
    assert is_null(2, bitmap) and all(not is_null(i, bitmap) for i in range(len(pa_a)) if i != 2)
    assert all([np_a_e == ref_e for np_a_e, ref_e in zip(np_a, ref)])


def test_create_str_array_long_with_null():
    # Bitmap spans two bytes once the array has 9+ elements, so this pins
    # that the copy guard never calls bool() on a multi-byte bitmap.
    vals = ["s%d" % i for i in range(9)] + [None]
    bitmap, np_a = create_str_array(pa.array(vals, type=pa.string()))
    assert is_null(9, bitmap) and all(not is_null(i, bitmap) for i in range(9))
    assert np_a[0] == "s0" and np_a[8] == "s8" and np_a[9] == ""
    sliced = pa.array(vals + ["tail"], type=pa.string())[1:]
    bitmap_s, np_s = create_str_array(sliced)
    assert is_null(8, bitmap_s) and not is_null(9, bitmap_s)


def test_create_str_array_all_null_bitmap_is_a_copy():
    # An all-null bitmap byte is zero (falsy); the copy must still happen
    # so the returned bitmap owns its memory instead of viewing the Arrow
    # buffer, which the caller may let die.
    bitmap, np_a = create_str_array(pa.array([None] * 8, type=pa.string()))
    assert all(is_null(i, bitmap) for i in range(8))
    assert bitmap.base is None


def test_structured_array_adapter():
    indices = pa.array([14, 89, None, 105], type=pa.int32())
    ratios = pa.array([1.41, None, 1.72, 9.99], type=pa.float64())
    struct_array = pa.StructArray.from_arrays([indices, ratios], ["indices", "ratios"])
    struct_bitmap, bitmap, data = structured_array_adapter(struct_array)
    assert struct_bitmap is None  # no struct-level nulls in this test
    indices_bitmap = bitmap["indices"]
    indices_data = data["indices"]
    assert len(indices_bitmap) == 1, "length of `indices` is less than 9"
    assert indices_bitmap[0] == int("1011", base=2)
    assert indices_data[0] == 14 and indices_data[1] == 89 and indices_data[3] == 105
    ratios_bitmap = bitmap["ratios"]
    ratios_data = data["ratios"]
    assert len(ratios_bitmap) == 1, "length of `ratios` is less than 9"
    assert ratios_bitmap[0] == int("1101", base=2)
    assert np.allclose([ratios_data[0], ratios_data[2], ratios_data[3]], [1.41, 1.72, 9.99])


def test_uniform_arrow_array_adapter_1():
    a = pa.array([141, None, 172, 314, 271], type=pa.int32())
    bitmap, data = uniform_arrow_array_adapter(a)
    assert data[0] == 141 and data[2] == 172 and data[3] == 314 and data[4] == 271
    assert (
        not is_null(0, bitmap) and is_null(1, bitmap) and not is_null(2, bitmap)
        and not is_null(3, bitmap) and not is_null(4, bitmap)
    )


def test_bitmap_at_offset_zero_does_not_alias_the_source():
    source = pa.array([1, 2, None, 4, 5, 6, 7, 8, 9], type=pa.int64())
    bitmap, _ = arrow_array_adapter(source)
    bitmap[0] = 0
    assert source.to_pylist() == [1, 2, None, 4, 5, 6, 7, 8, 9]


def test_short_offset_zero_slice_does_not_hand_back_the_parents_bits():
    parent = pa.array([1, 2, None] + list(range(4, 21)), type=pa.int64())
    bitmap, _ = arrow_array_adapter(parent.slice(0, 4))
    bitmap[0] = 0
    assert parent.to_pylist()[:4] == [1, 2, None, 4]


def test_bitmap_never_aliases_arrow_memory():
    # The offset branch repacks through np.packbits and so already owned its
    # memory; the offset-0 branch used to hand back a live view of the Arrow
    # validity buffer. Neither may reach a pyarrow object.
    source = pa.array([1, None, 3, 4, 5, 6, 7, 8, 9, 10], type=pa.int64())
    for offset in (0, 1, 3):
        bitmap, _ = arrow_array_adapter(source.slice(offset))
        assert owner_of(bitmap) is None, offset


def test_bitmap_survives_the_source_being_dropped():
    source = pa.array([1, 2, None, 4], type=pa.int64())
    bitmap, _ = arrow_array_adapter(source)
    before = bitmap.copy()
    del source
    gc.collect()
    assert bitmap.tolist() == before.tolist()


if __name__ == "__main__":
    test_create_str_array()
    test_create_str_array_long_with_null()
    test_create_str_array_all_null_bitmap_is_a_copy()
    test_structured_array_adapter()
    test_uniform_arrow_array_adapter_1()
