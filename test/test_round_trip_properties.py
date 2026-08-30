"""Round-trip invariants over the whole mapInArrow path.

The adapter side was tested in isolation and the output side was not tested at
all, which is how a trailing NUL came to be refused on the way in while leading
and interior NULs were silently destroyed on the way out. A value that a
pass-through UDF returns unchanged must arrive unchanged, and the column a UDF
names must be the column that arrives. Both are properties of the path, not of
either end, so they are asserted here rather than in the adapter tests.
"""
import numpy as np
import pyarrow as pa
import pytest
from numbarrow.core.mapinarrow_factory import make_mapinarrow_func


def _identity(data_dict, bitmap_dict, broadcasts):
    return {name: value for name, value in data_dict.items()}


def _round_trip(column, udf=_identity, name="c"):
    batch = pa.RecordBatch.from_arrays([column], names=[name])
    fn = make_mapinarrow_func(udf, input_columns=[name])
    return list(fn(iter([batch])))[0]


# Every shape a supported column can take, with no nulls so that value
# preservation is isolated from null handling.
COLUMNS = {
    "int64": pa.array([1, -2, 3], type=pa.int64()),
    "int32": pa.array([1, -2], type=pa.int32()),
    "float64": pa.array([1.5, -0.25], type=pa.float64()),
    "uint8": pa.array([0, 255], type=pa.uint8()),
    "bool": pa.array([True, False, True], type=pa.bool_()),
    "ascii": pa.array(["a", "bb", "ccc"], type=pa.string()),
    "empty strings": pa.array(["", "", ""], type=pa.string()),
    "cjk": pa.array(["中文", "字"], type=pa.string()),
    "astral": pa.array(["\U0001F600x", "y"], type=pa.string()),
    "interior nul": pa.array(["a\x00b", "plain"], type=pa.string()),
    "leading nul": pa.array(["\x00ab", "plain"], type=pa.string()),
    "large_string": pa.array(["a", "bb"], type=pa.large_string()),
}


@pytest.mark.parametrize("label", sorted(COLUMNS))
def test_a_value_survives_the_round_trip(label):
    column = COLUMNS[label]
    got = _round_trip(column).column("c")
    assert got.to_pylist() == column.to_pylist(), (
        f"{label}: went in as {column.to_pylist()!r}, came out as {got.to_pylist()!r}"
    )


def test_a_udf_names_its_output_columns():
    # Returned columns are bound by name, not by the order the dict happens to
    # iterate in, or two same-typed columns silently swap.
    def swap(data_dict, bitmap_dict, broadcasts):
        return {"second": np.array([10, 20], dtype=np.int64),
                "first": np.array([1, 2], dtype=np.int64)}

    batch = pa.RecordBatch.from_arrays([pa.array([0, 0], type=pa.int64())], names=["c"])
    out = list(make_mapinarrow_func(swap, input_columns=["c"])(iter([batch])))[0]
    assert out.column("first").to_pylist() == [1, 2]
    assert out.column("second").to_pylist() == [10, 20]


def test_a_string_column_keeps_its_width_across_the_round_trip():
    # The |U width is data-dependent, so a column whose widest value is under a
    # null must not size the output from a value no caller can see.
    column = pa.array(["ab", None, "cd"], type=pa.string())
    got = _round_trip(column).column("c")
    assert [v for v in got.to_pylist() if v] == ["ab", "cd"]
