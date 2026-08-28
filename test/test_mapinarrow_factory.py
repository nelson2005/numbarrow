import numpy as np
import pyarrow as pa
import pytest

from numbarrow.core.is_null import is_null, is_null_struct
from numbarrow.core.mapinarrow_factory import make_mapinarrow_func


def run_batch(batch, input_columns=None):
    """Run one RecordBatch through the factory and return what the UDF saw."""
    seen = {}

    def main(data_dict, bitmap_dict, broadcasts):
        seen["data"] = data_dict
        seen["bitmap"] = bitmap_dict
        return {"out": np.asarray(next(iter(data_dict.values())))}

    list(make_mapinarrow_func(main, input_columns=input_columns)(iter([batch])))
    return seen


def test_struct_null_row_reaches_the_udf():
    # Plain pa.array, from_pylist and StructArray.from_arrays(mask=...) all emit
    # a struct-null row with no child validity buffer, so the struct-level
    # bitmap is the only record that the row is null at all.
    col = pa.array([{"v": 10}, None, {"v": 30}], type=pa.struct([pa.field("v", pa.int64())]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    struct_bitmap = seen["bitmap"]["s"]
    assert struct_bitmap is not None
    assert seen["bitmap"]["v"] is None
    assert not is_null(0, struct_bitmap) and is_null(1, struct_bitmap) and not is_null(2, struct_bitmap)
    assert is_null_struct(1, struct_bitmap, seen["bitmap"]["v"])
    assert not is_null_struct(0, struct_bitmap, seen["bitmap"]["v"])


def test_struct_bitmap_is_none_when_no_row_is_null():
    col = pa.array([{"v": 10}, {"v": 30}], type=pa.struct([pa.field("v", pa.int64())]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    assert "s" in seen["bitmap"] and seen["bitmap"]["s"] is None
    assert seen["data"]["v"].tolist() == [10, 30]


def test_both_null_layers_are_distinguishable():
    inner = pa.array([1, None, 3], type=pa.int64())
    col = pa.StructArray.from_arrays([inner], ["v"], mask=pa.array([False, False, True]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    struct_bitmap, field_bitmap = seen["bitmap"]["s"], seen["bitmap"]["v"]
    assert struct_bitmap is not None and field_bitmap is not None
    assert not is_null_struct(0, struct_bitmap, field_bitmap)
    assert is_null_struct(1, struct_bitmap, field_bitmap)   # null field, valid row
    assert is_null_struct(2, struct_bitmap, field_bitmap)   # null row


def test_colliding_struct_field_name_raises():
    ids = pa.array([1, 2], type=pa.int64())
    orders = pa.array([{"id": 10, "total": 1.5}, {"id": 20, "total": 2.5}],
                      type=pa.struct([("id", pa.int64()), ("total", pa.float64())]))
    batch = pa.RecordBatch.from_arrays([ids, orders], names=["id", "order"])
    with pytest.raises(ValueError, match="'id'"):
        run_batch(batch)


def test_colliding_names_raise_whichever_column_comes_first():
    ids = pa.array([1, 2], type=pa.int64())
    orders = pa.array([{"id": 10}, {"id": 20}], type=pa.struct([("id", pa.int64())]))
    batch = pa.RecordBatch.from_arrays([ids, orders], names=["id", "order"])
    for order in (["id", "order"], ["order", "id"]):
        with pytest.raises(ValueError, match="'id'"):
            run_batch(batch, input_columns=order)


def test_struct_field_named_after_its_own_column_raises():
    col = pa.array([{"s": 1}], type=pa.struct([("s", pa.int64())]))
    with pytest.raises(ValueError, match="'s'"):
        run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))


def test_null_free_column_arrives_as_none():
    # The same schema over two batches, one with a null and one without. A UDF
    # indexing bitmap_dict must find the key in both.
    schema_col = "magnitude"
    with_null = pa.array([1.0, None, 3.0], type=pa.float64())
    without_null = pa.array([1.0, 2.0, 3.0], type=pa.float64())
    keys = []
    for col in (with_null, without_null):
        seen = run_batch(pa.RecordBatch.from_arrays([col], names=[schema_col]))
        keys.append(sorted(seen["bitmap"]))
        assert schema_col in seen["bitmap"]
    assert keys[0] == keys[1] == [schema_col]


def test_null_free_column_of_every_supported_top_level_type():
    columns = {
        "i32": pa.array([1, 2], type=pa.int32()),
        "i64": pa.array([1, 2], type=pa.int64()),
        "f64": pa.array([1.0, 2.0], type=pa.float64()),
        "b": pa.array([True, False], type=pa.bool_()),
        "s": pa.array(["a", "b"], type=pa.string()),
        "ls": pa.array(["a", "b"], type=pa.large_string()),
        "d32": pa.array([1, 2], type=pa.int32()).cast(pa.date32()),
        "d64": pa.array([86400000, 172800000], type=pa.int64()).cast(pa.date64()),
        "ts": pa.array([1, 2], type=pa.int64()).cast(pa.timestamp("ms")),
    }
    batch = pa.RecordBatch.from_arrays(list(columns.values()), names=list(columns))
    seen = run_batch(batch)
    assert sorted(seen["bitmap"]) == sorted(columns)
    assert all(seen["bitmap"][name] is None for name in columns)


def test_list_of_struct_column_keys_the_struct_bitmap_by_column_name():
    col = pa.array([[{"v": 1}, {"v": 2}], [{"v": 3}, {"v": 4}]],
                   type=pa.list_(pa.struct([("v", pa.int64())])))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["rows"]))
    assert sorted(seen["bitmap"]) == ["rows", "v"]
    assert sorted(seen["data"]) == ["v"]
    assert seen["data"]["v"].tolist() == [1, 2, 3, 4]
