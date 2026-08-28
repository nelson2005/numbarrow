import numpy as np
import pyarrow as pa
import pytest

from numbarrow.core.is_null import is_null
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


def nulls(bitmap, n):
    return [is_null(i, bitmap) for i in range(n)]


def test_struct_null_row_reaches_the_udf():
    # Plain pa.array, from_pylist and StructArray.from_arrays(mask=...) all emit
    # a struct-null row with no child validity buffer, so the field's own bitmap
    # cannot see it. The struct-level bits are folded in, so one is_null call
    # per field does.
    col = pa.array([{"v": 10}, None, {"v": 30}], type=pa.struct([pa.field("v", pa.int64())]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    assert sorted(seen["bitmap"]) == ["v"]
    assert nulls(seen["bitmap"]["v"], 3) == [False, True, False]


def test_no_bitmap_when_nothing_is_null_at_either_layer():
    col = pa.array([{"v": 10}, {"v": 30}], type=pa.struct([pa.field("v", pa.int64())]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    assert "v" in seen["bitmap"] and seen["bitmap"]["v"] is None
    assert seen["data"]["v"].tolist() == [10, 30]


def test_both_null_layers_are_folded_together():
    inner = pa.array([1, None, 3], type=pa.int64())
    col = pa.StructArray.from_arrays([inner], ["v"], mask=pa.array([False, False, True]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    #                      valid, null field, null row
    assert nulls(seen["bitmap"]["v"], 3) == [False, True, True]


def test_a_field_named_after_its_own_column_is_unambiguous():
    # One column, one field, one key: nothing is lost, so nothing should raise.
    col = pa.array([{"s": 1}], type=pa.struct([("s", pa.int64())]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    assert seen["data"]["s"].tolist() == [1]


def test_struct_column_named_after_another_columns_field():
    # A struct column called 'region' next to store: struct<region, sqft>. The
    # data keys are code/region/sqft, all distinct, so this must keep working.
    region = pa.array([{"code": 1}, {"code": 2}], type=pa.struct([("code", pa.int64())]))
    store = pa.array([{"region": 10, "sqft": 1.0}, {"region": 20, "sqft": 2.0}],
                     type=pa.struct([("region", pa.int64()), ("sqft", pa.float64())]))
    batch = pa.RecordBatch.from_arrays([region, store], names=["region", "store"])
    for order in (["region", "store"], ["store", "region"]):
        seen = run_batch(batch, input_columns=order)
        assert sorted(seen["data"]) == ["code", "region", "sqft"]
        assert seen["data"]["code"].tolist() == [1, 2]
        assert seen["data"]["region"].tolist() == [10, 20]


def test_duplicate_input_columns_is_harmless():
    # Naming a column twice produced the same arrays twice before the
    # duplicate-key check existed, so it must not become an error.
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2], type=pa.int64())], names=["a"])
    seen = run_batch(batch, input_columns=["a", "a"])
    assert seen["data"]["a"].tolist() == [1, 2]
    assert sorted(seen["bitmap"]) == ["a"]


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


def test_null_free_column_arrives_as_none():
    # The same schema over two batches, one with a null and one without. A UDF
    # indexing bitmap_dict must find the key in both.
    with_null = pa.array([1.0, None, 3.0], type=pa.float64())
    without_null = pa.array([1.0, 2.0, 3.0], type=pa.float64())
    keys = []
    for col in (with_null, without_null):
        seen = run_batch(pa.RecordBatch.from_arrays([col], names=["magnitude"]))
        keys.append(sorted(seen["bitmap"]))
        assert "magnitude" in seen["bitmap"]
    assert keys[0] == keys[1] == ["magnitude"]


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


def test_list_of_struct_column_keys_by_field_name():
    col = pa.array([[{"v": 1}, {"v": 2}], [{"v": 3}, {"v": 4}]],
                   type=pa.list_(pa.struct([("v", pa.int64())])))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["rows"]))
    assert sorted(seen["bitmap"]) == ["v"]
    assert sorted(seen["data"]) == ["v"]
    assert seen["data"]["v"].tolist() == [1, 2, 3, 4]


def test_empty_batch():
    batch = pa.RecordBatch.from_arrays([pa.array([], type=pa.int64())], names=["a"])
    seen = run_batch(batch)
    assert seen["data"]["a"].tolist() == []
    assert "a" in seen["bitmap"]


def test_zero_field_struct_column_is_refused_rather_than_dropped():
    # A struct with no fields contributes no key, so a requested column used to
    # vanish from both dicts without a word.
    z = pa.array([{}, {}], type=pa.struct([]))
    n = pa.array([1, 2], type=pa.int64())
    batch = pa.RecordBatch.from_arrays([z, n], names=["z", "n"])
    with pytest.raises(ValueError, match="'z'"):
        run_batch(batch)


def test_two_fields_of_one_struct_do_not_share_a_bitmap():
    # With the struct-level bits folded in and no field-level nulls, both
    # fields get the struct bitmap; without a per-field copy they would be the
    # same array, and a caller writing through one would change the other.
    inner_a = pa.array([1, 2], type=pa.int64())
    inner_b = pa.array([3, 4], type=pa.int64())
    col = pa.StructArray.from_arrays([inner_a, inner_b], ["a", "b"],
                                     mask=pa.array([False, True]))
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["s"]))
    bitmap_a, bitmap_b = seen["bitmap"]["a"], seen["bitmap"]["b"]
    assert bitmap_a is not bitmap_b
    assert bitmap_a.tolist() == bitmap_b.tolist()
    bitmap_a[0] = 0
    assert bitmap_b[0] != 0


def test_a_list_column_with_a_null_row_is_documented_as_unsupported():
    # The fold covers the flattened struct elements, never the outer list rows,
    # so a null outer row is invisible AND shifts the element-to-row mapping.
    # Pinned so the limitation cannot change silently.
    ty = pa.list_(pa.struct([("v", pa.int64())]))
    col = pa.array([[{"v": 1}], None, [{"v": 3}]], type=ty)
    assert col.null_count == 1
    seen = run_batch(pa.RecordBatch.from_arrays([col], names=["rows"]))
    assert seen["data"]["v"].tolist() == [1, 3]
    assert seen["bitmap"]["v"] is None
