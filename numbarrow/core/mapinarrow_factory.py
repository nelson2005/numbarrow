"""
Factory for PySpark ``mapInArrow`` UDF functions.

Bridges PySpark's Arrow-based batch processing with Numba JIT-compiled functions
by converting each :class:`pyarrow.RecordBatch` column through
:func:`~numbarrow.core.adapters.arrow_array_adapter` before passing the data
to a user-supplied computation function.
"""

import numpy as np
import pyarrow as pa

from typing import Callable

from numbarrow.core.adapters import arrow_array_adapter


def _fold_struct_validity(struct_bitmap, field_bitmap):
    """Combine a struct's own validity bits into one field's bits.

    A row that is null as a whole leaves its fields' own bitmaps untouched, so
    a field bitmap alone cannot see it. Arrow validity is 1 for valid, so the
    two layers combine with a bitwise and.
    """
    if struct_bitmap is None:
        return field_bitmap
    if field_bitmap is None:
        # Copied per field so that two fields do not share one array, where a
        # caller writing through one would change the other.
        return struct_bitmap.copy()
    return struct_bitmap & field_bitmap


def _claim_keys(target: dict, owners: dict[str, str], additions: dict, col: str, kind: str):
    """Merge one column's contribution into a batch-wide dict.

    Struct fields are keyed by field name rather than by column, so two
    columns can produce the same key. Overwriting one with the other loses a
    column and its null information silently, and misaligns the row count when
    the two columns have different lengths, so a collision raises instead.
    """
    for key, value in additions.items():
        owner = owners.get(key)
        if owner is not None:
            if owner == col:
                raise ValueError(
                    f"{kind}_dict key {key!r} is claimed twice by column {col!r}: a struct "
                    f"field shares its name with the column that contains it."
                )
            raise ValueError(
                f"{kind}_dict key {key!r} is claimed by column {owner!r} and again by column "
                f"{col!r}. Struct fields are keyed by field name, so a field sharing a name "
                f"with another column would replace it."
            )
        owners[key] = col
        target[key] = value


def make_mapinarrow_func(
    main_func: Callable,
    input_columns: list[str] | None = None,
    broadcasts: dict | None = None
):
    """
    Creates a function that can be given as an argument to `mapInArrow`

    :param main_func: called once per :class:`pyarrow.RecordBatch` as
        ``main_func(data_dict, bitmap_dict, broadcasts)``, returning a
        ``dict[str, np.ndarray]`` that is used to create a PyArrow
        ``RecordBatch``.

        ``data_dict`` maps a name to an array of data of a supported type.  A
        column of a uniform type contributes one entry under the column name;
        a structured column contributes one entry per field, under the field
        name.

        ``bitmap_dict`` maps the same names to uint8 aligned arrays of bitmap
        data, or to ``None`` where every value is valid.  Every name in
        ``data_dict`` is present, so a null-free batch is indexable exactly
        like a batch containing nulls.

        For a structured column the struct-level validity is folded into each
        field's bitmap, so one :func:`~numbarrow.core.is_null.is_null` call per
        field sees both a null field and a row that is null as a whole.  Adapt
        the column directly with
        :func:`~numbarrow.core.adapters.arrow_array_adapter` to get the two
        layers separately, for
        :func:`~numbarrow.core.is_null.is_null_struct`.

        A name may be claimed only once.  A struct field sharing a name with
        another selected column raises :class:`ValueError` rather than
        replacing it.

    :param input_columns: optional list of column names that will be expected
        to be needed for in `data_dict` for the calculation done by
        `main_func`. When not given, all columns in the iterated over PySpark
        DataFrame will be used.
    :param broadcasts: optional dictionary of broadcast values
    """
    broadcasts = broadcasts if broadcasts is not None else {}

    def _(iterator):
        for batch in iterator:
            data_dict: dict[str, np.ndarray] = {}
            bitmap_dict: dict[str, np.ndarray | None] = {}
            data_owners: dict[str, str] = {}
            bitmap_owners: dict[str, str] = {}
            requested = input_columns if input_columns is not None else batch.schema.names
            # dict.fromkeys keeps first-seen order. Naming a column twice was
            # harmless before the duplicate-key check existed, because the
            # second pass produced the same arrays, so it stays harmless.
            input_columns_ = list(dict.fromkeys(requested))
            for col in input_columns_:
                col_pa: pa.Array = batch.column(col)
                adapted = arrow_array_adapter(col_pa)
                if len(adapted) == 3:
                    struct_bitmap, col_bitmaps, col_datas = adapted
                    # The struct-level bitmap is folded into each field rather
                    # than published under the column's own name. Publishing it
                    # would claim a key that another column's field may already
                    # be called, turning schemas that worked into a hard error,
                    # and for a list-of-struct column the name would suggest it
                    # covers the outer list rows, which it does not.
                    col_bitmaps = {
                        name: _fold_struct_validity(struct_bitmap, field_bitmap)
                        for name, field_bitmap in col_bitmaps.items()
                    }
                else:
                    col_bitmap, col_data = adapted
                    col_bitmaps = {col: col_bitmap}
                    col_datas = {col: col_data}
                _claim_keys(data_dict, data_owners, col_datas, col, "data")
                _claim_keys(bitmap_dict, bitmap_owners, col_bitmaps, col, "bitmap")
            outputs = main_func(data_dict, bitmap_dict, broadcasts)
            yield pa.RecordBatch.from_pydict({col: pa.array(output) for col, output in outputs.items()})
    return _
