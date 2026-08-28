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
        # data_dict:   {name: np.ndarray}, keyed by column, or by field name
        #              for a struct column
        # bitmap_dict: the same names, each a uint8 bitmap or None where every
        #              value is valid; for a struct column the struct-level
        #              validity is folded into each field's bitmap
        # broadcasts:  {key: value}
        result = ...
        return {"output_col": result}

Every name in ``data_dict`` is also a key of ``bitmap_dict``, so a batch that
happens to contain no nulls is indexable exactly like one that does. For a
structured column the struct-level validity is folded into each field's bitmap,
so a row that is null as a whole is visible to one ``is_null`` call per field;
adapt the column directly to get the two layers separately. A name may
be claimed only once: a struct field sharing a name with another selected column
raises ``ValueError`` rather than replacing it.

    udf = make_mapinarrow_func(my_func, broadcasts={"scale": 1.5})
    df_out = df_in.mapInArrow(udf, output_schema)

See ``test_mapinarrow_spark.py`` in the `test suite
<https://github.com/Goykhman/numbarrow/tree/main/test>`_ for a complete runnable example.

Module
++++++

.. automodule:: numbarrow.core.mapinarrow_factory
   :members:
   :show-inheritance:
   :undoc-members:
