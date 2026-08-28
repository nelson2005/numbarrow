# numbarrow

Numba adapters for [PyArrow](https://arrow.apache.org/docs/python/) and [PySpark](https://spark.apache.org/docs/latest/api/python/).

numbarrow lets you work with Apache Arrow arrays directly inside Numba `@njit` compiled functions. It converts PyArrow arrays into NumPy views (zero-copy where possible) and extracts validity bitmaps for null handling — bridging PySpark's Arrow-based batch processing with high-performance JIT-compiled code.

## Installation

```bash
pip install numbarrow
```

Optional dependencies for PySpark and pandas support:

```bash
pip install numbarrow[test]       # adds pyspark and everything the tests need
pip install numbarrow[mapinarrow] # adds pandas, which pyspark's mapInArrow requires
```

The adapters themselves need only numba, numpy and pyarrow.

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
        if bitmap is None or not is_null(i, bitmap):
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
| `UInt8Array` | `uint8` | No (view) |
| `StringArray`, `LargeStringArray` | Fixed-width Unicode, width in characters | Yes (repacking) |
| `StructArray` | 3-tuple: struct bitmap, bitmaps by field, data by field | Per-field |
| `ListArray` (of structs) | 3-tuple: struct bitmap, bitmaps by field, data by field | Per-field |

Every other array type raises `NotImplementedError` naming the type, including a
`ListArray` whose elements are not structs.

Returned data arrays are read-only, matching
`pyarrow.Array.to_numpy(zero_copy_only=True)`, because they view Arrow buffers
the caller does not own. Declare numba signatures that receive them with
`readonly=True`, which accepts writable arrays as well, or leave the function
lazily typed and numba will infer it. Returned bitmaps own their memory and are
writable.

A uniform array adapts to a 2-tuple, `(bitmap, data)`, where `bitmap` is `None`
when the array has no validity buffer. A struct or list-of-struct array adapts
to a 3-tuple: the struct-level bitmap, then two dicts keyed by field name. The
struct-level bitmap is the only record of a row that is null as a whole, since
the fields of such a row carry no validity bits of their own; pass both layers
to `is_null_struct`.

## PySpark Integration

Use `make_mapinarrow_func` to create functions compatible with PySpark's `mapInArrow`:

```python
from numbarrow.core.mapinarrow_factory import make_mapinarrow_func

def compute(data_dict, bitmap_dict, broadcasts):
    # data_dict:   {name: np.ndarray}, one entry per column, or one per field
    #              for a struct column
    # bitmap_dict: the same names, each a uint8 bitmap or None where every
    #              value is valid; for a struct column the struct-level
    #              validity is folded into each field's bitmap
    result = data_dict["value"] * broadcasts["scale"]
    return {"output": result}

udf = make_mapinarrow_func(compute, broadcasts={"scale": 2.0})
df_in = ...           # caller-provided PySpark DataFrame
output_schema = ...   # caller-provided PySpark StructType
df_out = df_in.mapInArrow(udf, output_schema)
```

See [test/test_mapinarrow_spark.py](test/test_mapinarrow_spark.py) for a complete runnable example.

## Compatibility

| Dependency | Versions |
|---|---|
| Python | 3.12+ |
| numba | 0.60.0 – 0.67.0 |
| pyarrow | 16.1 – 24.0 |
| pyspark | 3.3 – 3.x (optional) |
| pandas | 1.5+ (optional, required by pyspark's `mapInArrow`) |

`pyproject.toml` is authoritative; the ranges above are the ones CI runs. CI
additionally exercises Python 3.10 and 3.11 with `--ignore-requires-python`,
because the package still builds and passes there. Treat those as regression
signal rather than as a supported configuration: pip refuses the install below
the declared floor. pyarrow 14 and 15 are excluded by measurement rather than
by policy, since they are built against numpy 1 and fail to import once numba
resolves numpy 2.

## Documentation

Full API documentation: [numbarrow docs](https://goykhman.github.io/numbarrow)

## License

See [LICENSE](LICENSE).
