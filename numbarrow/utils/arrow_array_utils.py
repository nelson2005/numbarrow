"""
Utilities for extracting data from PyArrow array buffers as NumPy arrays.

Handles uniform arrays (fixed-width elements), string arrays (variable-length
with offset buffers), struct arrays, and list-of-struct arrays. Validity
bitmaps are extracted as uint8 arrays for use with :func:`~numbarrow.core.is_null.is_null`.
"""

import ctypes
import numpy as np
import pyarrow as pa

from numbarrow.utils.utils import arrays_viewers


# The numpy view type for each Arrow type that has one, covering exactly the
# keys of ``arrays_viewers``. Resolved directly rather than through
# ``pa.DataType.to_pandas_dtype()``, which imports pandas, and so made a stock
# install unable to adapt an int or a float column.
# ``pa.bool_()`` is deliberately absent even though ``np.bool_`` is a key of
# ``arrays_viewers``: Arrow packs booleans one bit per element, so there is no
# uniform view of them, and the old ``to_pandas_dtype()`` lookup succeeded with
# an itemsize-1 viewer and returned wrong values. Boolean arrays go through the
# dispatcher's own handler, which unpacks the bits.
arrow_to_numpy_dtypes = {
    pa.float64(): np.float64,
    pa.int32(): np.int32,
    pa.int64(): np.int64,
    pa.uint8(): np.uint8,
}


def create_bitmap(bitmap_buf: pa.Buffer | None, offset: int = 0, length: int = 0):
    """ Create numpy array of uint8 type containing
    bit-map of valid array entries, adjusted for array offset.

    The returned array always owns its memory. The offset path below already
    produced a fresh array through ``np.packbits``; the offset-0 path copies
    so that ownership does not depend on a property the return value does not
    carry. Without the copy an offset-0 bitmap aliases the Arrow validity
    buffer, so writing to it nulls out the source, it reads as all-null once
    the source is collected, and a short slice hands back bits the slice does
    not own. """
    if bitmap_buf is None:
        return None
    bitmap_p = bitmap_buf.address
    bitmap_len = bitmap_buf.size
    bitmap_viewer = arrays_viewers[np.uint8]
    raw_bitmap = bitmap_viewer(bitmap_p, bitmap_len)
    if length == 0:
        return raw_bitmap[:0].copy()
    num_bytes = (length + 7) // 8
    if offset == 0:
        return raw_bitmap[:num_bytes].copy()
    # Re-pack bitmap bits starting from the offset bit position,
    # only reading the byte range that covers [offset, offset+length)
    first_byte = offset // 8
    last_byte = (offset + length + 7) // 8
    relevant = raw_bitmap[first_byte:last_byte]
    bit_offset = offset % 8
    bits = np.unpackbits(relevant, bitorder="little")
    sliced_bits = bits[bit_offset:bit_offset + length]
    pad = (-length) % 8
    if pad:
        sliced_bits = np.pad(sliced_bits, (0, pad), mode="constant")
    result = np.packbits(sliced_bits, bitorder="little")
    # Copied rather than sliced so that base is None on this path too: the
    # docstring above promises ownership, and a slice of the packbits array
    # keeps its parent alive instead.
    return result[:num_bytes].copy()


def create_str_array(pa_str_array: pa.StringArray | pa.LargeStringArray) -> tuple[np.ndarray | None, np.ndarray]:
    """Copy data from a densely packed PyArrow string array into a padded
    NumPy Unicode array.

    StringArray uses int32 offsets; LargeStringArray uses int64 offsets.
    """
    bitmap_buf, offsets_buf, data_buf = pa_str_array.buffers()
    data_p = data_buf.address
    offsets_p = offsets_buf.address
    offset_dtype = np.int64 if pa.types.is_large_string(pa_str_array.type) else np.int32
    offsets_len = offsets_buf.size // np.dtype(offset_dtype).itemsize
    offsets_array = arrays_viewers[offset_dtype](offsets_p, offsets_len)
    offset = pa_str_array.offset
    n = len(pa_str_array)
    logical_offsets = offsets_array[offset:offset + n + 1]
    if n == 0:
        # Read-only like the non-empty path below, so that the contract does
        # not depend on whether the batch happened to be empty.
        empty = np.empty((0,), dtype="|U1")
        empty.flags.writeable = False
        return create_bitmap(bitmap_buf, pa_str_array.offset, 0), empty
    bounds = np.asarray(logical_offsets, dtype=np.int64)
    # A null slot's offsets are not required to be empty: pc.if_else leaves the
    # masked-out value's bytes in place. Those bytes belong to no logical value,
    # so decoding them would hand the caller a value it explicitly masked out,
    # size the array from a string that is not there, and raise
    # UnicodeDecodeError on a spec-legal array whose dead bytes are not utf-8.
    live = np.ones(n, dtype=bool)
    if pa_str_array.null_count:
        live = pa_str_array.is_valid().to_numpy(zero_copy_only=False).astype(bool)
    # The width is a character count. A difference of two Arrow offsets counts
    # utf-8 bytes instead, which over-allocates by up to 4x on non-ASCII input,
    # and numpy holds the whole array resident at that width. In utf-8 every
    # byte that is not a 0b10xxxxxx continuation byte starts exactly one
    # character, so the character counts come from one vectorised pass over the
    # bytes rather than from a second decode of every element.
    span_start, span_stop = int(bounds[0]), int(bounds[n])
    if span_stop > span_start:
        raw = np.frombuffer(memoryview(data_buf), dtype=np.uint8,
                            count=span_stop - span_start, offset=span_start)
        starts_a_char = (raw & 0xC0) != 0x80
        # reduceat sums raw[idx[i]:idx[i+1]]; for an empty range it yields
        # raw[idx[i]] instead of 0, which can only overstate an empty slot by
        # one and so cannot make the width too small.
        # An empty trailing element starts at the very end of the span, and
        # reduceat rejects an index equal to the length. One padding element
        # makes that index legal; it can only ever be summed into an empty
        # range, which the next line zeroes. Clamping instead would be wrong,
        # since an element's start index is also the previous element's upper
        # bound, so lowering it drops a character from the element before.
        padded = np.append(starts_a_char, False)
        char_counts = np.add.reduceat(padded, bounds[:n] - span_start)
        char_counts = np.where(bounds[1:] > bounds[:n], char_counts, 0)
    else:
        char_counts = np.zeros(n, dtype=np.int64)
    item_sz = int(char_counts[live].max()) if live.any() else 0
    str_array = np.empty((n,), dtype=f"|U{item_sz}")
    for i in range(n):
        if not live[i]:
            continue
        start = int(bounds[i])
        length = int(bounds[i + 1]) - start
        str_array[i] = ctypes.string_at(data_p + start, length).decode("utf-8")
    bitmap = create_bitmap(bitmap_buf, offset, n)
    # A fresh numpy allocation, but read-only all the same, so that the
    # contract does not depend on which Arrow type the caller happened to pass.
    str_array.flags.writeable = False
    return bitmap, str_array


# Two-layer struct nullability inspired by Awkward Array's BitMaskedArray(RecordArray)
# design. See: https://awkward-array.org/doc/main/reference/generated/ak.contents.BitMaskedArray.html


def structured_array_adapter(struct_array: pa.StructArray) -> tuple[
    np.ndarray | None, dict[str, np.ndarray | None], dict[str, np.ndarray]
]:
    """
    NumPy adapter of PyArrow `StructArray`.

    Returns a 3-tuple:
    - struct-level validity bitmap (None if all rows valid)
    - dict mapping field names to per-field validity bitmaps
    - dict mapping field names to per-field value arrays
    """
    # Imported here rather than at module scope because the dispatcher's
    # handlers are defined in terms of the adapters in this module.
    from numbarrow.core.adapters import arrow_array_adapter

    assert isinstance(struct_array, pa.StructArray)
    data_type: pa.StructType = struct_array.type
    assert isinstance(data_type, pa.StructType)
    struct_bitmap_buf = struct_array.buffers()[0]
    struct_bitmap = create_bitmap(
        struct_bitmap_buf, struct_array.offset, len(struct_array)
    )
    names = [data_type[i].name for i in range(len(data_type))]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        # Arrow permits repeated field names; keying by name cannot represent
        # them, and struct_array.field(name) raises a bare KeyError naming
        # neither the struct nor the cause.
        raise NotImplementedError(
            f"Not implemented for a struct with repeated field names "
            f"{sorted(duplicates)} in {struct_array.type}"
        )
    bitmaps = {}
    datas = {}
    for field_ind in range(len(data_type)):
        field: pa.Field = data_type[field_ind]
        field_name = field.name
        pa_array = struct_array.field(field_ind)
        # Each child goes through the dispatcher rather than straight to
        # uniform_arrow_array_adapter. A boolean child is bit-packed, so the
        # uniform viewer would read it at one byte per element over a buffer
        # holding one bit per element; and a child with more than two buffers
        # has no uniform layout to view at all, so it needs its own handler or
        # the dispatcher's typed error.
        adapted = arrow_array_adapter(pa_array)
        if len(adapted) != 2:
            raise NotImplementedError(
                f"Not implemented for struct field {field_name!r} of type {pa_array.type}: "
                f"a structured child adapts to per-field dictionaries, which do not fit "
                f"a single field's bitmap and data"
            )
        bitmap, data = adapted
        bitmaps[field_name] = bitmap
        datas[field_name] = data
    return struct_bitmap, bitmaps, datas


def structured_list_array_adapter(list_array: pa.ListArray) -> tuple[
    np.ndarray | None, dict[str, np.ndarray | None], dict[str, np.ndarray]
]:
    """
    NumPy adapter of PyArrow array of same-length lists of structures.

    :param list_array: PyArrow array with elements being of `pa.ListType`.
        Each list is in turn of the same length, and each element of the list
        is of `pa.StructType`.

    Returns a 3-tuple of: the struct-level validity bitmap (or ``None`` if
    all values are valid), a dictionary mapping field names to per-field
    validity bitmaps (each ``None`` if all values are valid), and a
    dictionary mapping field names to the contiguous field data arrays.

    Whether a field's data is copied depends on the field's type. A
    fixed-width child is a zero-copy view over the contiguous
    `pa.StructArray` values; a boolean child is bit-unpacked and a string
    child is repacked into fixed-width Unicode, so both of those are copies.
    Every returned data array is read-only either way.

    The returned arrays are the flattened elements of every list in
    ``list_array``, with no offsets telling a caller where one row's elements
    end and the next row's begin. Mapping an element back to its row therefore
    assumes every list has the same length, which is what the ``:param:``
    contract above requires.
    """
    assert isinstance(list_array, pa.ListArray)
    values: pa.Array = list_array.values
    if not pa.types.is_struct(values.type):
        # The array itself is deliberately not interpolated: its repr carries
        # every value it holds, which grows without bound and puts the
        # caller's data into whatever log catches the traceback.
        raise NotImplementedError(
            f"Not implemented for a list array of {len(list_array)} rows of "
            f"type {list_array.type}: elements are {values.type}, not a struct"
        )
    # `values` is the whole child array and ignores this array's own offset,
    # so a sliced or offset list column would hand back the elements of rows
    # it does not contain. The offsets are absolute into `values`, so the
    # first and last of them bound exactly the elements these rows own.
    if len(list_array) == 0:
        # The Arrow spec lets a zero-length array carry null buffers, and
        # pyarrow's own full validation accepts one. Indexing a null offsets
        # buffer segfaults rather than raising, so there is nothing to read
        # here and nothing that needs reading.
        data_values: pa.StructArray = values.slice(0, 0)
    else:
        offsets = list_array.offsets
        start = offsets[0].as_py()
        stop = offsets[len(list_array)].as_py()
        data_values = values.slice(start, stop - start)
    return structured_array_adapter(data_values)


def uniform_arrow_array_adapter(pa_array: pa.Array) -> tuple[np.ndarray | None, np.ndarray]:
    """ NumPy adapter for PyArrow arrays with uniformly sized elements.

    Returns the validity bitmap, which owns its memory, and a zero-copy numpy
    view over the array's data buffer. The view is read-only: Arrow buffers are
    immutable by contract, and this is what pyarrow's own
    ``Array.to_numpy(zero_copy_only=True)`` returns. Declare numba signatures
    that receive it with ``readonly=True``, which accepts writable arrays too.
    """
    data_arrow_ty = pa_array.type
    data_np_ty = arrow_to_numpy_dtypes.get(data_arrow_ty, None)
    if data_np_ty is None:
        # Resolved before unpacking the buffers, so that a type with a
        # different buffer count reports its type rather than dying on the
        # unpack with "too many values to unpack".
        raise ValueError(
            f"There is no numpy view type for {data_arrow_ty} in "
            f"`arrow_array_utils.arrow_to_numpy_dtypes`. Add it?"
        )
    bitmap_buf, data_buf = pa_array.buffers()
    if data_buf is None:
        raise ValueError(f"{data_arrow_ty} array of length {len(pa_array)} has no data buffer")
    data_item_byte_size = np.dtype(data_np_ty).itemsize
    data_len = len(pa_array)
    if data_len == 0:
        # An empty array's offset may still point past the end of an empty
        # buffer, where np.frombuffer raises a bare "buffer is smaller than
        # requested size" naming neither the array nor the type.
        empty = np.empty((0,), dtype=data_np_ty)
        # Read-only like the non-empty path below, so that the contract does
        # not depend on whether the batch happened to be empty.
        empty.flags.writeable = False
        return create_bitmap(bitmap_buf, pa_array.offset, 0), empty
    # Viewed through the buffer rather than through its raw address, so that
    # the result's ``.base`` chain reaches the pyarrow buffer and keeps it
    # alive. A view built over a bare address owns nothing and refers to
    # nothing, so once the caller drops the last reference to the source the
    # result reads freed memory: correct at tiny sizes, wrong from a few
    # hundred elements, and a segfault once the allocation is large enough to
    # be returned to the system.
    data = np.frombuffer(
        memoryview(data_buf),
        dtype=data_np_ty,
        count=data_len,
        offset=pa_array.offset * data_item_byte_size
    )
    # np.frombuffer only reports read-only for buffers pyarrow marks immutable,
    # which depends on where the buffer came from: a locally built array is
    # mutable, while the same data after an Arrow IPC round trip, which is the
    # transport mapInArrow uses, is not. Clearing the flag makes the result the
    # same either way, instead of a UDF compiling in a unit test and failing on
    # a real Spark batch.
    data.flags.writeable = False
    bitmap = create_bitmap(bitmap_buf, pa_array.offset, len(pa_array))
    return bitmap, data
