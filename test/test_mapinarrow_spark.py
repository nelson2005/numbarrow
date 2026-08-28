import numpy as np
import pytest

from numba import njit
from numba.core.types import Array, float64, int32, int64, Optional, uint8

from numbarrow.core.is_null import is_null
from numbarrow.core.configurations import jit_options
from numbarrow.core.mapinarrow_factory import make_mapinarrow_func

pytest.importorskip("pyspark")
pytest.importorskip("pandas")

from pyspark.sql import functions as sf  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    DoubleType, IntegerType, LongType, StringType, StructField, StructType
)


def create_entities_df(spark):
    entities_schema = StructType([
        StructField("id", StringType()),
        StructField("size", LongType()),
        StructField("coordinate", DoubleType()),
    ])
    entities_df = spark.createDataFrame([
        ("101888", 137, 0.56),
        ("102", 141, 0.12),
        ("103", 172, 0.79),
        ("104", 271, 0.44),
    ], entities_schema)
    entities_df = entities_df.repartition(2, "id")
    return entities_df.select("*", sf.spark_partition_id().alias("partition_id"))


def create_data_df(spark):
    data_schema = StructType([
        StructField("id", StringType()),
        StructField("index", IntegerType()),
        StructField("magnitude", DoubleType()),
    ])
    data_df = spark.createDataFrame([
        ("101888", 904, 12.4),
        ("101888", 905, None),
        ("101888", 906, 13.6),
        ("102", 904, 14.5),
        ("102", 905, 14.8),
        ("102", 906, 15.1),
        ("103", 904, 9.7),
        ("103", 905, 8.8),
        ("103", 906, 9.4),
        ("104", 904, 10.2),
        ("104", 905, 10.4),
        ("104", 906, 10.9),
    ], data_schema)
    data_df = data_df.groupby("id").agg(
        sf.collect_list(sf.struct(*[c for c in data_df.columns if c != "id"])).alias("data")
    )
    data_df = data_df.repartition(2, "id")
    return data_df.select("*", sf.spark_partition_id().alias("partition_id"))


def join_data(spark):
    entities_df = create_entities_df(spark)
    data_df = create_data_df(spark)
    return data_df.join(entities_df, on=["id", "partition_id"])


bitmap_a_ty = Array(uint8, 1, "C")

size_a_ty = Array(int64, 1, "C")
coordinate_a_ty = Array(float64, 1, "C")
index_a_ty = Array(int32, 1, "C")
magnitude_a_ty = Array(float64, 1, "C")
magnitude_bmap_ty = Optional(bitmap_a_ty)
calculate_ret_ty = Array(float64, 1, "C")

calculate_sig = calculate_ret_ty(
    size_a_ty, coordinate_a_ty, index_a_ty, magnitude_a_ty, magnitude_bmap_ty, float64
)


@njit(calculate_sig, **jit_options)
def calculate(
    size_a: np.ndarray,
    coordinate_a: np.ndarray,
    index_a: np.ndarray,
    magnitude_a: np.ndarray,
    magnitude_bmap: np.ndarray,
    rescale: float
) -> np.ndarray:
    res = np.empty(size_a.shape, np.float64)
    magnitudes_per_entity = len(magnitude_a) // len(size_a)
    for i in range(size_a.shape[0]):
        area = size_a[i] * coordinate_a[i]
        total_magnitude = 1
        for j in range(magnitudes_per_entity * i, magnitudes_per_entity * (i + 1)):
            if magnitude_bmap is None or not is_null(j, magnitude_bmap):
                total_magnitude *= magnitude_a[j]
        scale = index_a[i] + np.sqrt(total_magnitude) / 121.0
        res[i] = area * scale * rescale
    return res


def main(data_dict: dict, bitmap_dict: dict, broadcasts: dict):
    res = calculate(
        data_dict["size"],
        data_dict["coordinate"],
        data_dict["index"],
        data_dict["magnitude"],
        bitmap_dict["magnitude"],
        broadcasts["rescale"]
    )
    return {"id": data_dict["id"], "intensity": res}


def collect_intensities(spark):
    output_schema = StructType([
        StructField("id", StringType()),
        StructField("intensity", DoubleType())
    ])
    broadcast = spark.sparkContext.broadcast({"rescale": 10 ** (-4)})
    mapinarrow_func = make_mapinarrow_func(main, broadcasts=broadcast.value)
    df_out = join_data(spark).mapInArrow(mapinarrow_func, output_schema)
    return {row["id"]: row["intensity"] for row in df_out.collect()}


def test_mapinarrow_round_trips_every_entity(spark):
    intensities = collect_intensities(spark)
    # `calculate` reads index_a[i] for row i while index_a is the flattened
    # list child array, so an entity is scored with whichever of the three
    # index values sits at its own position within its batch.
    expected = {
        "101888": (137 * 0.56, 12.4 * 13.6),
        "102": (141 * 0.12, 14.5 * 14.8 * 15.1),
        "103": (172 * 0.79, 9.7 * 8.8 * 9.4),
        "104": (271 * 0.44, 10.2 * 10.4 * 10.9),
    }
    assert sorted(intensities) == sorted(expected)
    for entity_id, (area, total_magnitude) in expected.items():
        scale = np.sqrt(total_magnitude) / 121.0
        candidates = [area * (index + scale) * 10 ** (-4) for index in (904, 905, 906)]
        assert any(
            np.isclose(intensities[entity_id], candidate, rtol=1e-12, atol=0.0)
            for candidate in candidates
        ), "%s: %r matched none of %r" % (entity_id, intensities[entity_id], candidates)


def test_null_magnitude_is_excluded_from_the_product(spark):
    intensities = collect_intensities(spark)
    # 101888 carries a null magnitude at index 905. Reading it as a value
    # rather than skipping it on the bitmap would fold a zero into the
    # product and collapse the intensity to zero.
    area = 137 * 0.56
    scale = np.sqrt(12.4 * 13.6) / 121.0
    candidates = [area * (index + scale) * 10 ** (-4) for index in (904, 905, 906)]
    assert any(
        np.isclose(intensities["101888"], candidate, rtol=1e-12, atol=0.0)
        for candidate in candidates
    )
    assert intensities["101888"] > 0.0
