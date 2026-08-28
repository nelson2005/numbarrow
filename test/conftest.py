import os
import shutil
import sys

import pytest


def spark_unavailable_reason():
    """Reason a SparkSession cannot start here, or None when one can."""
    if sys.platform != "linux":
        return "spark tests run on linux only, this platform is %s" % sys.platform
    # Mirrors how pyspark picks a java binary: a non-empty JAVA_HOME wins
    # outright and is never checked against the filesystem, so a JAVA_HOME
    # pointing nowhere fails even when java is on PATH. Only an unset
    # JAVA_HOME falls back to PATH.
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        runner = os.path.join(java_home, "bin", "java")
        if not os.access(runner, os.X_OK):
            return "JAVA_HOME is set but %s is not executable" % runner
        return None
    if shutil.which("java") is None:
        return "JAVA_HOME is unset and there is no java on PATH"
    return None


@pytest.fixture(scope="session")
def spark():
    reason = spark_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    # Imported here rather than at module scope so a missing pyspark skips
    # these tests instead of aborting collection for the whole suite.
    from pyspark.sql import SparkSession
    # Workers launch whatever python3 PATH resolves to unless this is set,
    # and that interpreter cannot import numbarrow.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    # Bind the driver to loopback rather than letting spark pick whichever
    # interface the hostname resolves to.
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    session = (
        SparkSession
        .builder
        .appName("numba arrow runner")
        .master("local[2]")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
