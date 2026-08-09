"""
Default configuration options for Numba JIT compilation used throughout numbarrow.
"""

import os
import json


def get_jit_options():
    """
    E.g., export NUMBARROW_JIT_OPTIONS='{"cache": false}'
    """
    as_str = os.environ.get("NUMBARROW_JIT_OPTIONS")
    if as_str is None or as_str.strip() == "":
        return {"cache": True}
    try:
        as_json = json.loads(as_str)
    except json.JSONDecodeError as e:
        raise ValueError("NUMBARROW_JIT_OPTIONS must be valid JSON") from e
    if not isinstance(as_json, dict):
        raise ValueError("NUMBARROW_JIT_OPTIONS must be a JSON object")
    return as_json


jit_options = get_jit_options()
