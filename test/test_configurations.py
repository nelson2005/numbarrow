import json
import os
import subprocess
import sys

PRINT_OPTIONS = "import json; from numbarrow.core.configurations import jit_options; print(json.dumps(jit_options))"


def run_with_env(env_value):
    env = {k: v for k, v in os.environ.items() if k != "NUMBARROW_JIT_OPTIONS"}
    if env_value is not None:
        env["NUMBARROW_JIT_OPTIONS"] = env_value
    return subprocess.run(
        [sys.executable, "-c", PRINT_OPTIONS], capture_output=True, text=True, env=env
    )


def test_jit_options_default():
    result = run_with_env(None)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"cache": True}


def test_jit_options_override():
    result = run_with_env('{"cache": false}')
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"cache": False}


def test_jit_options_empty_string_falls_back():
    result = run_with_env("")
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"cache": True}


def test_jit_options_invalid_json_rejected():
    result = run_with_env("{cache: false}")
    assert result.returncode != 0
    assert "NUMBARROW_JIT_OPTIONS must be valid JSON" in result.stderr


def test_jit_options_non_object_rejected():
    result = run_with_env("[1, 2]")
    assert result.returncode != 0
    assert "NUMBARROW_JIT_OPTIONS must be a JSON object" in result.stderr


if __name__ == "__main__":
    test_jit_options_default()
    test_jit_options_override()
    test_jit_options_empty_string_falls_back()
    test_jit_options_invalid_json_rejected()
    test_jit_options_non_object_rejected()
