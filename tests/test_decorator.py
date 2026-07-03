from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from func_capture import capture, func_capture


class CaptureDecoratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._had_env = "FUNC_CAPTURE" in os.environ
        self._old_env = os.environ.get("FUNC_CAPTURE")

    def tearDown(self) -> None:
        if self._had_env:
            assert self._old_env is not None
            os.environ["FUNC_CAPTURE"] = self._old_env
        else:
            os.environ.pop("FUNC_CAPTURE", None)

    def test_missing_env_returns_original_function(self) -> None:
        os.environ.pop("FUNC_CAPTURE", None)

        def target() -> str:
            return "plain"

        self.assertIs(capture(target), target)

    def test_empty_env_returns_original_function(self) -> None:
        os.environ["FUNC_CAPTURE"] = ""

        def target() -> str:
            return "plain"

        self.assertIs(capture(target), target)

    def test_unmatched_key_returns_original_function_without_loading_script(self) -> None:
        os.environ["FUNC_CAPTURE"] = "other.module.fn=/does/not/exist.py"

        def target() -> str:
            return "plain"

        self.assertIs(capture(target), target)

    def test_matching_qualified_function_uses_script_instrument(self) -> None:
        script = self._write_script(
            """
def instrument(func):
    def wrapper(*args, **kwargs):
        return ("captured", func(*args, **kwargs), func.__name__)
    return wrapper
"""
        )

        def target(left: int, right: int) -> int:
            return left + right

        key = f"{target.__module__}.{target.__qualname__}"
        os.environ["FUNC_CAPTURE"] = f"{key}={script}"

        decorated = capture(target)

        self.assertIsNot(decorated, target)
        self.assertEqual(decorated(2, 3), ("captured", 5, "target"))

    def test_explicit_key_can_select_instrumentation(self) -> None:
        script = self._write_script(
            """
def instrument(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs).upper()
    return wrapper
"""
        )
        os.environ["FUNC_CAPTURE"] = json.dumps({"chosen": str(script)})

        def target() -> str:
            return "captured"

        decorated = capture(key="chosen")(target)

        self.assertEqual(decorated(), "CAPTURED")

    def test_positional_key_can_select_instrumentation(self) -> None:
        script = self._write_script(
            """
def instrument(func):
    def wrapper(*args, **kwargs):
        return f"wrapped:{func(*args, **kwargs)}"
    return wrapper
"""
        )
        os.environ["FUNC_CAPTURE"] = f"chosen={script}"

        def target() -> str:
            return "value"

        decorated = capture("chosen")(target)

        self.assertEqual(decorated(), "wrapped:value")

    def test_missing_instrument_function_is_an_error_for_matching_function(self) -> None:
        script = self._write_script("VALUE = 1\n")

        def target() -> str:
            return "plain"

        key = f"{target.__module__}.{target.__qualname__}"
        os.environ["FUNC_CAPTURE"] = f"{key}={script}"

        with self.assertRaises(AttributeError):
            capture(target)

    def test_path_with_apostrophe_is_not_mangled(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        odd_dir = Path(temp_dir.name) / "o'brien"
        odd_dir.mkdir()
        script = odd_dir / "instrument.py"
        script.write_text(
            "def instrument(func):\n"
            "    def wrapper(*a, **k):\n"
            "        return ('ok', func(*a, **k))\n"
            "    return wrapper\n",
            encoding="utf-8",
        )

        def target() -> str:
            return "v"

        key = f"{target.__module__}.{target.__qualname__}"
        os.environ["FUNC_CAPTURE"] = f"{key}={script}"

        decorated = capture(target)

        self.assertEqual(decorated(), ("ok", "v"))

    def test_quoted_path_with_separator_is_not_split(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        odd_dir = Path(temp_dir.name) / "with,comma"
        odd_dir.mkdir()
        script = odd_dir / "instrument.py"
        script.write_text(
            "def instrument(func):\n"
            "    def wrapper(*a, **k):\n"
            "        return ('quoted', func(*a, **k))\n"
            "    return wrapper\n",
            encoding="utf-8",
        )

        def target() -> str:
            return "v"

        key = f"{target.__module__}.{target.__qualname__}"
        os.environ["FUNC_CAPTURE"] = f"{key}='{script}'"

        decorated = capture(target)

        self.assertEqual(decorated(), ("quoted", "v"))

    def test_escaped_separator_in_path_is_not_split(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        odd_dir = Path(temp_dir.name) / "with,comma"
        odd_dir.mkdir()
        script = odd_dir / "instrument.py"
        script.write_text(
            "def instrument(func):\n"
            "    def wrapper(*a, **k):\n"
            "        return ('escaped', func(*a, **k))\n"
            "    return wrapper\n",
            encoding="utf-8",
        )

        def target() -> str:
            return "v"

        key = f"{target.__module__}.{target.__qualname__}"
        escaped_script = str(script).replace(",", "\\,")
        os.environ["FUNC_CAPTURE"] = f"{key}={escaped_script}"

        decorated = capture(target)

        self.assertEqual(decorated(), ("escaped", "v"))

    def test_multiple_entries_are_parsed(self) -> None:
        script = self._write_script(
            """
def instrument(func):
    def wrapper(*args, **kwargs):
        return f"wrapped:{func(*args, **kwargs)}"
    return wrapper
"""
        )

        def target() -> str:
            return "value"

        key = f"{target.__module__}.{target.__qualname__}"
        # Trailing separators and blank lines must be tolerated.
        os.environ["FUNC_CAPTURE"] = f"other.fn=/nope.py;\n{key}={script},\n"

        decorated = capture(target)

        self.assertEqual(decorated(), "wrapped:value")

    def test_edited_script_is_reloaded_when_size_is_unchanged(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        script = Path(temp_dir.name) / "instrument.py"

        def write(tag: str) -> None:
            script.write_text(
                "def instrument(func):\n"
                "    def wrapper(*a, **k):\n"
                f"        return ('{tag}', func(*a, **k))\n"
                "    return wrapper\n",
                encoding="utf-8",
            )

        def target() -> str:
            return "v"

        key = f"{target.__module__}.{target.__qualname__}"
        os.environ["FUNC_CAPTURE"] = f"{key}={script}"

        write("first")
        self.assertEqual(capture(target)(), ("first", "v"))

        write("third")

        self.assertEqual(capture(target)(), ("third", "v"))

    def test_public_alias_points_to_capture(self) -> None:
        self.assertIs(func_capture, capture)

    def _write_script(self, source: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        script = Path(temp_dir.name) / "instrument.py"
        script.write_text(source.lstrip(), encoding="utf-8")
        return script


if __name__ == "__main__":
    unittest.main()
