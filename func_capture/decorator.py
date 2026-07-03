"""Opt-in function instrumentation controlled by ``FUNC_CAPTURE``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tokenize
import types
from typing import Any, Callable, Optional, TypeVar, Union, cast, overload

F = TypeVar("F", bound=Callable[..., Any])
Instrument = Callable[..., Any]

ENV_VAR = "FUNC_CAPTURE"


@overload
def capture(
    func: F,
    *instrument_positional_args: Any,
    key: Optional[str] = None,
    env_var: str = ENV_VAR,
    instrument_args: Optional[Iterable[Any]] = None,
    instrument_kwargs: Optional[Mapping[str, Any]] = None,
    **instrument_keyword_args: Any,
) -> F:
    ...


@overload
def capture(
    func: None = None,
    *instrument_positional_args: Any,
    key: Optional[str] = None,
    env_var: str = ENV_VAR,
    instrument_args: Optional[Iterable[Any]] = None,
    instrument_kwargs: Optional[Mapping[str, Any]] = None,
    **instrument_keyword_args: Any,
) -> Callable[[F], F]:
    ...


@overload
def capture(
    func: str,
    *instrument_positional_args: Any,
    key: None = None,
    env_var: str = ENV_VAR,
    instrument_args: Optional[Iterable[Any]] = None,
    instrument_kwargs: Optional[Mapping[str, Any]] = None,
    **instrument_keyword_args: Any,
) -> Callable[[F], F]:
    ...


def capture(
    func: Union[F, str, None] = None,
    *instrument_positional_args: Any,
    key: Optional[str] = None,
    env_var: str = ENV_VAR,
    instrument_args: Optional[Iterable[Any]] = None,
    instrument_kwargs: Optional[Mapping[str, Any]] = None,
    **instrument_keyword_args: Any,
) -> Union[F, Callable[[F], F]]:
    """Decorate a function with instrumentation selected by an environment var.

    ``FUNC_CAPTURE`` accepts either JSON object syntax or comma/semicolon/newline
    separated ``function_key=script.py`` entries. Without the environment var, or
    without a matching key, this returns the original function object unchanged.

    Function keys are matched against ``module.qualname`` first, then
    ``qualname``, and ``name``. An explicit key may be supplied
    as ``@capture("my.key")`` or ``@capture(key="my.key")``.

    Extra positional and keyword arguments are passed to the loaded
    ``instrument`` callable after the target function, allowing
    ``instrument(func, *args, **kwargs)`` scripts to expose custom behavior.
    Use ``instrument_args``/``instrument_kwargs`` when decorator syntax makes
    positional forwarding ambiguous.
    """

    if isinstance(func, str):
        if key is not None:
            raise TypeError("capture() got both a positional key and key=")
        key = func
        func = None

    def decorate(target: F) -> F:
        if env_var not in os.environ:
            return target

        spec = os.environ[env_var]
        if not spec.strip():
            return target

        configured = _parse_capture_spec(spec, env_var=env_var)
        script_path = _script_for_function(configured, target, explicit_key=key)
        if script_path is None:
            return target

        forwarded_args = _forwarded_instrument_args(
            instrument_positional_args,
            instrument_args=instrument_args,
        )
        forwarded_kwargs = _forwarded_instrument_kwargs(
            instrument_kwargs,
            instrument_keyword_args,
        )

        instrument = _load_instrument(script_path)
        instrumented = instrument(target, *forwarded_args, **forwarded_kwargs)
        if not callable(instrumented):
            raise TypeError(
                f"instrument() in {script_path!r} returned a non-callable "
                f"for {_function_key(target)!r}"
            )
        return cast(F, instrumented)

    if func is None:
        return decorate
    if not callable(func):
        raise TypeError("capture() expects a callable, a key string, or no positional argument")
    return decorate(cast(F, func))


func_capture = capture


def _forwarded_instrument_args(
    positional_args: tuple[Any, ...],
    *,
    instrument_args: Optional[Iterable[Any]],
) -> tuple[Any, ...]:
    if instrument_args is None:
        return positional_args
    try:
        return positional_args + tuple(instrument_args)
    except TypeError as exc:
        raise TypeError("capture() instrument_args must be iterable") from exc


def _forwarded_instrument_kwargs(
    instrument_kwargs: Optional[Mapping[str, Any]],
    keyword_args: dict[str, Any],
) -> dict[str, Any]:
    if instrument_kwargs is None:
        return dict(keyword_args)
    if not isinstance(instrument_kwargs, Mapping):
        raise TypeError("capture() instrument_kwargs must be a mapping")

    duplicate_keys = set(instrument_kwargs).intersection(keyword_args)
    if duplicate_keys:
        duplicates = ", ".join(sorted(repr(key) for key in duplicate_keys))
        raise TypeError(
            f"capture() got duplicate instrument keyword argument(s): {duplicates}"
        )

    forwarded = dict(instrument_kwargs)
    forwarded.update(keyword_args)
    non_string_keys = [key for key in forwarded if not isinstance(key, str)]
    if non_string_keys:
        keys = ", ".join(sorted(repr(key) for key in non_string_keys))
        raise TypeError(f"capture() instrument keyword names must be strings; got {keys}")
    return forwarded


def _parse_capture_spec(spec: str, *, env_var: str = ENV_VAR) -> dict[str, str]:
    stripped = spec.strip()
    if not stripped:
        return {}

    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{env_var} is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{env_var} JSON value must be an object")
        return _validate_mapping(parsed, env_var=env_var)

    parsed: dict[str, str] = {}
    for entry in _split_capture_entries(spec, env_var=env_var):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"{env_var} entries must use key=script.py syntax; got {entry!r}"
            )
        entry_key, script_path = entry.split("=", 1)
        entry_key = entry_key.strip()
        script_path = script_path.strip()
        if not entry_key:
            raise ValueError(f"{env_var} contains an empty function key")
        if not script_path:
            raise ValueError(f"{env_var} contains an empty script path for {entry_key!r}")
        parsed[entry_key] = script_path
    return parsed


def _split_capture_entries(spec: str, *, env_var: str) -> list[str]:
    entries: list[str] = []
    current: list[str] = []
    quote: Optional[str] = None
    quote_allowed = True
    index = 0

    while index < len(spec):
        char = spec[index]

        if quote is not None:
            if char == "\\" and index + 1 < len(spec) and spec[index + 1] == quote:
                current.append(spec[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
                quote_allowed = False
            else:
                current.append(char)
            index += 1
            continue

        if char == "\\" and index + 1 < len(spec) and spec[index + 1] in "\"',;\n":
            current.append(spec[index + 1])
            quote_allowed = False
            index += 2
            continue

        if char in "'\"" and quote_allowed:
            quote = char
            index += 1
            continue

        if char in ",;\n":
            entries.append("".join(current))
            current = []
            quote_allowed = True
            index += 1
            continue

        current.append(char)
        if char == "=":
            quote_allowed = True
        elif not char.isspace():
            quote_allowed = False
        index += 1

    if quote is not None:
        raise ValueError(f"{env_var} contains an unterminated quoted value")

    entries.append("".join(current))
    return entries


def _validate_mapping(parsed: dict[Any, Any], *, env_var: str) -> dict[str, str]:
    configured: dict[str, str] = {}
    for entry_key, script_path in parsed.items():
        if not isinstance(entry_key, str) or not entry_key:
            raise ValueError(f"{env_var} JSON keys must be non-empty strings")
        if not isinstance(script_path, str) or not script_path:
            raise ValueError(
                f"{env_var} JSON value for {entry_key!r} must be a non-empty string"
            )
        configured[entry_key] = script_path
    return configured


def _script_for_function(
    configured: dict[str, str],
    func: Callable[..., Any],
    *,
    explicit_key: Optional[str],
) -> Optional[str]:
    if explicit_key is not None:
        return configured.get(explicit_key)

    for key in _function_keys(func):
        script_path = configured.get(key)
        if script_path is not None:
            return script_path
    return None


def _function_keys(func: Callable[..., Any]) -> tuple[str, ...]:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", "")
    name = getattr(func, "__name__", "")

    keys: list[str] = []
    if module and qualname:
        keys.append(f"{module}.{qualname}")
    if qualname:
        keys.append(qualname)
    if name and name != qualname:
        keys.append(name)
    return tuple(keys)


def _function_key(func: Callable[..., Any]) -> str:
    keys = _function_keys(func)
    if keys:
        return keys[0]
    return repr(func)


def _load_instrument(script_path: str) -> Instrument:
    expanded = Path(os.path.expandvars(script_path)).expanduser()
    return _load_instrument_from_file(str(expanded.resolve()))


# Cache loaded instrument callables per resolved script path. The source bytes
# are hashed so edited scripts reload even when filesystem timestamp precision
# or Python bytecode caches would otherwise hide the change.
_INSTRUMENT_CACHE: dict[str, tuple[str, Instrument]] = {}


def _load_instrument_from_file(
    resolved_script_path: str,
) -> Instrument:
    script = Path(resolved_script_path)
    if not script.is_file():
        raise FileNotFoundError(f"instrumentation script not found: {script}")

    source_bytes = script.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    cached = _INSTRUMENT_CACHE.get(resolved_script_path)
    if cached is not None and cached[0] == source_hash:
        return cached[1]

    digest = hashlib.sha256(resolved_script_path.encode("utf-8")).hexdigest()
    module_name = f"_func_capture_instrument_{digest}"

    module = types.ModuleType(module_name)
    module.__file__ = str(script)
    module.__package__ = ""
    old_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source_bytes).readline)
        source = source_bytes.decode(encoding)
        exec(compile(source, str(script), "exec"), module.__dict__)
    except Exception:
        if old_module is not None:
            sys.modules[module_name] = old_module
        else:
            sys.modules.pop(module_name, None)
        raise

    instrument = getattr(module, "instrument", None)
    if not callable(instrument):
        raise AttributeError(f"instrumentation script {script} has no callable instrument()")

    _INSTRUMENT_CACHE[resolved_script_path] = (source_hash, instrument)
    return cast(Instrument, instrument)
