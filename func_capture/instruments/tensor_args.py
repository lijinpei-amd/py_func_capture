"""Capture tensor-call metadata and periodic replay snapshots.

This file is an instrumentation script for ``func_capture.capture``. Example:

    export FUNC_CAPTURE='atom.model_ops.v4_kernels.state_writes.update_compressor_states=/path/to/func_capture/instruments/tensor_args.py'
    export FUNC_CAPTURE_OUTPUT_DIR=/tmp/func_capture
    export FUNC_CAPTURE_FULL_EVERY_N=100

For every call, the wrapper appends a JSONL record containing non-tensor
arguments and tensor metadata. Each record also carries per-call *symbolic*
shapes: concrete dimensions are assigned fresh occurrence symbols (``d0``,
``d1``, ...), and dimensions that are themselves symbolic (e.g. ``SymInt`` under
tracing) are recorded by their string representation rather than forced to
``int``. Every ``FUNC_CAPTURE_FULL_EVERY_N`` calls, it also writes a ``.pt``
snapshot of the pre-call positional and keyword arguments with tensors copied to
CPU.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import functools
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import re
import threading
import time
import warnings
from typing import Any, Callable, Mapping, Optional, TypeVar, Union, cast

F = TypeVar("F", bound=Callable[..., Any])

OUTPUT_DIR_ENV = "FUNC_CAPTURE_OUTPUT_DIR"
FULL_EVERY_N_ENV = "FUNC_CAPTURE_FULL_EVERY_N"
STRICT_ENV = "FUNC_CAPTURE_STRICT"

DEFAULT_OUTPUT_DIR = "func_capture_out"
# Writing a full ``.pt`` snapshot copies every tensor argument to CPU, so a
# snapshot on *every* call would cripple a hot kernel. Default to an occasional
# snapshot; callers opt into denser capture via ``FUNC_CAPTURE_FULL_EVERY_N``.
DEFAULT_FULL_EVERY_N = 100

# Guard the recursive argument walkers against pathological inputs (reference
# cycles or extremely deep nesting) so instrumentation never blows the stack.
MAX_CAPTURE_DEPTH = 50

FORMAT_VERSION = 1
TENSOR_MARKER = "__func_capture_tensor_v1__"
# Sets are serialised through a marker wrapper: once tensors inside a set are
# replaced by (unhashable) saved-tensor dicts, the set itself can no longer be
# rebuilt directly, so we round-trip via an ordered ``items`` list instead.
SET_MARKER = "__func_capture_set_v1__"


@dataclass(frozen=True)
class CaptureConfig:
    output_dir: Path
    full_every_n: int
    strict: bool


def instrument(func: F) -> F:
    """Wrap ``func`` with tensor-aware capture instrumentation."""

    try:
        config = _read_config()
    except Exception as exc:
        if _strict_requested_after_config_error():
            raise
        # Instrumentation must never take down the program it observes. A bad
        # environment value degrades to a no-op wrapper with a warning.
        warnings.warn(
            f"func_capture: disabling capture for {_function_key(func)!r}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return func

    function_key = _function_key(func)
    capture_dir = config.output_dir / _safe_path_name(function_key)
    full_dir = capture_dir / "full"

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = None

    # State that must not be shared across processes: a forked worker inherits
    # the parent's pid-derived paths, lock, and call counter, which would
    # interleave JSONL lines and clobber ``.pt`` snapshots. Keyed by pid and
    # rebuilt whenever the current pid changes (i.e. after a fork).
    process_local: dict[str, Any] = {}

    def _process_local() -> tuple[int, Path, threading.Lock, dict[str, int]]:
        pid = os.getpid()
        if process_local.get("pid") != pid:
            process_local["pid"] = pid
            process_local["records_path"] = capture_dir / f"calls.{pid}.jsonl"
            process_local["lock"] = threading.Lock()
            process_local["state"] = {"call_index": 0}
            capture_dir.mkdir(parents=True, exist_ok=True)
            if config.full_every_n > 0:
                full_dir.mkdir(parents=True, exist_ok=True)
        return (
            process_local["pid"],
            process_local["records_path"],
            process_local["lock"],
            process_local["state"],
        )

    # Initialise eagerly so the common (no-fork) path never races on rebuild.
    _process_local()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        pid, records_path, lock, state = _process_local()

        with lock:
            state["call_index"] += 1
            call_index = state["call_index"]

        full_capture: Optional[dict[str, Any]] = None
        capture_error: Optional[dict[str, str]] = None

        if config.full_every_n > 0 and call_index % config.full_every_n == 0:
            full_name = f"call_{pid}_{call_index:09d}.pt"
            full_path = full_dir / full_name
            try:
                _save_full_call(
                    full_path,
                    func=func,
                    function_key=function_key,
                    call_index=call_index,
                    args=args,
                    kwargs=kwargs,
                )
                full_capture = {
                    "path": str(full_path.relative_to(capture_dir)),
                    "format": "torch-save",
                }
            except Exception as exc:  # pragma: no cover - strict mode re-raises.
                capture_error = _error_record(exc)
                if config.strict:
                    raise

        try:
            record = _call_record(
                func=func,
                function_key=function_key,
                signature=signature,
                call_index=call_index,
                args=args,
                kwargs=kwargs,
                full_capture=full_capture,
                capture_error=capture_error,
            )
            _append_jsonl(records_path, record, lock)
        except Exception as exc:  # pragma: no cover - strict mode re-raises.
            if config.strict:
                raise
            # Best-effort error record; a failure here (e.g. a full disk) must
            # not escape and break the wrapped function in non-strict mode.
            try:
                _append_jsonl(
                    records_path,
                    {
                        "version": FORMAT_VERSION,
                        "event": "capture_error",
                        "function": function_key,
                        "call_index": call_index,
                        "time_ns": time.time_ns(),
                        "process_id": pid,
                        "error": _error_record(exc),
                    },
                    lock,
                )
            except Exception:
                pass

        return func(*args, **kwargs)

    return cast(F, wrapper)


def load_full_call(
    path: Union[os.PathLike[str], str],
    *,
    tensor_device: str = "original",
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Load a full-call snapshot and return replayable ``(args, kwargs)``.

    ``tensor_device`` controls where saved tensors are restored:

    - ``"original"``: move tensors back to the device recorded at capture time.
      If that device is unavailable on the current machine the tensor is left
      on CPU rather than raising.
    - ``"cpu"``: leave tensors on CPU.
    - any other string: pass it to ``Tensor.to(...)`` as the target device.
    """

    snapshot = _load_snapshot(path)
    return restore_full_call(snapshot, tensor_device=tensor_device)


def restore_full_call(
    snapshot: Mapping[str, Any],
    *,
    tensor_device: str = "original",
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Restore ``(args, kwargs)`` from a loaded full-call snapshot."""

    args = tuple(_restore_saved_value(value, tensor_device) for value in snapshot["args"])
    kwargs = {
        key: _restore_saved_value(value, tensor_device)
        for key, value in snapshot["kwargs"].items()
    }
    return args, kwargs


def replay_full_call(
    func: Callable[..., Any],
    path: Union[os.PathLike[str], str],
    *,
    tensor_device: str = "original",
) -> Any:
    """Load a full-call snapshot and invoke ``func(*args, **kwargs)``."""

    args, kwargs = load_full_call(path, tensor_device=tensor_device)
    return func(*args, **kwargs)


def _read_config() -> CaptureConfig:
    output_dir = Path(os.environ.get(OUTPUT_DIR_ENV, DEFAULT_OUTPUT_DIR)).expanduser()
    strict = _env_bool(STRICT_ENV, default=False)
    full_every_n = _env_int(FULL_EVERY_N_ENV, DEFAULT_FULL_EVERY_N)
    if full_every_n < 0:
        raise ValueError(f"{FULL_EVERY_N_ENV} must be >= 0")
    return CaptureConfig(
        output_dir=output_dir,
        full_every_n=full_every_n,
        strict=strict,
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _strict_requested_after_config_error() -> bool:
    try:
        return _env_bool(STRICT_ENV, default=False)
    except ValueError:
        return True


def _call_record(
    *,
    func: Callable[..., Any],
    function_key: str,
    signature: Optional[inspect.Signature],
    call_index: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    full_capture: Optional[dict[str, Any]],
    capture_error: Optional[dict[str, str]],
) -> dict[str, Any]:
    bound = _bind_arguments(signature, args, kwargs)
    symbolizer = _ShapeSymbolizer()
    arguments = {
        name: _value_metadata(value, symbolizer=symbolizer, path=name)
        for name, value in bound.items()
    }

    record: dict[str, Any] = {
        "version": FORMAT_VERSION,
        "event": "call",
        "function": function_key,
        "module": getattr(func, "__module__", ""),
        "qualname": getattr(func, "__qualname__", getattr(func, "__name__", "")),
        "call_index": call_index,
        "time_ns": time.time_ns(),
        "process_id": os.getpid(),
        "thread_id": threading.get_ident(),
        "arguments": arguments,
        "tensor_shape_symbols": symbolizer.values_by_symbol,
    }
    if full_capture is not None:
        record["full_capture"] = full_capture
    if capture_error is not None:
        record["capture_error"] = capture_error
    return record


def _bind_arguments(
    signature: Optional[inspect.Signature],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if signature is not None:
        try:
            bound = signature.bind(*args, **kwargs)
            # Record defaulted parameters the caller omitted so the capture
            # reflects the effective call, not just the explicit arguments.
            bound.apply_defaults()
            return dict(bound.arguments)
        except TypeError:
            pass

    bound: dict[str, Any] = {f"arg{index}": value for index, value in enumerate(args)}
    bound.update({f"kw:{key}": value for key, value in kwargs.items()})
    return bound


def _value_metadata(
    value: Any,
    *,
    symbolizer: "_ShapeSymbolizer",
    path: str,
    depth: int = 0,
    seen: tuple[int, ...] = (),
) -> dict[str, Any]:
    if _is_tensor(value):
        return _tensor_metadata(value, symbolizer=symbolizer)

    if depth >= MAX_CAPTURE_DEPTH:
        return {"kind": "truncated", "type": _type_name(value), "reason": "max_depth"}

    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        if id(value) in seen:
            return {"kind": "cycle", "type": _type_name(value)}
        seen = seen + (id(value),)

    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "type": _type_name(value),
            "items": [
                [
                    _json_friendly_value(key),
                    _value_metadata(
                        child,
                        symbolizer=symbolizer,
                        path=f"{path}.{key}",
                        depth=depth + 1,
                        seen=seen,
                    ),
                ]
                for key, child in value.items()
            ],
        }

    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "type": _type_name(value),
            "items": [
                _value_metadata(
                    child,
                    symbolizer=symbolizer,
                    path=f"{path}_{index}",
                    depth=depth + 1,
                    seen=seen,
                )
                for index, child in enumerate(value)
            ],
        }

    if isinstance(value, (set, frozenset)):
        return {
            "kind": "set",
            "type": _type_name(value),
            "items": [
                _value_metadata(
                    child,
                    symbolizer=symbolizer,
                    path=f"{path}_{index}",
                    depth=depth + 1,
                    seen=seen,
                )
                for index, child in enumerate(sorted(value, key=repr))
            ],
        }

    return {
        "kind": "value",
        "type": _type_name(value),
        **_json_friendly_value(value),
    }


def _tensor_metadata(value: Any, *, symbolizer: "_ShapeSymbolizer") -> dict[str, Any]:
    shape = _tensor_shape(value)
    stride = _tensor_stride(value)
    metadata: dict[str, Any] = {
        "kind": "tensor",
        "shape": shape,
        "symbolic_shape": [symbolizer.symbol_for(dim) for dim in shape],
        "dtype": str(getattr(value, "dtype", "")),
        "device": str(getattr(value, "device", "")),
        "layout": str(getattr(value, "layout", "")),
        "requires_grad": bool(getattr(value, "requires_grad", False)),
        "is_contiguous": _safe_tensor_bool(value, "is_contiguous"),
    }
    if stride is not None:
        metadata["stride"] = stride
    storage_offset = _safe_tensor_call(value, "storage_offset")
    if storage_offset is not None:
        metadata["storage_offset"] = _coerce_dim(storage_offset)
    return metadata


class _ShapeSymbolizer:
    def __init__(self) -> None:
        self._values_by_symbol: dict[str, Any] = {}
        self._next_generated = 0

    @property
    def values_by_symbol(self) -> dict[str, Any]:
        return dict(self._values_by_symbol)

    def symbol_for(self, dim: Any) -> str:
        if not isinstance(dim, int) or isinstance(dim, bool):
            return str(dim)

        symbol = f"d{self._next_generated}"
        self._next_generated += 1
        self._values_by_symbol[symbol] = dim
        return symbol


def _save_full_call(
    path: Path,
    *,
    func: Callable[..., Any],
    function_key: str,
    call_index: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    snapshot = {
        "version": FORMAT_VERSION,
        "function": function_key,
        "module": getattr(func, "__module__", ""),
        "qualname": getattr(func, "__qualname__", getattr(func, "__name__", "")),
        "call_index": call_index,
        "time_ns": time.time_ns(),
        "process_id": os.getpid(),
        "args": tuple(_saved_value(value) for value in args),
        "kwargs": {key: _saved_value(value) for key, value in kwargs.items()},
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    _save_snapshot(snapshot, tmp_path)
    os.replace(tmp_path, path)


def _saved_value(value: Any, depth: int = 0, seen: tuple[int, ...] = ()) -> Any:
    if _is_tensor(value):
        copied = _copy_tensor_to_cpu(value)
        metadata = _tensor_metadata(value, symbolizer=_ShapeSymbolizer())
        _reconcile_saved_metadata(metadata, copied)
        return {
            TENSOR_MARKER: True,
            "metadata": metadata,
            "tensor": copied,
        }

    if depth >= MAX_CAPTURE_DEPTH:
        return _safe_repr(value)

    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        if id(value) in seen:
            return _safe_repr(value)
        seen = seen + (id(value),)

    if isinstance(value, Mapping):
        return {key: _saved_value(child, depth + 1, seen) for key, child in value.items()}
    if _is_namedtuple(value):
        return type(value)(*(_saved_value(child, depth + 1, seen) for child in value))
    if isinstance(value, list):
        return [_saved_value(child, depth + 1, seen) for child in value]
    if isinstance(value, tuple):
        return tuple(_saved_value(child, depth + 1, seen) for child in value)
    if isinstance(value, (set, frozenset)):
        return {
            SET_MARKER: "frozenset" if isinstance(value, frozenset) else "set",
            "items": [_saved_value(child, depth + 1, seen) for child in value],
        }
    return value


def _restore_saved_value(value: Any, tensor_device: str) -> Any:
    if _is_saved_tensor(value):
        tensor = value["tensor"]
        metadata = value.get("metadata", {})
        target_device = _target_device(metadata, tensor_device)
        if target_device is not None and hasattr(tensor, "to"):
            try:
                tensor = tensor.to(target_device)
            except Exception:
                # Recorded device is unavailable here; keep the CPU tensor.
                pass
        if metadata.get("requires_grad") and hasattr(tensor, "requires_grad_"):
            try:
                tensor = tensor.requires_grad_(True)
            except RuntimeError:
                pass
        return tensor
    if _is_saved_set(value):
        items = [_restore_saved_value(child, tensor_device) for child in value["items"]]
        return frozenset(items) if value[SET_MARKER] == "frozenset" else set(items)
    if isinstance(value, Mapping):
        return {key: _restore_saved_value(child, tensor_device) for key, child in value.items()}
    if _is_namedtuple(value):
        return type(value)(*(_restore_saved_value(child, tensor_device) for child in value))
    if isinstance(value, list):
        return [_restore_saved_value(child, tensor_device) for child in value]
    if isinstance(value, tuple):
        return tuple(_restore_saved_value(child, tensor_device) for child in value)
    return value


def _target_device(metadata: Mapping[str, Any], tensor_device: str) -> Optional[str]:
    if tensor_device == "cpu":
        return None
    if tensor_device == "original":
        device = metadata.get("device")
        if isinstance(device, str) and device:
            return device
        return None
    return tensor_device


def _is_saved_tensor(value: Any) -> bool:
    # Require the full wrapper shape so a user mapping that merely happens to
    # contain the marker key is never mistaken for a captured tensor.
    return (
        isinstance(value, Mapping)
        and value.get(TENSOR_MARKER) is True
        and "tensor" in value
        and "metadata" in value
    )


def _is_saved_set(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get(SET_MARKER) in {"set", "frozenset"}


def _is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")


def _reconcile_saved_metadata(metadata: dict[str, Any], copied: Any) -> None:
    """Align a snapshot's layout metadata with the tensor actually saved.

    The CPU copy may not preserve the source layout (e.g. an overlapping or
    unsupported-stride tensor is materialised contiguously), so the recorded
    stride/contiguity/offset must describe the saved tensor rather than the
    original to avoid metadata that contradicts the data.
    """

    if not _is_tensor(copied):
        return
    stride = _tensor_stride(copied)
    if stride is not None:
        metadata["stride"] = stride
    is_contiguous = _safe_tensor_bool(copied, "is_contiguous")
    if is_contiguous is not None:
        metadata["is_contiguous"] = is_contiguous
    storage_offset = _safe_tensor_call(copied, "storage_offset")
    if storage_offset is not None:
        metadata["storage_offset"] = _coerce_dim(storage_offset)


def _copy_tensor_to_cpu(value: Any) -> Any:
    torch = _get_torch()
    no_grad = getattr(torch, "no_grad", None) if torch is not None else None
    context = no_grad() if callable(no_grad) else nullcontext()

    with context:
        source = value.detach() if hasattr(value, "detach") else value
        shape = _tensor_shape(source)
        stride = _tensor_stride(source)

        # Preserve the exact strided layout only when it is safe: an overlapping
        # (e.g. broadcast/expanded, stride-0) tensor cannot be copied into an
        # equally strided destination without undefined behaviour, and a
        # symbolic shape/stride cannot be passed to ``empty_strided`` at all.
        if (
            torch is not None
            and stride is not None
            and _all_ints(shape)
            and _all_ints(stride)
            and not _strided_overlaps(shape, stride)
            and hasattr(torch, "empty_strided")
        ):
            try:
                copied = torch.empty_strided(
                    tuple(shape),
                    tuple(stride),
                    dtype=getattr(source, "dtype", None),
                    device="cpu",
                )
                copied.copy_(source, non_blocking=False)
                return copied
            except Exception:
                pass

        copied = source.cpu() if hasattr(source, "cpu") else source
        return copied.clone() if hasattr(copied, "clone") else copied


def _strided_overlaps(shape: list[Any], stride: list[Any]) -> bool:
    # A zero stride on a dimension larger than 1 aliases the same memory across
    # that dimension (tensors from ``expand``/broadcast), which makes an
    # in-place strided copy undefined.
    for dim, dim_stride in zip(shape, stride):
        if isinstance(dim, int) and isinstance(dim_stride, int) and dim > 1 and dim_stride == 0:
            return True
    return False


def _all_ints(values: list[Any]) -> bool:
    return all(isinstance(item, int) for item in values)


def _save_snapshot(snapshot: Mapping[str, Any], path: Path) -> None:
    torch = _get_torch()
    if torch is not None and hasattr(torch, "save"):
        torch.save(snapshot, path)
        return
    with path.open("wb") as file:
        pickle.dump(snapshot, file, protocol=pickle.HIGHEST_PROTOCOL)


def _load_snapshot(path: Union[os.PathLike[str], str]) -> Mapping[str, Any]:
    torch = _get_torch()
    if torch is not None and hasattr(torch, "load"):
        try:
            return cast(Mapping[str, Any], torch.load(path, map_location="cpu", weights_only=False))
        except TypeError:
            return cast(Mapping[str, Any], torch.load(path, map_location="cpu"))
    with Path(path).open("rb") as file:
        return cast(Mapping[str, Any], pickle.load(file))


def _append_jsonl(path: Path, record: Mapping[str, Any], lock: threading.Lock) -> None:
    # Serialise the whole line (including the trailing newline) into a single
    # write so an O_APPEND write stays atomic and lines are never torn.
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as file:
            file.write(line)


def _is_tensor(value: Any) -> bool:
    torch = _get_torch()
    tensor_type = getattr(torch, "Tensor", None) if torch is not None else None
    return tensor_type is not None and isinstance(value, tensor_type)


def _get_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _coerce_dim(dim: Any) -> Any:
    # Concrete Python sizes/strides stay plain ints. Non-int shape objects are
    # deliberately not coerced through int(): for torch.SymInt that can
    # materialise/specialise a dynamic dimension.
    if isinstance(dim, int) and not isinstance(dim, bool):
        return dim
    return str(dim)


def _tensor_shape(value: Any) -> list[Any]:
    return [_coerce_dim(dim) for dim in getattr(value, "shape", ())]


def _tensor_stride(value: Any) -> Optional[list[Any]]:
    stride = _safe_tensor_call(value, "stride")
    if stride is None:
        return None
    return [_coerce_dim(dim) for dim in stride]


def _safe_tensor_call(value: Any, name: str) -> Any:
    method = getattr(value, name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _safe_tensor_bool(value: Any, name: str) -> Optional[bool]:
    result = _safe_tensor_call(value, name)
    if result is None:
        return None
    return bool(result)


def _json_friendly_value(value: Any) -> dict[str, Any]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"value": value}
    if isinstance(value, bytes):
        return {"repr": repr(value)}
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return {"repr": _safe_repr(value)}
    return {"value": value}


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception as exc:
        return f"<repr failed: {type(exc).__name__}: {exc}>"


def _type_name(value: Any) -> str:
    cls = type(value)
    if cls.__module__ == "builtins":
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def _function_key(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", "")
    if module and qualname:
        return f"{module}.{qualname}"
    return qualname or getattr(func, "__name__", repr(func))


def _safe_path_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "function"


def _error_record(exc: Exception) -> dict[str, str]:
    return {
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
    }
