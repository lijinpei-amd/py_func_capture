"""Capture tensor-call metadata and periodic replay snapshots.

This file is an instrumentation script for ``func_capture.capture``. Example:

    export FUNC_CAPTURE='atom.model_ops.v4_kernels.state_writes.update_compressor_states=/path/to/func_capture/instruments/tensor_args.py'
    export FUNC_CAPTURE_OUTPUT_DIR=/tmp/func_capture
    export FUNC_CAPTURE_FULL_EVERY_N=100

Or pass a YAML config path through ``func_capture.capture``:

    @capture("my.function", config_path="/path/to/tensor_capture.yaml")

Example YAML:

    capture:
      mode: metadata_and_tensors  # metadata | metadata_and_tensors
      frequency: 100

By default, the wrapper appends a JSONL record containing non-tensor arguments
and tensor metadata for every call, buffering those records in memory and
flushing them to disk periodically. Every ``FUNC_CAPTURE_FULL_EVERY_N`` calls,
it also writes a ``.pt`` snapshot of the pre-call positional and keyword
arguments with tensors copied to CPU. A YAML config can switch to metadata-only
capture or capture metadata plus tensor contents on a different call frequency.
Send ``SIGUSR1`` to reload active configs and ``SIGUSR2`` to flush buffered
metadata records immediately.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import atexit
import functools
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import re
import signal
import threading
import time
import warnings
from types import FrameType
from typing import Any, Callable, Mapping, Optional, TypeVar, Union, cast
import weakref

F = TypeVar("F", bound=Callable[..., Any])

OUTPUT_DIR_ENV = "FUNC_CAPTURE_OUTPUT_DIR"
FULL_EVERY_N_ENV = "FUNC_CAPTURE_FULL_EVERY_N"
STRICT_ENV = "FUNC_CAPTURE_STRICT"
CONFIG_PATH_ENV = "FUNC_CAPTURE_CONFIG"
# Comma/space-separated parameter names whose tensor *contents* are saved in a
# full snapshot. Any tensor argument not named here is written as a metadata
# stub (shape/dtype/stride only) and rehydrated with random data on load. When
# unset, every tensor is saved in full (the original behaviour).
KEEP_TENSORS_ENV = "FUNC_CAPTURE_KEEP_TENSORS"

DEFAULT_OUTPUT_DIR = "func_capture_out"
# Writing a full ``.pt`` snapshot copies every tensor argument to CPU, so a
# snapshot on *every* call would cripple a hot kernel. Default to an occasional
# snapshot; callers opt into denser capture via ``FUNC_CAPTURE_FULL_EVERY_N``.
DEFAULT_FULL_EVERY_N = 100
DEFAULT_METADATA_EVERY_N = 1
DEFAULT_METADATA_FLUSH_INTERVAL_SECONDS = 1.0

# Guard the recursive argument walkers against pathological inputs (reference
# cycles or extremely deep nesting) so instrumentation never blows the stack.
MAX_CAPTURE_DEPTH = 50

FORMAT_VERSION = 1
TENSOR_MARKER = "__func_capture_tensor_v1__"
# A stub records a tensor's shape/dtype/stride/device but *not* its data. It is
# written for arguments excluded from ``keep_tensors`` so a snapshot can shrink
# from gigabytes (weights) to kilobytes (routing only). Stubs are rehydrated
# with random tensors of the recorded spec at load time — enough for a
# performance replay, where kernel timing depends on shape/dtype/routing, not
# on the actual weight/activation values.
TENSOR_STUB_MARKER = "__func_capture_tensor_stub_v1__"
# Sets are serialised through a marker wrapper: once tensors inside a set are
# replaced by (unhashable) saved-tensor dicts, the set itself can no longer be
# rebuilt directly, so we round-trip via an ordered ``items`` list instead.
SET_MARKER = "__func_capture_set_v1__"


@dataclass(frozen=True)
class CaptureConfig:
    output_dir: Path
    metadata_every_n: int
    full_every_n: int
    strict: bool
    # ``None`` means "save every tensor's contents" (original behaviour). A set
    # (possibly empty) means "save contents only for these parameter names; stub
    # everything else". Frozen so the dataclass stays hashable.
    keep_tensors: Optional[frozenset[str]] = None


def instrument(
    func: F,
    config_path: Optional[Union[os.PathLike[str], str]] = None,
) -> F:
    """Wrap ``func`` with tensor-aware capture instrumentation."""

    try:
        config = _read_config(config_path)
    except Exception as exc:
        if _strict_requested_after_config_error(config_path):
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

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = None

    runtime = _CaptureRuntime(
        func=func,
        function_key=function_key,
        signature=signature,
        config_path=config_path,
        config=config,
    )

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        runtime.capture(args, kwargs)
        return func(*args, **kwargs)

    return cast(F, wrapper)


class _CaptureRuntime:
    def __init__(
        self,
        *,
        func: Callable[..., Any],
        function_key: str,
        signature: Optional[inspect.Signature],
        config_path: Optional[Union[os.PathLike[str], str]],
        config: CaptureConfig,
    ) -> None:
        self.func = func
        self.function_key = function_key
        self.signature = signature
        self.config_path = config_path
        self._config = config
        self._safe_name = _safe_path_name(function_key)
        self._lock = threading.RLock()
        # Serialises the actual JSONL writes so a background-flusher write and an
        # on-demand (SIGUSR2/atexit) flush of the same file cannot interleave and
        # tear lines. Kept separate from ``_lock`` so slow disk I/O never blocks
        # the hot path that only buffers records. Reentrant so a re-entrant flush
        # on the same thread cannot self-deadlock.
        self._io_lock = threading.RLock()
        self._pid: Optional[int] = None
        self._call_index = 0
        self._capture_dir: Optional[Path] = None
        self._full_dir: Optional[Path] = None
        self._records_path: Optional[Path] = None
        self._record_buffers: dict[Path, list[Mapping[str, Any]]] = {}
        self._flush_event = threading.Event()
        self._flusher_thread: Optional[threading.Thread] = None
        self._flusher_pid: Optional[int] = None
        self._last_flush_error: Optional[str] = None

        with self._lock:
            self._ensure_process_locked()
            self._start_flusher_thread_locked()
        _register_capture_state(self)

    def capture(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        # Never touch arguments while a CUDA/HIP graph is being captured: reading
        # tensor contents (repr/json of routing objects) or copying to CPU issues
        # a disallowed operation that invalidates the capture. Such calls also
        # replay from the graph without re-entering Python, so recording them is
        # pointless. Skip entirely and let the eager calls be captured instead.
        if _is_cuda_graph_capturing():
            return
        with self._lock:
            self._ensure_process_locked()
            self._start_flusher_thread_locked()
            config = self._config
            if config.metadata_every_n == 0 and config.full_every_n == 0:
                return

            self._call_index += 1
            call_index = self._call_index
            pid = self._require_pid_locked()
            records_path = self._require_records_path_locked()
            capture_dir = self._require_capture_dir_locked()
            full_dir = self._require_full_dir_locked()

        full_capture: Optional[dict[str, Any]] = None
        capture_error: Optional[dict[str, str]] = None

        if _captures_call(config.full_every_n, call_index):
            full_name = f"call_{pid}_{call_index:09d}.pt"
            full_path = full_dir / full_name
            try:
                full_dir.mkdir(parents=True, exist_ok=True)
                _save_full_call(
                    full_path,
                    func=self.func,
                    function_key=self.function_key,
                    call_index=call_index,
                    args=args,
                    kwargs=kwargs,
                    signature=self.signature,
                    keep_tensors=config.keep_tensors,
                )
                full_capture = {
                    "path": str(full_path.relative_to(capture_dir)),
                    "format": "torch-save",
                }
            except Exception as exc:  # pragma: no cover - strict mode re-raises.
                capture_error = _error_record(exc)
                if config.strict:
                    raise

        if (
            _captures_call(config.metadata_every_n, call_index)
            or full_capture is not None
            or capture_error is not None
        ):
            try:
                record = _call_record(
                    func=self.func,
                    function_key=self.function_key,
                    signature=self.signature,
                    call_index=call_index,
                    args=args,
                    kwargs=kwargs,
                    full_capture=full_capture,
                    capture_error=capture_error,
                )
                self.buffer_record(records_path, record)
            except Exception as exc:  # pragma: no cover - strict mode re-raises.
                if config.strict:
                    raise
                # Best-effort error record; failures while buffering must not
                # escape and break the wrapped function in non-strict mode.
                try:
                    self.buffer_record(
                        records_path,
                        {
                            "version": FORMAT_VERSION,
                            "event": "capture_error",
                            "function": self.function_key,
                            "call_index": call_index,
                            "time_ns": time.time_ns(),
                            "process_id": pid,
                            "error": _error_record(exc),
                        },
                    )
                except Exception:
                    pass

    def buffer_record(self, path: Path, record: Mapping[str, Any]) -> None:
        with self._lock:
            self._record_buffers.setdefault(path, []).append(record)

    def flush(self) -> None:
        with self._lock:
            self._ensure_process_locked()
            buffers = self._record_buffers
            self._record_buffers = {}

        if not buffers:
            return

        failed: dict[Path, list[Mapping[str, Any]]] = {}
        with self._io_lock:
            for path, records in buffers.items():
                try:
                    _append_jsonl_records(path, records)
                except Exception as exc:
                    failed[path] = records
                    self._warn_flush_error(path, exc)

        if failed:
            with self._lock:
                for path, records in failed.items():
                    existing = self._record_buffers.get(path, [])
                    self._record_buffers[path] = records + existing
        else:
            # Everything landed; allow a future identical error to warn again.
            with self._lock:
                self._last_flush_error = None

    def reload_config(self) -> None:
        try:
            config = _read_config(self.config_path)
        except Exception as exc:
            warnings.warn(
                f"func_capture: keeping existing capture config for "
                f"{self.function_key!r}; reload failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        with self._lock:
            self._config = config
            self._set_paths_locked(os.getpid(), reset_call_index=False)
            self._start_flusher_thread_locked()

    def _start_flusher_thread_locked(self) -> None:
        pid = os.getpid()
        if (
            self._flusher_pid == pid
            and self._flusher_thread is not None
            and self._flusher_thread.is_alive()
        ):
            return

        self._flusher_pid = pid
        self._flusher_thread = threading.Thread(
            target=_capture_runtime_flush_loop,
            args=(weakref.ref(self), pid),
            name=f"func_capture_metadata_flush_{pid}",
            daemon=True,
        )
        self._flusher_thread.start()

    def _ensure_process_locked(self) -> None:
        pid = os.getpid()
        if self._pid != pid:
            # A forked child inherits the parent's in-memory buffers. The child
            # must not flush the parent's already-captured metadata.
            self._record_buffers = {}
            self._set_paths_locked(pid, reset_call_index=True)

    def _set_paths_locked(self, pid: int, *, reset_call_index: bool) -> None:
        self._pid = pid
        if reset_call_index:
            self._call_index = 0
        self._capture_dir = self._config.output_dir / self._safe_name
        self._full_dir = self._capture_dir / "full"
        self._records_path = self._capture_dir / f"calls.{pid}.jsonl"
        if self._config.metadata_every_n > 0 or self._config.full_every_n > 0:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
        if self._config.full_every_n > 0:
            self._full_dir.mkdir(parents=True, exist_ok=True)

    def _require_pid_locked(self) -> int:
        if self._pid is None:
            raise RuntimeError("capture runtime has no process id")
        return self._pid

    def _require_capture_dir_locked(self) -> Path:
        if self._capture_dir is None:
            raise RuntimeError("capture runtime has no capture directory")
        return self._capture_dir

    def _require_full_dir_locked(self) -> Path:
        if self._full_dir is None:
            raise RuntimeError("capture runtime has no full-capture directory")
        return self._full_dir

    def _require_records_path_locked(self) -> Path:
        if self._records_path is None:
            raise RuntimeError("capture runtime has no records path")
        return self._records_path

    def _warn_flush_error(self, path: Path, exc: Exception) -> None:
        message = f"{path}: {type(exc).__name__}: {exc}"
        with self._lock:
            if self._last_flush_error == message:
                return
            self._last_flush_error = message
        warnings.warn(
            f"func_capture: failed to flush metadata records for "
            f"{self.function_key!r}: {message}",
            RuntimeWarning,
            stacklevel=2,
        )


def _capture_runtime_flush_loop(
    runtime_ref: weakref.ReferenceType[_CaptureRuntime],
    pid: int,
) -> None:
    while True:
        runtime = runtime_ref()
        if runtime is None or os.getpid() != pid:
            return
        flush_event = runtime._flush_event
        del runtime

        flush_event.wait(DEFAULT_METADATA_FLUSH_INTERVAL_SECONDS)
        flush_event.clear()

        runtime = runtime_ref()
        if runtime is None or os.getpid() != pid:
            return
        runtime.flush()
        del runtime


_CAPTURE_STATES: weakref.WeakSet[_CaptureRuntime] = weakref.WeakSet()
_CAPTURE_STATES_LOCK = threading.RLock()
_SIGNAL_HANDLERS_LOCK = threading.Lock()
_PREVIOUS_SIGNAL_HANDLERS: dict[int, Any] = {}

# Signal handlers run on the main thread at (nearly) arbitrary points, so they
# must not take locks, do I/O, parse YAML, or emit warnings — all of which the
# reload/flush work does. The handler therefore only records what was asked and
# wakes a dedicated worker thread that performs the real work at a safe point.
_SIGNAL_REQUEST_COUNTS = {"reload": 0, "flush": 0}
_SIGNAL_PROCESSED_COUNTS = {"reload": 0, "flush": 0}
_SIGNAL_WORK_EVENT = threading.Event()
_SIGNAL_WORKER_LOCK = threading.Lock()
_SIGNAL_WORKER_THREAD: Optional[threading.Thread] = None
_SIGNAL_WORKER_PID: Optional[int] = None
# Bumped after every dispatch cycle so callers (and tests) can wait for a raised
# signal to be fully processed instead of racing the worker thread.
_SIGNAL_DISPATCH_CV = threading.Condition(threading.Lock())
_SIGNAL_DISPATCH_COUNT = 0

_ATEXIT_REGISTERED = False


def _register_capture_state(state: _CaptureRuntime) -> None:
    with _CAPTURE_STATES_LOCK:
        _CAPTURE_STATES.add(state)
    _register_atexit_flush()
    _install_signal_handlers()


def _register_atexit_flush() -> None:
    # Flush buffered records on clean interpreter exit; the background flusher is
    # a daemon thread and is killed at shutdown without draining its buffer, so
    # without this the most recently buffered records (and the JSONL entries that
    # point at synchronously-written ``.pt`` snapshots) would be lost.
    global _ATEXIT_REGISTERED
    with _SIGNAL_WORKER_LOCK:
        if _ATEXIT_REGISTERED:
            return
        _ATEXIT_REGISTERED = True
    atexit.register(_flush_all_capture_states)


def _capture_states_snapshot() -> list[_CaptureRuntime]:
    with _CAPTURE_STATES_LOCK:
        return list(_CAPTURE_STATES)


def _flush_all_capture_states() -> None:
    for state in _capture_states_snapshot():
        state.flush()


def _reload_all_capture_states() -> None:
    for state in _capture_states_snapshot():
        state.reload_config()


def _signal_dispatch_loop(pid: int) -> None:
    global _SIGNAL_DISPATCH_COUNT
    while True:
        if os.getpid() != pid:
            return
        _SIGNAL_WORK_EVENT.wait()
        if os.getpid() != pid:
            return
        while True:
            # Clear before reading the counters: a signal that arrives after
            # this point re-sets the event. Counters make that signal visible
            # even if it lands while this worker is dispatching older work.
            _SIGNAL_WORK_EVENT.clear()
            reload_count = _SIGNAL_REQUEST_COUNTS["reload"]
            flush_count = _SIGNAL_REQUEST_COUNTS["flush"]
            do_reload = reload_count > _SIGNAL_PROCESSED_COUNTS["reload"]
            do_flush = flush_count > _SIGNAL_PROCESSED_COUNTS["flush"]

            if do_reload:
                _reload_all_capture_states()
                _SIGNAL_PROCESSED_COUNTS["reload"] = reload_count
            if do_flush:
                _flush_all_capture_states()
                _SIGNAL_PROCESSED_COUNTS["flush"] = flush_count

            with _SIGNAL_DISPATCH_CV:
                _SIGNAL_DISPATCH_COUNT += 1
                _SIGNAL_DISPATCH_CV.notify_all()

            if (
                _SIGNAL_REQUEST_COUNTS["reload"]
                == _SIGNAL_PROCESSED_COUNTS["reload"]
                and _SIGNAL_REQUEST_COUNTS["flush"]
                == _SIGNAL_PROCESSED_COUNTS["flush"]
            ):
                break


def _ensure_signal_dispatcher() -> None:
    global _SIGNAL_WORKER_THREAD, _SIGNAL_WORKER_PID
    pid = os.getpid()
    with _SIGNAL_WORKER_LOCK:
        if (
            _SIGNAL_WORKER_PID == pid
            and _SIGNAL_WORKER_THREAD is not None
            and _SIGNAL_WORKER_THREAD.is_alive()
        ):
            return
        _SIGNAL_WORKER_PID = pid
        _SIGNAL_WORKER_THREAD = threading.Thread(
            target=_signal_dispatch_loop,
            args=(pid,),
            name=f"func_capture_signal_dispatch_{pid}",
            daemon=True,
        )
        _SIGNAL_WORKER_THREAD.start()


def _signal_dispatch_count() -> int:
    with _SIGNAL_DISPATCH_CV:
        return _SIGNAL_DISPATCH_COUNT


def _wait_for_signal_dispatch(previous_count: int, timeout: float) -> bool:
    with _SIGNAL_DISPATCH_CV:
        return _SIGNAL_DISPATCH_CV.wait_for(
            lambda: _SIGNAL_DISPATCH_COUNT > previous_count,
            timeout=timeout,
        )


def _install_signal_handlers() -> None:
    signal_numbers = _capture_signal_numbers()
    if not signal_numbers:
        return
    with _SIGNAL_HANDLERS_LOCK:
        for signum in signal_numbers:
            if signum in _PREVIOUS_SIGNAL_HANDLERS:
                continue
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, _handle_capture_signal)
            except (OSError, ValueError, AttributeError):
                # signal.signal only works on the main thread; degrade quietly.
                continue
            _PREVIOUS_SIGNAL_HANDLERS[signum] = previous
    _ensure_signal_dispatcher()


def _capture_signal_numbers() -> tuple[int, ...]:
    numbers: list[int] = []
    for name in ("SIGUSR1", "SIGUSR2"):
        signum = getattr(signal, name, None)
        if signum is not None:
            numbers.append(int(signum))
    return tuple(numbers)


def _handle_capture_signal(signum: int, frame: Optional[FrameType]) -> None:
    # Keep the Python signal handler minimal: record the request and wake the
    # worker thread that performs lock-taking, I/O, YAML parsing, and warnings.
    if signum == int(getattr(signal, "SIGUSR1", -1)):
        _SIGNAL_REQUEST_COUNTS["reload"] += 1
    elif signum == int(getattr(signal, "SIGUSR2", -1)):
        _SIGNAL_REQUEST_COUNTS["flush"] += 1
    _SIGNAL_WORK_EVENT.set()

    previous = _PREVIOUS_SIGNAL_HANDLERS.get(signum)
    if callable(previous) and previous is not _handle_capture_signal:
        previous(signum, frame)


def _reset_capture_state_after_fork_in_child() -> None:
    # Only the forking thread survives into the child, so any lock another thread
    # (the flusher or signal worker) held at fork time is inherited locked and
    # would deadlock the child. Replace every lock with a fresh one and drop the
    # stale helper-thread handles; the per-runtime pid check rebuilds paths and
    # restarts threads on the next call.
    global _CAPTURE_STATES_LOCK, _SIGNAL_HANDLERS_LOCK, _SIGNAL_WORKER_LOCK
    global _SIGNAL_WORK_EVENT, _SIGNAL_WORKER_THREAD, _SIGNAL_WORKER_PID
    global _SIGNAL_DISPATCH_CV, _SIGNAL_DISPATCH_COUNT

    _CAPTURE_STATES_LOCK = threading.RLock()
    _SIGNAL_HANDLERS_LOCK = threading.Lock()
    _SIGNAL_WORKER_LOCK = threading.Lock()
    _SIGNAL_WORK_EVENT = threading.Event()
    _SIGNAL_WORKER_THREAD = None
    _SIGNAL_WORKER_PID = None
    _SIGNAL_DISPATCH_CV = threading.Condition(threading.Lock())
    _SIGNAL_DISPATCH_COUNT = 0
    _SIGNAL_REQUEST_COUNTS["reload"] = 0
    _SIGNAL_REQUEST_COUNTS["flush"] = 0
    _SIGNAL_PROCESSED_COUNTS["reload"] = 0
    _SIGNAL_PROCESSED_COUNTS["flush"] = 0

    states = list(_CAPTURE_STATES)
    for state in states:
        state._lock = threading.RLock()
        state._io_lock = threading.RLock()
        state._flush_event = threading.Event()
        state._flusher_thread = None
        state._flusher_pid = None
        state._record_buffers = {}
        state._pid = None

    if states:
        _ensure_signal_dispatcher()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        after_in_child=_reset_capture_state_after_fork_in_child,
    )


def load_full_call(
    path: Union[os.PathLike[str], str],
    *,
    tensor_device: str = "original",
    map_location: Any = "cpu",
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Load a full-call snapshot and return replayable ``(args, kwargs)``.

    ``tensor_device`` controls where saved tensors are restored:

    - ``"original"``: move tensors back to the device recorded at capture time.
      If that device is unavailable on the current machine the tensor is left
      on CPU rather than raising.
    - ``"cpu"``: leave tensors on CPU.
    - any other string: pass it to ``Tensor.to(...)`` as the target device.

    ``map_location`` is forwarded to ``torch.load``. It governs where tensors
    nested inside *opaque* pickled arguments (e.g. a ``RoutingData`` object)
    land — the ``tensor_device`` restore only relocates the framework's own
    saved-tensor wrappers, not tensors hidden inside third-party objects. Set it
    to the replay device (e.g. ``"cuda:0"``) so those nested tensors are usable.
    """

    snapshot = _load_snapshot(path, map_location=map_location)
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


_MISSING = object()


def _read_config(
    config_path: Optional[Union[os.PathLike[str], str]] = None,
) -> CaptureConfig:
    output_dir = Path(os.environ.get(OUTPUT_DIR_ENV, DEFAULT_OUTPUT_DIR)).expanduser()
    strict = _env_bool(STRICT_ENV, default=False)
    full_every_n = _env_int(FULL_EVERY_N_ENV, DEFAULT_FULL_EVERY_N)
    keep_tensors = _env_name_set(KEEP_TENSORS_ENV)
    config = CaptureConfig(
        output_dir=output_dir,
        metadata_every_n=DEFAULT_METADATA_EVERY_N,
        full_every_n=full_every_n,
        strict=strict,
        keep_tensors=keep_tensors,
    )

    effective_config_path = config_path
    if effective_config_path is None:
        env_config_path = os.environ.get(CONFIG_PATH_ENV)
        if env_config_path is not None and env_config_path.strip():
            effective_config_path = env_config_path

    if effective_config_path is not None:
        config = _apply_yaml_config(config, _expand_path(effective_config_path))

    _validate_capture_interval("metadata_every_n", config.metadata_every_n)
    _validate_capture_interval(FULL_EVERY_N_ENV, config.full_every_n)
    return config


def _apply_yaml_config(base: CaptureConfig, path: Path) -> CaptureConfig:
    root = _load_yaml_mapping(path)
    capture_section = _optional_mapping(root, "capture")

    output_dir = base.output_dir
    output_dir_value = _config_value(root, {}, ("output_dir",))
    if output_dir_value is not _MISSING:
        output_dir = _expand_path(_config_path_value(output_dir_value, "output_dir"))

    strict = base.strict
    strict_value = _config_value(root, {}, ("strict",))
    if strict_value is not _MISSING:
        strict = _coerce_bool(strict_value, "strict")

    metadata_every_n = base.metadata_every_n
    full_every_n = base.full_every_n

    mode_value = _config_value(root, capture_section, ("mode", "capture_mode"))
    frequency_value = _config_value(
        root,
        capture_section,
        ("frequency", "every_n", "capture_every_n"),
    )
    frequency = (
        None
        if frequency_value is _MISSING
        else _coerce_int(frequency_value, "frequency")
    )

    if mode_value is not _MISSING:
        metadata_every_n, full_every_n = _capture_intervals_for_mode(
            mode_value,
            frequency=frequency,
            base=base,
        )
    else:
        metadata_value = _config_value(
            root,
            capture_section,
            ("metadata", "capture_metadata"),
        )
        tensor_contents_value = _config_value(
            root,
            capture_section,
            (
                "tensor_contents",
                "capture_tensor_contents",
                "capture_tensors",
                "tensors",
                "full",
            ),
        )

        if metadata_value is not _MISSING:
            metadata_every_n = (
                _frequency_or_default(frequency, base.metadata_every_n)
                if _coerce_bool(metadata_value, "capture_metadata")
                else 0
            )
        elif frequency is not None:
            metadata_every_n = frequency

        if tensor_contents_value is not _MISSING:
            full_every_n = (
                _frequency_or_default(
                    frequency, base.full_every_n, fallback=DEFAULT_FULL_EVERY_N
                )
                if _coerce_bool(tensor_contents_value, "capture_tensor_contents")
                else 0
            )
        # A bare ``frequency`` only thins the metadata stream; it must not
        # silently change the (much more expensive) tensor-snapshot cadence,
        # which stays at whatever the environment/base config selected.

    explicit_metadata_every_n = _config_value(
        root,
        capture_section,
        ("metadata_every_n",),
    )
    if explicit_metadata_every_n is not _MISSING:
        metadata_every_n = _coerce_int(
            explicit_metadata_every_n,
            "metadata_every_n",
        )

    explicit_full_every_n = _config_value(
        root,
        capture_section,
        ("tensor_contents_every_n", "full_every_n"),
    )
    if explicit_full_every_n is not _MISSING:
        full_every_n = _coerce_int(explicit_full_every_n, "full_every_n")

    keep_tensors = base.keep_tensors
    keep_tensors_value = _config_value(
        root,
        capture_section,
        ("keep_tensors", "keep_tensor_contents", "full_tensors"),
    )
    if keep_tensors_value is not _MISSING:
        keep_tensors = _coerce_name_set(keep_tensors_value, "keep_tensors")

    return CaptureConfig(
        output_dir=output_dir,
        metadata_every_n=metadata_every_n,
        full_every_n=full_every_n,
        strict=strict,
        keep_tensors=keep_tensors,
    )


def _load_yaml_mapping(path: Path) -> Mapping[Any, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"capture config not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read func_capture tensor_args config files"
        ) from exc

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"capture config {path} must contain a YAML mapping")
    return loaded


def _optional_mapping(root: Mapping[Any, Any], key: str) -> Mapping[Any, Any]:
    value = root.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"capture config field {key!r} must be a mapping")
    return value


def _config_value(
    root: Mapping[Any, Any],
    capture_section: Mapping[Any, Any],
    names: tuple[str, ...],
) -> Any:
    for source in (capture_section, root):
        for name in names:
            if name in source:
                return source[name]
    return _MISSING


def _capture_intervals_for_mode(
    value: Any,
    *,
    frequency: Optional[int],
    base: CaptureConfig,
) -> tuple[int, int]:
    if value is False:
        return 0, 0
    if not isinstance(value, str):
        raise ValueError(f"capture mode must be a string or false, got {value!r}")

    mode = value.strip().lower().replace("-", "_").replace(" ", "_")
    if mode in {"metadata", "metadata_only", "meta"}:
        return _frequency_or_default(frequency, base.metadata_every_n), 0
    if mode in {
        "metadata_and_tensors",
        "metadata_with_tensors",
        "metadata_plus_tensors",
        "tensors",
        "tensor_contents",
        "full",
        "all",
    }:
        if frequency is None:
            return (
                _frequency_or_default(None, base.metadata_every_n),
                _frequency_or_default(
                    None, base.full_every_n, fallback=DEFAULT_FULL_EVERY_N
                ),
            )
        return frequency, frequency
    if mode in {"off", "none", "disabled", "false"}:
        return 0, 0

    raise ValueError(
        "capture mode must be one of metadata, metadata_and_tensors, or off; "
        f"got {value!r}"
    )


def _frequency_or_default(
    frequency: Optional[int],
    default: int,
    *,
    fallback: int = DEFAULT_METADATA_EVERY_N,
) -> int:
    if frequency is not None:
        return frequency
    if default > 0:
        return default
    # The caller asked to enable a capture stream but neither the config nor the
    # environment gave a cadence. ``fallback`` lets tensor-snapshot callers land
    # on the sparse ``DEFAULT_FULL_EVERY_N`` instead of the every-call metadata
    # default, which would copy every tensor to CPU on a hot path.
    return fallback


def _expand_path(value: Union[os.PathLike[str], str]) -> Path:
    return Path(os.path.expandvars(os.fspath(value))).expanduser()


def _config_path_value(value: Any, name: str) -> Union[os.PathLike[str], str]:
    if isinstance(value, (str, os.PathLike)):
        return value
    raise ValueError(f"{name} must be a path string, got {value!r}")


def _captures_call(every_n: int, call_index: int) -> bool:
    return every_n > 0 and call_index % every_n == 0


def _validate_capture_interval(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return _coerce_int(value, name)


def _env_name_set(name: str) -> Optional[frozenset[str]]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return _coerce_name_set(value, name)


def _coerce_name_set(value: Any, name: str) -> frozenset[str]:
    if isinstance(value, str):
        parts = re.split(r"[,\s]+", value.strip())
        return frozenset(part for part in parts if part)
    if isinstance(value, (list, tuple, set, frozenset)):
        names: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"{name} entries must be strings, got {item!r}")
            item = item.strip()
            if item:
                names.append(item)
        return frozenset(names)
    raise ValueError(f"{name} must be a string or list of strings, got {value!r}")


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return _coerce_bool(value, name)


def _coerce_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    raise ValueError(f"{name} must be an integer, got {value!r}")


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _strict_requested_after_config_error(
    config_path: Optional[Union[os.PathLike[str], str]] = None,
) -> bool:
    try:
        if _env_bool(STRICT_ENV, default=False):
            return True
    except ValueError:
        return True
    effective_config_path = config_path
    if effective_config_path is None:
        env_config_path = os.environ.get(CONFIG_PATH_ENV)
        if env_config_path is not None and env_config_path.strip():
            effective_config_path = env_config_path
    if effective_config_path is None:
        return False
    try:
        root = _load_yaml_mapping(_expand_path(effective_config_path))
        strict_value = _config_value(root, {}, ("strict",))
        if strict_value is _MISSING:
            return False
        return _coerce_bool(strict_value, "strict")
    except Exception:
        return False


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
    arguments = {
        name: _value_metadata(value, path=name)
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
    path: str,
    depth: int = 0,
    seen: tuple[int, ...] = (),
) -> dict[str, Any]:
    if _is_tensor(value):
        return _tensor_metadata(value)

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


def _tensor_metadata(value: Any) -> dict[str, Any]:
    shape = _tensor_shape(value)
    stride = _tensor_stride(value)
    metadata: dict[str, Any] = {
        "kind": "tensor",
        "shape": shape,
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


def _save_full_call(
    path: Path,
    *,
    func: Callable[..., Any],
    function_key: str,
    call_index: int,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    signature: Optional[inspect.Signature] = None,
    keep_tensors: Optional[frozenset[str]] = None,
) -> None:
    # When ``keep_tensors`` is set, only arguments whose (bound) parameter name
    # is listed keep their tensor contents; all other tensors are written as
    # metadata-only stubs. ``keep_tensors is None`` preserves everything.
    positional_names = _positional_param_names(signature)

    def save_arg(index: int, value: Any) -> Any:
        name = positional_names[index] if index < len(positional_names) else None
        return _saved_value_with_policy(value, _keep_real(name, keep_tensors))

    def save_kwarg(name: str, value: Any) -> Any:
        return _saved_value_with_policy(value, _keep_real(name, keep_tensors))

    snapshot = {
        "version": FORMAT_VERSION,
        "function": function_key,
        "module": getattr(func, "__module__", ""),
        "qualname": getattr(func, "__qualname__", getattr(func, "__name__", "")),
        "call_index": call_index,
        "time_ns": time.time_ns(),
        "process_id": os.getpid(),
        "args": tuple(save_arg(index, value) for index, value in enumerate(args)),
        "kwargs": {key: save_kwarg(key, value) for key, value in kwargs.items()},
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    _save_snapshot(snapshot, tmp_path)
    os.replace(tmp_path, path)


def _keep_real(name: Optional[str], keep_tensors: Optional[frozenset[str]]) -> bool:
    if keep_tensors is None:
        return True
    return name is not None and name in keep_tensors


def _positional_param_names(signature: Optional[inspect.Signature]) -> list[str]:
    if signature is None:
        return []
    names: list[str] = []
    for param in signature.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            names.append(param.name)
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            # Positional args beyond this point can't be name-mapped; stop so
            # they fall through to the "unknown name" (stub-if-filtering) case.
            break
    return names


def _saved_value_with_policy(value: Any, keep_real: bool) -> Any:
    return _saved_value(value) if keep_real else _stub_value(value)


def _stub_value(value: Any, depth: int = 0, seen: tuple[int, ...] = ()) -> Any:
    """Like ``_saved_value`` but tensors become metadata-only stubs.

    Container structure and non-tensor leaves are preserved so a stubbed
    argument still reconstructs with the right shape (e.g. a list of tensors).
    """

    if _is_tensor(value):
        return {TENSOR_STUB_MARKER: True, "metadata": _tensor_metadata(value)}

    if depth >= MAX_CAPTURE_DEPTH:
        return _safe_repr(value)

    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        if id(value) in seen:
            return _safe_repr(value)
        seen = seen + (id(value),)

    if isinstance(value, Mapping):
        return {key: _stub_value(child, depth + 1, seen) for key, child in value.items()}
    if _is_namedtuple(value):
        return type(value)(*(_stub_value(child, depth + 1, seen) for child in value))
    if isinstance(value, list):
        return [_stub_value(child, depth + 1, seen) for child in value]
    if isinstance(value, tuple):
        return tuple(_stub_value(child, depth + 1, seen) for child in value)
    if isinstance(value, (set, frozenset)):
        return {
            SET_MARKER: "frozenset" if isinstance(value, frozenset) else "set",
            "items": [_stub_value(child, depth + 1, seen) for child in value],
        }
    return value


def _saved_value(value: Any, depth: int = 0, seen: tuple[int, ...] = ()) -> Any:
    if _is_tensor(value):
        copied = _copy_tensor_to_cpu(value)
        metadata = _tensor_metadata(value)
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
    if _is_stub_tensor(value):
        return _synthesize_tensor(value.get("metadata", {}), tensor_device)
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


def _is_stub_tensor(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get(TENSOR_STUB_MARKER) is True
        and "metadata" in value
    )


def _synthesize_tensor(metadata: Mapping[str, Any], tensor_device: str) -> Any:
    """Build a random tensor matching a stub's recorded shape/dtype/layout.

    Contents are random because a performance replay only needs the kernel to
    see the right shapes, dtypes, strides, and device — not the original data.
    """

    torch = _get_torch()
    if torch is None:
        raise RuntimeError("torch is required to rehydrate a tensor stub")

    shape = metadata.get("shape", [])
    if not _all_ints(shape):
        raise ValueError(
            f"cannot synthesize a tensor for stub with non-integer shape {shape!r}"
        )
    dtype = _parse_dtype(torch, metadata.get("dtype"))
    device = _target_device(metadata, tensor_device) or "cpu"

    stride = metadata.get("stride")
    storage_offset = metadata.get("storage_offset", 0)
    use_strided = (
        isinstance(stride, list)
        and _all_ints(stride)
        and isinstance(storage_offset, int)
        and not _is_contiguous_stride(shape, stride)
        and not _strided_overlaps(shape, stride)
    )

    try:
        if use_strided:
            span = storage_offset + _strided_storage_span(shape, stride)
            flat = _random_tensor(torch, [span], dtype, device)
            tensor = torch.as_strided(flat, tuple(shape), tuple(stride), storage_offset)
        else:
            tensor = _random_tensor(torch, shape, dtype, device)
    except Exception:
        # Recorded device/layout unavailable here; fall back to a plain CPU
        # tensor of the right shape/dtype rather than failing the replay.
        tensor = _random_tensor(torch, shape, dtype, "cpu")

    if metadata.get("requires_grad") and getattr(tensor, "is_floating_point", None):
        try:
            if tensor.is_floating_point():
                tensor = tensor.requires_grad_(True)
        except (RuntimeError, AttributeError):
            pass
    return tensor


def _random_tensor(torch: Any, shape: Any, dtype: Any, device: str) -> Any:
    shape = tuple(shape)
    if dtype is None:
        return torch.zeros(shape, device=device)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, device=device, dtype=torch.bool)
    if getattr(dtype, "is_floating_point", False):
        try:
            return torch.randn(shape, device=device, dtype=dtype)
        except (RuntimeError, TypeError):
            # Some low-precision float dtypes (fp8 variants) don't support randn
            # directly; generate in float32 and cast.
            return torch.randn(shape, device=device, dtype=torch.float32).to(dtype)
    if getattr(dtype, "is_complex", False):
        return torch.randn(shape, device=device, dtype=dtype)
    # Integer / unsigned dtypes: fill the low byte so values stay valid for
    # every width (uint8 packed MXFP4, index-like ints, etc.).
    high = 256 if dtype is torch.uint8 else 128
    return torch.randint(0, high, shape, device=device, dtype=dtype)


def _parse_dtype(torch: Any, dtype_str: Any) -> Any:
    if not isinstance(dtype_str, str) or not dtype_str:
        return None
    name = dtype_str[len("torch.") :] if dtype_str.startswith("torch.") else dtype_str
    dtype = getattr(torch, name, None)
    # Guard against attribute names that aren't actually dtypes.
    if dtype is not None and type(dtype).__name__ == "dtype":
        return dtype
    return None


def _is_contiguous_stride(shape: list[Any], stride: list[Any]) -> bool:
    if len(shape) != len(stride):
        return False
    expected = 1
    for dim, dim_stride in zip(reversed(shape), reversed(stride)):
        if not isinstance(dim, int) or not isinstance(dim_stride, int):
            return False
        if dim != 1 and dim_stride != expected:
            return False
        expected *= dim if dim > 0 else 1
    return True


def _strided_storage_span(shape: list[Any], stride: list[Any]) -> int:
    # Number of elements the last valid index reaches (plus one) for a strided
    # tensor, so the backing storage is large enough for ``as_strided``.
    span = 1
    for dim, dim_stride in zip(shape, stride):
        if dim > 0:
            span += (dim - 1) * dim_stride
    return span


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
        # equally strided destination without undefined behaviour, and
        # non-integer shape/stride values cannot be passed to ``empty_strided``.
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


def _load_snapshot(
    path: Union[os.PathLike[str], str],
    map_location: Any = "cpu",
) -> Mapping[str, Any]:
    torch = _get_torch()
    if torch is not None and hasattr(torch, "load"):
        try:
            return cast(Mapping[str, Any], torch.load(path, map_location=map_location, weights_only=False))
        except TypeError:
            return cast(Mapping[str, Any], torch.load(path, map_location=map_location))
    with Path(path).open("rb") as file:
        return cast(Mapping[str, Any], pickle.load(file))


def _append_jsonl(path: Path, record: Mapping[str, Any], lock: threading.Lock) -> None:
    with lock:
        _append_jsonl_records(path, [record])


def _append_jsonl_records(path: Path, records: list[Mapping[str, Any]]) -> None:
    if not records:
        return
    # Serialise the whole line (including the trailing newline) into a single
    # write so an O_APPEND write stays atomic and lines are never torn.
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(payload)


def _is_tensor(value: Any) -> bool:
    torch = _get_torch()
    tensor_type = getattr(torch, "Tensor", None) if torch is not None else None
    return tensor_type is not None and isinstance(value, tensor_type)


def _get_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _is_cuda_graph_capturing() -> bool:
    # True while a CUDA/HIP graph capture is in progress on the current stream.
    # Guarded so the check itself never raises on CPU-only builds or older torch.
    torch = _get_torch()
    if torch is None:
        return False
    cuda = getattr(torch, "cuda", None)
    is_capturing = getattr(cuda, "is_current_stream_capturing", None)
    if not callable(is_capturing):
        return False
    try:
        return bool(is_capturing())
    except Exception:
        return False


def _coerce_dim(dim: Any) -> Any:
    # Concrete Python sizes/strides stay plain ints. Non-int shape objects are
    # stringified so metadata remains JSON-serialisable without invoking
    # ``__int__``.
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
