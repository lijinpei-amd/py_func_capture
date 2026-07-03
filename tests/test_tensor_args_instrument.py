from __future__ import annotations

import collections
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import tempfile
import time
import unittest

from func_capture import capture
from func_capture.instruments import tensor_args


# Module-level so torch.save's pickle fallback can serialise instances.
Plan = collections.namedtuple("Plan", ["count", "tensor"])


class FakeDynamicDim:
    def __init__(self, name: str) -> None:
        self.name = name

    def __int__(self) -> int:
        raise AssertionError("non-int dimensions must not be coerced to int")

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        dtype: str = "torch.float32",
        device: str = "cuda:0",
        stride=None,
        requires_grad: bool = False,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.layout = "torch.strided"
        self.requires_grad = requires_grad
        self._stride = tuple(stride) if stride is not None else self._contiguous_stride()

    def stride(self):
        return self._stride

    def storage_offset(self):
        return 0

    def is_contiguous(self):
        return self._stride == self._contiguous_stride()

    def detach(self):
        return self

    def cpu(self):
        return FakeTensor(
            self.shape,
            dtype=self.dtype,
            device="cpu",
            stride=self._stride,
            requires_grad=self.requires_grad,
        )

    def clone(self):
        return FakeTensor(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            stride=self._stride,
            requires_grad=self.requires_grad,
        )

    def copy_(self, other, non_blocking=False):
        self.shape = other.shape
        self.dtype = other.dtype
        self.requires_grad = other.requires_grad
        return self

    def to(self, device):
        return FakeTensor(
            self.shape,
            dtype=self.dtype,
            device=device,
            stride=self._stride,
            requires_grad=self.requires_grad,
        )

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self

    def _contiguous_stride(self):
        stride = []
        running = 1
        for dim in reversed(self.shape):
            stride.append(running)
            running *= dim
        return tuple(reversed(stride))


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    Tensor = FakeTensor

    @staticmethod
    def no_grad():
        return FakeNoGrad()

    empty_strided_calls = 0

    @staticmethod
    def empty_strided(shape, stride, *, dtype=None, device=None):
        FakeTorch.empty_strided_calls += 1
        return FakeTensor(shape, dtype=dtype, device=device, stride=stride)

    @staticmethod
    def save(snapshot, path):
        with Path(path).open("wb") as file:
            pickle.dump(snapshot, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path, map_location=None, weights_only=False):
        with Path(path).open("rb") as file:
            return pickle.load(file)


@contextmanager
def fake_torch_module():
    old_torch = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch()
    try:
        yield
    finally:
        if old_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = old_torch


@contextmanager
def patched_env(**updates):
    old_values = {key: os.environ.get(key) for key in updates}
    os.environ.update({key: str(value) for key, value in updates.items()})
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TensorArgsInstrumentTests(unittest.TestCase):
    def test_records_metadata_every_call_and_full_args_every_n_calls(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(
            kv,
            score,
            ape,
            kv_state,
            score_state,
            *,
            write_plan,
            num_write: int,
            state_slot_mapping,
            ratio: int,
            overlap: bool,
        ):
            return "called"

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
            FUNC_CAPTURE_FULL_EVERY_N=2,
        ):
            wrapped = tensor_args.instrument(target)
            for _ in range(3):
                self.assertEqual(
                    wrapped(
                        FakeTensor((3, 8), stride=(16, 1)),
                        FakeTensor((3, 8), stride=(16, 1)),
                        FakeTensor((2, 8)),
                        FakeTensor((4, 5, 8)),
                        FakeTensor((4, 5, 8)),
                        write_plan=FakeTensor((6, 4), dtype="torch.int32"),
                        num_write=6,
                        state_slot_mapping=FakeTensor((7,), dtype="torch.int32"),
                        ratio=2,
                        overlap=True,
                    ),
                    "called",
                )

            tensor_args._flush_all_capture_states()
            record_files = list(Path(temp_dir.name).glob("*/calls.*.jsonl"))
            self.assertEqual(len(record_files), 1)
            records = [
                json.loads(line)
                for line in record_files[0].read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual([record["call_index"] for record in records], [1, 2, 3])
            self.assertNotIn("full_capture", records[0])
            self.assertIn("full_capture", records[1])
            self.assertNotIn("full_capture", records[2])

            second_call = records[1]
            self.assertEqual(second_call["arguments"]["ratio"]["value"], 2)
            self.assertEqual(second_call["arguments"]["overlap"]["value"], True)
            self.assertNotIn("tensor_shape_symbols", second_call)
            self.assertNotIn("symbolic_shape", second_call["arguments"]["kv"])
            self.assertEqual(second_call["arguments"]["kv"]["shape"], [3, 8])
            self.assertEqual(second_call["arguments"]["kv"]["stride"], [16, 1])

            full_path = record_files[0].parent / second_call["full_capture"]["path"]
            self.assertTrue(full_path.exists())
            args, kwargs = tensor_args.load_full_call(full_path, tensor_device="cpu")
            self.assertEqual(args[0].shape, (3, 8))
            self.assertEqual(args[0].device, "cpu")
            self.assertEqual(args[0].stride(), (16, 1))
            self.assertEqual(kwargs["num_write"], 6)
            self.assertEqual(kwargs["ratio"], 2)

    def test_yaml_config_metadata_only_frequency(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / "capture.yaml"
        config_path.write_text(
            f"""
output_dir: {temp_dir.name}
capture:
  mode: metadata
  frequency: 2
""".lstrip(),
            encoding="utf-8",
        )

        def target(x):
            return x

        with fake_torch_module():
            wrapped = tensor_args.instrument(target, config_path=config_path)
            wrapped(FakeTensor((1,)))
            wrapped(FakeTensor((2,)))
            wrapped(FakeTensor((3,)))

        record_file, records = self._records(temp_dir.name)
        self.assertEqual([record["call_index"] for record in records], [2])
        self.assertNotIn("full_capture", records[0])
        self.assertFalse((record_file.parent / "full").exists())

    def test_yaml_config_captures_tensor_contents_on_frequency(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / "capture.yaml"
        config_path.write_text(
            f"""
output_dir: {temp_dir.name}
capture:
  mode: metadata_and_tensors
  frequency: 2
""".lstrip(),
            encoding="utf-8",
        )

        def target(x):
            return x

        with fake_torch_module():
            wrapped = tensor_args.instrument(target, config_path=config_path)
            wrapped(FakeTensor((1,)))
            wrapped(FakeTensor((2,)))
            wrapped(FakeTensor((3,)))

        record_file, records = self._records(temp_dir.name)
        self.assertEqual([record["call_index"] for record in records], [2])
        self.assertIn("full_capture", records[0])
        full_path = record_file.parent / records[0]["full_capture"]["path"]
        self.assertTrue(full_path.exists())
        args, _kwargs = tensor_args.load_full_call(full_path, tensor_device="cpu")
        self.assertEqual(args[0].shape, (2,))

    def test_config_path_can_be_forwarded_through_capture(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / "capture.yaml"
        config_path.write_text(
            f"""
output_dir: {temp_dir.name}
capture:
  mode: metadata
  frequency: 2
""".lstrip(),
            encoding="utf-8",
        )

        def target(x):
            return x

        script_path = Path(tensor_args.__file__).resolve()
        with fake_torch_module(), patched_env(FUNC_CAPTURE=f"chosen={script_path}"):
            wrapped = capture("chosen", config_path=config_path)(target)
            wrapped(FakeTensor((1,)))
            wrapped(FakeTensor((2,)))

        _record_file, records = self._records(temp_dir.name)
        self.assertEqual([record["call_index"] for record in records], [2])
        self.assertNotIn("full_capture", records[0])

    def test_metadata_records_are_buffered_until_flush(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(x):
            return x

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
            FUNC_CAPTURE_FULL_EVERY_N=0,
        ):
            wrapped = tensor_args.instrument(target)
            wrapped(FakeTensor((1,)))

            self.assertEqual(list(Path(temp_dir.name).glob("*/calls.*.jsonl")), [])
            tensor_args._flush_all_capture_states()

        _record_file, records = self._records(temp_dir.name)
        self.assertEqual([record["call_index"] for record in records], [1])

    @unittest.skipUnless(
        hasattr(signal, "SIGUSR2") and hasattr(signal, "raise_signal"),
        "SIGUSR2 is not available on this platform",
    )
    def test_sigusr2_flushes_buffered_metadata(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(x):
            return x

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
            FUNC_CAPTURE_FULL_EVERY_N=0,
        ):
            wrapped = tensor_args.instrument(target)
            wrapped(FakeTensor((1,)))

            self.assertEqual(list(Path(temp_dir.name).glob("*/calls.*.jsonl")), [])
            before = tensor_args._signal_dispatch_count()
            signal.raise_signal(signal.SIGUSR2)
            self.assertTrue(
                tensor_args._wait_for_signal_dispatch(before, timeout=5.0)
            )

        record_files = list(Path(temp_dir.name).glob("*/calls.*.jsonl"))
        self.assertEqual(len(record_files), 1)
        records = [
            json.loads(line)
            for line in record_files[0].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([record["call_index"] for record in records], [1])

    @unittest.skipUnless(
        hasattr(signal, "SIGUSR1") and hasattr(signal, "raise_signal"),
        "SIGUSR1 is not available on this platform",
    )
    def test_sigusr1_reloads_yaml_config(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / "capture.yaml"

        def write_config(frequency: int) -> None:
            config_path.write_text(
                f"""
output_dir: {temp_dir.name}
capture:
  mode: metadata
  frequency: {frequency}
""".lstrip(),
                encoding="utf-8",
            )

        def target(x):
            return x

        write_config(10)
        with fake_torch_module():
            wrapped = tensor_args.instrument(target, config_path=config_path)
            wrapped(FakeTensor((1,)))

            write_config(1)
            before = tensor_args._signal_dispatch_count()
            signal.raise_signal(signal.SIGUSR1)
            self.assertTrue(
                tensor_args._wait_for_signal_dispatch(before, timeout=5.0)
            )
            wrapped(FakeTensor((2,)))

        _record_file, records = self._records(temp_dir.name)
        self.assertEqual([record["call_index"] for record in records], [2])

    @unittest.skipUnless(
        hasattr(signal, "SIGUSR1") and hasattr(signal, "raise_signal"),
        "SIGUSR1 is not available on this platform",
    )
    def test_sigusr1_can_enable_initially_disabled_config(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / "capture.yaml"

        def write_config(mode: str) -> None:
            config_path.write_text(
                f"""
output_dir: {temp_dir.name}
capture:
  mode: {mode}
  frequency: 1
""".lstrip(),
                encoding="utf-8",
            )

        def target(x):
            return x

        write_config("off")
        with fake_torch_module():
            wrapped = tensor_args.instrument(target, config_path=config_path)
            self.assertIsNot(wrapped, target)
            wrapped(FakeTensor((1,)))

            write_config("metadata")
            before = tensor_args._signal_dispatch_count()
            signal.raise_signal(signal.SIGUSR1)
            self.assertTrue(
                tensor_args._wait_for_signal_dispatch(before, timeout=5.0)
            )
            wrapped(FakeTensor((2,)))

        _record_file, records = self._records(temp_dir.name)
        self.assertEqual([record["call_index"] for record in records], [1])

    def test_metadata_records_flush_periodically(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        original_interval = tensor_args.DEFAULT_METADATA_FLUSH_INTERVAL_SECONDS

        def target(x):
            return x

        try:
            tensor_args.DEFAULT_METADATA_FLUSH_INTERVAL_SECONDS = 0.01
            with fake_torch_module(), patched_env(
                FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
                FUNC_CAPTURE_FULL_EVERY_N=0,
            ):
                wrapped = tensor_args.instrument(target)
                wrapped(FakeTensor((1,)))

                deadline = time.time() + 1.0
                record_files = []
                while time.time() < deadline:
                    record_files = list(Path(temp_dir.name).glob("*/calls.*.jsonl"))
                    if record_files:
                        break
                    time.sleep(0.01)
        finally:
            tensor_args.DEFAULT_METADATA_FLUSH_INTERVAL_SECONDS = original_interval

        self.assertEqual(len(record_files), 1)
        records = [
            json.loads(line)
            for line in record_files[0].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([record["call_index"] for record in records], [1])

    def test_mode_tensors_without_frequency_uses_sparse_full_default(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / "capture.yaml"
        config_path.write_text(
            "capture:\n  mode: metadata_and_tensors\n",
            encoding="utf-8",
        )

        # Snapshots explicitly disabled in the environment, mode asks for
        # tensors but gives no cadence: full capture must fall back to the sparse
        # DEFAULT_FULL_EVERY_N, not to the every-call metadata default.
        with patched_env(FUNC_CAPTURE_FULL_EVERY_N="0"):
            config = tensor_args._read_config(config_path)

        self.assertEqual(config.full_every_n, tensor_args.DEFAULT_FULL_EVERY_N)
        self.assertEqual(
            config.metadata_every_n, tensor_args.DEFAULT_METADATA_EVERY_N
        )

    def test_bare_frequency_does_not_change_snapshot_cadence(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / "capture.yaml"
        config_path.write_text(
            "capture:\n  frequency: 5\n",
            encoding="utf-8",
        )

        # A bare frequency thins metadata only; the snapshot cadence stays at the
        # environment/base value instead of being silently pulled down to 5.
        with patched_env(FUNC_CAPTURE_FULL_EVERY_N="200"):
            config = tensor_args._read_config(config_path)

        self.assertEqual(config.metadata_every_n, 5)
        self.assertEqual(config.full_every_n, 200)

    @unittest.skipUnless(hasattr(os, "fork"), "fork is not available")
    def test_capture_works_in_forked_child(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(x):
            return x

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
            FUNC_CAPTURE_FULL_EVERY_N=0,
        ):
            wrapped = tensor_args.instrument(target)
            wrapped(FakeTensor((1,)))

            pid = os.fork()
            if pid == 0:  # pragma: no cover - runs only in the forked child.
                exit_code = 0
                try:
                    wrapped(FakeTensor((2,)))
                    tensor_args._flush_all_capture_states()
                except BaseException:
                    exit_code = 1
                finally:
                    os._exit(exit_code)

            deadline = time.time() + 10.0
            status = None
            while time.time() < deadline:
                waited_pid, waited_status = os.waitpid(pid, os.WNOHANG)
                if waited_pid == pid:
                    status = waited_status
                    break
                time.sleep(0.02)

            if status is None:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                self.fail("forked child hung (fork deadlock in capture runtime)")

            self.assertEqual(os.waitstatus_to_exitcode(status), 0)

        tensor_args._flush_all_capture_states()
        # The child wrote its own pid-scoped records file without clobbering the
        # parent's buffered records; both should be present after flushing.
        record_files = sorted(Path(temp_dir.name).glob("*/calls.*.jsonl"))
        self.assertEqual(len(record_files), 2)

    def _records(self, temp_dir):
        self._flush_capture_states()
        record_files = list(Path(temp_dir).glob("*/calls.*.jsonl"))
        self.assertEqual(len(record_files), 1)
        return record_files[0], [
            json.loads(line)
            for line in record_files[0].read_text(encoding="utf-8").splitlines()
        ]

    def _flush_capture_states(self) -> None:
        # ``capture()`` execs the instrument script into a distinct module
        # object, so runtimes may be registered in a different copy of this
        # module than the one imported here. Flush every loaded copy.
        target = Path(tensor_args.__file__).resolve()
        for module in list(sys.modules.values()):
            module_file = getattr(module, "__file__", None)
            if module_file is None or Path(module_file).resolve() != target:
                continue
            flush = getattr(module, "_flush_all_capture_states", None)
            if callable(flush):
                flush()

    def test_apply_defaults_records_omitted_default_arguments(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(a, b=10, *, overlap=False):
            return a

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
            FUNC_CAPTURE_FULL_EVERY_N=0,
        ):
            wrapped = tensor_args.instrument(target)
            wrapped(1)

        _, records = self._records(temp_dir.name)
        arguments = records[0]["arguments"]
        # b and overlap were omitted by the caller but must still be recorded.
        self.assertEqual(arguments["b"]["value"], 10)
        self.assertEqual(arguments["overlap"]["value"], False)

    def test_namedtuple_round_trips_with_type_preserved(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(plan):
            return plan

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
            FUNC_CAPTURE_FULL_EVERY_N=1,
        ):
            wrapped = tensor_args.instrument(target)
            wrapped(Plan(count=3, tensor=FakeTensor((2, 2))))

            record_file, records = self._records(temp_dir.name)
            full_path = record_file.parent / records[0]["full_capture"]["path"]
            (args, _kwargs) = tensor_args.load_full_call(full_path, tensor_device="cpu")

        restored = args[0]
        self.assertIsInstance(restored, Plan)
        self.assertEqual(restored.count, 3)
        self.assertEqual(restored.tensor.shape, (2, 2))

    def test_frozenset_captures_tensors_and_round_trips(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(items):
            return items

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
            FUNC_CAPTURE_FULL_EVERY_N=1,
        ):
            wrapped = tensor_args.instrument(target)
            wrapped(frozenset({FakeTensor((5,))}))

            record_file, records = self._records(temp_dir.name)
            # The tensor inside the frozenset must be detected in the metadata.
            self.assertEqual(records[0]["arguments"]["items"]["kind"], "set")
            item_kinds = [item["kind"] for item in records[0]["arguments"]["items"]["items"]]
            self.assertIn("tensor", item_kinds)

            full_path = record_file.parent / records[0]["full_capture"]["path"]
            (args, _kwargs) = tensor_args.load_full_call(full_path, tensor_device="cpu")

        self.assertIsInstance(args[0], frozenset)
        restored = next(iter(args[0]))
        self.assertEqual(restored.device, "cpu")

    def test_expanded_tensor_skips_empty_strided(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(broadcast):
            return broadcast

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
            FUNC_CAPTURE_FULL_EVERY_N=1,
        ):
            FakeTorch.empty_strided_calls = 0
            wrapped = tensor_args.instrument(target)
            # stride-0 dim of size > 1 => overlapping, must not use empty_strided.
            wrapped(FakeTensor((3, 4), stride=(0, 1)))

            self.assertEqual(FakeTorch.empty_strided_calls, 0)

            record_file, records = self._records(temp_dir.name)
            full_path = record_file.parent / records[0]["full_capture"]["path"]
            (args, _kwargs) = tensor_args.load_full_call(full_path, tensor_device="cpu")

        self.assertEqual(args[0].shape, (3, 4))

    def test_strict_mode_reraises_capture_error(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(x):
            return x

        original_save = tensor_args._save_full_call

        def boom(*_args, **_kwargs):
            raise RuntimeError("save failed")

        with fake_torch_module():
            tensor_args._save_full_call = boom
            try:
                with patched_env(
                    FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
                    FUNC_CAPTURE_FULL_EVERY_N=1,
                    FUNC_CAPTURE_STRICT="1",
                ):
                    wrapped = tensor_args.instrument(target)
                    with self.assertRaises(RuntimeError):
                        wrapped(FakeTensor((2,)))

                # Non-strict swallows the same failure and records it instead.
                with patched_env(
                    FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
                    FUNC_CAPTURE_FULL_EVERY_N=1,
                    FUNC_CAPTURE_STRICT="0",
                ):
                    wrapped = tensor_args.instrument(target)
                    self.assertEqual(wrapped(FakeTensor((2,))).shape, (2,))
            finally:
                tensor_args._save_full_call = original_save

        _, records = self._records(temp_dir.name)
        self.assertIn("capture_error", records[-1])

    def test_invalid_config_degrades_to_passthrough(self) -> None:
        def target(x):
            return x

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR="unused",
            FUNC_CAPTURE_FULL_EVERY_N="-1",
        ):
            with self.assertWarns(RuntimeWarning):
                wrapped = tensor_args.instrument(target)
            # Bad config must not crash instrumentation; return the original.
            self.assertIs(wrapped, target)

    def test_invalid_config_reraises_when_strict_requested(self) -> None:
        def target(x):
            return x

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR="unused",
            FUNC_CAPTURE_FULL_EVERY_N="-1",
            FUNC_CAPTURE_STRICT="1",
        ):
            with self.assertRaises(ValueError):
                tensor_args.instrument(target)

    def test_non_int_dimensions_are_not_forced_to_int(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(x):
            return x

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
            FUNC_CAPTURE_FULL_EVERY_N=0,
        ):
            wrapped = tensor_args.instrument(target)
            wrapped(FakeTensor((FakeDynamicDim("s0"), 4), stride=(4, 1)))

        _, records = self._records(temp_dir.name)
        metadata = records[0]["arguments"]["x"]
        self.assertEqual(metadata["shape"], ["s0", 4])
        self.assertNotIn("symbolic_shape", metadata)
        self.assertNotIn("tensor_shape_symbols", records[0])

    def test_reference_cycle_does_not_recurse_infinitely(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(data):
            return data

        cyclic: list = []
        cyclic.append(cyclic)

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
            FUNC_CAPTURE_FULL_EVERY_N=1,
        ):
            wrapped = tensor_args.instrument(target)
            # Must not raise RecursionError.
            wrapped(cyclic)

        _, records = self._records(temp_dir.name)
        self.assertEqual(records[0]["arguments"]["data"]["kind"], "sequence")


if __name__ == "__main__":
    unittest.main()
