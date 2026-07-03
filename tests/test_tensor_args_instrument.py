from __future__ import annotations

import collections
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

from func_capture.instruments import tensor_args


# Module-level so torch.save's pickle fallback can serialise instances.
Plan = collections.namedtuple("Plan", ["count", "tensor"])


class FakeSymInt:
    def __init__(self, name: str) -> None:
        self.name = name

    def __int__(self) -> int:
        raise AssertionError("symbolic dimensions must not be coerced to int")

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
            kv_dim_symbol = second_call["arguments"]["kv"]["symbolic_shape"][1]
            ape_dim_symbol = second_call["arguments"]["ape"]["symbolic_shape"][1]
            ape_ratio_symbol = second_call["arguments"]["ape"]["symbolic_shape"][0]
            write_plan_n_symbol = second_call["arguments"]["write_plan"]["symbolic_shape"][0]
            self.assertNotEqual(ape_ratio_symbol, "ratio")
            self.assertNotEqual(write_plan_n_symbol, "num_write")
            self.assertNotEqual(ape_dim_symbol, kv_dim_symbol)
            self.assertEqual(second_call["tensor_shape_symbols"][ape_ratio_symbol], 2)
            self.assertEqual(second_call["tensor_shape_symbols"][ape_dim_symbol], 8)
            self.assertEqual(second_call["tensor_shape_symbols"][kv_dim_symbol], 8)
            self.assertEqual(second_call["tensor_shape_symbols"][write_plan_n_symbol], 6)
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


    def _records(self, temp_dir):
        record_files = list(Path(temp_dir).glob("*/calls.*.jsonl"))
        self.assertEqual(len(record_files), 1)
        return record_files[0], [
            json.loads(line)
            for line in record_files[0].read_text(encoding="utf-8").splitlines()
        ]

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

    def test_symbolic_dimensions_are_not_forced_to_int(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        def target(x):
            return x

        with fake_torch_module(), patched_env(
            FUNC_CAPTURE_OUTPUT_DIR=temp_dir.name,
            FUNC_CAPTURE_FULL_EVERY_N=0,
        ):
            wrapped = tensor_args.instrument(target)
            wrapped(FakeTensor((FakeSymInt("s0"), 4), stride=(4, 1)))

        _, records = self._records(temp_dir.name)
        metadata = records[0]["arguments"]["x"]
        self.assertEqual(metadata["shape"], ["s0", 4])
        self.assertEqual(metadata["symbolic_shape"][0], "s0")
        concrete_symbol = metadata["symbolic_shape"][1]
        self.assertEqual(records[0]["tensor_shape_symbols"][concrete_symbol], 4)

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
