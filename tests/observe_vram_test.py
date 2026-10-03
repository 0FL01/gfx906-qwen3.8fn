#!/usr/bin/env python3
"""Synthetic sysfs and tiny CPU children only; no target/GPU qualification."""

import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "observe_vram.py"
spec = importlib.util.spec_from_file_location("observe_vram", SCRIPT)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class ObserveVramTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.devices = [self.root / "gpu0", self.root / "gpu1"]
        for index, path in enumerate(self.devices):
            path.mkdir()
            (path / "mem_info_vram_total").write_bytes(b"1000\n")
            (path / "mem_info_vram_used").write_bytes(str(100 + index).encode() + b"\n")
        self.output = self.root / "observation.jsonl"

    def argv(self, code="pass", command=None, interval="0.01"):
        return ["--output", str(self.output), "--device-0", str(self.devices[0]),
                "--device-1", str(self.devices[1]), "--interval", interval, "--"] + (
                    command if command is not None else [sys.executable, "-c", code])

    def run_observer(self, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return observer.main(self.argv(**kwargs))

    def rows(self):
        raw = self.output.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        lines = raw.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(line) + 1 <= observer.MAX_RECORD_BYTES for line in lines))

        def reject(value):
            raise ValueError(value)

        return [json.loads(line, parse_constant=reject) for line in lines]

    def assert_failed_before_child(self):
        with mock.patch.object(observer.subprocess, "Popen") as popen:
            self.assertNotEqual(self.run_observer(), 0)
            popen.assert_not_called()

    def test_real_tiny_child_inherited_streams_and_source(self):
        request_log = self.root / "request.jsonl"
        error_log = self.root / "request.stderr"
        command = [sys.executable, "-c", "import sys; print('request'); print('child error', file=sys.stderr)"]
        with request_log.open("wb") as stdout, error_log.open("wb") as stderr:
            result = subprocess.run([sys.executable, str(SCRIPT)] + self.argv(command=command),
                                    stdout=stdout, stderr=stderr, check=False)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(request_log.read_bytes(), b"request\n")
        self.assertEqual(error_log.read_bytes(), b"child error\n")
        source, footer = self.rows()
        self.assertEqual(source["kind"], "vram_source")
        self.assertEqual(source["command_argv"], command)
        self.assertEqual(source["interval_seconds"], 0.01)
        self.assertEqual(source["scope"], observer.SCOPE)
        self.assertEqual(source["mapping_attestation"], "caller_supplied_labels_no_HIP_index_attestation")
        self.assertEqual(source["devices"][0]["supplied_path"], str(self.devices[0]))
        self.assertEqual(source["devices"][1]["resolved_path"], str(self.devices[1].resolve()))
        self.assertEqual(footer["kind"], "vram_complete")
        self.assertTrue(footer["observation_complete"])
        self.assertTrue(footer["post_wait_sampled"])
        self.assertGreaterEqual(footer["sample_rounds"], 3)
        self.assertTrue(math.isfinite(footer["completed_child_elapsed_seconds"]))
        self.assertGreaterEqual(footer["completed_child_elapsed_seconds"], 0)
        for index, device in enumerate(footer["devices"]):
            self.assertEqual(device["total_bytes"], 1000)
            self.assertEqual(device["sample_count"], footer["sample_rounds"])
            self.assertEqual(device["observed_max_used_bytes"], 100 + index)
            self.assertEqual(device["observed_min_free_bytes"], 900 - index)

    def test_fresh_output_does_not_overwrite_or_launch(self):
        self.output.write_bytes(b"raw/model/config/journal must remain intact\n")
        original = self.output.read_bytes()
        self.assert_failed_before_child()
        self.assertEqual(self.output.read_bytes(), original)

    def test_existing_output_symlink_directory_fifo_rejected(self):
        target = self.root / "raw"
        target.write_bytes(b"original")
        self.output.symlink_to(target)
        self.assert_failed_before_child()
        self.assertEqual(target.read_bytes(), b"original")
        self.output.unlink()
        self.output.mkdir()
        self.assert_failed_before_child()
        self.output.rmdir()
        os.mkfifo(self.output)
        self.assert_failed_before_child()

    def test_resolved_alias_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.devices[0], target_is_directory=True)
        self.devices[1] = alias
        self.assert_failed_before_child()
        self.assertFalse(self.output.exists())

    def test_malformed_unsigned_bounded_and_ascii(self):
        path = self.devices[1] / "mem_info_vram_used"
        for value in (b"", b"-1\n", b"+1\n", b"1.0\n", b"NaN\n", b"inf\n", b" 1\n",
                      b"1 \n", b"1\n\n", b"1\x00\n", "１２\n".encode(), b"1" * 100,
                      str(1 << 64).encode() + b"\n"):
            with self.subTest(value=value):
                path.write_bytes(value)
                self.assert_failed_before_child()
                self.assertFalse(self.output.exists())

    def test_bounds_total_and_attribute_types(self):
        used = self.devices[0] / "mem_info_vram_used"
        total = self.devices[0] / "mem_info_vram_total"
        for total_value, used_value in ((0, 0), (1000, 1001)):
            total.write_text(str(total_value))
            used.write_text(str(used_value))
            self.assert_failed_before_child()
        total.write_bytes(b"1000\n")
        used.unlink()
        self.assert_failed_before_child()
        used.mkdir()
        self.assert_failed_before_child()
        used.rmdir()
        os.mkfifo(used)
        self.assert_failed_before_child()
        used.unlink()
        used.symlink_to(total)
        self.assert_failed_before_child()
        self.assertFalse(self.output.exists())

    def test_unsigned_boundary_values_and_zero_free(self):
        for index, path in enumerate(self.devices):
            (path / "mem_info_vram_total").write_text(str(observer.MAX_VALUE))
            (path / "mem_info_vram_used").write_text(str(observer.MAX_VALUE if index else 0))
        self.assertEqual(self.run_observer(), 0)
        _, footer = self.rows()
        self.assertEqual(footer["devices"][0]["observed_max_used_bytes"], 0)
        self.assertEqual(footer["devices"][1]["observed_min_free_bytes"], 0)

    def test_directory_mappings_and_exact_two_required(self):
        self.devices[1] = self.root / "missing"
        self.assert_failed_before_child()
        self.devices[1].write_bytes(b"not a directory")
        self.assert_failed_before_child()
        cases = [[], ["--output", str(self.output)],
                 ["--output", str(self.output), "--device-0", str(self.devices[0]), "--", sys.executable]]
        for argv in cases:
            with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(observer.subprocess, "Popen") as popen:
                with self.assertRaises(SystemExit):
                    observer.main(argv)
                popen.assert_not_called()

    def test_interval_and_explicit_separator(self):
        for text in ("nan", "inf", "-inf", "0", "-1", "0.0001", "10.1", "nope"):
            with self.subTest(text=text), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    observer.main(self.argv(interval=text))
        for argv in (self.argv()[:-4], self.argv()[:-3]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    observer.main(argv)
        self.assertEqual(observer.interval_seconds("0.001"), 0.001)
        self.assertEqual(observer.interval_seconds("10"), 10)
        self.assertFalse(self.output.exists())

    def test_nonzero_child_propagated_no_success_footer(self):
        self.assertEqual(self.run_observer(code="raise SystemExit(7)"), 7)
        _, footer = self.rows()
        self.assertEqual(footer["kind"], "vram_failure")
        self.assertEqual(footer["child_returncode"], 7)
        self.assertEqual(footer["observer_returncode"], 7)
        self.assertIsNone(footer["observer_error"])
        self.assertTrue(footer["observation_complete"])

    def test_execution_error_preserved(self):
        self.assertEqual(self.run_observer(command=[str(self.root / "no-command")]), 1)
        _, footer = self.rows()
        self.assertEqual(footer["kind"], "vram_failure")
        self.assertFalse(footer["child_started"])
        self.assertIsNone(footer["child_returncode"])
        self.assertIsNone(footer["completed_child_elapsed_seconds"])
        self.assertEqual(footer["observer_error"]["stage"], "spawn")
        self.assertEqual(footer["observer_error"]["type"], "FileNotFoundError")

    def inject_child(self, mutate):
        real_popen = observer.subprocess.Popen

        def start(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            mutate()
            return child

        return mock.patch.object(observer.subprocess, "Popen", side_effect=start)

    def test_sample_failure_drains_real_child_and_keeps_nonzero(self):
        done = self.root / "drained"
        used = self.devices[0] / "mem_info_vram_used"
        code = f"import time,pathlib; time.sleep(0.06); pathlib.Path({str(done)!r}).write_text('done'); raise SystemExit(9)"
        with self.inject_child(used.unlink):
            self.assertEqual(self.run_observer(code=code), 9)
        self.assertEqual(done.read_text(), "done")
        _, footer = self.rows()
        self.assertEqual(footer["kind"], "vram_failure")
        self.assertEqual(footer["child_returncode"], 9)
        self.assertFalse(footer["observation_complete"])
        self.assertEqual(footer["observer_error"]["stage"], "sample")
        self.assertGreaterEqual(footer["completed_child_elapsed_seconds"], 0.06)

    def test_changed_total_drains_zero_child(self):
        done = self.root / "drained"
        total = self.devices[1] / "mem_info_vram_total"
        code = f"import time,pathlib; time.sleep(0.03); pathlib.Path({str(done)!r}).write_text('done')"
        with self.inject_child(lambda: total.write_bytes(b"2000\n")):
            self.assertEqual(self.run_observer(code=code), 1)
        self.assertTrue(done.exists())
        _, footer = self.rows()
        self.assertEqual(footer["child_returncode"], 0)
        self.assertEqual(footer["kind"], "vram_failure")
        self.assertIn("total changed", footer["observer_error"]["message"])
        self.assertEqual(footer["devices"][1]["total_bytes"], 1000)
        self.assertEqual(footer["devices"][1]["sample_count"], 1)

    def test_source_output_failure_never_starts_child(self):
        with mock.patch.object(observer, "write_record", side_effect=OSError("disk full")):
            self.assert_failed_before_child()
        self.assertNotIn(b"vram_complete", self.output.read_bytes())

    def test_final_output_failure_has_no_success_footer(self):
        done = self.root / "drained"
        write = observer.write_record

        def fail_complete(fd, record):
            if record["kind"] == "vram_complete":
                os.write(fd, observer.encode_record(record))
                raise OSError("final output failed after partial write")
            write(fd, record)

        code = f"import pathlib; pathlib.Path({str(done)!r}).write_text('done')"
        with mock.patch.object(observer, "write_record", side_effect=fail_complete):
            self.assertEqual(self.run_observer(code=code), 1)
        self.assertTrue(done.exists())
        _, footer = self.rows()
        self.assertEqual(footer["kind"], "vram_failure")
        self.assertEqual(footer["observer_error"]["stage"], "final_output")

    def test_close_failure_repairs_success_footer(self):
        close = observer.os.close
        failed = False

        def fail_output_close(fd):
            nonlocal failed
            path = Path(f"/proc/self/fd/{fd}").resolve()
            close(fd)
            if path == self.output and not failed:
                failed = True
                raise OSError("output close failed")

        with mock.patch.object(observer.os, "close", side_effect=fail_output_close):
            self.assertEqual(self.run_observer(), 1)
        self.assertTrue(failed)
        _, footer = self.rows()
        self.assertEqual(footer["kind"], "vram_failure")
        self.assertEqual(footer["observer_error"]["stage"], "close_output")

    def test_close_repair_does_not_touch_replacement_file(self):
        close = observer.os.close
        replaced = False

        def replace_output(fd):
            nonlocal replaced
            path = Path(f"/proc/self/fd/{fd}").resolve()
            close(fd)
            if path == self.output and not replaced:
                replaced = True
                self.output.rename(self.root / "original_observer_output")
                self.output.write_bytes(b"replacement must be untouched")
                raise OSError("output close failed")

        with mock.patch.object(observer.os, "close", side_effect=replace_output):
            self.assertEqual(self.run_observer(), 1)
        self.assertTrue(replaced)
        self.assertEqual(self.output.read_bytes(), b"replacement must be untouched")

    def test_persistent_final_output_failure_stderr_and_source_only(self):
        write = observer.write_record

        def fail_footer(fd, record):
            if record["kind"] != "vram_source":
                raise OSError("disk remains full")
            write(fd, record)

        with mock.patch.object(observer, "write_record", side_effect=fail_footer), contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(observer.main(self.argv()), 1)
        self.assertIn("final_output", stderr.getvalue())
        self.assertIn("failure_output", stderr.getvalue())
        self.assertEqual(len(self.output.read_bytes().splitlines()), 1)
        self.assertNotIn(b"vram_complete", self.output.read_bytes())

    def test_write_record_rolls_back_failed_fsync_and_rejects_nonfinite(self):
        with self.output.open("xb", buffering=0) as stream:
            observer.write_record(stream.fileno(), {"kind": "source"})
            original = self.output.read_bytes()
            with mock.patch.object(observer.os, "fsync", side_effect=OSError("flush failed")):
                with self.assertRaises(OSError):
                    observer.write_record(stream.fileno(), {"kind": "vram_complete"})
            self.assertEqual(self.output.read_bytes(), original)
            for value in (math.nan, math.inf, -math.inf):
                with self.assertRaises(ValueError):
                    observer.write_record(stream.fileno(), {"value": value})

    def test_bounded_source_no_child_and_no_output(self):
        command = [sys.executable, "x" * observer.MAX_RECORD_BYTES]
        with mock.patch.object(observer.subprocess, "Popen") as popen:
            self.assertEqual(self.run_observer(command=command), 1)
            popen.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_monotonic_elapsed_and_aggregate_pre_during_post(self):
        sequences = iter(((100, 101), (400, 300), (800, 500), (250, 900)))
        read = observer.read_unsigned
        counts = [0, 0]
        values = []

        def synthetic_read(path):
            if path.name == "mem_info_vram_total":
                return read(path)
            index = self.devices.index(path.parent)
            if index == 0:
                values[:] = next(sequences)
            counts[index] += 1
            return values[index]

        child = mock.Mock(returncode=0)
        child.wait.side_effect = [subprocess.TimeoutExpired("command", 0.01), 0, 0]
        with mock.patch.object(observer, "read_unsigned", side_effect=synthetic_read), \
                mock.patch.object(observer.subprocess, "Popen", return_value=child) as popen, \
                mock.patch.object(observer.time, "monotonic", side_effect=[50.0, 52.25]), \
                mock.patch.object(observer.time, "time", side_effect=AssertionError("wall clock used")):
            self.assertEqual(self.run_observer(), 0)
        popen.assert_called_once_with([sys.executable, "-c", "pass"], shell=False)
        self.assertEqual(child.wait.call_args_list, [mock.call(timeout=0.01),
                                                    mock.call(timeout=0.01), mock.call()])
        _, footer = self.rows()
        self.assertEqual(footer["completed_child_elapsed_seconds"], 2.25)
        self.assertEqual(footer["sample_rounds"], 4)
        self.assertEqual(counts, [4, 4])
        self.assertEqual([d["observed_max_used_bytes"] for d in footer["devices"]], [800, 900])
        self.assertEqual([d["observed_min_free_bytes"] for d in footer["devices"]], [200, 100])

    def test_post_wait_failure_and_interrupted_wait_never_succeed(self):
        child = mock.Mock(returncode=0)
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        with mock.patch.object(observer.subprocess, "Popen", return_value=child):
            self.assertEqual(self.run_observer(), 1)
        child.kill.assert_not_called()
        child.terminate.assert_not_called()
        self.assertEqual(child.wait.call_count, 2)
        _, footer = self.rows()
        self.assertEqual(footer["kind"], "vram_failure")
        self.assertTrue(footer["post_wait_sampled"])
        self.assertEqual(footer["observer_error"]["type"], "KeyboardInterrupt")
        self.output.unlink()
        read = observer.read_unsigned
        used_reads = 0

        def fail_post(path):
            nonlocal used_reads
            if path == self.devices[0] / "mem_info_vram_used":
                used_reads += 1
                if used_reads == 3:
                    raise OSError("post-wait sample failed")
            return read(path)

        with mock.patch.object(observer, "read_unsigned", side_effect=fail_post), \
                mock.patch.object(observer.subprocess, "Popen", return_value=mock.Mock(returncode=0)):
            self.assertEqual(self.run_observer(), 1)
        _, footer = self.rows()
        self.assertEqual(footer["kind"], "vram_failure")
        self.assertFalse(footer["post_wait_sampled"])
        self.assertEqual(footer["observer_error"]["stage"], "post_wait_sample")

    def test_signal_to_observer_drains_own_attached_child(self):
        ready, done = self.root / "ready", self.root / "done"
        code = (f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); "
                f"time.sleep(0.15); pathlib.Path({str(done)!r}).touch()")
        with (self.root / "stderr").open("wb") as stderr:
            wrapper = subprocess.Popen([sys.executable, str(SCRIPT)] + self.argv(code=code),
                                       stdout=subprocess.DEVNULL, stderr=stderr)
            try:
                while not ready.exists() and wrapper.poll() is None:
                    time.sleep(0.001)
                wrapper.send_signal(signal.SIGTERM)
            finally:
                status = wrapper.wait()
        self.assertTrue(done.exists())
        self.assertEqual(status, 1)
        _, footer = self.rows()
        self.assertEqual(footer["kind"], "vram_failure")
        self.assertEqual(footer["child_returncode"], 0)
        self.assertEqual(footer["observer_error"]["stage"], "signal")


if __name__ == "__main__":
    unittest.main()
