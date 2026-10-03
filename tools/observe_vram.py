#!/usr/bin/env python3
"""Sample two explicit AMD sysfs mappings around one attached argv command.

Child stdout/stderr are inherited; redirect the request log independently.
This observes global driver VRAM, not an exact instantaneous peak, HIP-owned
allocations, phases, performance, reference correctness or MTP. The caller must
qualify the mapping to HIP indices and keep the command attached to its workload
(including Docker). On observer failure/signals we wait for that command without
terminating it; detached workloads cannot be drained by this helper.

At most two records, each <= 1 MiB, go to an exclusively created regular output:
vram_source then vram_complete or vram_failure. Preflight errors do not launch a
child. Unwritable output is an error reported on stderr, never a success footer.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time


SCOPE = "sampled_global_driver_VRAM_not_exact_instantaneous_peak"
MAX_RECORD_BYTES = 1024 * 1024
MAX_VALUE = (1 << 64) - 1
VALUE_BYTES = 21  # At most 20 ASCII decimal digits and one optional newline.


def interval_seconds(text):
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("interval must be finite, in [0.001, 10] seconds") from error
    if not math.isfinite(value) or not 0.001 <= value <= 10:
        raise argparse.ArgumentTypeError("interval must be finite, in [0.001, 10] seconds")
    return value


def read_unsigned(path):
    # Nonblocking open plus fstat rejects a substituted FIFO/device before read.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{path}: expected a regular sysfs attribute")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(VALUE_BYTES + 1)
        if not re.fullmatch(rb"[0-9]{1,20}\n?", raw):
            raise ValueError(f"{path}: expected bounded ASCII unsigned decimal")
        value = int(raw)
        if value > MAX_VALUE:
            raise ValueError(f"{path}: exceeds uint64")
        return value
    finally:
        os.close(fd)


class Device:
    def __init__(self, label, supplied):
        self.label, self.supplied = label, str(supplied)
        self.path = Path(supplied).resolve(strict=True)
        if not self.path.is_dir():
            raise ValueError(f"{self.path}: expected a device directory")
        self.total = None
        self.count = 0
        self.max_used = self.min_free = None

    def sample(self):
        total = read_unsigned(self.path / "mem_info_vram_total")
        used = read_unsigned(self.path / "mem_info_vram_used")
        if not 0 < total or used > total:
            raise ValueError(f"{self.path}: require total > 0 and used <= total")
        if self.total is not None and total != self.total:
            raise ValueError(f"{self.path}: VRAM total changed")
        self.total = total
        self.max_used = used if self.count == 0 else max(self.max_used, used)
        self.min_free = total - used if self.count == 0 else min(self.min_free, total - used)
        self.count += 1

    def source(self):
        return {"label": self.label, "supplied_path": self.supplied,
                "resolved_path": str(self.path),
                "used_file": str(self.path / "mem_info_vram_used"),
                "total_file": str(self.path / "mem_info_vram_total"),
                "total_bytes": self.total}

    def aggregate(self):
        return {"label": self.label, "total_bytes": self.total,
                "sample_count": self.count, "observed_max_used_bytes": self.max_used,
                "observed_min_free_bytes": self.min_free}


def encode_record(record):
    raw = (json.dumps(record, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")) + "\n").encode("ascii")
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("JSONL record exceeds 1 MiB")
    return raw


def write_record(fd, record):
    raw = encode_record(record)
    start = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        view = memoryview(raw)
        while view:
            size = os.write(fd, view)
            if size <= 0:
                raise OSError("output write made no progress")
            view = view[size:]
        os.fsync(fd)
    except BaseException:
        # A partial or failed final write must not leave a success footer.
        try:
            os.ftruncate(fd, start)
            os.lseek(fd, start, os.SEEK_SET)
        except OSError as error:
            diagnostic(error_record("output_rollback", error))
        raise


def diagnostic(error):
    try:
        print("observe_vram: " + json.dumps(error, ensure_ascii=True, allow_nan=False),
              file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def error_record(stage, error):
    return {"stage": stage, "type": type(error).__name__, "message": str(error)[:1024]}


def observe(args):
    devices = [Device("device_0", args.device_0), Device("device_1", args.device_1)]
    if devices[0].path == devices[1].path or devices[0].path.samefile(devices[1].path):
        raise ValueError("device mappings must resolve to distinct directories")
    # Validate both mappings and all attributes before creating output/spawning.
    for device in devices:
        device.sample()
    source = {"kind": "vram_source", "protocol": 1, "scope": SCOPE,
              "mapping_attestation": "caller_supplied_labels_no_HIP_index_attestation",
              "value_source": "AMD_sysfs_mem_info_vram_used_and_mem_info_vram_total",
              "sampling_window": "pre_spawn_through_child_lifetime_and_post_wait",
              "elapsed_scope": "monotonic_before_Popen_through_completed_child_wait_includes_spawn_and_observer_overhead",
              "command_argv": args.command, "interval_seconds": args.interval,
              "devices": [device.source() for device in devices]}
    encode_record(source)  # Bound argv/path serialization before starting anything.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    child = None
    failure = None
    elapsed = None
    rounds = 1
    post_wait_sampled = False
    handlers = {}
    stage = "source_output"

    def fail(where, error):
        nonlocal failure
        if failure is None:
            failure = error_record(where, error)
            diagnostic(failure)  # Observable immediately, even during a long drain.

    def interrupted(signum, frame):
        # Do not raise inside Popen: it may already have spawned but not returned.
        fail("signal", RuntimeError(f"received signal {signum}; waiting for own child"))

    try:
        output_info = os.fstat(fd)
        if not stat.S_ISREG(output_info.st_mode):
            raise ValueError("output must be a freshly created regular file")
        write_record(fd, source)
        source_end = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            stage = "signal_setup"
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                handlers[sig] = signal.signal(sig, interrupted)
            if failure is None:
                stage = "spawn"
                begin = time.monotonic()
                child = subprocess.Popen(args.command, shell=False)
                while failure is None:
                    stage = "sample"
                    for device in devices:
                        device.sample()
                    rounds += 1
                    stage = "wait"
                    try:
                        child.wait(timeout=args.interval)
                        break
                    except subprocess.TimeoutExpired:
                        pass
        except BaseException as error:
            fail(stage, error)
        finally:
            if child is not None:
                # No kill/terminate/timeout: an attached Docker client must drain.
                while True:
                    try:
                        child.wait()
                        elapsed = time.monotonic() - begin
                        if not math.isfinite(elapsed) or elapsed < 0:
                            raise ValueError("invalid completed monotonic elapsed")
                        break
                    except BaseException as error:
                        fail("drain", error)
                        if child.returncode is not None:
                            elapsed = None
                            break
                try:
                    for device in devices:
                        device.sample()
                    rounds += 1
                    post_wait_sampled = True
                except BaseException as error:
                    fail("post_wait_sample", error)
            for sig, previous in handlers.items():
                signal.signal(sig, previous)
        child_code = child.returncode if child is not None else None
        status = (128 - child_code if child_code < 0 else child_code) if child_code else 0
        if failure is not None or child is None:
            status = status or 1

        def footer():
            return {"kind": "vram_failure" if status else "vram_complete",
                    "protocol": 1, "scope": SCOPE,
                    "observation_complete": failure is None and post_wait_sampled,
                    "child_started": child is not None, "child_returncode": child_code,
                    "observer_returncode": status,
                    "completed_child_elapsed_seconds": elapsed,
                    "sample_rounds": rounds, "post_wait_sampled": post_wait_sampled,
                    "devices": [device.aggregate() for device in devices],
                    "observer_error": failure}

        try:
            write_record(fd, footer())
        except BaseException as error:
            fail("final_output", error)
            status = status or 1
            try:
                os.ftruncate(fd, source_end)
                os.lseek(fd, source_end, os.SEEK_SET)
                write_record(fd, footer())
            except BaseException as output_error:
                diagnostic(error_record("failure_output", output_error))
        closing_fd, fd = fd, None
        try:
            os.close(closing_fd)
        except BaseException as error:
            fail("close_output", error)
            status = status or 1
            try:
                # close(2) may release the fd even on error. Repair only our own
                # exclusively created inode, never a replacement raw/config file.
                fd = os.open(args.output, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                current = os.fstat(fd)
                if (current.st_dev, current.st_ino) != (output_info.st_dev, output_info.st_ino):
                    raise ValueError("output was replaced; refusing footer repair")
                os.ftruncate(fd, source_end)
                os.lseek(fd, source_end, os.SEEK_SET)
                write_record(fd, footer())
            except BaseException as output_error:
                diagnostic(error_record("failure_output", output_error))
        return status
    finally:
        if fd is not None:
            os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh observer JSONL, never appended")
    parser.add_argument("--device-0", type=Path, required=True, help="caller-mapped AMD sysfs device directory")
    parser.add_argument("--device-1", type=Path, required=True, help="caller-mapped AMD sysfs device directory")
    parser.add_argument("--interval", type=interval_seconds, default=0.1, metavar="SECONDS")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        parser.parse_args(argv)  # Includes normal --help handling.
        parser.error("explicit -- COMMAND [ARG ...] is required")
    split = argv.index("--")
    args = parser.parse_args(argv[:split])
    args.command = argv[split + 1:]
    if not args.command:
        parser.error("missing command after --")
    try:
        return observe(args)
    except BaseException as error:
        diagnostic(error_record("preflight_or_output", error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
