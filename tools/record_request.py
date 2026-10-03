#!/usr/bin/env python3
"""Validate protocol-1 core-session full requests and append ONE r4_request.

Only emitted structured records are evidence. Legacy session JSON is rejected.
PP, first-output and TG are completed wall intervals, with sampling/CLI IO in
their declared scopes; load is separate. Rates use actual tokens, not requested
tokens or GPU-event sums. This is candidate non-MTP request accounting, not an
MTP, long-context, independent-reference, peak-VRAM or paired-speedup proof.
--results must explicitly name the caller's canonical ROOT/results.jsonl.

Optional --vram-log attaches a complete two-device observe_vram protocol only.
The command join accepts a direct attached Docker run, the existing image,
known AMD device/IPC/security options, /core and read-only /models bind mounts,
and /core/build/core-session as entrypoint. It compares structured native argv
to the request source; it does not attest binary/mount identity or HIP indices.
Observer elapsed includes Docker startup, load and cleanup, not GPU events or
per-phase peaks. VRAM remains sampled global driver usage, not HIP ownership.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

if __package__:
    from .record_memory import (append_result as append_record, exact, expect,
                                finite_float, integer, invalid_constant, number,
                                require, unique_object)
else:
    from record_memory import (append_result as append_record, exact, expect,
                               finite_float, integer, invalid_constant, number,
                               require, unique_object)


MAX_RAW_BYTES = 16 * 1024 * 1024
MAX_CAPACITY, VOCAB, EOS = 131072, 248320, 248046
Q40, Q41 = 2764800, 2867200
COUNTERS = ("expert_hits", "expert_misses", "expert_upload_bytes")
SCOPE = "candidate_non_mtp_full_request_no_speed_claim"
TIMING_SCOPE = "completed_wall_sampling_and_cli_io_excludes_load_cleanup"
PP_SCOPE = "completed_prompt_calls_and_row_io"
TG_SCOPE = "remaining_output_forwards_sampling_and_io"
MAX_VRAM_BYTES = 2 * 1024 * 1024
MAX_VRAM_RECORD_BYTES = 1024 * 1024
UINT64_MAX = (1 << 64) - 1
VRAM_SCOPE = "sampled_global_driver_VRAM_not_exact_instantaneous_peak"
VRAM_MAPPING = "caller_supplied_labels_no_HIP_index_attestation"
VRAM_COMMAND_JOIN = "direct_Docker_core_session_structured_argv_matches_request_source_only_no_binary_or_mount_identity_attestation"
DOCKER_IMAGE = "llama.cpp-gfx906:cmake-4.4.3"


def cli_text(value, label, empty=False):
    # JSON escaping permits quoted/control characters in real argv paths. NUL
    # and lone surrogates cannot represent the driver's UTF-8 argv strings.
    require(type(value) is str and (empty or bool(value)) and "\0" not in value and
            not any(0xd800 <= ord(c) <= 0xdfff for c in value), label + ": invalid argv string")
    return value


def records(path):
    """Bound both the pre-read extent and the actual read of a growing file."""
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), str(path) + ": not a regular file")
    require(info.st_size <= MAX_RAW_BYTES, str(path) + ": oversized input")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        require(stat.S_ISREG(opened.st_mode), str(path) + ": not a regular file")
        require(opened.st_size <= MAX_RAW_BYTES, str(path) + ": oversized input")
        raw = stream.read(MAX_RAW_BYTES + 1)
    require(len(raw) <= MAX_RAW_BYTES, str(path) + ": oversized input")
    require(raw.endswith(b"\n"), str(path) + ": missing completed final line")
    lines = raw.split(b"\n")[:-1]
    require(len(lines) >= 2, str(path) + ": missing source/complete records")
    rows = []
    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line.decode("utf-8"), object_pairs_hook=unique_object,
                             parse_float=finite_float, parse_constant=invalid_constant)
            require(type(obj) is dict, "expected an object")
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{path}:row {i}: {error}") from error
        rows.append(obj)
    return rows


def ids(value, label, low=0, high=MAX_CAPACITY):
    require(type(value) is list and low <= len(value) <= high, label + ": invalid ID count")
    for i, token in enumerate(value):
        integer(token, f"{label}[{i}]", 0, VOCAB - 1)
    return value


def source(obj):
    expect(obj, {
        "kind": "session_request_source", "protocol": 1,
        "runtime": "own_48_layer_HIP", "kv": "Q4_0", "candidate": True,
        "mtp": False, "mtp_acceptance": None, "session_start": "fresh",
        "prefix_reuse": False, "expert_cache_warmness": "unknown",
        "timing_scope": TIMING_SCOPE, "pp_scope": PP_SCOPE, "tg_scope": TG_SCOPE,
    }, ("revision", "dirty", "model", "series", "config", "sampling", "prompt_ids"), "source")
    revision = obj["revision"]
    require(type(revision) is str and len(revision) == 40 and
            all(c in "0123456789abcdefABCDEF" for c in revision), "source: invalid compiled revision")
    require(type(obj["dirty"]) is bool, "source: invalid boolean dirty")
    cli_text(obj["model"], "source.model")
    config = obj["config"]
    expect(config, {}, ("capacity", "expert_slots", "max_batch_tokens", "requested_output_tokens",
                       "ignore_eos", "trace", "trace_directory", "logits_path", "diagnostic_rows"), "config")
    capacity = integer(config["capacity"], "config.capacity", 4, MAX_CAPACITY)
    require(capacity % 4 == 0, "config: capacity must be a multiple of four")
    integer(config["expert_slots"], "config.expert_slots", 1, 512)
    chunk = integer(config["max_batch_tokens"], "config.max_batch_tokens", 1, 1024)
    requested = integer(config["requested_output_tokens"], "config.requested_output_tokens", 0, MAX_CAPACITY)
    for key in ("ignore_eos", "trace", "diagnostic_rows"):
        require(type(config[key]) is bool, "config." + key + ": expected boolean")
    trace = cli_text(config["trace_directory"], "config.trace_directory", empty=True)
    logits = cli_text(config["logits_path"], "config.logits_path", empty=True)
    exact(config["trace"], bool(trace), "config.trace")
    require(not trace or chunk == 1, "config: trace requires chunk one")
    prompt = ids(obj["prompt_ids"], "source.prompt_ids", 1, capacity)
    require(len(prompt) + max(requested - 1, 0) <= capacity, "config: requested forwards exceed capacity")
    sampling = obj["sampling"]
    require(type(sampling) is dict, "sampling: expected an object")
    mode = sampling.get("mode")
    if mode == "stochastic":
        expect(sampling, {"mode": "stochastic", "rng": "mt19937_64_high53_ascending_id_cdf",
                          "baseline_rng_equivalent": False},
               ("seed", "temperature", "top_p", "top_k"), "sampling")
        integer(sampling["seed"], "sampling.seed")
        temperature = number(sampling["temperature"], "sampling.temperature")
        top_p = number(sampling["top_p"], "sampling.top_p")
        require(temperature > 0 and 0 < top_p <= 1, "sampling: invalid filters")
        top_k = integer(sampling["top_k"], "sampling.top_k", 0, VOCAB)
        primary = temperature == 1.0 and top_p == .95 and top_k == 20
        exact(obj["series"], "primary_sampling" if primary else "custom_sampling", "source.series")
        require(requested > 0, "sampling: stochastic request needs positive output limit")
    else:
        expect(sampling, {"mode": "greedy_diagnostic"}, label="sampling")
        exact(obj["series"], "greedy_diagnostic", "source.series")
    exact(config["diagnostic_rows"], mode != "stochastic" or bool(trace) or bool(logits),
          "config.diagnostic_rows")


def counter_delta(obj, previous, consumed, width, label):
    for key in COUNTERS:
        integer(obj[key], label + "." + key)
    delta = {key: obj[key] - previous[key] for key in COUNTERS}
    require(all(v >= 0 for v in delta.values()), label + ": expert counters decreased")
    require(obj["expert_hits"] + obj["expert_misses"] == 480 * consumed and
            delta["expert_hits"] + delta["expert_misses"] == 480 * width,
            label + ": incomplete expert assignment accounting")
    misses, uploaded = delta["expert_misses"], delta["expert_upload_bytes"]
    require(misses * Q40 <= uploaded <= misses * Q41,
            label + ": expert upload bytes outside per-miss payload bounds")


def chronological_le(lower, upper, label):
    # setprecision(17) round-trips the driver's binary64 values. Only rounding
    # of sums of disjoint intervals needs a few ulps, not a time/speed tolerance.
    number(lower, label + ".lower")
    number(upper, label + ".upper")
    require(lower <= upper or lower - upper <= 4 * math.ulp(max(lower, upper)),
            label + ": impossible timing chronology")


def completed(obj, origin):
    config = origin["config"]
    requested, prompt = config["requested_output_tokens"], origin["prompt_ids"]
    expect(obj, {
        "kind": "session_request_complete", "protocol": 1, "candidate": True,
        "mtp": False, "mtp_acceptance": None, "scope": SCOPE, "passed": True,
    }, ("input_tokens", "output_tokens", "consumed_tokens", "pp_calls", "tg_forwards", "random_draws",
        "load_ms", "pp_ms", "tg_ms", "total_ms", "first_output_ms", "first_output_after_pp_ms",
        "stop_reason", "pending_token", *COUNTERS, "generated_ids"), "complete")
    output_count = integer(obj["output_tokens"], "complete.output_tokens", 0, requested)
    outputs = ids(obj["generated_ids"], "complete.generated_ids", output_count, output_count)
    forwards = max(output_count - 1, 0)
    for key, want in {
        "input_tokens": len(prompt), "consumed_tokens": len(prompt) + forwards,
        "pp_calls": (len(prompt) + config["max_batch_tokens"] - 1) // config["max_batch_tokens"],
        "tg_forwards": forwards,
        "random_draws": output_count if origin["sampling"]["mode"] == "stochastic" else 0,
        "pending_token": outputs[-1] if outputs else None,
    }.items():
        exact(obj[key], want, "complete." + key)
    if not outputs:
        require(requested == 0, "complete: missing requested output")
        exact(obj["stop_reason"], "prompt_only", "complete.stop_reason")
        exact(obj["first_output_ms"], None, "complete.first_output_ms")
    else:
        if not config["ignore_eos"]:
            require(EOS not in outputs[:-1], "complete: output after stopping EOS")
        stopped_eos = outputs[-1] == EOS and not config["ignore_eos"]
        require(stopped_eos or output_count == requested, "complete: short output without EOS")
        exact(obj["stop_reason"], "eos" if stopped_eos else "output_limit", "complete.stop_reason")
        number(obj["first_output_ms"], "complete.first_output_ms")
    for key in ("load_ms", "pp_ms", "tg_ms", "total_ms", "first_output_after_pp_ms"):
        number(obj[key], "complete." + key)
    require(obj["pp_ms"] > 0, "complete: PP elapsed must be positive")
    chronological_le(obj["pp_ms"], obj["total_ms"], "complete.PP/total")
    if outputs:
        chronological_le(math.fsum((obj["pp_ms"], obj["first_output_after_pp_ms"])),
                         obj["first_output_ms"], "complete.PP/first_output")
        chronological_le(math.fsum((obj["first_output_ms"], obj["tg_ms"])),
                         obj["total_ms"], "complete.first_output/TG/total")
    else:
        require(obj["first_output_after_pp_ms"] == 0, "complete: prompt-only first-output time is not zero")
    require(obj["tg_ms"] > 0 if forwards else obj["tg_ms"] == 0,
            "complete: TG elapsed inconsistent with remaining forwards")
    counter_delta(obj, dict.fromkeys(COUNTERS, 0), obj["consumed_tokens"], obj["consumed_tokens"], "complete")


def timeline(rows, origin, footer):
    """Match the driver's rows-then-window PP, output, (TG, output)* order."""
    diagnostic = origin["config"]["diagnostic_rows"]
    prompt, outputs = origin["prompt_ids"], footer["generated_ids"]
    cursor, position, last_argmax = 1, 0, None
    previous = dict.fromkeys(COUNTERS, 0)
    times = {"pp": [], "tg": []}

    def take():
        nonlocal cursor
        require(cursor < len(rows) - 1, "timeline: missing request record")
        row = rows[cursor]
        cursor += 1
        return row

    def consume(tokens, phase):
        nonlocal position, previous, last_argmax
        if diagnostic:
            for i, token in enumerate(tokens):
                row = take()
                expect(row, {"kind": "session_request_row", "protocol": 1, "phase": phase,
                             "position": position + i, "token": token, "finite": True},
                       ("argmax",), "row")
                last_argmax = integer(row["argmax"], "row.argmax", 0, VOCAB - 1)
            window = take()
            expect(window, {"kind": "session_request_window", "protocol": 1, "phase": phase,
                            "first_position": position, "rows": len(tokens)},
                   ("completed_ms", *COUNTERS), "window")
            times[phase].append(number(window["completed_ms"], "window.completed_ms"))
            counter_delta(window, previous, position + len(tokens), len(tokens), "window")
            previous = {key: window[key] for key in COUNTERS}
        position += len(tokens)

    chunk = origin["config"]["max_batch_tokens"]
    for first in range(0, len(prompt), chunk):
        consume(prompt[first:first + chunk], "pp")
    for i, token in enumerate(outputs):
        if i:
            consume([outputs[i - 1]], "tg")
        expect(take(), {"kind": "session_request_output", "protocol": 1, "index": i, "token": token},
               label="output")
        if origin["sampling"]["mode"] == "greedy_diagnostic":
            exact(token, last_argmax, "output.greedy_argmax")
    require(cursor == len(rows) - 1, "timeline: unexpected extra request records")
    exact(position, footer["consumed_tokens"], "timeline.consumed_tokens")
    if diagnostic:
        for key in COUNTERS:
            exact(footer[key], previous[key], "complete.final_window." + key)
        for phase in ("pp", "tg"):
            chronological_le(math.fsum(times[phase]), footer[phase + "_ms"], phase + ".completed_calls/wall")
    return {
        "row_count": position if diagnostic else 0,
        "window_count": footer["pp_calls"] + footer["tg_forwards"] if diagnostic else 0,
        "pp_completed_call_ms_sum": math.fsum(times["pp"]) if diagnostic else None,
        "tg_completed_call_ms_sum": math.fsum(times["tg"]) if diagnostic else None,
    }


def rate(count, elapsed, label):
    if not count:
        return None
    value = count * 1000 / elapsed
    return number(value, label)


def distinct_paths(paths):
    """Reject lexical, symlink and hardlink aliases before reading inputs."""
    paths = [Path(path).resolve() for path in paths]
    for i, path in enumerate(paths):
        for previous in paths[:i]:
            require(path != previous and not (path.exists() and previous.exists() and
                    os.path.samefile(path, previous)), "request/VRAM/results paths must be distinct")


def vram_records(path):
    """Exactly two finished bounded rows; nonblocking open rejects FIFO races."""
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), str(path) + ": not a regular file")
    require(info.st_size <= MAX_VRAM_BYTES, str(path) + ": oversized VRAM input")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode), str(path) + ": not a regular file")
        require(opened.st_size <= MAX_VRAM_BYTES, str(path) + ": oversized VRAM input")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(MAX_VRAM_BYTES + 1)
    finally:
        os.close(fd)
    require(len(raw) <= MAX_VRAM_BYTES, str(path) + ": oversized VRAM input")
    require(raw.endswith(b"\n"), str(path) + ": missing completed final line")
    lines = raw.split(b"\n")[:-1]
    require(len(lines) == 2, str(path) + ": expected exactly two VRAM JSONL rows")
    rows = []
    for i, line in enumerate(lines, 1):
        require(len(line) + 1 <= MAX_VRAM_RECORD_BYTES, str(path) + ": oversized VRAM record")
        try:
            obj = json.loads(line.decode("utf-8"), object_pairs_hook=unique_object,
                             parse_float=finite_float, parse_constant=invalid_constant)
            require(type(obj) is dict, "expected an object")
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{path}:row {i}: {error}") from error
        rows.append(obj)
    return rows


def absolute_posix(value, label):
    """Validate recorded remote paths lexically, never resolve on this host."""
    path = PurePosixPath(cli_text(value, label))
    require(path.is_absolute() and not value.startswith("//") and str(path) == value and
            ".." not in path.parts, label + ": expected a normalized absolute POSIX path")
    return path


def docker_native_argv(argv):
    """Closed attached Docker shape: no shell, detachment or argv overrides."""
    require(type(argv) is list and len(argv) >= 3, "VRAM command: expected structured argv")
    for arg in argv:
        cli_text(arg, "VRAM command argument")
    require(argv[0] in ("docker", "/usr/bin/docker") and argv[1] == "run",
            "VRAM command: require direct Docker run")
    fixed = {"--group-add": "video", "--ipc": "host",
             "--security-opt": "seccomp=unconfined", "--entrypoint": "/core/build/core-session"}
    seen, devices, mounts = set(), set(), set()
    cursor = 2
    while cursor < len(argv) and argv[cursor].startswith("-"):
        flag = argv[cursor]
        cursor += 1
        require(flag in (*fixed, "--rm", "--name", "--device", "-v"),
                "VRAM command: unsupported Docker option " + flag)
        if flag not in ("--device", "-v"):
            require(flag not in seen, "VRAM command: duplicate Docker option " + flag)
            seen.add(flag)
        if flag == "--rm":
            continue
        require(cursor < len(argv), "VRAM command: missing Docker option value")
        value = argv[cursor]
        cursor += 1
        if flag in fixed:
            exact(value, fixed[flag], "VRAM Docker " + flag)
        elif flag == "--name":
            require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value) is not None,
                    "VRAM command: invalid Docker name")
        elif flag == "--device":
            require(value in ("/dev/kfd", "/dev/dri") and value not in devices,
                    "VRAM command: unexpected/duplicate Docker device")
            devices.add(value)
        else:
            parts = value.split(":")
            require(len(parts) in (2, 3), "VRAM command: expected bind mount")
            absolute_posix(parts[0], "VRAM Docker mount source")
            target = parts[1]
            require(target not in mounts and
                    ((target == "/core" and (len(parts) == 2 or parts[2] == "rw")) or
                     (target == "/models" and len(parts) == 3 and parts[2] == "ro")),
                    "VRAM command: unexpected/duplicate Docker mount")
            mounts.add(target)
    require(set(fixed) | {"--rm"} <= seen and devices == {"/dev/kfd", "/dev/dri"} and
            mounts == {"/core", "/models"}, "VRAM command: incomplete attached Docker prefix")
    require(cursor < len(argv) and argv[cursor] == DOCKER_IMAGE, "VRAM command: unsupported Docker image")
    return argv[cursor + 1:]


def native_integer(text, label, unsigned=False):
    # from_chars accepts neither whitespace nor '+'. Strip leading zeroes before
    # int conversion so an oversized decimal cannot bypass the explicit bounds.
    require(re.fullmatch(r"[0-9]+" if unsigned else r"-?[0-9]+", text) is not None,
            label + ": invalid native integer")
    negative = text.startswith("-")
    digits = text[1:] if negative else text
    digits = digits.lstrip("0") or "0"
    require(len(digits) <= (20 if unsigned else 10), label + ": native integer overflow")
    value = int(digits) * (-1 if negative else 1)
    return integer(value, label, 0 if unsigned else -(1 << 31), UINT64_MAX if unsigned else (1 << 31) - 1)


def native_float(text, label):
    require(re.fullmatch(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text) is not None,
            label + ": invalid native float")
    return number(float(text), label)


def join_native_command(argv, origin):
    """Mirror session_cli.hpp defaults/grammar; reject every duplicate flag."""
    args = docker_native_argv(argv)
    values = {"--capacity": 4096, "--slots": 112, "--generate": 0, "--prefill-chunk": 1,
              "--trace": "", "--logits": "", "--sample": False, "--ignore-eos": False,
              "--seed": None, "--temperature": 1., "--top-p": .95, "--top-k": 20}
    seen, model, prompt = set(), None, []
    cursor = 0
    while cursor < len(args):
        arg = args[cursor]
        cursor += 1
        if arg in values:
            require(arg not in seen, "VRAM native command: duplicate flag " + arg)
            seen.add(arg)
            if arg in ("--sample", "--ignore-eos"):
                values[arg] = True
                continue
            require(cursor < len(args), "VRAM native command: missing option value " + arg)
            value = args[cursor]
            cursor += 1
            if arg in ("--trace", "--logits"):
                values[arg] = value
            elif arg in ("--temperature", "--top-p"):
                values[arg] = native_float(value, "VRAM native " + arg)
            else:
                values[arg] = native_integer(value, "VRAM native " + arg, unsigned=arg == "--seed")
        elif arg.startswith("--"):
            raise ValueError("VRAM native command: unsupported flag " + arg)
        elif model is None:
            model = arg
        else:
            require(len(prompt) < MAX_CAPACITY, "VRAM native command: too many prompt IDs")
            prompt.append(native_integer(arg, "VRAM native token ID"))
    require(values["--sample"] or "--prefill-chunk" in seen,
            "VRAM native command: would not emit request protocol")
    if not values["--sample"]:
        require(not seen.intersection(("--seed", "--temperature", "--top-p", "--top-k")),
                "VRAM native command: filters/seed require --sample")
    else:
        require(values["--seed"] is not None, "VRAM native command: --sample requires --seed")
    exact(model, origin["model"], "VRAM native model")
    exact(prompt, origin["prompt_ids"], "VRAM native prompt IDs")
    config = origin["config"]
    for key, value in {
        "capacity": values["--capacity"], "expert_slots": values["--slots"],
        "max_batch_tokens": values["--prefill-chunk"], "requested_output_tokens": values["--generate"],
        "ignore_eos": values["--ignore-eos"], "trace": bool(values["--trace"]),
        "trace_directory": values["--trace"], "logits_path": values["--logits"],
        "diagnostic_rows": not values["--sample"] or bool(values["--trace"]) or bool(values["--logits"]),
    }.items():
        exact(value, config[key], "VRAM native config." + key)
    sampling = origin["sampling"]
    exact(values["--sample"], sampling["mode"] == "stochastic", "VRAM native sampling mode")
    if values["--sample"]:
        for flag, key in (("--seed", "seed"), ("--top-k", "top_k")):
            exact(values[flag], sampling[key], "VRAM native sampling." + key)
        # Native binary64 filters may serialize as integer JSON numbers (e.g. 1).
        for flag, key in (("--temperature", "temperature"), ("--top-p", "top_p")):
            require(values[flag] == sampling[key], "VRAM native sampling." + key + ": mismatch")


def vram_observation(path, origin, native_footer):
    observed_source, footer = vram_records(path)
    expect(observed_source, {
        "kind": "vram_source", "protocol": 1, "scope": VRAM_SCOPE,
        "mapping_attestation": VRAM_MAPPING,
        "value_source": "AMD_sysfs_mem_info_vram_used_and_mem_info_vram_total",
        "sampling_window": "pre_spawn_through_child_lifetime_and_post_wait",
        "elapsed_scope": "monotonic_before_Popen_through_completed_child_wait_includes_spawn_and_observer_overhead",
    }, ("command_argv", "interval_seconds", "devices"), "VRAM source")
    interval = number(observed_source["interval_seconds"], "VRAM interval")
    require(.001 <= interval <= 10, "VRAM interval: outside observer bounds")
    expect(footer, {
        "kind": "vram_complete", "protocol": 1, "scope": VRAM_SCOPE,
        "observation_complete": True, "child_started": True, "child_returncode": 0,
        "observer_returncode": 0, "post_wait_sampled": True, "observer_error": None,
    }, ("completed_child_elapsed_seconds", "sample_rounds", "devices"), "VRAM complete")
    rounds = integer(footer["sample_rounds"], "VRAM sample_rounds", 3, UINT64_MAX)
    elapsed = number(footer["completed_child_elapsed_seconds"], "VRAM completed child elapsed")
    # The attached child includes startup/load/request/cleanup; only a lower
    # bound is justified. Divide before adding to avoid overflowing millisecond sums.
    chronological_le(math.fsum((native_footer["load_ms"] / 1000, native_footer["total_ms"] / 1000)),
                     elapsed, "VRAM native load+request/child elapsed")
    resolved_paths = set()
    for devices, label in ((observed_source["devices"], "VRAM source"), (footer["devices"], "VRAM complete")):
        require(type(devices) is list and len(devices) == 2, label + ": expected exactly two devices")
    for i, (device, aggregate) in enumerate(zip(observed_source["devices"], footer["devices"])):
        label = f"VRAM device_{i}"
        expect(device, {"label": f"device_{i}"},
               ("supplied_path", "resolved_path", "used_file", "total_file", "total_bytes"), label + " source")
        cli_text(device["supplied_path"], label + ".supplied_path")
        resolved = absolute_posix(device["resolved_path"], label + ".resolved_path")
        require(resolved not in resolved_paths, "VRAM devices must have distinct resolved paths")
        resolved_paths.add(resolved)
        exact(device["used_file"], str(resolved / "mem_info_vram_used"), label + ".used_file")
        exact(device["total_file"], str(resolved / "mem_info_vram_total"), label + ".total_file")
        total = integer(device["total_bytes"], label + ".total_bytes", 1, UINT64_MAX)
        expect(aggregate, {"label": f"device_{i}", "total_bytes": total, "sample_count": rounds},
               ("observed_max_used_bytes", "observed_min_free_bytes"), label + " complete")
        used = integer(aggregate["observed_max_used_bytes"], label + ".observed_max_used_bytes", 0, total)
        exact(aggregate["observed_min_free_bytes"], total - used, label + ".observed_min_free_bytes")
    join_native_command(observed_source["command_argv"], origin)
    return {"scope": VRAM_SCOPE, "mapping_attestation": VRAM_MAPPING,
            "command_join_scope": VRAM_COMMAND_JOIN, "source": observed_source, "complete": footer, "passed": True}


def collect(raw_path, vram_path=None):
    """Return one compact result only after the complete chronology validates."""
    raw_path = Path(raw_path).resolve()
    if vram_path is not None:
        vram_path = Path(vram_path).resolve()
        distinct_paths((raw_path, vram_path))
    rows = records(raw_path)
    origin, footer = rows[0], rows[-1]
    source(origin)
    completed(footer, origin)
    diagnostics = timeline(rows, origin, footer)
    raw_logs = {"request": str(raw_path)}
    if origin["config"]["logits_path"]:
        # This protects a reported binary artifact against journal aliasing;
        # its contents are not independently inspected or qualified here.
        raw_logs["logits_artifact"] = str(Path(origin["config"]["logits_path"]).resolve())
    result = {
        "kind": "r4_request", "protocol": 1,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": origin["revision"], "dirty": origin["dirty"], "runtime": origin["runtime"],
        "model": origin["model"], "raw_logs": raw_logs, "scope": SCOPE,
        "mtp": False, "mtp_acceptance": None, "performance_claim": False, "speedup_claim": False,
        "paired_ab_claim": False, "long_context_qualification_claim": False,
        "peak_vram_qualification_claim": False, "independent_reference_claim": False,
        "timing_scope": TIMING_SCOPE, "pp_scope": PP_SCOPE, "tg_scope": TG_SCOPE,
        "throughput_scope": "observed_completed_wall_rates_actual_prompt_and_outputs; TG excludes first output; load separate",
        "source": origin, "output_ids": footer["generated_ids"], "complete": footer,
        "counts": {**{key: footer[key] for key in ("input_tokens", "output_tokens", "consumed_tokens",
                                                   "pp_calls", "tg_forwards", "random_draws")},
                   "requested_output_tokens": origin["config"]["requested_output_tokens"],
                   "records": len(rows)},
        "timings_ms": {key: footer[key] for key in ("load_ms", "pp_ms", "first_output_ms",
                                                   "first_output_after_pp_ms", "tg_ms", "total_ms")},
        "throughput": {
            "pp_tokens_per_second": rate(footer["input_tokens"], footer["pp_ms"], "PP rate"),
            "tg_output_tokens_per_second": rate(footer["tg_forwards"], footer["tg_ms"], "TG rate"),
            "request_output_tokens_per_second": rate(footer["output_tokens"], footer["total_ms"], "request rate"),
        },
        "expert_counters": {key: footer[key] for key in COUNTERS},
        "diagnostics": diagnostics, "passed": True,
    }
    if vram_path is not None:
        result["vram_observation"] = vram_observation(vram_path, origin, footer)
        raw_logs["vram"] = str(vram_path)
    return result


def append_result(destination, record):
    """Keep diagnostic captures outside the explicit journal, then locked append."""
    trace = record["source"]["config"]["trace_directory"]
    if trace:
        require(not Path(destination).resolve().is_relative_to(Path(trace).resolve()),
                "results would write inside input trace captures")
    append_record(destination, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--vram-log", type=Path, help="optional complete observer JSONL for a direct Docker request")
    parser.add_argument("--results", type=Path, required=True, help="explicit canonical ROOT journal path")
    args = parser.parse_args(argv)
    try:
        if args.vram_log is not None:
            distinct_paths((args.raw, args.vram_log, args.results))
            record = collect(args.raw, args.vram_log)
        else:
            record = collect(args.raw)
        append_result(args.results, record)
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print("record_request: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
