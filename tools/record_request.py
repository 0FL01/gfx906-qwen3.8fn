#!/usr/bin/env python3
"""Validate protocol-1 core-session full requests and append ONE r4_request.

Only emitted structured records are evidence. Legacy session JSON is rejected.
PP, first-output and TG are completed wall intervals, with sampling/CLI IO in
their declared scopes; load is separate. Rates use actual tokens, not requested
tokens or GPU-event sums. This is candidate non-MTP request accounting, not an
MTP, long-context, independent-reference, peak-VRAM or paired-speedup proof.
--results must explicitly name the caller's canonical ROOT/results.jsonl.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
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


def collect(raw_path):
    """Return one compact result only after the complete chronology validates."""
    raw_path = Path(raw_path).resolve()
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
    return {
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
    parser.add_argument("--results", type=Path, required=True, help="explicit canonical ROOT journal path")
    args = parser.parse_args(argv)
    try:
        append_result(args.results, collect(args.raw))
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print("record_request: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
