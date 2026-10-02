#!/usr/bin/env python3
"""Validate seven core-session-batch-test JSONL rows and append one R4a record.

The frozen protocol proves short-window full-logit self-parity and observed
one-slot reuse. It does not qualify PP, performance, or an independent HF path.
"""

import argparse
import datetime
import json
from pathlib import Path
import stat
import struct
import sys

if __package__:
    from .record_memory import (append_result, exact, expect, finite_float,
                                integer, invalid_constant, number, require,
                                text, unique_object)
else:
    from record_memory import (append_result, exact, expect, finite_float,
                               integer, invalid_constant, number, require,
                               text, unique_object)


MAX_RAW_BYTES = 2 * 1024 * 1024
CAPACITY, TEACHER_ROWS, TAIL_ROWS, VOCAB = 40, 32, 8, 248320
ROUTES_PER_ROW, INHERITED_HITS = 48 * 10, 48
Q4_0_BYTES, Q4_1_BYTES = 2764800, 2867200
RAM_PAYLOAD = (6 * Q4_1_BYTES + 42 * Q4_0_BYTES) * 512
HOST_LOGIT_BYTES = 3 * VOCAB * 4
PINNED_BYTES = 3 * 4 * 2560 * 4
WORKSPACE_FLOOR = 24 * 36864 * 4 + 960 * 36 + HOST_LOGIT_BYTES
RUNTIME, MODEL = "own_48_layer_HIP", "qwen38-keep1-Q4_0.gguf"
SCHEDULES = ("reference_n1", "replay_n1", "replay_n2", "replay_n3",
             "replay_mixed_123_prefix5")
COUNTERS = ("expert_hits", "expert_misses", "expert_upload_bytes")
ROUTE_FIELDS = ("hits", "misses", "upload_bytes")
CATEGORIES = ("weights", "expert_slots", "qsa_kv", "qsa_index", "gdn_state", "ple_state", "workspace")
ZERO_STATS = {"consumed_tokens": 0, **dict.fromkeys(COUNTERS, 0), "last_completed_ms": 0}
ZERO_ROUTES = dict.fromkeys(ROUTE_FIELDS, 0)


def json_integer(value):
    # max_digits10 ostream output spells a negative-zero double as "-0".
    # Preserve that sign for timing comparisons; unsigned counters cannot use it.
    return -0.0 if value == "-0" else int(value)


def records(path):
    """Bound even a growing input, and require exactly seven newline-finished rows."""
    require(stat.S_ISREG(path.stat().st_mode), str(path) + ": not a regular file")
    require(path.stat().st_size <= MAX_RAW_BYTES, str(path) + ": oversized input")
    with path.open("rb") as stream:
        raw = stream.read(MAX_RAW_BYTES + 1)
    require(len(raw) <= MAX_RAW_BYTES, str(path) + ": oversized input")
    require(raw.endswith(b"\n"), str(path) + ": missing completed final line")
    lines = raw.split(b"\n")[:-1]
    require(len(lines) == 7, str(path) + ": expected exactly seven JSONL rows")
    rows = []
    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line.decode("utf-8"), object_pairs_hook=unique_object,
                             parse_int=json_integer, parse_float=finite_float, parse_constant=invalid_constant)
            require(type(obj) is dict, "expected an object")
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{path}:row {i}: {error}") from error
        rows.append(obj)
    return rows


def source(obj):
    expect(obj, {
        "kind": "session_batch_source", "protocol": 1, "runtime": RUNTIME,
        "config": {"capacity": 40, "expert_slots": 1, "max_batch_tokens": 3, "trace": False},
        "teacher_ids": [248044, *range(100, 131)], "capacity_tail_ids": list(range(131, 139)),
        "schedule_order": list(SCHEDULES), "scope": "exact_causal_short_window_replay_before_PP",
        "comparison": "all_FP32_bits_no_tolerance",
        "reference_scope": "retained_sequential_N1_same_Session_unchanged_weights_and_gates",
        "sampling": "teacher_forced", "performance_claim": False, "session_instances": 1,
        "reset_clears_weight_slots": False, "cache_start_equality_claim": False,
        "reuse_scope": "matching_teacher_suffix_after_first_grouped_window_initial_slots_overwritten",
        "grouping_scope": "within_call_only_all_48_layers_10_routes_per_row",
        "counter_definitions": {
            "routes_per_row": ROUTES_PER_ROW, "max_inherited_slot_hits_per_window": INHERITED_HITS,
            "timeline_rows_per_schedule": CAPACITY, "teacher_rows_per_schedule": TEACHER_ROWS,
            "tail_rows_per_schedule": TAIL_ROWS, "logit_values_per_row": VOCAB, "invalid_windows_per_replay": 7,
            "bitwise_count_scope": "timeline_and_reset_probe_only_excludes_preservation_rechecks",
            "route_stats_scope": "timeline_cumulative_reset_probe_separate",
            "within_window_hit_lower_bound_scope": "sum_max_0_window_hits_minus_48_all_teacher_windows",
        },
        "geometry": {
            "min_q8_blocks_per_device": 960, "q8_block_bytes": 36, "scratch_buffers_per_device": 24,
            "min_scratch_floats_per_buffer": 36864, "min_host_logit_bytes": HOST_LOGIT_BYTES,
            "pinned_handoff_bytes": PINNED_BYTES, "min_workspace_aggregate_bytes_per_device": WORKSPACE_FLOOR,
            "individual_q8_scratch_geometry_observable": False, "workspace_check": "aggregate_floor_only",
            "free_vram_equality_required": False,
            "ledger_scope": "Session_owned_buffers_and_reported_host_capacities_excludes_allocator_metadata",
        },
        "preallocated_reference_bytes": TEACHER_ROWS * VOCAB * 4,
        "preallocated_tail_reference_bytes": TAIL_ROWS * VOCAB * 4,
        "preallocated_snapshot_bytes": HOST_LOGIT_BYTES,
    }, ("revision", "dirty", "model", "model_bytes"), "source")
    revision = obj["revision"]
    require(type(revision) is str and len(revision) == 40 and
            all(c in "0123456789abcdefABCDEF" for c in revision), "source: invalid compiled revision")
    require(type(obj["dirty"]) is bool, "source: invalid boolean dirty")
    require(Path(text(obj["model"], "source.model")).name == MODEL, "source: unexpected model variant")
    integer(obj["model_bytes"], "source.model_bytes", 1, (1 << 63) - 1)


def layout(index):
    """Reproduce requested widths, including the two separate tail boundaries."""
    offset, mixed = 0, 0
    windows = []

    def requested():
        nonlocal mixed
        if index <= 1:
            return 1
        if index <= 3:
            return index
        if offset < 5:
            return 1
        width = (2, 3, 1)[mixed % 3]
        mixed += 1
        return width

    for boundary in (TEACHER_ROWS, CAPACITY - 1, CAPACITY):
        while offset < boundary:
            width = requested()
            rows = min(width, boundary - offset)
            windows.append((offset, rows, width))
            offset += rows
    return windows


def stats(obj, consumed, label):
    expect(obj, {"consumed_tokens": consumed}, (*COUNTERS, "last_completed_ms"), label)
    for key in COUNTERS:
        integer(obj[key], label + "." + key)
    number(obj["last_completed_ms"], label + ".last_completed_ms")
    if consumed == 0:
        require(all(obj[key] == 0 for key in (*COUNTERS, "last_completed_ms")), label + ": reset stats are not zero")


def same_stats(actual, expected, label):
    stats(actual, expected["consumed_tokens"], label)
    for key in COUNTERS:
        exact(actual[key], expected[key], label + "." + key)
    # C++ emits double with max_digits10. Integer-looking double JSON is valid;
    # signed zero still matters for the fixture's unchanged-timing proof.
    require(struct.pack("!d", actual["last_completed_ms"]) == struct.pack("!d", expected["last_completed_ms"]),
            label + ": completed timing bits changed")


def routes(obj, rows, label):
    expect(obj, {}, ROUTE_FIELDS, label)
    for key in ROUTE_FIELDS:
        integer(obj[key], label + "." + key)
    hits, misses, uploaded = (obj[key] for key in ROUTE_FIELDS)
    require(hits + misses == ROUTES_PER_ROW * rows, label + ": incomplete 480*N route accounting")
    require(misses * Q4_0_BYTES <= uploaded <= misses * Q4_1_BYTES,
            label + ": upload bytes inconsistent with miss payloads")
    if rows == 1:
        require(hits <= INHERITED_HITS and misses >= 48 * 9, label + ": one-slot N1 route bound")


def sum_routes(windows):
    return {key: sum(w["delta"][key] for w in windows) for key in ROUTE_FIELDS}


def window(obj, shape, previous, reference, label):
    offset, rows, requested = shape
    expect(obj, {"offset": offset, "rows": rows, "requested_rows": requested, "routes": rows * ROUTES_PER_ROW},
           ("delta", "n1_reference_delta", "stats"), label)
    routes(obj["delta"], rows, label + ".delta")
    routes(obj["n1_reference_delta"], rows, label + ".n1_reference_delta")
    exact(obj["n1_reference_delta"], sum_routes(reference[offset:offset + rows]), label + ".matching_N1")
    stats(obj["stats"], offset + rows, label + ".stats")
    exact(previous["consumed_tokens"], offset, label + ".previous_consumed")
    for counter, key in zip(COUNTERS, ROUTE_FIELDS):
        exact(obj["stats"][counter], previous[counter] + obj["delta"][key], label + "." + counter)


def memory(obj, label, loaded=None):
    expect(obj, {"capacity": CAPACITY, "expert_slots": 1, "ownership_verified": True,
                 "ram_expert_payload": RAM_PAYLOAD, "pinned_handoff": PINNED_BYTES,
                 "expert_payload_reads": 48 * 3, "expert_payload_bytes_read": RAM_PAYLOAD},
           ("ram_expert_capacity", "host_embedding_capacity", "host_logit_capacity", "devices"), label)
    integer(obj["ram_expert_capacity"], label + ".ram_expert_capacity", RAM_PAYLOAD)
    integer(obj["host_embedding_capacity"], label + ".host_embedding_capacity", 1)
    integer(obj["host_logit_capacity"], label + ".host_logit_capacity", HOST_LOGIT_BYTES)
    devices = obj["devices"]
    require(type(devices) is list and len(devices) == 2, label + ": expected both static owners")
    for i, d in enumerate(devices):
        where = f"{label}.devices[{i}]"
        expect(d, {"device": i, "first_layer": i * 24, "last_layer": i * 24 + 23,
                   "gdn_layers": 18, "qsa_layers": 6},
               (*CATEGORIES, "owned_bytes", "owned_peak_bytes", "owned_buffers", "total_vram", "free_vram"), where)
        for key in CATEGORIES:
            integer(d[key], where + "." + key, 0 if key == "ple_state" else 1)
        integer(d["workspace"], where + ".workspace", WORKSPACE_FLOOR)
        require(d["ple_state"] > 0 if i == 0 else d["ple_state"] == 0, where + ": PLE owner")
        total = integer(d["total_vram"], where + ".total_vram", 1)
        owned = integer(d["owned_bytes"], where + ".owned_bytes", 1, total)
        exact(owned, sum(d[key] for key in CATEGORIES), where + ".category_sum")
        integer(d["owned_buffers"], where + ".owned_buffers", 1)
        integer(d["owned_peak_bytes"], where + ".owned_peak_bytes", owned)
        integer(d["free_vram"], where + ".free_vram", 1, total)
        if loaded is not None:
            for key in d.keys() - {"free_vram"}:
                exact(d[key], loaded["devices"][i][key], where + ".steady." + key)
    if loaded is not None:
        for key in obj.keys() - {"devices"}:
            exact(obj[key], loaded[key], label + ".steady." + key)


def invalid_proofs(proofs, windows, index, reset_window, label):
    require(type(proofs) is list and len(proofs) == (0 if index == 0 else 7), label + ": invalid proof count")
    if index == 0:
        return 0
    ordinary = next(i for i, w in enumerate(windows)
                    if index != 4 or (w["offset"] + w["rows"] >= 5 and w["rows"] == 3))
    cases = [("empty", 0, ordinary, False), ("length4", 4, ordinary, False),
             ("negative_last", 3, ordinary, False), ("vocabulary_last", 3, ordinary, False),
             ("capacity39_length2", 2, len(windows) - 2, False),
             ("capacity39_length3", 3, len(windows) - 2, False),
             ("capacity40_length2", 2, len(windows) - 1, True)]
    preserved = 0
    for i, (proof, (name, rows, prior_index, reset)) in enumerate(zip(proofs, cases)):
        prior = windows[prior_index]
        continuation = reset_window if reset else windows[prior_index + 1]
        count = prior["rows"] * VOCAB
        where = f"{label}[{i}]"
        expect(proof, {
            "case": name, "rows": rows, "prior_rows": prior["rows"], "preserved_logit_values": count,
            "exception": "invalid_argument", "stats_bitwise_preserved": True, "logits_bitwise_preserved": True,
            "memory_ledger_preserved": True, "consumed_advance": 0, "continued": True,
            "reset_before_continuation": reset, "continuation_rows": continuation["rows"],
            "continuation_bitwise_logit_values": continuation["rows"] * VOCAB,
        }, ("stats_before_and_after",), where)
        same_stats(proof["stats_before_and_after"], prior["stats"], where + ".stats_before_and_after")
        require(continuation["offset"] == (0 if reset else prior["stats"]["consumed_tokens"]),
                where + ": wrong continuation position")
        preserved += count
    return preserved


def reuse(obj, windows, index, reference, label):
    teacher = [w for w in windows if w["offset"] < TEACHER_ROWS]
    lower_bound = sum(max(0, w["delta"]["hits"] - INHERITED_HITS) for w in teacher)
    if index < 2:
        expect(obj, {"applicable": False, "first_row": 0, "rows": 0, "n1_reference": ZERO_ROUTES,
                     "candidate": ZERO_ROUTES, "hits_gained": 0, "misses_saved": 0,
                     "teacher_within_window_hit_lower_bound": 0, "initial_cache_contents_matched": False,
                     "performance_claim": False}, label=label)
        return
    first = next(w["offset"] + w["rows"] for w in teacher if w["rows"] > 1)
    baseline = sum_routes(reference[first:TEACHER_ROWS])
    candidate = sum_routes([w for w in teacher if w["offset"] >= first])
    gained, saved = candidate["hits"] - baseline["hits"], baseline["misses"] - candidate["misses"]
    require(lower_bound > 0 and gained > 0 and saved > 0, label + ": no positive matching-suffix/within-window reuse")
    expect(obj, {"applicable": True, "first_row": first, "rows": TEACHER_ROWS - first,
                 "n1_reference": baseline, "candidate": candidate, "hits_gained": gained, "misses_saved": saved,
                 "teacher_within_window_hit_lower_bound": lower_bound, "initial_cache_contents_matched": False,
                 "performance_claim": False}, label=label)


def schedule(obj, index, reference, loaded):
    label = SCHEDULES[index]
    expect(obj, {"kind": "session_batch_schedule", "protocol": 1, "index": index, "schedule": label,
                 "reference": index == 0, "occupied_prefix_n1_rows": 5 if index == 4 else 0, "passed": True},
           ("counts", "initial_stats", "teacher_stats", "timeline_stats", "windows", "reuse", "invalid_proofs",
            "reset_probe", "final_reset_stats", "memory"), label)
    shapes = layout(index)
    windows = obj["windows"]
    require(type(windows) is list and len(windows) == len(shapes), label + ": incorrect window count")
    # Validate the reference's shape and schema before using any of its counters.
    if index == 0:
        for i, w in enumerate(windows):
            expect(w, {"offset": i, "rows": 1, "requested_rows": 1, "routes": ROUTES_PER_ROW},
                   ("delta", "n1_reference_delta", "stats"), f"{label}.windows[{i}]")
            routes(w["delta"], 1, f"{label}.windows[{i}].delta")
        reference = windows
    stats(obj["initial_stats"], 0, label + ".initial_stats")
    stats(obj["final_reset_stats"], 0, label + ".final_reset_stats")
    previous = obj["initial_stats"]
    for i, (w, shape) in enumerate(zip(windows, shapes)):
        window(w, shape, previous, reference, f"{label}.windows[{i}]")
        previous = w["stats"]
    teacher = [w for w in windows if w["offset"] < TEACHER_ROWS]
    same_stats(obj["teacher_stats"], teacher[-1]["stats"], label + ".teacher_stats")
    same_stats(obj["timeline_stats"], windows[-1]["stats"], label + ".timeline_stats")
    reset = obj["reset_probe"]
    has_reset = index > 0
    expect(reset, {"rows": int(has_reset), "finite_logit_values": VOCAB if has_reset else 0,
                   "bitwise_compared_logit_values": VOCAB if has_reset else 0}, ("window",), label + ".reset_probe")
    if has_reset:
        window(reset["window"], (0, 1, 1), ZERO_STATS, reference, label + ".reset_probe.window")
    else:
        exact(reset["window"], None, label + ".reset_probe.window")
    preserved = invalid_proofs(obj["invalid_proofs"], windows, index, reset["window"], label + ".invalid_proofs")
    reuse(obj["reuse"], windows, index, reference, label + ".reuse")
    expect(obj["counts"], {
        "teacher_rows": TEACHER_ROWS, "tail_rows": TAIL_ROWS, "rows": CAPACITY, "finite_rows": CAPACITY,
        "bitwise_compared_rows": CAPACITY if has_reset else 0,
        "teacher_finite_logit_values": TEACHER_ROWS * VOCAB,
        "teacher_bitwise_compared_logit_values": TEACHER_ROWS * VOCAB if has_reset else 0,
        "tail_finite_logit_values": TAIL_ROWS * VOCAB,
        "tail_bitwise_compared_logit_values": TAIL_ROWS * VOCAB if has_reset else 0,
        "finite_logit_values": CAPACITY * VOCAB, "bitwise_compared_logit_values": CAPACITY * VOCAB if has_reset else 0,
        "windows": len(windows), "teacher_windows": len(teacher), "tail_windows": len(windows) - len(teacher),
        "partial_windows": sum(rows < requested for _, rows, requested in shapes), "routes": CAPACITY * ROUTES_PER_ROW,
        "invalid_windows": 7 if has_reset else 0, "rejection_preserved_logit_values": preserved,
    }, label=label + ".counts")
    snapshots = 1 + len(windows) + 1 + (7 * 2 + 2 if has_reset else 0)
    audit = obj["memory"]
    expect(audit, {
        "snapshot_count": snapshots, "steady_comparisons_to_loaded": snapshots,
        "stats_preserved_comparisons": snapshots,
        "bitwise_preserved_logit_values": CAPACITY * VOCAB + 2 * preserved + (VOCAB if has_reset else 0),
        "steady_categories_counts_peak_host_and_payload_reads": True,
    }, ("minimum_free_vram_bytes", "final_snapshot", "loaded_anchor_snapshot"), label + ".memory")
    if index == 0:
        loaded = audit["loaded_anchor_snapshot"]
        memory(loaded, label + ".memory.loaded_anchor_snapshot")
    else:
        exact(audit["loaded_anchor_snapshot"], None, label + ".memory.loaded_anchor_snapshot")
    memory(audit["final_snapshot"], label + ".memory.final_snapshot", loaded)
    minima = audit["minimum_free_vram_bytes"]
    require(type(minima) is list and len(minima) == 2, label + ": expected both minimum free VRAM values")
    for i, minimum in enumerate(minima):
        upper = min(loaded["devices"][i]["free_vram"], audit["final_snapshot"]["devices"][i]["free_vram"])
        integer(minimum, f"{label}.memory.minimum_free_vram_bytes[{i}]", 1, upper)
    return reference, loaded


def complete(obj, schedules):
    counts = [s["counts"] for s in schedules]
    audits = [s["memory"] for s in schedules]
    resets = [s["reset_probe"] for s in schedules]
    expect(obj, {
        "kind": "session_batch_complete", "protocol": 1, "schedule_count": len(SCHEDULES), "record_count": 7,
        "timeline_rows": 5 * CAPACITY, "teacher_rows": 5 * TEACHER_ROWS, "tail_rows": 5 * TAIL_ROWS,
        "finite_rows": 5 * CAPACITY, "bitwise_compared_rows": 4 * CAPACITY, "timeline_routes": 5 * CAPACITY * ROUTES_PER_ROW,
        "window_count": sum(c["windows"] for c in counts), "finite_logit_values": 5 * CAPACITY * VOCAB,
        "bitwise_compared_logit_values": 4 * CAPACITY * VOCAB,
        "reset_probe_rows": sum(r["rows"] for r in resets), "reset_probe_routes": 4 * ROUTES_PER_ROW,
        "reset_probe_finite_logit_values": 4 * VOCAB, "reset_probe_bitwise_compared_logit_values": 4 * VOCAB,
        "invalid_window_rejections": sum(c["invalid_windows"] for c in counts),
        "rejection_preserved_logit_values": sum(c["rejection_preserved_logit_values"] for c in counts),
        "loaded_memory_snapshot_count": 1, "schedule_memory_snapshot_count": sum(a["snapshot_count"] for a in audits),
        "memory_snapshot_count": 1 + sum(a["snapshot_count"] for a in audits),
        "memory_bitwise_preserved_logit_values": sum(a["bitwise_preserved_logit_values"] for a in audits),
        "positive_grouped_reuse_schedules": 3, "session_instances": 1, "raii_session_cleanup_completed": True,
        "owned_buffer_release_measured": False, "performance_claim": False, "passed": True,
    }, label="complete")


def collect(raw_path):
    """Return ONE r4a_grouped_session record after full validation; never write inputs."""
    raw_path = Path(raw_path).resolve()
    rows = records(raw_path)
    origin, schedules, footer = rows[0], rows[1:6], rows[6]
    source(origin)
    reference, loaded = None, None
    for i, obj in enumerate(schedules):
        reference, loaded = schedule(obj, i, reference, loaded)
    complete(footer, schedules)
    return {
        "kind": "r4a_grouped_session", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": origin["revision"], "dirty": origin["dirty"], "runtime": RUNTIME,
        "model_variant": "qwen38-keep1-Q4_0", "model": origin["model"], "raw_logs": {"batch": str(raw_path)},
        "scope": "short-window N1/N2/N3 and occupied-prefix mixed replay: full-vocabulary bitwise self-parity on one own48 Session; matching-suffix one-slot reuse diagnostics; not PP, performance, or independent HF qualification",
        "performance_claim": False, "pp_qualified": False, "independent_hf_claim": False, "mtp": False,
        "timing_scope": "last_completed_ms is finite nonnegative correctness metadata; invalid windows preserve its double bits; not performance evidence",
        "expert_read_scope": "instrumented logical expert payload loads and bytes, steady after construction; not a syscall trace or physical SSD-read trace",
        "memory_scope": "steady Session-owned categories/counts/peaks and reported host capacities; aggregate Q8/scratch workspace floor only, no individual-buffer-size proof or full-RAM accounting; cleanup footer follows RAII destruction, no after-destruction owned-buffer release measurement",
        "fixture_gates": {"frozen_before_execution": True, "logits": "all_FP32_bits_no_tolerance",
                          "reuse": "each grouped schedule: positive matching-suffix hits gained and misses saved AND positive sum(max(0, teacher_window_hits-48))",
                          "workspace": "aggregate_floor_only", "owned_buffer_release_measured": False},
        "source": origin, "schedules": schedules, "complete": footer, "passed": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        append_result(args.results, collect(args.raw))
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print("record_batch: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
