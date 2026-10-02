#!/usr/bin/env python3
"""Validate the complete core-memory protocol and append one R3c evidence record.

This is capacity/ownership/reset/replay qualification, not an occupied-128K or
performance measurement. Inputs are bounded structured JSONL, never free text.
"""

import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import sys


MAX_RAW_BYTES = 2 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1
CAPACITY, SLOTS, VOCAB = 131072, 112, 248320
MIN_HEADROOM = 1024 * 1024 * 1024
RUNTIME = "own_48_layer_HIP"
SCOPE = "r3_allocation_owner_headroom_reset_replay"
MODEL = "qwen38-keep1-Q4_0.gguf"
PHASES = ("loaded", "step0", "step1", "reset", "replay0", "replay1")
CONSUMED = (0, 1, 2, 0, 1, 2)
TOKEN_IDS = [248044, 100]
Q4_0_EXPERT_BYTES, Q4_1_EXPERT_BYTES = 2764800, 2867200
GEOMETRY = {
    "qsa_kv_bytes_per_gpu": 432 * 1024 * 1024,
    "qsa_pooled_index_bytes_per_gpu": 96 * 1024 * 1024,
    "qsa_index_tail_bytes_per_gpu": 9216,
    "qsa_index_bytes_per_gpu": 96 * 1024 * 1024 + 9216,
    "gdn_state_bytes_per_gpu": 18 * (786432 + 30720) * 4,
    "ple_state_bytes": [368640, 0],
    "q4_1_down_expert_bytes": Q4_1_EXPERT_BYTES,
    "q4_0_down_expert_bytes": Q4_0_EXPERT_BYTES,
    "expert_slots_bytes": [7500595200, 7431782400],
    "ram_expert_payload_bytes": 68262297600,
    "expert_payload_reads": 144,
    "host_logit_payload_bytes": 993280,
    "pinned_handoff_bytes": 40960,
}
LIMITS = {
    "gpu_categories": "live_Session_Buffer_allocations_only",
    "gpu_private_allocations": "HIP_context_rocBLAS_visible_only_in_total_free_VRAM",
    "workspace": "includes_staged_recurrent_conv_PLE_state",
    "host_capacities": "reported_payload_buffers_only_excludes_metadata_allocator_overhead",
    "host_embedding_payload": "not_exposed_by_SessionMemory_capacity_checked_nonzero_and_steady",
    "rss": "getrusage_RUSAGE_SELF_ru_maxrss_Linux_KiB_times_1024",
    "context": "capacity_allocation_only_two_consumed_tokens_per_pass",
    "mtp": False,
}
CATEGORIES = ("weights", "expert_slots", "qsa_kv", "qsa_index", "gdn_state", "ple_state", "workspace")
STEADY_DEVICE_FIELDS = (*CATEGORIES, "owned_bytes", "owned_peak_bytes", "owned_buffers")
COUNTERS = ("expert_hits", "expert_misses", "expert_upload_bytes")


def require(condition, label):
    if not condition:
        raise ValueError(label)


def exact(actual, expected, label):
    require(type(actual) is type(expected), label + ": incorrect type")
    if type(expected) is dict:
        expect(actual, expected, label=label)
    elif type(expected) is list:
        require(len(actual) == len(expected), label + ": incorrect length")
        for i, (got, want) in enumerate(zip(actual, expected)):
            exact(got, want, f"{label}[{i}]")
    else:
        require(actual == expected, label + ": unexpected value")


def expect(obj, fixed, variable=(), label="memory"):
    require(type(obj) is dict and set(obj) == set(fixed) | set(variable),
            label + ": unexpected or missing fields")
    for key, value in fixed.items():
        exact(obj[key], value, label + "." + key)


def integer(value, label, low=0, high=UINT64_MAX):
    require(type(value) is int and low <= value <= high, label + ": invalid integer")
    return value


def number(value, label):
    require(type(value) in (int, float), label + ": expected a nonboolean number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and value >= 0, label + ": invalid finite number")
    return value


def text(value, label):
    require(type(value) is str and 0 < len(value) <= 4096 and
            not any(ord(c) < 32 or 0xd800 <= ord(c) <= 0xdfff for c in value),
            label + ": invalid string")
    return value


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "duplicate JSON key: " + key)
        obj[key] = value
    return obj


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite JSON number")
    return result


def invalid_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def records(path):
    """Read at most 2 MiB, including for growing logs; require eight finished rows."""
    path = Path(path)
    require(stat.S_ISREG(path.stat().st_mode), str(path) + ": not a regular file")
    require(path.stat().st_size <= MAX_RAW_BYTES, str(path) + ": oversized input")
    with path.open("rb") as stream:
        raw = stream.read(MAX_RAW_BYTES + 1)
    require(len(raw) <= MAX_RAW_BYTES, str(path) + ": oversized input")
    require(raw.endswith(b"\n"), str(path) + ": missing completed final line")
    lines = raw.split(b"\n")[:-1]
    require(len(lines) == 8, str(path) + ": expected exactly eight JSONL rows")
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


def probes(value, label, before=None):
    require(type(value) is list and len(value) == 2, label + ": expected both devices")
    for i, device in enumerate(value):
        where = f"{label}[{i}]"
        expect(device, {"device": i, "wave_size": 64},
               ("name", "arch", "compute_units", "async_engines", "properties_total_vram",
                "total_vram", "free_vram"), where)
        text(device["name"], where + ".name")
        arch = text(device["arch"], where + ".arch")
        require(arch == "gfx906" or arch.startswith("gfx906:"), where + ": requires gfx906")
        integer(device["compute_units"], where + ".compute_units", 1, (1 << 31) - 1)
        integer(device["async_engines"], where + ".async_engines", high=(1 << 31) - 1)
        total = integer(device["total_vram"], where + ".total_vram", 1)
        exact(device["properties_total_vram"], total, where + ".properties_total_vram")
        integer(device["free_vram"], where + ".free_vram", MIN_HEADROOM, total)
        if before is not None:
            for key in device.keys() - {"free_vram"}:
                exact(device[key], before[i][key], where + "." + key)


def source(obj):
    expect(obj, {
        "kind": "memory_source", "protocol": 1, "build_provenance_complete": True,
        "runtime": RUNTIME, "scope": SCOPE, "capacity": CAPACITY, "expert_slots": SLOTS,
        "session_instances": 1, "trace": False, "sampling": "teacher_forced",
        "token_ids": TOKEN_IDS, "vocabulary": VOCAB, "min_free_headroom_bytes": MIN_HEADROOM,
        "gates_frozen_before_execution": True, "occupied_128k_qualified": False,
        "performance_claim": False, "limits": LIMITS, "expected_geometry": GEOMETRY,
    }, ("revision", "dirty", "model_path", "before_load", "caller_device", "peak_rss_bytes"), "source")
    revision = obj["revision"]
    require(type(revision) is str and len(revision) == 40 and
            all(c in "0123456789abcdefABCDEF" for c in revision), "source: invalid compiled revision")
    require(type(obj["dirty"]) is bool, "source: invalid boolean dirty")
    model = Path(text(obj["model_path"], "source.model_path"))
    require(model.is_absolute() and model.name == MODEL, "source: expected absolute keep1-Q4_0 model path")
    integer(obj["caller_device"], "source.caller_device", high=1)
    integer(obj["peak_rss_bytes"], "source.peak_rss_bytes", 1)
    probes(obj["before_load"], "source.before_load")


def device_memory(devices, before, label):
    require(type(devices) is list and len(devices) == 2, label + ": expected both owners")
    for i, device in enumerate(devices):
        where = f"{label}[{i}]"
        expect(device, {
            "device": i, "first_layer": i * 24, "last_layer": i * 24 + 23,
            "gdn_layers": 18, "qsa_layers": 6,
            "expert_slots": GEOMETRY["expert_slots_bytes"][i],
            "qsa_kv": GEOMETRY["qsa_kv_bytes_per_gpu"],
            "qsa_index": GEOMETRY["qsa_index_bytes_per_gpu"],
            "gdn_state": GEOMETRY["gdn_state_bytes_per_gpu"],
            "ple_state": GEOMETRY["ple_state_bytes"][i], "total_vram": before[i]["total_vram"],
        }, ("weights", "workspace", "owned_bytes", "owned_peak_bytes", "owned_buffers", "free_vram"), where)
        # Weights, workspace and allocation counts are observed, not compile-time guesses.
        for key in ("weights", "workspace", "owned_buffers"):
            integer(device[key], where + "." + key, 1)
        owned = integer(device["owned_bytes"], where + ".owned_bytes", 1)
        require(sum(device[key] for key in CATEGORIES) == owned, where + ": category sum differs from owned bytes")
        integer(device["owned_peak_bytes"], where + ".owned_peak_bytes", owned)
        free = integer(device["free_vram"], where + ".free_vram", MIN_HEADROOM, device["total_vram"])
        require(owned + free <= device["total_vram"], where + ": owned bytes exceed observed used VRAM")


def host_memory(obj, label):
    expect(obj, {
        "ram_expert_payload": GEOMETRY["ram_expert_payload_bytes"],
        "pinned_handoff": GEOMETRY["pinned_handoff_bytes"],
        "expert_payload_reads": GEOMETRY["expert_payload_reads"],
        "expert_payload_bytes_read": GEOMETRY["ram_expert_payload_bytes"],
    }, ("ram_expert_capacity", "host_embedding_capacity", "host_logit_capacity"), label)
    integer(obj["ram_expert_capacity"], label + ".ram_expert_capacity", obj["ram_expert_payload"])
    integer(obj["host_embedding_capacity"], label + ".host_embedding_capacity", 1)
    integer(obj["host_logit_capacity"], label + ".host_logit_capacity", GEOMETRY["host_logit_payload_bytes"])


def route_stats(obj, consumed, previous, label):
    expect(obj, {"consumed_tokens": consumed}, (*COUNTERS, "last_completed_ms"), label)
    for key in COUNTERS:
        integer(obj[key], label + "." + key)
    number(obj["last_completed_ms"], label + ".last_completed_ms")
    if consumed == 0:
        for key in (*COUNTERS, "last_completed_ms"):
            require(obj[key] == 0, label + ": initial/reset stats are not zero")
        return
    deltas = {key: obj[key] - previous[key] for key in COUNTERS}
    require(all(delta >= 0 for delta in deltas.values()), label + ": expert counters decreased")
    hits, misses, uploaded = (deltas[key] for key in COUNTERS)
    require(hits <= 480 and misses <= 480 and hits + misses == 480 and
            obj["expert_hits"] + obj["expert_misses"] == consumed * 480,
            label + ": incomplete expert route accounting")
    require(misses * Q4_0_EXPERT_BYTES <= uploaded <= misses * Q4_1_EXPERT_BYTES,
            label + ": expert uploads inconsistent with miss payloads")


def snapshot(obj, i, before, loaded, previous, previous_rss):
    where = PHASES[i]
    has_logits, replay = CONSUMED[i] > 0, i >= 4
    expect(obj, {
        "kind": "memory_snapshot", "phase": where, "capacity": CAPACITY, "expert_slots": SLOTS,
        "consumed_tokens": CONSUMED[i], "token_id": TOKEN_IDS[CONSUMED[i] - 1] if has_logits else None,
        "ownership_verified": True,
        "checks": {
            "geometry": True, "headroom": True, "steady_allocations": True if i else None,
            "zero_stats": None if has_logits else True, "finite_logits": True if has_logits else None,
            "finite_logit_values": VOCAB if has_logits else 0, "bitwise_replay": True if replay else None,
            "bitwise_replay_compared_values": VOCAB if replay else 0,
            "snapshot_stats_preserved": True, "snapshot_logits_preserved": True if has_logits else None,
            "snapshot_logit_compared_values": VOCAB if has_logits else 0,
            "snapshot_caller_device_preserved": True, "snapshot_caller_device": i % 2,
        },
    }, ("stats", "devices", "host", "peak_rss_bytes"), where)
    route_stats(obj["stats"], CONSUMED[i], previous, where + ".stats")
    device_memory(obj["devices"], before, where + ".devices")
    host_memory(obj["host"], where + ".host")
    integer(obj["peak_rss_bytes"], where + ".peak_rss_bytes", max(1, previous_rss))
    if loaded is not None:
        for device, baseline in zip(obj["devices"], loaded["devices"]):
            for key in STEADY_DEVICE_FIELDS:
                exact(device[key], baseline[key], where + ".steady." + key)
        exact(obj["host"], loaded["host"], where + ".steady.host")


def complete(obj, origin, snapshots):
    expect(obj, {
        "kind": "memory_complete", "protocol": 1, "scope": SCOPE, "capacity": CAPACITY,
        "expert_slots": SLOTS, "session_instances": 1, "consumed_tokens": 2,
        "baseline_consumed_tokens": 2, "replay_consumed_tokens": 2,
        "occupied_128k_qualified": False, "performance_claim": False,
        "checks": {
            "completed_steps": 4, "snapshot_count": 6, "steady_memory_comparisons": 5,
            "initial_stats_zero": True, "reset_consumed_and_stats_zero": True, "all_outputs_finite": True,
            "finite_logit_values": 4 * VOCAB, "bitwise_replay": True,
            "bitwise_replay_compared_values": 2 * VOCAB, "snapshot_stats_checks": 6,
            "snapshot_device_checks": 6, "snapshot_logit_compared_values": 4 * VOCAB,
            "ownership_verified": True, "steady_categories_counts_peak_host_and_payload_reads": True,
            "min_free_headroom_bytes": MIN_HEADROOM, "expert_payload_reads": GEOMETRY["expert_payload_reads"],
            "expert_payload_bytes_read": GEOMETRY["ram_expert_payload_bytes"],
            "caller_device_restored": True, "owned_bytes_recovered_both_devices": True,
        }, "passed": True,
    }, ("after_destruction", "cleanup", "peak_rss_bytes"), "complete")
    probes(obj["after_destruction"], "complete.after_destruction", origin["before_load"])
    integer(obj["peak_rss_bytes"], "complete.peak_rss_bytes", snapshots[-1]["peak_rss_bytes"])
    cleanup = obj["cleanup"]
    expect(cleanup, {
        "both_devices_synchronized": True, "device_reset_used": False,
        "caller_device_before": origin["caller_device"], "caller_device_after": origin["caller_device"],
    }, ("devices",), "complete.cleanup")
    require(type(cleanup["devices"]) is list and len(cleanup["devices"]) == 2,
            "complete.cleanup: expected both devices")
    for i, proof in enumerate(cleanup["devices"]):
        last = snapshots[-1]["devices"][i]
        after_free = obj["after_destruction"][i]["free_vram"]
        recovered = after_free - last["free_vram"]
        require(recovered >= last["owned_bytes"], f"complete.cleanup[{i}]: insufficient free-byte recovery")
        expect(proof, {
            "device": i, "before_load_free_vram": origin["before_load"][i]["free_vram"],
            "last_snapshot_free_vram": last["free_vram"], "after_destroy_free_vram": after_free,
            "required_owned_bytes": last["owned_bytes"], "free_recovered_bytes": recovered,
            "recovered_minus_owned_bytes": recovered - last["owned_bytes"],
            "min_snapshot_free_vram": min(s["devices"][i]["free_vram"] for s in snapshots),
            "owned_bytes_recovered": True,
        }, label=f"complete.cleanup[{i}]")


def collect(raw_path):
    """Return ONE r3c_memory record after full validation; never write inputs."""
    raw_path = Path(raw_path).resolve()
    rows = records(raw_path)
    origin, snapshots, footer = rows[0], rows[1:7], rows[7]
    source(origin)
    previous, previous_rss = None, origin["peak_rss_bytes"]
    for i, row in enumerate(snapshots):
        snapshot(row, i, origin["before_load"], snapshots[0] if i else None, previous, previous_rss)
        previous, previous_rss = row["stats"], row["peak_rss_bytes"]
    complete(footer, origin, snapshots)
    return {
        "kind": "r3c_memory", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": origin["revision"], "dirty": origin["dirty"], "runtime": RUNTIME,
        "model_variant": "qwen38-keep1-Q4_0", "model": origin["model_path"],
        "raw_logs": {"memory": str(raw_path)},
        "scope": "capacity allocation only: one own48 HIP Session, capacity131072, static24/24, slots112; two consumed teacher tokens per pass, reset and bitwise self-replay; not occupied128K qualification or inference speedup",
        "occupied_128k_qualified": False, "performance_claim": False, "mtp": False,
        "timing_scope": "last_completed_ms is finite nonnegative correctness metadata, not performance evidence",
        "expert_read_scope": "instrumented logical expert payload loads and byte counters, steady after load; not a syscall trace or physical SSD-read trace",
        "ram_scope": "reported host payload-buffer capacities and observed process RSS high-water; embedding payload is not exposed; no full-RAM accounting is inferred",
        "fixture_gates": {"min_free_headroom_bytes": MIN_HEADROOM, "frozen_before_execution": True,
                          "cleanup": "after_destroy_free_vram-last_snapshot_free_vram>=owned_bytes; no tolerance",
                          "replay": "two full-vocabulary bitwise self-replays"},
        "source": origin, "snapshots": snapshots, "complete": footer, "passed": True,
    }


def append_result(destination, record):
    """Serialize/validate before opening the journal; lock one complete append."""
    payload = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    destination = Path(destination).resolve()
    inputs = [Path(p).resolve() for p in record["raw_logs"].values()]
    inputs.append(Path(record["model"]).resolve())
    for path in inputs:
        require(destination != path and not (destination.exists() and path.exists() and os.path.samefile(destination, path)),
                "results would overwrite an input artifact")
    if destination.exists():
        require(stat.S_ISREG(destination.stat().st_mode), "results is not a regular file")
    with destination.open("ab+", buffering=0) as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        opened = os.fstat(output.fileno())
        require(stat.S_ISREG(opened.st_mode), "results is not a regular file")
        for path in inputs:
            require(not (path.exists() and os.path.samestat(opened, path.stat())),
                    "results would overwrite an input artifact")
        output.seek(0, os.SEEK_END)
        end = output.tell()
        if end:
            output.seek(-1, os.SEEK_END)
            require(output.read(1) == b"\n", "results has an incomplete final line")
        try:
            require(output.write(payload) == len(payload), "incomplete result append")
            output.flush()
        except (OSError, ValueError):
            output.truncate(end)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = collect(args.raw)
        append_result(args.results, record)
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print("record_memory: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
