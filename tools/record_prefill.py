#!/usr/bin/env python3
"""Validate frozen core-prefill-test protocol 1; append ONE short-PP record.

Self-parity only: four 40-row timelines, not independent HF, long PP, speed,
group>128, MTP or measured buffer release. --results is the explicit caller's
canonical journal path; there is deliberately no inferred/default journal.
"""

import argparse
import datetime
import json
import math
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
RECORD_COUNT = 92
CAPACITY, TEACHER_ROWS, VOCAB, MAX_BATCH = 40, 32, 248320, 32
ABSOLUTE_GATE, RELATIVE_GATE = .02, .002
Q40, Q41 = 2764800, 2867200
RAM_PAYLOAD = (6 * Q41 + 42 * Q40) * 512
HOST_LOGIT_BYTES = MAX_BATCH * VOCAB * 4
PINNED_HANDOFF = MAX_BATCH * 4 * 2560 * 4
STAGE_BYTES = 16 * Q41
PINNED_STAGING = 4 * STAGE_BYTES
WORKSPACE_FLOOR = (24 * MAX_BATCH * 12288 * 4 + MAX_BATCH * 320 * 36 +
                   HOST_LOGIT_BYTES + MAX_BATCH * 10 * 2560 * 4 + 2 * STAGE_BYTES)
PHASES = ("reference_n1", "chunk4", "chunk32", "occupied_prefix5_chunk17_remainder10")
RUNTIME = "own_48_layer_HIP"
COUNTERS = ("expert_hits", "expert_misses", "expert_upload_bytes")
ROUTE_FIELDS = ("hits", "misses", "upload_bytes")
CATEGORIES = ("weights", "expert_slots", "qsa_kv", "qsa_index", "gdn_state", "ple_state", "workspace")
ZERO_STATS = {"consumed_tokens": 0, **dict.fromkeys(COUNTERS, 0),
              "last_completed_ms": 0, "last_completed_ms_bits": 0}


def json_integer(value):
    # The ostream spells negative-zero doubles as -0; do not lose their bits.
    return -0.0 if value == "-0" else int(value)


def records(path):
    """Bound even a growing regular input; accept only 92 finished JSON objects."""
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), str(path) + ": not a regular file")
    require(info.st_size <= MAX_RAW_BYTES, str(path) + ": oversized input")
    with path.open("rb") as stream:
        raw = stream.read(MAX_RAW_BYTES + 1)
    require(len(raw) <= MAX_RAW_BYTES, str(path) + ": oversized input")
    require(raw.endswith(b"\n"), str(path) + ": missing completed final line")
    lines = raw.split(b"\n")[:-1]
    require(len(lines) == RECORD_COUNT, str(path) + ": expected exactly 92 JSONL rows")
    rows = []
    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line.decode("utf-8"), object_pairs_hook=unique_object,
                             parse_int=json_integer, parse_float=finite_float,
                             parse_constant=invalid_constant)
            require(type(obj) is dict, "expected an object")
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{path}:row {i}: {error}") from error
        rows.append(obj)
    return rows


def source(obj):
    expect(obj, {
        "kind": "prefill_source", "protocol": 1, "runtime": RUNTIME,
        "model_source": "caller_supplied_GGUF_path_no_checksum_attestation",
        "config": {"capacity": 40, "expert_slots": 1, "max_batch_tokens": 32, "trace": False},
        "teacher_ids": [248044, *range(100, 131)], "continuation_ids": list(range(131, 139)),
        "phase_order": list(PHASES), "continuation_schedule": "eight_N1_calls_each_phase",
        "occupied_prefix_schedule": "five_N1_calls", "chunk4_teacher_schedule": "eight_N4_calls",
        "gate": {"absolute": ABSOLUTE_GATE, "relative": RELATIVE_GATE,
                 "formula": "abs(actual-reference)<=.02+.002*abs(reference)",
                 "all_finite_required": True, "allowed_violations": 0,
                 "bit_equality_required": False, "argmax_equality_required": False, "cli_adjustable": False},
        "reference_scope": "all_40_full_vocabulary_rows_retained_sequential_N1_same_Session",
        "weight_values": "unchanged_loaded_GGUF", "weight_precision": "unchanged_loaded_tensor_types",
        "kv": "Q4_0_K_and_V", "sampling": "teacher_forced", "session_instances": 1,
        "reset_clears_logical_state_and_statistics": True, "reset_retains_expert_cache": True,
        "cold_cache_equality_claim": False,
        "candidate_scope": "first_end_to_end_PP_gate_current_unqualified_candidate",
        "independent_HF_reference": False, "R4_complete_claim": False,
        "qualification_4K_16K_claim": False, "performance_claim": False,
        "timing_scope": "diagnostic_completed_full_window_wall_time_excludes_comparison_and_memory_snapshots",
        "counter_scope": "480_assignments_per_row_hits_include_within_call_group_reuse_misses_count_uploaded_payloads",
        "read_counter_scope": "constructor_expert_payload_reads_not_physical_SSD_syscall_trace",
        "geometry": {"logical_candidate_limit": 1024, "projection_expert_microtile_limit": 128,
                     "expert_band": 16, "stages_per_device": 2,
                     "pinned_expert_staging_bytes": PINNED_STAGING, "pinned_handoff_bytes": PINNED_HANDOFF,
                     "host_logit_min_bytes": HOST_LOGIT_BYTES,
                     "workspace_aggregate_min_bytes_per_device": WORKSPACE_FLOOR,
                     "individual_workspace_geometry_observable": False, "free_vram_equality_required": False,
                     "memory_scope": "Session_owned_Buffer_ledger_and_reported_host_capacities_excludes_allocator_metadata"},
        "preallocated_reference_bytes": CAPACITY * VOCAB * 4,
        "preallocated_preservation_snapshot_bytes": HOST_LOGIT_BYTES, "jsonl_after_session_cleanup": True,
    }, ("revision", "dirty", "model", "model_bytes"), "source")
    revision = obj["revision"]
    require(type(revision) is str and len(revision) == 40 and
            all(c in "0123456789abcdefABCDEF" for c in revision), "source: invalid compiled revision")
    require(type(obj["dirty"]) is bool, "source: invalid boolean dirty")
    model = Path(text(obj["model"], "source.model"))
    require(model.is_absolute() and model.name == "qwen38-keep1-Q4_0.gguf",
            "source: expected absolute keep1-Q4_0 model path")
    integer(obj["model_bytes"], "source.model_bytes", 1, (1 << 63) - 1)


def layout(index):
    widths = (([1] * 32), ([4] * 8), [32], ([1] * 5 + [17, 10]))[index] + [1] * 8
    offset, shapes = 0, []
    for width in widths:
        shapes.append((offset, width))
        offset += width
    return shapes


def stats(obj, consumed, label):
    expect(obj, {"consumed_tokens": consumed}, (*COUNTERS, "last_completed_ms", "last_completed_ms_bits"), label)
    for key in COUNTERS:
        integer(obj[key], label + "." + key)
    timing = number(obj["last_completed_ms"], label + ".last_completed_ms")
    bits = integer(obj["last_completed_ms_bits"], label + ".last_completed_ms_bits")
    exact(bits, struct.unpack("!Q", struct.pack("!d", timing))[0], label + ".timing_bits")
    if consumed == 0:
        for key in (*COUNTERS, "last_completed_ms_bits"):
            exact(obj[key], 0, label + ".reset." + key)
    else:
        require(timing > 0, label + ": completed time is not positive")
    require(obj["expert_hits"] + obj["expert_misses"] == 480 * consumed,
            label + ": cumulative routes are not 480*consumed")


def same_stats(actual, expected, label):
    stats(actual, expected["consumed_tokens"], label)
    for key in ("consumed_tokens", *COUNTERS, "last_completed_ms_bits"):
        exact(actual[key], expected[key], label + "." + key)


def routes(obj, rows, label):
    expect(obj, {}, ROUTE_FIELDS, label)
    for key in ROUTE_FIELDS:
        integer(obj[key], label + "." + key)
    require(obj["hits"] + obj["misses"] == 480 * rows, label + ": routes are not 480*N")
    require(obj["misses"] * Q40 <= obj["upload_bytes"] <= obj["misses"] * Q41,
            label + ": upload bytes outside unchanged miss payload range")


def memory(obj, label, loaded=None):
    expect(obj, {"capacity": CAPACITY, "expert_slots": 1, "ownership_verified": True,
                 "ram_expert_payload": RAM_PAYLOAD, "pinned_handoff": PINNED_HANDOFF,
                 "pinned_expert_staging": PINNED_STAGING, "expert_payload_reads": 144,
                 "expert_payload_bytes_read": RAM_PAYLOAD},
           ("ram_expert_capacity", "host_embedding_capacity", "host_logit_capacity", "devices"), label)
    integer(obj["ram_expert_capacity"], label + ".ram_expert_capacity", RAM_PAYLOAD)
    integer(obj["host_embedding_capacity"], label + ".host_embedding_capacity", 1)
    integer(obj["host_logit_capacity"], label + ".host_logit_capacity", HOST_LOGIT_BYTES)
    devices = obj["devices"]
    require(type(devices) is list and len(devices) == 2, label + ": expected both static owners")
    for i, d in enumerate(devices):
        where = f"{label}.devices[{i}]"
        expect(d, {"device": i, "first_layer": 24 * i, "last_layer": 24 * i + 23,
                   "gdn_layers": 18, "qsa_layers": 6,
                   "expert_slots": (6 * Q41 + 18 * Q40) if i == 0 else 24 * Q40,
                   "qsa_kv": 6 * CAPACITY * 32 * 18,
                   "qsa_index": 6 * (CAPACITY // 4 * 128 + 384) * 4,
                   "gdn_state": 18 * (786432 + 30720) * 4, "ple_state": 368640 if i == 0 else 0},
               ("weights", "workspace", "owned_bytes", "owned_peak_bytes", "owned_buffers", "total_vram", "free_vram"), where)
        integer(d["weights"], where + ".weights", 1)
        integer(d["workspace"], where + ".workspace", WORKSPACE_FLOOR)
        owned = integer(d["owned_bytes"], where + ".owned_bytes", 1)
        exact(owned, sum(d[key] for key in CATEGORIES), where + ".category_sum")
        integer(d["owned_peak_bytes"], where + ".owned_peak_bytes", owned)
        integer(d["owned_buffers"], where + ".owned_buffers", 1)
        total = integer(d["total_vram"], where + ".total_vram", 1)
        free = integer(d["free_vram"], where + ".free_vram", 1, total)
        require(owned <= total - free, where + ": owned bytes exceed observed used VRAM")
        if loaded is not None:
            for key in d.keys() - {"free_vram"}:
                exact(d[key], loaded["devices"][i][key], where + ".steady." + key)
    if loaded is not None:
        for key in obj.keys() - {"devices"}:
            exact(obj[key], loaded[key], label + ".steady." + key)


def coordinate(obj, offset, rows, label):
    expect(obj, {}, ("position", "vocabulary_index"), label)
    integer(obj["position"], label + ".position", offset, offset + rows - 1)
    integer(obj["vocabulary_index"], label + ".vocabulary_index", 0, VOCAB - 1)


def close(actual, expected, label):
    # Only roundoff in emitted summary arithmetic, never slack on the logit gate.
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label + ": inconsistent sum")


def leq(actual, bound, label, terms=0):
    # Positive double sums in inspect_logits may contain up to 32*VOCAB terms.
    # Allow their bounded accumulation roundoff, not a new acceptance tolerance.
    roundoff = max(1e-12, (terms + 4) * sys.float_info.epsilon)
    require(actual <= bound or math.isclose(actual, bound, rel_tol=roundoff, abs_tol=0),
            label + ": inconsistent metric")


def errors(obj, offset, rows, reference, label):
    compared = reference is not None
    count = rows * VOCAB
    expect(obj, {
        "values": count, "finite_values": count, "nonfinite_actual": 0, "nonfinite_reference": 0,
        "compared_values": count if compared else 0, "finite_pairs": count if compared else 0,
        "metric_scope": "finite_pairs_only_nonfinite_pairs_also_violate", "violations": 0,
        "bit_equality_required": False, "argmax_equality_required": False,
        "argmax_compared_rows": rows if compared else 0, "first_violation_coordinate": None,
    }, ("actual_maxabs", "actual_rms", "maxabs", "rms", "maxboundratio", "bit_mismatches",
        "argmax_agree_rows", "maxabs_coordinate", "maxboundratio_coordinate"), label)
    amax = number(obj["actual_maxabs"], label + ".actual_maxabs")
    arms = number(obj["actual_rms"], label + ".actual_rms")
    require(amax <= float.fromhex("0x1.fffffep+127"), label + ": actual norm exceeds finite FP32 range")
    leq(arms, amax, label + ".actual_rms<=max", count)
    leq(amax / math.sqrt(count), arms, label + ".actual_rms>=one_max", count)
    integer(obj["bit_mismatches"], label + ".bit_mismatches", 0, count if compared else 0)
    integer(obj["argmax_agree_rows"], label + ".argmax_agree_rows", 0, rows if compared else 0)
    if not compared:
        for key in ("maxabs", "rms", "maxboundratio", "maxabs_coordinate", "maxboundratio_coordinate"):
            exact(obj[key], None, label + "." + key)
        if offset == 0:
            require(amax > 0 and arms > 0, label + ": first reference N1 smoke has no positive finite logits")
        return
    for key in ("maxabs", "rms", "maxboundratio"):
        number(obj[key], label + "." + key)
    for key in ("maxabs_coordinate", "maxboundratio_coordinate"):
        coordinate(obj[key], offset, rows, label + "." + key)
    maximum, rms, ratio = (obj[key] for key in ("maxabs", "rms", "maxboundratio"))
    rmax = max(w["errors"]["actual_maxabs"] for w in reference)
    # N1 row RMS is emitted with max_digits10. hypot avoids squaring overflow.
    rrms = math.hypot(*(w["errors"]["actual_rms"] for w in reference)) / math.sqrt(rows)
    bound = ABSOLUTE_GATE + RELATIVE_GATE * rmax
    require(ratio <= 1 and maximum <= bound, label + ": frozen full-output gate failed")
    leq(rms, maximum, label + ".rms<=max", count)
    leq(maximum / math.sqrt(count), rms, label + ".rms>=one_max", count)
    leq(rms, maximum * math.sqrt(obj["bit_mismatches"] / count), label + ".rms_vs_mismatch_count", count)
    leq(maximum / bound, ratio, label + ".ratio_lower")
    leq(ratio, maximum / ABSOLUTE_GATE, label + ".ratio_upper")
    leq(abs(amax - rmax), maximum + 1e-12 * max(amax, rmax), label + ".actual_reference_max_norm")
    # Candidate sums span a whole window; retained N1 summaries sum each row
    # separately. Even identical logit bits need not give identical double RMS.
    rms_roundoff = (count + VOCAB + 8) * sys.float_info.epsilon * max(arms, rrms)
    leq(abs(arms - rrms), rms + rms_roundoff, label + ".actual_reference_rms_norm")
    if maximum == 0:
        exact(ratio == 0 and rms == 0, True, label + ".zero_error_metrics")
        exact(obj["argmax_agree_rows"], rows, label + ".zero_error_argmax")
        for key in ("maxabs_coordinate", "maxboundratio_coordinate"):
            exact(obj[key], {"position": offset, "vocabulary_index": 0}, label + ".zero_error." + key)
    else:
        require(obj["bit_mismatches"] > 0 and rms > 0 and ratio > 0, label + ": positive error missing diagnostic counts")
    # Different signed-zero bits can mismatch with zero numerical error. Neither
    # bit identity nor argmax identity is an acceptance gate for nonzero error.


def window(obj, index, position, shape, previous, reference, loaded):
    offset, rows = shape
    label = f"{PHASES[index]}.window[{position}]"
    expect(obj, {
        "kind": "prefill_window", "protocol": 1, "phase_index": index, "phase": PHASES[index],
        "window_index": position, "offset": offset, "rows": rows,
        "segment": "teacher" if offset < TEACHER_ROWS else "continuation", "reference": index == 0,
        "completed_call": True, "expected_routes": 480 * rows,
        "memory_stats_and_full_active_span_preserved": True,
        "first_input_n1_smoke": index == 0 and offset == 0, "passed": True, "failure": None,
    }, ("completed_window_wall_ms", "stats_before", "stats_after", "route_delta",
        "retained_n1_route_delta", "errors", "memory_snapshot"), label)
    require(number(obj["completed_window_wall_ms"], label + ".wall_ms") > 0, label + ": nonpositive wall time")
    same_stats(obj["stats_before"], previous, label + ".stats_before")
    stats(obj["stats_after"], offset + rows, label + ".stats_after")
    routes(obj["route_delta"], rows, label + ".route_delta")
    for counter, key in zip(COUNTERS, ROUTE_FIELDS):
        exact(obj["stats_after"][counter], previous[counter] + obj["route_delta"][key], label + "." + counter)
    routes(obj["retained_n1_route_delta"], rows, label + ".retained_n1_route_delta")
    retained = obj["route_delta"] if index == 0 else {
        key: sum(w["route_delta"][key] for w in reference[offset:offset + rows]) for key in ROUTE_FIELDS}
    exact(obj["retained_n1_route_delta"], retained, label + ".matching_N1_routes")
    errors(obj["errors"], offset, rows, None if index == 0 else reference[offset:offset + rows], label + ".errors")
    memory(obj["memory_snapshot"], label + ".memory_snapshot", loaded)


def invalid_proofs(proofs, windows, index):
    label = PHASES[index] + ".invalid_proofs"
    require(type(proofs) is list and len(proofs) == (5 if index else 0), label + ": invalid proof count")
    if not index:
        return 0
    prior_index = next(i for i, w in enumerate(windows) if w["rows"] > 1)
    cases = [("empty", 0, prior_index), ("length33", 33, prior_index),
             ("negative_last", 3, prior_index), ("oov_last", 3, prior_index),
             ("capacity39_length2", 2, len(windows) - 2)]
    preserved = 0
    for i, (proof, (name, input_rows, prior_index)) in enumerate(zip(proofs, cases)):
        prior, continuation = windows[prior_index:prior_index + 2]
        count = prior["rows"] * VOCAB
        where = f"{label}[{i}]"
        expect(proof, {
            "case": name, "input_rows": input_rows, "offset": prior["offset"] + prior["rows"],
            "prior_rows": prior["rows"], "preserved_fullspan_values": count, "exception": "invalid_argument",
            "all_public_stats_bitwise_preserved": True, "fullspan_bits_preserved": True,
            "memory_ledger_preserved": True, "pinned_expert_staging_preserved": True,
            "free_vram_equality_required": False, "consumed_increment": 0, "continued_without_reset": True,
            "continuation_offset": continuation["offset"], "continuation_rows": continuation["rows"],
            "continuation_compared_values": continuation["rows"] * VOCAB, "continuation_violations": 0,
        }, ("stats_before_and_after",), where)
        same_stats(proof["stats_before_and_after"], prior["stats_after"], where + ".stats_before_and_after")
        # The next accepted window already has matching before-stats, full-logit
        # comparisons and a steady memory snapshot; no reset row may intervene.
        preserved += count
    return preserved


def phase(obj, index, windows):
    label = PHASES[index]
    expect(obj, {
        "kind": "prefill_phase", "protocol": 1, "phase_index": index, "phase": label,
        "windows": len(windows), "teacher_rows": 32, "continuation_rows": 8,
        "finite_logit_values": CAPACITY * VOCAB, "compared_logit_values": CAPACITY * VOCAB if index else 0,
        "bit_mismatches_diagnostic": sum(w["errors"]["bit_mismatches"] for w in windows), "violations": 0,
        "timing_includes_rejection_reset_comparison_memory": False, "passed": True,
    }, ("completed_window_wall_ms_sum", "teacher_stats", "final_stats", "invalid_proofs"), label)
    close(number(obj["completed_window_wall_ms_sum"], label + ".wall_sum"),
          sum(w["completed_window_wall_ms"] for w in windows), label + ".wall_sum")
    teacher = next(w for w in windows if w["offset"] + w["rows"] == TEACHER_ROWS)
    same_stats(obj["teacher_stats"], teacher["stats_after"], label + ".teacher_stats")
    same_stats(obj["final_stats"], windows[-1]["stats_after"], label + ".final_stats")
    return invalid_proofs(obj["invalid_proofs"], windows, index)


def collect(raw_path):
    """Return one r4b_short_prefill after complete validation; no file writes."""
    raw_path = Path(raw_path).resolve()
    rows = records(raw_path)
    origin, loaded_row, footer = rows[0], rows[1], rows[-1]
    source(origin)
    expect(loaded_row, {"kind": "prefill_loaded_memory", "protocol": 1, "passed": True},
           ("stats", "memory"), "loaded")
    same_stats(loaded_row["stats"], ZERO_STATS, "loaded.stats")
    loaded = loaded_row["memory"]
    memory(loaded, "loaded.memory")
    resets, phases, windows, reference = [], [], [], None
    memories = [loaded]
    cursor, preserved = 2, 0
    for index, name in enumerate(PHASES):
        reset = rows[cursor]
        cursor += 1
        expect(reset, {"kind": "prefill_reset", "protocol": 1, "phase_index": index,
                       "phase": name, "expert_cache_retained": True, "passed": True},
               ("stats", "memory"), name + ".reset")
        same_stats(reset["stats"], ZERO_STATS, name + ".reset.stats")
        memory(reset["memory"], name + ".reset.memory", loaded)
        memories.append(reset["memory"])
        resets.append(reset)
        shapes = layout(index)
        current = rows[cursor:cursor + len(shapes)]
        cursor += len(shapes)
        previous = reset["stats"]
        for position, (obj, shape) in enumerate(zip(current, shapes)):
            window(obj, index, position, shape, previous, reference, loaded)
            previous = obj["stats_after"]
            memories.append(obj["memory_snapshot"])
        if index == 0:
            reference = current
        summary = rows[cursor]
        cursor += 1
        preserved += phase(summary, index, current)
        phases.append(summary)
        windows.extend(current)
    final = rows[cursor]
    cursor += 1
    require(cursor == len(rows) - 1, "incorrect protocol order/count")
    expect(final, {"kind": "prefill_final_reset", "protocol": 1, "passed": True},
           ("stats", "memory"), "final_reset")
    same_stats(final["stats"], ZERO_STATS, "final_reset.stats")
    memory(final["memory"], "final_reset.memory", loaded)
    memories.append(final["memory"])
    expect(footer, {
        "kind": "prefill_complete", "protocol": 1, "phase_count": 4, "record_count": RECORD_COUNT,
        "timeline_rows": 160, "teacher_rows": 128, "continuation_rows": 32, "window_count": 80,
        "finite_logit_values": 160 * VOCAB, "compared_logit_values": 120 * VOCAB,
        "violations": 0, "bit_mismatches_diagnostic": sum(w["errors"]["bit_mismatches"] for w in windows),
        "bit_equality_required": False, "invalid_window_rejections": 15,
        "rejection_preserved_fullspan_values": preserved,
        "memory_snapshot_count": len(memories) + 2 * 15,
        "memory_preserved_fullspan_values": 160 * VOCAB + 2 * preserved,
        "steady_categories_allocated_bytes_counts_peak_host_pinned_and_payload_reads": True,
        "session_instances": 1, "raii_session_cleanup_completed": True, "owned_buffer_release_measured": False,
        "reference_scope": "same_session_N1_self_parity", "first_PP_gate_passed": True,
        "R4_complete_claim": False, "qualification_4K_16K_claim": False, "independent_HF_reference": False,
        "performance_claim": False, "passed": True,
    }, ("completed_window_wall_ms_sum", "minimum_free_vram_bytes"), "complete")
    close(number(footer["completed_window_wall_ms_sum"], "complete.wall_sum"),
          sum(w["completed_window_wall_ms"] for w in windows), "complete.wall_sum")
    minima = footer["minimum_free_vram_bytes"]
    require(type(minima) is list and len(minima) == 2, "complete: expected both free VRAM minima")
    for i, minimum in enumerate(minima):
        # Thirty rejection observations are summarized, not serialized. The
        # reported minimum may be below every one of the 86 visible snapshots.
        upper = min(m["devices"][i]["free_vram"] for m in memories)
        integer(minimum, f"complete.minimum_free_vram_bytes[{i}]", 1, upper)
    return {
        "kind": "r4b_short_prefill", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": origin["revision"], "dirty": origin["dirty"], "runtime": RUNTIME,
        "model_variant": "qwen38-keep1-Q4_0", "model": origin["model"], "raw_logs": {"prefill": str(raw_path)},
        "scope": "qualified short PP self-parity: retained same-Session N1 reference40, chunk4/chunk32/occupied-prefix5 then17+10, eight N1 continuations each; not full R4, independent HF, 4K/16K, group>128, performance, MTP or release-recovery qualification",
        "short_pp_qualified": True, "R4_complete_claim": False, "qualification_4K_16K_claim": False,
        "group_over_128_qualified": False, "independent_hf_claim": False, "performance_claim": False, "mtp": False,
        "timing_scope": "diagnostic completed caller window wall sums and last_completed_ms; excludes comparison/memory/reset/rejection work; not performance evidence",
        "expert_read_scope": "instrumented constructor payload reads and bytes, steady after load; not a syscall trace or physical SSD-read trace",
        "memory_scope": "116 observations, 86 serialized snapshots plus 30 rejection observations summarized by preservation proofs; steady Session-owned categories/counts/peaks and reported host/pinned capacities; workspace aggregate floor only, no individual-buffer-size proof or full-RAM accounting; RAII cleanup reported, no measured owned-buffer release/recovery",
        "fixture_gates": {"frozen_before_execution": True, "absolute": ABSOLUTE_GATE, "relative": RELATIVE_GATE,
                          "formula": origin["gate"]["formula"], "all_finite_required": True, "allowed_violations": 0,
                          "bit_equality_required": False, "argmax_equality_required": False,
                          "workspace": "aggregate_floor_only", "owned_buffer_release_measured": False},
        "source": origin, "loaded_memory": loaded_row, "resets": resets, "windows": windows,
        "phases": phases, "final_reset": final, "complete": footer,
        "records": rows, "passed": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True,
                        help="explicit canonical journal path supplied by caller")
    args = parser.parse_args(argv)
    try:
        append_result(args.results, collect(args.raw))
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print("record_prefill: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
