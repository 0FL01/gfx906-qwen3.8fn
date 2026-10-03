#!/usr/bin/env python3
"""Validate core-prefill-long-test protocol 1; append ONE r4b_long_prefill.

Exactly 93 (4096 teacher) or 129 (16384 teacher) finished records: source,
loaded memory, three (reset, checkpoints/windows, phase), final-reset memory,
cleanup-complete. Reference N1 calls are checked in-process and compressed at
1024-row boundaries; candidates use 1024 chunks and prefix5 then997 chunks.
This is same-Session full-logit self-parity, not independent HF, performance,
full R4, MTP, physical expert tiles, individual buffers or full-RAM accounting.
--results is the explicit caller's canonical ROOT/results.jsonl, never inferred.
"""

import argparse
import datetime
import json
import math
from pathlib import Path
import stat
import sys

if __package__:
    from .record_memory import (append_result as append_record, exact, expect,
                                finite_float, integer, invalid_constant, number,
                                require, text, unique_object)
    from .record_prefill import close, json_integer, leq, same_stats, stats
else:
    from record_memory import (append_result as append_record, exact, expect,
                               finite_float, integer, invalid_constant, number,
                               require, text, unique_object)
    from record_prefill import close, json_integer, leq, same_stats, stats


MAX_RAW_BYTES = 16 * 1024 * 1024
TEACHERS = (4096, 16384)
VOCAB, MAX_BATCH, SLOTS, CONTINUATION = 248320, 1024, 112, 32
Q40, Q41 = 2764800, 2867200
RAM_PAYLOAD = (6 * Q41 + 42 * Q40) * 512
HOST_LOGIT_BYTES = MAX_BATCH * VOCAB * 4
PINNED_HANDOFF, PINNED_STAGING = MAX_BATCH * 4 * 2560 * 4, 4 * 16 * Q41
WORKSPACE_FLOOR = (24 * MAX_BATCH * 12288 * 4 + MAX_BATCH * 320 * 36 +
                   HOST_LOGIT_BYTES + MAX_BATCH * 10 * 2560 * 4 + 2 * 16 * Q41)
FLOAT_GUARD_BYTES, INPUT_WORKSPACE_BYTES = 256, (MAX_BATCH + 1 + 32) * 4
REJECTION_VALUES = (3 * (1024 + 997) + 2) * VOCAB
PHASES = ("reference_n1", "canonical1024", "occupied5_then997")
COUNTERS = ("expert_hits", "expert_misses", "expert_upload_bytes")
GROUP_FIELDS = ("last_max_expert_group_assignments", "expert_groups_gt128")
CATEGORIES = ("weights", "expert_slots", "qsa_kv", "qsa_index", "gdn_state", "ple_state", "workspace")
ZERO_STATS = {"consumed_tokens": 0, **dict.fromkeys(COUNTERS, 0),
              "last_completed_ms": 0, "last_completed_ms_bits": 0}
ZERO_GROUPS = dict.fromkeys(GROUP_FIELDS, 0)
ERROR_COUNTS = ("values", "finite_values", "compared_values", "finite_pairs", "nonfinite_actual",
                "nonfinite_reference", "violations", "bit_mismatches_diagnostic",
                "argmax_rows_diagnostic", "argmax_agree_diagnostic")
ERROR_METRICS = ("maxabs", "rms", "maxboundratio")
ERROR_COORDINATES = ("maxabs_coordinate", "maxboundratio_coordinate")
VIOLATION_FIELDS = ("first_violation_coordinate", "first_violation_actual",
                    "first_violation_reference", "first_violation_bound")
FP32_MAX = float.fromhex("0x1.fffffep+127")
COVERAGE_SCOPE = "actual_accepted_teacher_rows_logical_causal_visibility_not_GPU_selected_ID_trace"


def counts(teacher):
    total = teacher + CONTINUATION
    mixed = (teacher - 5 + 996) // 997
    windows = [total, teacher // MAX_BATCH + CONTINUATION, 5 + mixed + CONTINUATION]
    return total, mixed, windows, 10 + teacher // MAX_BATCH + 1 + sum(windows[1:])


def records(path):
    """Bound before and during reading; never read an unbounded/growing log."""
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), str(path) + ": not a regular file")
    require(info.st_size <= MAX_RAW_BYTES, str(path) + ": oversized input")
    with path.open("rb") as stream:
        raw = stream.read(MAX_RAW_BYTES + 1)
    require(len(raw) <= MAX_RAW_BYTES, str(path) + ": oversized input")
    require(raw.endswith(b"\n"), str(path) + ": missing completed final line")
    lines = raw.split(b"\n")[:-1]
    require(len(lines) in (93, 129), str(path) + ": expected exactly 93 or 129 JSONL rows")
    result = []
    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line.decode("utf-8"), object_pairs_hook=unique_object,
                             parse_int=json_integer, parse_float=finite_float,
                             parse_constant=invalid_constant)
            require(type(obj) is dict, "expected an object")
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{path}:row {i}: {error}") from error
        result.append(obj)
    return result


def source(obj):
    require(type(obj) is dict, "source: expected object")
    teacher = obj.get("teacher_id_count")
    require(type(teacher) is int and teacher in TEACHERS, "source: expected 4096 or 16384 teacher rows")
    total, mixed, windows, record_count = counts(teacher)
    reference_bytes = total * VOCAB * 4
    expect(obj, {
        "kind": "prefill_long_source", "protocol": 1, "runtime": "own_48_layer_HIP",
        "session_instances": 1,
        "config": {"capacity": total, "expert_slots": SLOTS, "max_batch_tokens": MAX_BATCH, "trace": False},
        "large_RAM_opt_in": "explicit_ROWS_4096_or_16384",
        "teacher_family": "new_BOS_then_monotone_teacher_IDs_not_original_user_prompt_or_baseline_parity",
        "source_ID_formula": "id[p]=248044 if p==0 else 99+p; 0<=p<ROWS+32",
        "source_id_count": total, "teacher_id_count": teacher, "teacher_BOS": 248044,
        "teacher_non_BOS_first": 100, "teacher_last": teacher + 98,
        "continuation_id_count": CONTINUATION, "continuation_first": teacher + 99,
        "continuation_last": total + 98, "every_source_ID_compile_time_and_runtime_checked": True,
        "expected_windows_per_phase": windows, "phase_order": list(PHASES),
        "teacher_schedules": ["ROWS_N1", "ROWS/1024_N1024", "5_N1_then_ceil((ROWS-5)/997)_chunks_final_remainder"],
        "mixed_final_remainder": teacher - 5 - (mixed - 1) * 997,
        "continuation_schedule": "32_N1_each_phase", "baseline_checkpoint_count": teacher // MAX_BATCH + 1,
        "expected_success_records": record_count, "expected_success_windows": sum(windows),
        "expected_finite_values": 3 * total * VOCAB, "expected_compared_values": 2 * total * VOCAB,
        "expected_memory_observations": sum(windows) + 21, "expected_invalid_rejections": 8,
        "gate": {"absolute": .02, "relative": .002,
                 "formula": "abs(actual-reference)<=.02+.002*abs(reference)",
                 "all_finite_required": True, "allowed_violations": 0, "bit_equality_required": False,
                 "argmax_equality_required": False, "cli_adjustable": False},
        "reference_scope": "all_ROWS_plus32_full_vocabulary_logits_retained_sequential_N1_same_Session",
        "vocabulary": VOCAB, "weights": "unchanged_loaded_GGUF_values_and_tensor_types",
        "kv": "Q4_0_K_and_V", "sampling": "teacher_forced", "reset_retains_expert_cache": True,
        "cold_cache_equality_claim": False, "independent_HF_reference": False,
        "timing_scope": "diagnostic_correctness_completed_call_wall_excludes_comparison_memory_reset_rejection_not_PP_or_full_request_performance",
        "counter_scope": "480_assignments_per_row_hits_include_within_call_reuse_misses_count_uploaded_groups",
        "uploaded_group_payload_min_bytes": Q40, "uploaded_group_payload_max_bytes": Q41,
        "expert_payload_reads_required": 144, "expert_payload_bytes_read_required": RAM_PAYLOAD,
        "read_counter_scope": "constructor_payload_reads_not_physical_SSD_trace",
        "preallocated_reference_bytes": reference_bytes, "preallocated_preservation_bytes": HOST_LOGIT_BYTES,
        "float_guard_elements_each_end": 16, "float_guard_total_bytes": FLOAT_GUARD_BYTES,
        "guarded_input_workspace_bytes": INPUT_WORKSPACE_BYTES,
        "known_host_min_bytes_before_embedding_metadata_runtime": RAM_PAYLOAD + reference_bytes +
        2 * HOST_LOGIT_BYTES + PINNED_HANDOFF + PINNED_STAGING + FLOAT_GUARD_BYTES + INPUT_WORKSPACE_BYTES,
        "host_accounting": "expert_RAM_plus_reference_plus_preservation_plus_Session_logits_plus_pinned_handoff_and_expert_staging_no_weight_copy",
        "workspace_aggregate_min_bytes_per_device": WORKSPACE_FLOOR,
        "individual_buffer_capacities_observable": False, "physical_128_column_tile_proven": False,
        "coverage_scope": "actual_teacher_rows_and_call_offsets_cover_visible2047..2056_each_mod4_not_selected_ID_trace_or_synthetic_QKV",
        "performance_claim": False, "peak_VRAM_qualification_claim": False, "R4_complete_claim": False,
        "remaining_R4_evidence": "separate_full_request_512_output_benchmark_and_peak_VRAM_qualification",
    }, ("revision", "dirty", "model", "model_bytes"), "source")
    revision = obj["revision"]
    require(type(revision) is str and len(revision) == 40 and
            all(c in "0123456789abcdefABCDEF" for c in revision), "source: invalid compiled revision")
    require(type(obj["dirty"]) is bool, "source: invalid boolean dirty")
    model = Path(text(obj["model"], "source.model"))
    require(model.is_absolute() and model.name == "qwen38-keep1-Q4_0.gguf",
            "source: expected absolute keep1-Q4_0 model path")
    integer(obj["model_bytes"], "source.model_bytes", 1, (1 << 63) - 1)
    return teacher


def layout(teacher, index, compressed=False):
    if index == 0:
        widths = [MAX_BATCH] * (teacher // MAX_BATCH) + [CONTINUATION] if compressed else [1] * (teacher + CONTINUATION)
    elif index == 1:
        widths = [MAX_BATCH] * (teacher // MAX_BATCH) + [1] * CONTINUATION
    else:
        mixed = (teacher - 5 + 996) // 997
        widths = [1] * 5 + [997] * (mixed - 1) + [teacher - 5 - (mixed - 1) * 997] + [1] * CONTINUATION
    offset, shapes = 0, []
    for width in widths:
        shapes.append((offset, width))
        offset += width
    return shapes


def group_stats(obj, label):
    expect(obj, {}, GROUP_FIELDS, label)
    for key in GROUP_FIELDS:
        integer(obj[key], label + "." + key)


def same_groups(obj, previous, label):
    group_stats(obj, label)
    exact(obj, previous, label)


def completed_stats(obj, previous, offset, rows, label):
    stats(obj, offset + rows, label)
    deltas = [integer(obj[key] - previous[key], label + ".delta." + key) for key in COUNTERS]
    hits, misses, uploaded = deltas
    require(hits <= 480 * rows and misses <= 480 * rows and hits + misses == 480 * rows,
            label + ": routes are not 480*N")
    require(misses * Q40 <= uploaded <= misses * Q41, label + ": upload bytes outside unchanged per-miss payload range")


def completed_groups(obj, previous, rows, label):
    group_stats(obj, label)
    maximum = integer(obj[GROUP_FIELDS[0]], label + ".maximum", 1, rows)
    delta = integer(obj[GROUP_FIELDS[1]] - previous[GROUP_FIELDS[1]], label + ".delta", 0, 480 * rows // 129)
    exact(delta > 0, maximum > 128, label + ".threshold_max_consistency")
    return maximum, delta


def coordinate(obj, offset, rows, label):
    expect(obj, {}, ("position", "vocabulary_index"), label)
    integer(obj["position"], label + ".position", offset, offset + rows - 1)
    integer(obj["vocabulary_index"], label + ".vocabulary_index", 0, VOCAB - 1)


def errors(obj, offset, rows, compared_rows, label, values_rows=None):
    """Validate the emitted comparison metrics, not unavailable reference norms."""
    values = (rows if values_rows is None else values_rows) * VOCAB
    pairs = compared_rows * VOCAB
    expect(obj, {"values": values, "finite_values": values, "compared_values": pairs, "finite_pairs": pairs,
                 "nonfinite_actual": 0, "nonfinite_reference": 0, "violations": 0,
                 "argmax_rows_diagnostic": compared_rows, **dict.fromkeys(VIOLATION_FIELDS)},
           ("bit_mismatches_diagnostic", "argmax_agree_diagnostic", *ERROR_METRICS, *ERROR_COORDINATES), label)
    mismatches = integer(obj["bit_mismatches_diagnostic"], label + ".bits", 0, pairs)
    integer(obj["argmax_agree_diagnostic"], label + ".argmax", 0, compared_rows)
    if not pairs:
        for key in (*ERROR_METRICS, *ERROR_COORDINATES):
            exact(obj[key], None, label + "." + key)
        return
    maximum, rms, ratio = [number(obj[key], label + "." + key) for key in ERROR_METRICS]
    require(ratio <= 1, label + ": frozen full-logit bound ratio exceeds one")
    require(maximum <= 2 * FP32_MAX, label + ": error exceeds finite FP32 difference range")
    leq(rms, maximum, label + ".rms<=max", pairs)
    leq(maximum / math.sqrt(pairs), rms, label + ".rms>=one_max", pairs)
    leq(rms, maximum * math.sqrt(mismatches / pairs), label + ".rms_vs_bits", pairs)
    # Protocol does not report actual/reference norms or values at either max.
    # Only these necessary ratio bounds are observable; no invented .02 maxabs gate.
    leq(maximum / (.02 + .002 * FP32_MAX), ratio, label + ".ratio_lower")
    leq(ratio, maximum / .02, label + ".ratio_upper")
    for key in ERROR_COORDINATES:
        coordinate(obj[key], offset, rows, label + "." + key)
    if maximum == 0:
        require(rms == 0 and ratio == 0, label + ": inconsistent zero error metrics")
        exact(obj["argmax_agree_diagnostic"], compared_rows, label + ".zero_error_argmax")
        for key in ERROR_COORDINATES:
            exact(obj[key], {"position": offset, "vocabulary_index": 0}, label + ".zero_error." + key)
    else:
        require(mismatches > 0 and rms > 0 and ratio > 0, label + ": positive error missing diagnostics")


def aggregate_errors(obj, components, offset, rows, compared_rows, label, values_rows=None):
    errors(obj, offset, rows, compared_rows, label, values_rows)
    for key in ERROR_COUNTS:
        exact(obj[key], sum(e[key] for e in components), label + ".sum." + key)
    paired = [e for e in components if e["finite_pairs"]]
    if not paired:
        return
    for metric, coord in zip(("maxabs", "maxboundratio"), ERROR_COORDINATES):
        winner = max(paired, key=lambda e: e[metric])  # C++ add uses >, retaining FIRST ties.
        require(obj[metric] == winner[metric], label + ".maximum." + metric)
        exact(obj[coord], winner[coord], label + ".maximum." + coord)
    rms = math.hypot(*(e["rms"] * math.sqrt(e["finite_pairs"]) for e in paired)) / math.sqrt(obj["finite_pairs"])
    require(math.isclose(obj["rms"], rms, rel_tol=1e-12, abs_tol=0), label + ": inconsistent RMS aggregate")


def memory(obj, total, label, loaded=None):
    expect(obj, {"capacity": total, "expert_slots": SLOTS, "ownership_verified": True,
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
                   "expert_slots": SLOTS * ((6 * Q41 + 18 * Q40) if i == 0 else 24 * Q40),
                   "qsa_kv": 6 * total * 32 * 18, "qsa_index": 6 * (total // 4 * 128 + 384) * 4,
                   "gdn_state": 18 * (786432 + 30720) * 4, "ple_state": 368640 if i == 0 else 0},
               ("weights", "workspace", "owned_bytes", "owned_peak_bytes", "owned_buffers", "total_vram", "free_vram"), where)
        integer(d["weights"], where + ".weights", 1)
        integer(d["workspace"], where + ".workspace", WORKSPACE_FLOOR)
        owned = integer(d["owned_bytes"], where + ".owned_bytes", 1)
        exact(owned, sum(d[key] for key in CATEGORIES), where + ".category_sum")
        integer(d["owned_peak_bytes"], where + ".owned_peak_bytes", owned)
        integer(d["owned_buffers"], where + ".owned_buffers", 1)
        total_vram = integer(d["total_vram"], where + ".total_vram", 1)
        free = integer(d["free_vram"], where + ".free_vram", 1, total_vram)
        require(owned <= total_vram - free, where + ": owned bytes exceed observed used VRAM")
        if loaded is not None:
            for key in d.keys() - {"free_vram"}:
                exact(d[key], loaded["devices"][i][key], where + ".steady." + key)
    if loaded is not None:
        for key in obj.keys() - {"devices"}:
            exact(obj[key], loaded[key], label + ".steady." + key)


def memory_row(obj, event, index, total, loaded=None, reported=None):
    label = f"{event}[{index}]"
    expect(obj, {"kind": "prefill_long_memory", "protocol": 1, "event": event,
                 "phase_index": index, "host_accounting_excludes_metadata_allocator_runtime_overhead": True,
                 "passed": True}, ("stats", "route_stats", "memory", "reported_known_host_and_fixture_bytes"), label)
    same_stats(obj["stats"], ZERO_STATS, label + ".stats")
    same_groups(obj["route_stats"], ZERO_GROUPS, label + ".route_stats")
    memory(obj["memory"], total, label + ".memory", loaded)
    m = obj["memory"]
    floor = sum(m[key] for key in ("ram_expert_capacity", "host_embedding_capacity", "host_logit_capacity",
                                  "pinned_handoff", "pinned_expert_staging"))
    floor += total * VOCAB * 4 + HOST_LOGIT_BYTES + FLOAT_GUARD_BYTES + INPUT_WORKSPACE_BYTES
    # Guarded vector capacities are not exposed, only their combined reported sum.
    integer(obj["reported_known_host_and_fixture_bytes"], label + ".known_host", floor)
    if reported is not None:
        exact(obj["reported_known_host_and_fixture_bytes"], reported, label + ".steady.known_host")


def checkpoint(obj, position, shape, teacher, previous, previous_groups, loaded):
    offset, rows = shape
    label = f"reference_n1.checkpoint[{position}]"
    expect(obj, {"kind": "prefill_long_baseline_checkpoint", "protocol": 1, "phase_index": 0,
                 "checkpoint_index": position, "offset": offset, "rows": rows, "windows": rows,
                 "segment": "teacher" if offset < teacher else "continuation",
                 "max_expert_group_count": 1, "expert_groups_gt128": 0,
                 "every_N1_call_finite_stats_routes_memory_guards_checked": True, "passed": True},
           ("stats_before", "stats_after", "route_stats_before", "route_stats_after",
            "correctness_completed_call_wall_ms_sum", "errors", "memory"), label)
    same_stats(obj["stats_before"], previous, label + ".stats_before")
    completed_stats(obj["stats_after"], previous, offset, rows, label + ".stats_after")
    same_groups(obj["route_stats_before"], previous_groups, label + ".route_stats_before")
    same_groups(obj["route_stats_after"], {GROUP_FIELDS[0]: 1, GROUP_FIELDS[1]: 0}, label + ".route_stats_after")
    require(number(obj["correctness_completed_call_wall_ms_sum"], label + ".wall_sum") > 0, label + ": nonpositive time")
    errors(obj["errors"], offset, rows, 0, label + ".errors")
    memory(obj["memory"], teacher + CONTINUATION, label + ".memory", loaded)


def window(obj, index, position, shape, teacher, previous, previous_groups, loaded):
    offset, rows = shape
    label = f"{PHASES[index]}.window[{position}]"
    expect(obj, {"kind": "prefill_long_window", "protocol": 1, "phase_index": index, "phase": PHASES[index],
                 "window_index": position, "offset": offset, "rows": rows,
                 "stage": "memory_and_full_live_span_preservation", "completed_call": True,
                 "segment": "teacher" if offset < teacher else "continuation", "expected_routes": 480 * rows,
                 "completed_blocks_before": offset // 4, "completed_blocks_after": (offset + rows) // 4,
                 "tail_before": offset % 4, "tail_after": (offset + rows) % 4,
                 "stats_and_full_live_span_preserved": True, "passed": True},
           ("stats_before", "stats_after", "route_stats_before", "route_stats_after", "expert_groups_gt128_delta",
            "correctness_completed_call_wall_ms", "errors", "memory"), label)
    same_stats(obj["stats_before"], previous, label + ".stats_before")
    completed_stats(obj["stats_after"], previous, offset, rows, label + ".stats_after")
    same_groups(obj["route_stats_before"], previous_groups, label + ".route_stats_before")
    _, delta = completed_groups(obj["route_stats_after"], previous_groups, rows, label + ".route_stats_after")
    exact(obj["expert_groups_gt128_delta"], delta, label + ".delta")
    require(number(obj["correctness_completed_call_wall_ms"], label + ".wall_ms") > 0, label + ": nonpositive time")
    errors(obj["errors"], offset, rows, rows, label + ".errors")
    memory(obj["memory"], teacher + CONTINUATION, label + ".memory", loaded)


def coverage(teacher, index):
    visible, multirow, mod4, starts, occupied, crossings, budget = 0, 0, 0, 0, 0, 0, False
    for offset, rows in layout(teacher, index):
        if offset >= teacher:
            continue
        for position in range(max(offset + 1, 2047), min(offset + rows, 2056) + 1):
            visible |= 1 << (position - 2047)
            mod4 |= 1 << (position % 4)
            if rows > 1:
                multirow |= 1 << (position - 2047)
        crossings += offset // 4 != (offset + rows) // 4
        if offset and rows > 1:
            occupied += 1
            starts |= 1 << (offset % 4)
            budget |= offset < 2052 <= offset + rows
    return {"visible2047_2056_mask": visible, "visible_mod4_mask": mod4,
            "multirow_visible2047_2056_mask": multirow, "occupied_chunk_start_mod4_mask": starts,
            "occupied_chunks": occupied, "completed_block_crossing_windows": crossings,
            "occupied_budget2052_crossing": budget, "all_boundary_visibilities_observed": visible == 1023 and mod4 == 15,
            "scope": COVERAGE_SCOPE}


def invalid_proofs(proofs, windows, index, teacher):
    label = PHASES[index] + ".invalid_proofs"
    require(type(proofs) is list and len(proofs) == (4 if index else 0), label + ": incorrect proof count")
    if not index:
        return 0
    total = teacher + CONTINUATION
    first_multi = next(i for i, w in enumerate(windows) if w["rows"] > 1)
    cases = [("oversized1025", 1025, first_multi), ("negative_last1024", 1024, first_multi),
             ("oov_last1024", 1024, first_multi), ("capacity_remaining1_length2", 2, len(windows) - 2)]
    preserved = 0
    for i, (proof, (name, input_rows, prior_index)) in enumerate(zip(proofs, cases)):
        prior, continued = windows[prior_index:prior_index + 2]
        offset, count = prior["offset"] + prior["rows"], prior["rows"] * VOCAB
        where = f"{label}[{i}]"
        expect(proof, {"case": name, "input_rows": input_rows, "offset": offset, "prior_rows": prior["rows"],
                       "input_fits_remaining_capacity": input_rows <= total - offset,
                       "preserved_fullspan_values": count, "exception": "invalid_argument",
                       "all_public_stats_and_route_stats_bitwise_preserved": True,
                       "fullspan_bits_preserved": True, "memory_ledger_preserved": True, "guards_preserved": True,
                       "continued_without_reset": True, "continuation_offset": continued["offset"],
                       "continuation_rows": continued["rows"], "continuation_compared_values": continued["rows"] * VOCAB,
                       "continuation_violations": 0}, ("stats_before_and_after", "route_stats_before_and_after"), where)
        same_stats(proof["stats_before_and_after"], prior["stats_after"], where + ".stats")
        same_groups(proof["route_stats_before_and_after"], prior["route_stats_after"], where + ".routes")
        # The VERY NEXT accepted window has the same before-stats and compares
        # ALL rows. Nothing here asserts preservation after that span expires.
        preserved += count
    return preserved


def phase(obj, index, current, teacher):
    total, _, windows, _ = counts(teacher)
    label = PHASES[index]
    maximum = max(w["max_expert_group_count"] if index == 0 else w["route_stats_after"][GROUP_FIELDS[0]] for w in current)
    threshold = sum(w["expert_groups_gt128"] if index == 0 else w["expert_groups_gt128_delta"] for w in current)
    expect(obj, {"kind": "prefill_long_phase", "protocol": 1, "phase_index": index, "phase": label,
                 "windows": windows[index], "teacher_rows": teacher, "continuation_rows": CONTINUATION,
                 "max_expert_group_count": maximum, "expert_groups_gt128": threshold,
                 "coverage": coverage(teacher, index), "passed": True},
           ("errors", "teacher_stats", "final_stats", "teacher_route_stats", "final_route_stats",
            "correctness_completed_call_wall_ms_sum", "invalid_proofs"), label)
    aggregate_errors(obj["errors"], [w["errors"] for w in current], 0, total, total if index else 0, label + ".errors")
    time_key = "correctness_completed_call_wall_ms" + ("_sum" if index == 0 else "")
    close(number(obj["correctness_completed_call_wall_ms_sum"], label + ".wall_sum"),
          sum(w[time_key] for w in current), label + ".wall_sum")
    teacher_window = next(w for w in current if w["offset"] + w["rows"] == teacher)
    for prefix, w in (("teacher", teacher_window), ("final", current[-1])):
        same_stats(obj[prefix + "_stats"], w["stats_after"], label + "." + prefix + "_stats")
        same_groups(obj[prefix + "_route_stats"], w["route_stats_after"], label + "." + prefix + "_route_stats")
    if index:
        require(maximum > 128 and threshold > 0, label + ": actual logical group>128 was not proven")
    return invalid_proofs(obj["invalid_proofs"], current, index, teacher)


def collect(raw_path):
    """Validate every emitted record; return one compact aggregate, no writes."""
    raw_path = Path(raw_path).resolve()
    rows = records(raw_path)
    origin = rows[0]
    teacher = source(origin)
    total, _, windows, record_count = counts(teacher)
    exact(len(rows), record_count, "source/actual record count")
    memory_row(rows[1], "loaded", 0, total)
    loaded, reported = rows[1]["memory"], rows[1]["reported_known_host_and_fixture_bytes"]
    memory_events, memories, phases, cursor, preserved = [rows[1]], [loaded], [], 2, 0
    for index in range(3):
        reset = rows[cursor]
        cursor += 1
        memory_row(reset, "reset", index, total, loaded, reported)
        memory_events.append(reset)
        memories.append(reset["memory"])
        shapes = layout(teacher, index, compressed=True)
        current = rows[cursor:cursor + len(shapes)]
        cursor += len(shapes)
        previous, previous_groups = reset["stats"], reset["route_stats"]
        for position, (obj, shape) in enumerate(zip(current, shapes)):
            if index == 0:
                checkpoint(obj, position, shape, teacher, previous, previous_groups, loaded)
            else:
                window(obj, index, position, shape, teacher, previous, previous_groups, loaded)
            previous, previous_groups = obj["stats_after"], obj["route_stats_after"]
            memories.append(obj["memory"])
        summary = rows[cursor]
        cursor += 1
        preserved += phase(summary, index, current, teacher)
        phases.append(summary)
    final = rows[cursor]
    cursor += 1
    require(cursor == len(rows) - 1, "incorrect protocol order/count")
    memory_row(final, "final_reset", 2, total, loaded, reported)
    memory_events.append(final)
    memories.append(final["memory"])
    footer = rows[-1]
    maximum, threshold = max(p["max_expert_group_count"] for p in phases), sum(p["expert_groups_gt128"] for p in phases)
    exact(preserved, REJECTION_VALUES, "complete.rejection_values")
    expect(footer, {
        "kind": "prefill_long_complete", "protocol": 1, "teacher_rows_per_phase": teacher,
        "continuation_rows_per_phase": CONTINUATION, "phase_count": 3, "timeline_rows": 3 * total,
        "window_count": sum(windows), "record_count": record_count,
        "max_expert_group_count": maximum, "expert_groups_gt128": threshold,
        "invalid_window_rejections": 8, "rejection_preserved_fullspan_values": preserved,
        "memory_observation_count": sum(windows) + 21,
        "memory_preserved_fullspan_values": 3 * total * VOCAB + 2 * preserved,
        "steady_owners_categories_Buffer_counts_peaks_and_host_read_ledger": True,
        "fixture_guards_preserved": True, "raii_session_and_fixture_cleanup_completed": True,
        "owned_buffer_release_measured": False, "full_prefill_self_parity_passed": True,
        "logical_group_gt128_proven": True, "physical_128_column_tile_proven": False,
        "independent_HF_reference": False, "cold_cache_equality_claim": False,
        "performance_claim": False, "peak_VRAM_qualification_claim": False, "R4_complete_claim": False, "passed": True,
    }, ("errors", "minimum_free_vram_bytes", "correctness_completed_call_wall_ms_sum"), "complete")
    aggregate_errors(footer["errors"], [p["errors"] for p in phases], 0, total, 2 * total,
                     "complete.errors", values_rows=3 * total)
    close(number(footer["correctness_completed_call_wall_ms_sum"], "complete.wall_sum"),
          sum(p["correctness_completed_call_wall_ms_sum"] for p in phases), "complete.wall_sum")
    minima = footer["minimum_free_vram_bytes"]
    require(type(minima) is list and len(minima) == 2, "complete: expected both free VRAM minima")
    for i, minimum in enumerate(minima):
        # Per-N1 and rejection observations are checked in-process, not emitted.
        upper = min(m["devices"][i]["free_vram"] for m in memories)
        integer(minimum, f"complete.minimum_free_vram_bytes[{i}]", 1, upper)
    return {
        "kind": "r4b_long_prefill", "protocol": 1,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": origin["revision"], "dirty": origin["dirty"], "runtime": origin["runtime"],
        "model_variant": "qwen38-keep1-Q4_0", "model": origin["model"], "model_bytes": origin["model_bytes"],
        "raw_logs": {"prefill_long": str(raw_path)},
        "scope": f"long{teacher // 1024}K same-Session full-logit self-parity; {teacher} teacher+32 continuation rows per phase; observed logical expert group>128 and logical causal boundaries; not full R4 or performance evidence",
        "long_prefill_self_parity_qualified": True, "teacher_rows_per_phase": teacher,
        "continuation_rows_per_phase": CONTINUATION, "logical_group_gt128_qualified": True,
        "logical_causal_boundary_coverage_qualified": True, "GPU_selected_ID_trace_qualified": False,
        "physical_128_column_tile_proven": False, "individual_buffer_capacities_qualified": False,
        "full_ram_accounting_qualified": False, "owned_buffer_release_measured": False,
        "independent_hf_claim": False, "R4_complete_claim": False, "performance_claim": False,
        "peak_VRAM_qualification_claim": False, "speedup_claim": False, "cold_cache_equality_claim": False, "mtp": False,
        "timing_scope": origin["timing_scope"], "expert_read_scope": origin["read_counter_scope"],
        "coverage_scope": origin["coverage_scope"],
        "memory_scope": f"{sum(windows) + 21} in-process observations; {len(memories)} serialized memory ledgers; reference N1 checkpoints and rejection proofs summarize remaining observations; aggregate workspace floor only, no individual-buffer, full-RAM or measured-release proof",
        "source": origin, "phases": phases, "memory": memory_events, "complete": footer,
        "window_summary": {"windows": sum(windows), "serialized_baseline_checkpoints": teacher // MAX_BATCH + 1,
                           "serialized_candidate_windows": sum(windows[1:]), **footer["errors"],
                           "bit_equality_required": False, "argmax_equality_required": False,
                           "max_expert_group_count": maximum, "expert_groups_gt128": threshold,
                           "routes": {key: sum(p["final_stats"][key] for p in phases) for key in COUNTERS}},
        "passed": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="explicit canonical journal path supplied by caller")
    args = parser.parse_args(argv)
    try:
        append_record(args.results, collect(args.raw))
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print("record_prefill_long: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
