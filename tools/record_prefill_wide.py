#!/usr/bin/env python3
"""Validate core-prefill-wide-test JSONL protocol 1; append ONE r4b_wide_prefill.

Exactly 1181 rows: source, loaded memory, four (reset, chronological windows,
phase), final-reset memory, cleanup-complete footer. The 1169 windows cover
1056 N1 reference calls, 8*128/7*129+121/1024 teachers and 32 N1 continuations
per phase. Only observed completed-call diagnostics prove logical group>128.
No physical 128-column kernel/staging, individual-buffer capacity, full-RAM,
release, independent HF, 4K/16K, full R4, MTP or throughput qualification.
--results is the caller's explicit canonical ROOT/results.jsonl, never inferred.
"""

import argparse
import datetime
import json
from pathlib import Path
import stat
import sys

if __package__:
    from .record_memory import (append_result as append_record, exact, expect,
                                finite_float, integer, invalid_constant, number,
                                require, text, unique_object)
    from .record_prefill import close, errors, json_integer, same_stats, stats
else:
    from record_memory import (append_result as append_record, exact, expect,
                               finite_float, integer, invalid_constant, number,
                               require, text, unique_object)
    from record_prefill import close, errors, json_integer, same_stats, stats


MAX_RAW_BYTES = 8 * 1024 * 1024
RECORD_COUNT, WINDOW_COUNT, MEMORY_COUNT = 1181, 1169, 1205
CAPACITY, TEACHER_ROWS, VOCAB, MAX_BATCH = 1056, 1024, 248320, 1024
Q40, Q41 = 2764800, 2867200
RAM_PAYLOAD = (6 * Q41 + 42 * Q40) * 512
HOST_LOGIT_BYTES = MAX_BATCH * VOCAB * 4
PINNED_HANDOFF, PINNED_STAGING = MAX_BATCH * 4 * 2560 * 4, 4 * 16 * Q41
WORKSPACE_FLOOR = (24 * MAX_BATCH * 12288 * 4 + MAX_BATCH * 320 * 36 +
                   HOST_LOGIT_BYTES + MAX_BATCH * 10 * 2560 * 4 + 2 * 16 * Q41)
PHASES = ("reference_n1", "chunk128", "chunk129", "single1024")
COUNTERS = ("expert_hits", "expert_misses", "expert_upload_bytes")
ROUTE_FIELDS = ("hits", "misses", "upload_bytes")
GROUP_FIELDS = ("last_max_expert_group_assignments", "expert_groups_gt128")
CATEGORIES = ("weights", "expert_slots", "qsa_kv", "qsa_index", "gdn_state", "ple_state", "workspace")
ZERO_STATS = {"consumed_tokens": 0, **dict.fromkeys(COUNTERS, 0),
              "last_completed_ms": 0, "last_completed_ms_bits": 0}
ZERO_GROUPS = dict.fromkeys(GROUP_FIELDS, 0)


def records(path):
    """Bound before loading and again while reading, including growing files."""
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), str(path) + ": not a regular file")
    require(info.st_size <= MAX_RAW_BYTES, str(path) + ": oversized input")
    with path.open("rb") as stream:
        raw = stream.read(MAX_RAW_BYTES + 1)
    require(len(raw) <= MAX_RAW_BYTES, str(path) + ": oversized input")
    require(raw.endswith(b"\n"), str(path) + ": missing completed final line")
    lines = raw.split(b"\n")[:-1]
    require(len(lines) == RECORD_COUNT, str(path) + ": expected exactly 1181 JSONL rows")
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
        "kind": "prefill_wide_source", "protocol": 1, "runtime": "own_48_layer_HIP",
        "model_source": "caller_supplied_GGUF_path_no_checksum_attestation",
        "config": {"capacity": 1056, "expert_slots": 1, "max_batch_tokens": 1024, "trace": False},
        "teacher_ids": [248044, *range(100, 1123)], "continuation_ids": list(range(1123, 1155)),
        "teacher_id_count": 1024, "continuation_id_count": 32, "source_id_count": 1056,
        "all_source_ids_exact_compile_time_checked": True,
        "expected_windows_per_phase": [1056, 40, 40, 33], "phase_order": list(PHASES),
        "teacher_schedules": ["1024_N1", "8_N128", "7_N129_then_N121", "1_N1024"],
        "continuation_schedule": "32_N1_each_phase",
        "occupied_prefix_scope": "later_chunk128_and_chunk129_calls_have_real_prior_history",
        "gate": {"absolute": .02, "relative": .002,
                 "formula": "abs(actual-reference)<=.02+.002*abs(reference)",
                 "all_finite_required": True, "allowed_violations": 0,
                 "bit_equality_required": False, "argmax_equality_required": False, "cli_adjustable": False},
        "reference_scope": "all_1056_full_vocabulary_rows_retained_sequential_N1_same_Session",
        "weight_values": "unchanged_loaded_GGUF", "weight_precision": "unchanged_loaded_tensor_types",
        "kv": "Q4_0_K_and_V", "sampling": "teacher_forced", "session_instances": 1,
        "reset_clears_logical_state_and_statistics": True, "reset_retains_expert_cache": True,
        "cold_cache_equality_claim": False, "speedup_claim": False, "independent_HF_reference": False,
        "MTP_qualification_claim": False, "qualification_4K_16K_claim": False, "R4_complete_claim": False,
        "timing_scope": "correctness_only_completed_call_wall_time_excludes_comparison_memory_reset_and_rejection_not_PP_throughput",
        "counter_scope": "480_assignments_per_row_hits_include_within_call_group_reuse_misses_count_uploaded_payloads",
        "uploaded_group_payload_bytes": {"min": Q40, "max": Q41},
        "individual_group_counts_observable": False, "completed_call_group_diagnostics_observable": True,
        "max_expert_group_count": None, "group_gt128_proven": False,
        "group_gt128_evidence": "requires_observed_completed_call_max_and_gt128_counter_not_chunk_length",
        "group_diagnostic_scope": "logical_assignments_to_one_expert_in_one_layer_max_across48_layers_last_successful_call_cumulative_groups_strictly_gt128_reset_zero",
        "read_counter_scope": "constructor_expert_payload_reads_not_physical_SSD_syscall_trace",
        "geometry": {
            "logical_chunk_limit": 1024, "projection_expert_chronological_microtile_limit": 8,
            "gdn_slice_limit": 128, "expert_band": 16, "stages_per_device": 2,
            "pinned_expert_staging_bytes": PINNED_STAGING, "pinned_handoff_bytes": PINNED_HANDOFF,
            "host_logit_min_bytes": HOST_LOGIT_BYTES,
            "workspace_aggregate_min_bytes_per_device": WORKSPACE_FLOOR,
            "individual_buffer_capacities_observable": False,
            "individual_buffer_capacity_qualification": "unavailable_from_memory_API_aggregate_floor_only",
            "component_128_staging_repack_qualified": False,
            "component_128_staging_repack_evidence": "not_inferred_from_N1024_or_aggregate_memory_floor",
            "free_vram_equality_required": False,
            "memory_scope": "Session_owned_Buffer_ledger_and_reported_host_capacities_excludes_allocator_metadata"},
        "preallocated_reference_bytes": CAPACITY * VOCAB * 4,
        "preallocated_preservation_snapshot_bytes": HOST_LOGIT_BYTES,
        "fixture_guard_elements_each_end": 16, "fixture_float_guard_bytes": 256,
        "preallocated_guarded_input_elements": 1057,
        "expected_success_windows": WINDOW_COUNT, "expected_success_records": RECORD_COUNT,
        "jsonl_after_session_cleanup": True,
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
    widths = ([1] * 1024, [128] * 8, [129] * 7 + [121], [1024])[index] + [1] * 32
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


def completed_groups(before, after, rows, label):
    group_stats(after, label)
    maximum = integer(after[GROUP_FIELDS[0]], label + ".maximum", 1, rows)
    delta = after[GROUP_FIELDS[1]] - before[GROUP_FIELDS[1]]
    integer(delta, label + ".threshold_delta", 0, 480 * rows // 129)
    exact(delta > 0, maximum > 128, label + ".threshold_max_consistency")
    if rows <= 128:
        exact(delta, 0, label + ".N_le128")
    return maximum, delta


def routes(obj, rows, label):
    expect(obj, {}, (*ROUTE_FIELDS, "miss_payload_min_bytes", "miss_payload_max_bytes"), label)
    for key in ROUTE_FIELDS:
        integer(obj[key], label + "." + key)
    exact(obj["hits"] + obj["misses"], 480 * rows, label + ".assignments")
    exact(obj["miss_payload_min_bytes"], obj["misses"] * Q40, label + ".payload_min")
    exact(obj["miss_payload_max_bytes"], obj["misses"] * Q41, label + ".payload_max")
    integer(obj["miss_payload_max_bytes"], label + ".payload_max")
    require(obj["miss_payload_min_bytes"] <= obj["upload_bytes"] <= obj["miss_payload_max_bytes"],
            label + ": upload bytes outside unchanged per-miss-group payload range")


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


def memory_row(obj, event, index, loaded=None):
    label = event + ("" if index is None else f"[{index}]")
    expect(obj, {"kind": "prefill_wide_memory", "protocol": 1, "event": event,
                 "phase_index": index, "passed": True}, ("stats", "route_stats", "memory"), label)
    same_stats(obj["stats"], ZERO_STATS, label + ".stats")
    same_groups(obj["route_stats"], ZERO_GROUPS, label + ".route_stats")
    memory(obj["memory"], label + ".memory", loaded)


def window(obj, index, position, shape, previous, previous_groups, reference, loaded):
    offset, rows = shape
    label = f"{PHASES[index]}.window[{position}]"
    expect(obj, {
        "kind": "prefill_wide_window", "protocol": 1, "phase_index": index, "phase": PHASES[index],
        "window_index": position, "offset": offset, "rows": rows,
        "segment": "teacher" if offset < TEACHER_ROWS else "continuation", "reference": index == 0,
        "completed_call": True, "expected_routes": 480 * rows,
        "memory_stats_and_full_active_span_preserved": True,
        "first_input_n1_smoke": index == 0 and offset == 0,
        "passed": True, "failure_stage": None, "failure": None,
    }, ("correctness_completed_call_wall_ms", "stats_before", "stats_after", "route_delta",
        "retained_n1_route_delta", "route_stats_before", "route_stats_after", "expert_groups_gt128_delta",
        "max_expert_group_count", "group_gt128_proven", "errors", "memory_snapshot"), label)
    require(number(obj["correctness_completed_call_wall_ms"], label + ".wall_ms") > 0, label + ": nonpositive wall time")
    same_stats(obj["stats_before"], previous, label + ".stats_before")
    stats(obj["stats_after"], offset + rows, label + ".stats_after")
    routes(obj["route_delta"], rows, label + ".route_delta")
    for counter, key in zip(COUNTERS, ROUTE_FIELDS):
        exact(obj["stats_after"][counter], previous[counter] + obj["route_delta"][key], label + "." + counter)
    same_groups(obj["route_stats_before"], previous_groups, label + ".route_stats_before")
    maximum, delta = completed_groups(previous_groups, obj["route_stats_after"], rows, label + ".route_stats_after")
    exact(obj["expert_groups_gt128_delta"], delta, label + ".threshold_delta")
    exact(obj["max_expert_group_count"], maximum, label + ".maximum")
    exact(obj["group_gt128_proven"], maximum > 128 and delta > 0, label + ".proof")
    routes(obj["retained_n1_route_delta"], rows, label + ".retained_n1_route_delta")
    retained = obj["route_delta"] if index == 0 else {
        key: sum(w["route_delta"][key] for w in reference[offset:offset + rows])
        for key in (*ROUTE_FIELDS, "miss_payload_min_bytes", "miss_payload_max_bytes")}
    exact(obj["retained_n1_route_delta"], retained, label + ".matching_N1_routes")
    errors(obj["errors"], offset, rows, None if index == 0 else reference[offset:offset + rows], label + ".errors")
    memory(obj["memory_snapshot"], label + ".memory_snapshot", loaded)


def invalid_proofs(proofs, windows, index):
    label = PHASES[index] + ".invalid_proofs"
    require(type(proofs) is list and len(proofs) == (5 if index else 0), label + ": incorrect proof count")
    if not index:
        return 0
    remaining = CAPACITY - windows[0]["rows"]
    cases = [("empty", 0, 0), ("length1025", 1025, 0),
             ("negative_last", min(MAX_BATCH, remaining), 0),
             ("oov_last", min(MAX_BATCH, remaining), 0),
             ("capacity1055_length2", 2, len(windows) - 2)]
    preserved = 0
    for i, (proof, (name, input_rows, prior_index)) in enumerate(zip(proofs, cases)):
        prior, continuation = windows[prior_index:prior_index + 2]
        offset, count = prior["offset"] + prior["rows"], prior["rows"] * VOCAB
        where = f"{label}[{i}]"
        expect(proof, {
            "case": name, "input_rows": input_rows, "offset": offset, "prior_rows": prior["rows"],
            "input_fits_remaining_capacity": input_rows <= CAPACITY - offset,
            "preserved_fullspan_values": count, "exception": "invalid_argument",
            "all_public_stats_bitwise_preserved": True, "route_stats_preserved": True,
            "fullspan_bits_preserved": True, "memory_ledger_preserved": True,
            "pinned_expert_staging_preserved": True, "fixture_guards_preserved": True,
            "free_vram_equality_required": False, "consumed_increment": 0, "continued_without_reset": True,
            "continuation_offset": continuation["offset"], "continuation_rows": continuation["rows"],
            "continuation_compared_values": continuation["rows"] * VOCAB, "continuation_violations": 0,
        }, ("stats_before_and_after", "route_stats_before_and_after"), where)
        same_stats(proof["stats_before_and_after"], prior["stats_after"], where + ".stats")
        same_groups(proof["route_stats_before_and_after"], prior["route_stats_after"], where + ".route_stats")
        # The four first-window rejections precede the VERY NEXT accepted call:
        # its before-stats match, full comparison passes, and no expired 1024-row
        # span is used after that call. Near capacity preserves the latest N1.
        preserved += count
    return preserved


def phase(obj, index, windows):
    label = PHASES[index]
    maximum = max(w["max_expert_group_count"] for w in windows)
    threshold = sum(w["expert_groups_gt128_delta"] for w in windows)
    expect(obj, {
        "kind": "prefill_wide_phase", "protocol": 1, "phase_index": index, "phase": label,
        "windows": len(windows), "teacher_rows": 1024, "continuation_rows": 32,
        "expected_routes": 480 * CAPACITY, "finite_logit_values": CAPACITY * VOCAB,
        "compared_logit_values": CAPACITY * VOCAB if index else 0,
        "bit_mismatches_diagnostic": sum(w["errors"]["bit_mismatches"] for w in windows), "violations": 0,
        "timing_includes_rejection_reset_comparison_memory": False,
        "max_expert_group_count": maximum, "expert_groups_gt128": threshold,
        "group_gt128_proven": maximum > 128 and threshold > 0, "passed": True,
    }, ("correctness_completed_call_wall_ms_sum", "teacher_stats", "final_stats",
        "teacher_route_stats", "final_route_stats", "invalid_proofs"), label)
    close(number(obj["correctness_completed_call_wall_ms_sum"], label + ".wall_sum"),
          sum(w["correctness_completed_call_wall_ms"] for w in windows), label + ".wall_sum")
    teacher = next(w for w in windows if w["offset"] + w["rows"] == TEACHER_ROWS)
    for key, window_key, w in (("teacher_stats", "stats_after", teacher), ("final_stats", "stats_after", windows[-1])):
        same_stats(obj[key], w[window_key], label + "." + key)
    same_groups(obj["teacher_route_stats"], teacher["route_stats_after"], label + ".teacher_route_stats")
    same_groups(obj["final_route_stats"], windows[-1]["route_stats_after"], label + ".final_route_stats")
    return invalid_proofs(obj["invalid_proofs"], windows, index)


def collect(raw_path):
    """Validate ALL raw windows; return compact aggregate without writing files."""
    raw_path = Path(raw_path).resolve()
    rows = records(raw_path)
    origin = rows[0]
    source(origin)
    memory_row(rows[1], "loaded", None)
    loaded = rows[1]["memory"]
    memory_events, memories = [rows[1]], [loaded]
    phases, windows, reference, cursor, preserved = [], [], None, 2, 0
    for index in range(4):
        reset = rows[cursor]
        cursor += 1
        memory_row(reset, "reset", index, loaded)
        memory_events.append(reset)
        memories.append(reset["memory"])
        shapes = layout(index)
        current = rows[cursor:cursor + len(shapes)]
        cursor += len(shapes)
        previous, previous_groups = reset["stats"], reset["route_stats"]
        for position, (obj, shape) in enumerate(zip(current, shapes)):
            window(obj, index, position, shape, previous, previous_groups, reference, loaded)
            previous, previous_groups = obj["stats_after"], obj["route_stats_after"]
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
    memory_row(final, "final_reset", None, loaded)
    memory_events.append(final)
    memories.append(final["memory"])
    single = phases[3]
    require(single["max_expert_group_count"] > 128 and single["expert_groups_gt128"] > 0,
            "single1024: actual logical group>128 was not proven")
    footer = rows[-1]
    maximum = max(w["max_expert_group_count"] for w in windows)
    threshold = sum(w["expert_groups_gt128_delta"] for w in windows)
    expect(footer, {
        "kind": "prefill_wide_complete", "protocol": 1, "phase_count": 4, "record_count": RECORD_COUNT,
        "timeline_rows": 4224, "teacher_rows": 4096, "continuation_rows": 128,
        "timeline_routes": 2027520, "window_count": WINDOW_COUNT,
        "finite_logit_values": 1048903680, "compared_logit_values": 786677760, "violations": 0,
        "bit_mismatches_diagnostic": sum(w["errors"]["bit_mismatches"] for w in windows),
        "bit_equality_required": False, "argmax_equality_required": False,
        "invalid_window_rejections": 15, "rejection_preserved_fullspan_values": preserved,
        "memory_snapshot_count": len(memories) + 30, "memory_preserved_fullspan_values": 1048903680 + 2 * preserved,
        "steady_categories_owners_allocated_bytes_buffer_counts_peak_host_pinned_and_payload_reads": True,
        "fixture_guards_and_capacities_preserved": True, "individual_buffer_capacities_qualified": False,
        "component_128_staging_repack_qualified": False, "session_instances": 1,
        "raii_session_and_fixture_cleanup_completed": True, "owned_buffer_release_measured": False,
        "reference_scope": "same_session_N1_self_parity", "wide1024_full_logit_gate_passed": True,
        "max_expert_group_count": maximum, "expert_groups_gt128": threshold,
        "single1024_max_expert_group_count": single["max_expert_group_count"],
        "single1024_expert_groups_gt128": single["expert_groups_gt128"], "group_gt128_proven": True,
        "group_gt128_evidence": "observed_completed_call_route_stats_single1024_max_gt128_and_positive_threshold_count",
        "cold_cache_equality_claim": False, "speedup_claim": False, "R4_complete_claim": False,
        "qualification_4K_16K_claim": False, "independent_HF_reference": False,
        "MTP_qualification_claim": False, "performance_claim": False, "passed": True,
    }, ("correctness_completed_call_wall_ms_sum", "minimum_free_vram_bytes"), "complete")
    exact(footer["memory_snapshot_count"], MEMORY_COUNT, "complete.memory_count")
    exact(preserved, 1273136640, "complete.rejection_values")
    close(number(footer["correctness_completed_call_wall_ms_sum"], "complete.wall_sum"),
          sum(w["correctness_completed_call_wall_ms"] for w in windows), "complete.wall_sum")
    minima = footer["minimum_free_vram_bytes"]
    require(type(minima) is list and len(minima) == 2, "complete: expected both free VRAM minima")
    for i, minimum in enumerate(minima):
        # Thirty rejection snapshots are checked in the driver but not serialized.
        upper = min(m["devices"][i]["free_vram"] for m in memories)
        integer(minimum, f"complete.minimum_free_vram_bytes[{i}]", 1, upper)
    return {
        "kind": "r4b_wide_prefill", "protocol": 1,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": origin["revision"], "dirty": origin["dirty"], "runtime": origin["runtime"],
        "model_variant": "qwen38-keep1-Q4_0", "model": origin["model"],
        "raw_logs": {"prefill_wide": str(raw_path)},
        "scope": "wide-boundary same-Session full-logit self-parity and observed logical expert group>128; 1024 teacher+32 continuation rows per phase; not full R4 or performance evidence",
        "wide1024_self_parity_qualified": True, "logical_group_gt128_qualified": True,
        "individual_buffer_capacities_qualified": False, "component_128_staging_repack_qualified": False,
        "full_ram_accounting_qualified": False, "owned_buffer_release_measured": False,
        "independent_hf_claim": False, "qualification_4K_16K_claim": False, "R4_complete_claim": False,
        "performance_claim": False, "speedup_claim": False, "cold_cache_equality_claim": False, "mtp": False,
        "timing_scope": origin["timing_scope"],
        "memory_scope": "1205 observations: 1175 serialized, 30 rejection observations summarized; steady owner/category/buffer-count/peak/host/pinned/read ledgers; aggregate workspace floor only; no individual-buffer or full-RAM/release proof",
        "expert_read_scope": origin["read_counter_scope"],
        "source": origin, "phases": phases, "memory": memory_events, "complete": footer,
        "window_summary": {
            "windows": WINDOW_COUNT, "finite_logit_values": footer["finite_logit_values"],
            "compared_logit_values": footer["compared_logit_values"], "violations": 0,
            "maxabs": max(w["errors"]["maxabs"] for w in windows if w["reference"] is False),
            "maxboundratio": max(w["errors"]["maxboundratio"] for w in windows if w["reference"] is False),
            "bit_mismatches_diagnostic": footer["bit_mismatches_diagnostic"],
            "argmax_agree_rows_diagnostic": sum(w["errors"]["argmax_agree_rows"] for w in windows),
            "argmax_compared_rows": 3168, "bit_equality_required": False, "argmax_equality_required": False,
            "routes": {key: sum(w["route_delta"][key] for w in windows) for key in ROUTE_FIELDS},
            "max_expert_group_count": maximum, "expert_groups_gt128": threshold},
        "passed": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="explicit canonical journal path")
    args = parser.parse_args(argv)
    try:
        append_record(args.results, collect(args.raw))
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print("record_prefill_wide: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
