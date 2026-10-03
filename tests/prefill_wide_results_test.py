"""Deterministic wide protocol evidence; no model, HIP, weights or journal needed."""

import copy
import datetime
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import record_prefill_wide as MODULE

SCRIPT = ROOT / "tools/record_prefill_wide.py"
REVISION = "0123456789abcdef0123456789abcdef01234567"
VOCAB = 248320
NAMES = ["reference_n1", "chunk128", "chunk129", "single1024"]
COUNTERS = ("expert_hits", "expert_misses", "expert_upload_bytes")
ROUTE_KEYS = ("hits", "misses", "upload_bytes")
CATEGORIES = ("weights", "expert_slots", "qsa_kv", "qsa_index", "gdn_state", "ple_state", "workspace")


def zero_stats():
    return {"consumed_tokens": 0, **dict.fromkeys(COUNTERS, 0),
            "last_completed_ms": 0, "last_completed_ms_bits": 0}


def zero_groups():
    return {"last_max_expert_group_assignments": 0, "expert_groups_gt128": 0}


def route(rows, hits):
    misses = 480 * rows - hits
    q41 = misses // 8
    return {"hits": hits, "misses": misses,
            "upload_bytes": (misses - q41) * 2764800 + q41 * 2867200,
            "miss_payload_min_bytes": misses * 2764800, "miss_payload_max_bytes": misses * 2867200}


def metrics(offset, rows, compared):
    # Exact realizable norms: every 64th reference value is16, all others zero;
    # count is divisible by64, so max16/RMS2. Identical candidate values really
    # have zero error, no bit mismatch, and the FIRST max coordinate at offset/0.
    return {
        "values": rows * VOCAB, "finite_values": rows * VOCAB,
        "nonfinite_actual": 0, "nonfinite_reference": 0, "actual_maxabs": 16., "actual_rms": 2.,
        "compared_values": rows * VOCAB if compared else 0, "finite_pairs": rows * VOCAB if compared else 0,
        "metric_scope": "finite_pairs_only_nonfinite_pairs_also_violate",
        "maxabs": 0 if compared else None, "rms": 0 if compared else None,
        "maxboundratio": 0 if compared else None, "violations": 0, "bit_mismatches": 0,
        "bit_equality_required": False, "argmax_agree_rows": rows if compared else 0,
        "argmax_compared_rows": rows if compared else 0, "argmax_equality_required": False,
        "maxabs_coordinate": {"position": offset, "vocabulary_index": 0} if compared else None,
        "maxboundratio_coordinate": {"position": offset, "vocabulary_index": 0} if compared else None,
        "first_violation_coordinate": None,
    }


def fixture():
    """Independent exact driver schema, with realizable logical route cardinalities."""
    source = {
        "kind": "prefill_wide_source", "protocol": 1, "revision": REVISION, "dirty": True,
        "model": '/models/данные "quoted"/qwen38-keep1-Q4_0.gguf', "model_bytes": 75399121792,
        "model_source": "caller_supplied_GGUF_path_no_checksum_attestation", "runtime": "own_48_layer_HIP",
        "config": {"capacity": 1056, "expert_slots": 1, "max_batch_tokens": 1024, "trace": False},
        "teacher_ids": [248044, *range(100, 1123)], "continuation_ids": list(range(1123, 1155)),
        "teacher_id_count": 1024, "continuation_id_count": 32, "source_id_count": 1056,
        "all_source_ids_exact_compile_time_checked": True, "expected_windows_per_phase": [1056, 40, 40, 33],
        "phase_order": NAMES.copy(), "teacher_schedules": ["1024_N1", "8_N128", "7_N129_then_N121", "1_N1024"],
        "continuation_schedule": "32_N1_each_phase",
        "occupied_prefix_scope": "later_chunk128_and_chunk129_calls_have_real_prior_history",
        "gate": {"absolute": .02, "relative": .002, "formula": "abs(actual-reference)<=.02+.002*abs(reference)",
                 "all_finite_required": True, "allowed_violations": 0, "bit_equality_required": False,
                 "argmax_equality_required": False, "cli_adjustable": False},
        "reference_scope": "all_1056_full_vocabulary_rows_retained_sequential_N1_same_Session",
        "weight_values": "unchanged_loaded_GGUF", "weight_precision": "unchanged_loaded_tensor_types",
        "kv": "Q4_0_K_and_V", "sampling": "teacher_forced", "session_instances": 1,
        "reset_clears_logical_state_and_statistics": True, "reset_retains_expert_cache": True,
        "cold_cache_equality_claim": False, "speedup_claim": False, "independent_HF_reference": False,
        "MTP_qualification_claim": False, "qualification_4K_16K_claim": False, "R4_complete_claim": False,
        "timing_scope": "correctness_only_completed_call_wall_time_excludes_comparison_memory_reset_and_rejection_not_PP_throughput",
        "counter_scope": "480_assignments_per_row_hits_include_within_call_group_reuse_misses_count_uploaded_payloads",
        "uploaded_group_payload_bytes": {"min": 2764800, "max": 2867200},
        "individual_group_counts_observable": False, "completed_call_group_diagnostics_observable": True,
        "max_expert_group_count": None, "group_gt128_proven": False,
        "group_gt128_evidence": "requires_observed_completed_call_max_and_gt128_counter_not_chunk_length",
        "group_diagnostic_scope": "logical_assignments_to_one_expert_in_one_layer_max_across48_layers_last_successful_call_cumulative_groups_strictly_gt128_reset_zero",
        "read_counter_scope": "constructor_expert_payload_reads_not_physical_SSD_syscall_trace",
        "geometry": {"logical_chunk_limit": 1024, "projection_expert_chronological_microtile_limit": 8,
                     "gdn_slice_limit": 128, "expert_band": 16, "stages_per_device": 2,
                     "pinned_expert_staging_bytes": 183500800, "pinned_handoff_bytes": 41943040,
                     "host_logit_min_bytes": 1017118720, "workspace_aggregate_min_bytes_per_device": 2433482752,
                     "individual_buffer_capacities_observable": False,
                     "individual_buffer_capacity_qualification": "unavailable_from_memory_API_aggregate_floor_only",
                     "component_128_staging_repack_qualified": False,
                     "component_128_staging_repack_evidence": "not_inferred_from_N1024_or_aggregate_memory_floor",
                     "free_vram_equality_required": False,
                     "memory_scope": "Session_owned_Buffer_ledger_and_reported_host_capacities_excludes_allocator_metadata"},
        "preallocated_reference_bytes": 1048903680, "preallocated_preservation_snapshot_bytes": 1017118720,
        "fixture_guard_elements_each_end": 16, "fixture_float_guard_bytes": 256,
        "preallocated_guarded_input_elements": 1057, "expected_success_windows": 1169,
        "expected_success_records": 1181, "jsonl_after_session_cleanup": True,
    }
    devices = []
    for i in (0, 1):
        d = {"device": i, "first_layer": 24 * i, "last_layer": 24 * i + 23,
             "gdn_layers": 18, "qsa_layers": 6, "weights": 1100000000 + i * 12345678,
             "expert_slots": [66969600, 66355200][i], "qsa_kv": 3649536, "qsa_index": 820224,
             "gdn_state": 58834944, "ple_state": [368640, 0][i], "workspace": 2433482752 + i * 43210,
             "owned_buffers": 1001 + i * 37, "total_vram": 17163091968}
        d["owned_bytes"] = sum(d[key] for key in CATEGORIES)
        d["owned_peak_bytes"] = d["owned_bytes"] + 4096
        d["free_vram"] = d["total_vram"] - d["owned_bytes"] - 64 * 1024 * 1024
        devices.append(d)
    loaded = {"capacity": 1056, "expert_slots": 1, "ownership_verified": True,
              "ram_expert_capacity": 68262297600 + 4096, "ram_expert_payload": 68262297600,
              "host_embedding_capacity": 521472000, "host_logit_capacity": 1017118720 + 4096,
              "pinned_handoff": 41943040, "pinned_expert_staging": 183500800,
              "expert_payload_reads": 144, "expert_payload_bytes_read": 68262297600, "devices": devices}
    records = [source]

    def snapshot():
        m = copy.deepcopy(loaded)
        for d in m["devices"]:
            d["free_vram"] -= len(records) * 1024
        return m

    def memory_row(event, index):
        return {"kind": "prefill_wide_memory", "protocol": 1, "event": event, "phase_index": index,
                "stats": zero_stats(), "route_stats": zero_groups(), "memory": snapshot(), "passed": True}

    records.append(memory_row("loaded", None))
    schedules = [[1] * 1056, [128] * 8 + [1] * 32, [129] * 7 + [121] + [1] * 32, [1024] + [1] * 32]
    reference, preserved, wall_sum, all_windows, phases = None, 0, 0, [], []
    for index, (name, widths) in enumerate(zip(NAMES, schedules)):
        records.append(memory_row("reset", index))
        offset, previous, groups_previous, windows = 0, zero_stats(), zero_groups(), []
        for position, rows in enumerate(widths):
            # Ten unique experts per token. N1 has480 singleton misses (with
            # ten same-expert choices per layer). N128 groups ten repeated
            # experts per layer: 480 misses, 480*(N-1) hits. N129 alternates
            # this and a cyclic distributed selection (max<=3, no big group).
            # N1024 repeats the ten experts: actual max1024, 480 big groups.
            distributed = rows == 129 and position % 2 == 1
            miss_count = 48 * 512 if distributed else 480
            delta = route(rows, 480 * rows - miss_count)
            timing = 50.25 + index / 4 + offset / 8 + rows
            after = {"consumed_tokens": offset + rows,
                     **{c: previous[c] + delta[k] for c, k in zip(COUNTERS, ROUTE_KEYS)},
                     "last_completed_ms": timing,
                     "last_completed_ms_bits": struct.unpack("!Q", struct.pack("!d", timing))[0]}
            maximum = 3 if distributed else rows
            group_delta = 480 if rows > 128 and not distributed else 0
            groups_after = {"last_max_expert_group_assignments": maximum,
                            "expert_groups_gt128": groups_previous["expert_groups_gt128"] + group_delta}
            retained = copy.deepcopy(delta) if index == 0 else {
                key: sum(w["route_delta"][key] for w in reference[offset:offset + rows])
                for key in (*ROUTE_KEYS, "miss_payload_min_bytes", "miss_payload_max_bytes")}
            w = {"kind": "prefill_wide_window", "protocol": 1, "phase_index": index, "phase": name,
                 "window_index": position, "offset": offset, "rows": rows,
                 "segment": "teacher" if offset < 1024 else "continuation", "reference": index == 0,
                 "completed_call": True, "correctness_completed_call_wall_ms": timing + 1.5,
                 "stats_before": copy.deepcopy(previous), "stats_after": after,
                 "expected_routes": 480 * rows, "route_delta": delta, "retained_n1_route_delta": retained,
                 "route_stats_before": copy.deepcopy(groups_previous), "route_stats_after": groups_after,
                 "expert_groups_gt128_delta": group_delta, "max_expert_group_count": maximum,
                 "group_gt128_proven": group_delta > 0, "errors": metrics(offset, rows, index > 0),
                 "memory_snapshot": snapshot(), "memory_stats_and_full_active_span_preserved": True,
                 "first_input_n1_smoke": index == 0 and offset == 0,
                 "passed": True, "failure_stage": None, "failure": None}
            records.append(w)
            windows.append(w)
            previous, groups_previous = after, groups_after
            offset += rows
        if index == 0:
            reference = windows
        proofs = []
        if index:
            remaining = 1056 - widths[0]
            cases = [("empty", 0, 0), ("length1025", 1025, 0), ("negative_last", remaining, 0),
                     ("oov_last", remaining, 0), ("capacity1055_length2", 2, len(windows) - 2)]
            for case, input_rows, prior_index in cases:
                prior, continuation = windows[prior_index:prior_index + 2]
                consumed = prior["offset"] + prior["rows"]
                proofs.append({
                    "case": case, "input_rows": input_rows, "offset": consumed, "prior_rows": prior["rows"],
                    "input_fits_remaining_capacity": input_rows <= 1056 - consumed,
                    "preserved_fullspan_values": prior["rows"] * VOCAB,
                    "stats_before_and_after": copy.deepcopy(prior["stats_after"]),
                    "route_stats_before_and_after": copy.deepcopy(prior["route_stats_after"]),
                    "exception": "invalid_argument", "all_public_stats_bitwise_preserved": True,
                    "route_stats_preserved": True, "fullspan_bits_preserved": True, "memory_ledger_preserved": True,
                    "pinned_expert_staging_preserved": True, "fixture_guards_preserved": True,
                    "free_vram_equality_required": False, "consumed_increment": 0, "continued_without_reset": True,
                    "continuation_offset": continuation["offset"], "continuation_rows": continuation["rows"],
                    "continuation_compared_values": continuation["rows"] * VOCAB, "continuation_violations": 0})
        preserved += sum(p["preserved_fullspan_values"] for p in proofs)
        phase_wall = sum(w["correctness_completed_call_wall_ms"] for w in windows)
        wall_sum += phase_wall
        teacher = next(w for w in windows if w["offset"] + w["rows"] == 1024)
        maximum = max(w["max_expert_group_count"] for w in windows)
        threshold = groups_previous["expert_groups_gt128"]
        summary = {"kind": "prefill_wide_phase", "protocol": 1, "phase_index": index, "phase": name,
                   "windows": len(windows), "teacher_rows": 1024, "continuation_rows": 32,
                   "expected_routes": 506880, "finite_logit_values": 262225920,
                   "compared_logit_values": 262225920 if index else 0,
                   "bit_mismatches_diagnostic": 0, "violations": 0,
                   "correctness_completed_call_wall_ms_sum": phase_wall,
                   "timing_includes_rejection_reset_comparison_memory": False,
                   "teacher_stats": copy.deepcopy(teacher["stats_after"]), "final_stats": copy.deepcopy(previous),
                   "teacher_route_stats": copy.deepcopy(teacher["route_stats_after"]),
                   "final_route_stats": copy.deepcopy(groups_previous), "max_expert_group_count": maximum,
                   "expert_groups_gt128": threshold, "group_gt128_proven": maximum > 128 and threshold > 0,
                   "invalid_proofs": proofs, "passed": True}
        records.append(summary)
        phases.append(summary)
        all_windows.extend(windows)
    records.append(memory_row("final_reset", None))
    minima = [min(r.get("memory", r.get("memory_snapshot"))["devices"][i]["free_vram"]
                  for r in records if "memory" in r or "memory_snapshot" in r) - 1024 for i in (0, 1)]
    records.append({
        "kind": "prefill_wide_complete", "protocol": 1, "phase_count": 4, "record_count": 1181,
        "timeline_rows": 4224, "teacher_rows": 4096, "continuation_rows": 128, "timeline_routes": 2027520,
        "window_count": 1169, "finite_logit_values": 1048903680, "compared_logit_values": 786677760,
        "violations": 0, "bit_mismatches_diagnostic": 0, "bit_equality_required": False, "argmax_equality_required": False,
        "correctness_completed_call_wall_ms_sum": wall_sum, "invalid_window_rejections": 15,
        "rejection_preserved_fullspan_values": preserved, "memory_snapshot_count": 1205,
        "memory_preserved_fullspan_values": 1048903680 + 2 * preserved, "minimum_free_vram_bytes": minima,
        "steady_categories_owners_allocated_bytes_buffer_counts_peak_host_pinned_and_payload_reads": True,
        "fixture_guards_and_capacities_preserved": True, "individual_buffer_capacities_qualified": False,
        "component_128_staging_repack_qualified": False, "session_instances": 1,
        "raii_session_and_fixture_cleanup_completed": True, "owned_buffer_release_measured": False,
        "reference_scope": "same_session_N1_self_parity", "wide1024_full_logit_gate_passed": True,
        "max_expert_group_count": 1024, "expert_groups_gt128": sum(p["expert_groups_gt128"] for p in phases),
        "single1024_max_expert_group_count": 1024, "single1024_expert_groups_gt128": 480, "group_gt128_proven": True,
        "group_gt128_evidence": "observed_completed_call_route_stats_single1024_max_gt128_and_positive_threshold_count",
        "cold_cache_equality_claim": False, "speedup_claim": False, "R4_complete_claim": False,
        "qualification_4K_16K_claim": False, "independent_HF_reference": False,
        "MTP_qualification_claim": False, "performance_claim": False, "passed": True})
    return records


def encoded(rows):
    return ("".join(json.dumps(r, allow_nan=False, separators=(",", ":")) + "\n" for r in rows)).encode()


def target(value, path):
    for key in path:
        value = value[key]
    return value


class PrefillWideResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = fixture()
        cls.original_raw = encoded(cls.original)

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="prefill-wide-results-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.raw, self.results = self.root / "wide.jsonl", self.root / "results.jsonl"
        self.rows = copy.deepcopy(self.original)
        self.raw.write_bytes(self.original_raw)
        self.windows = [[i for i, r in enumerate(self.rows) if r["kind"] == "prefill_wide_window" and r["phase_index"] == p]
                        for p in range(4)]
        self.phases = [i for i, r in enumerate(self.rows) if r["kind"] == "prefill_wide_phase"]

    def collect(self, rows=None):
        if rows is not None:
            self.raw.write_bytes(encoded(rows))
        return MODULE.collect(self.raw)

    def reject(self, rows):
        self.raw.write_bytes(encoded(rows))
        before = self.raw.read_bytes()
        with self.assertRaises(ValueError):
            self.collect()
        self.assertEqual(self.raw.read_bytes(), before)
        self.assertFalse(self.results.exists())

    def change(self, row, path, value):
        # Mutate one leaf and restore it rather than cloning four MiB hundreds
        # of times. Every rejection still parses and validates a real raw file.
        obj = target(self.rows[row], path[:-1])
        old = obj[path[-1]]
        obj[path[-1]] = value
        try:
            self.reject(self.rows)
        finally:
            obj[path[-1]] = old

    def cli(self, destination=None, args=None):
        command = [sys.executable, "-B", str(SCRIPT)]
        command += args if args is not None else ["--raw", self.raw.name, "--results", str(destination or self.results)]
        return subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30, check=False)

    def test_valid_collect_compact_aggregate_provenance_and_nonclaims(self):
        result = self.collect()
        self.assertEqual(len(self.rows), 1181)
        self.assertEqual(result["kind"], "r4b_wide_prefill")
        self.assertEqual(result["protocol"], 1)
        self.assertEqual(result["revision"], REVISION)
        self.assertIs(result["dirty"], True)
        self.assertEqual(result["source"], self.rows[0])
        self.assertEqual(result["phases"], [self.rows[i] for i in self.phases])
        self.assertEqual(result["memory"], [r for r in self.rows if r["kind"] == "prefill_wide_memory"])
        self.assertEqual(result["complete"], self.rows[-1])
        self.assertNotIn("records", result)
        self.assertNotIn("windows", result)
        self.assertEqual(result["raw_logs"], {"prefill_wide": str(self.raw.resolve())})
        self.assertEqual(datetime.datetime.fromisoformat(result["timestamp"]).utcoffset(), datetime.timedelta(0))
        self.assertEqual(result["window_summary"]["max_expert_group_count"], 1024)
        self.assertEqual(result["window_summary"]["expert_groups_gt128"], 2400)
        self.assertEqual(result["window_summary"]["maxboundratio"], 0)
        for key in ("individual_buffer_capacities_qualified", "component_128_staging_repack_qualified",
                    "full_ram_accounting_qualified", "owned_buffer_release_measured", "independent_hf_claim",
                    "qualification_4K_16K_claim", "R4_complete_claim", "performance_claim", "speedup_claim",
                    "cold_cache_equality_claim", "mtp"):
            self.assertIs(result[key], False)
        self.assertIn("not_PP_throughput", result["timing_scope"])
        self.assertEqual(self.raw.read_bytes(), self.original_raw)

    def test_exact_ids_counts_timeline_memory_and_fullspan_lifetime(self):
        result = self.collect()
        self.assertEqual(self.rows[0]["teacher_ids"] + self.rows[0]["continuation_ids"], [248044, *range(100, 1155)])
        self.assertEqual([len(w) for w in self.windows], [1056, 40, 40, 33])
        self.assertEqual([self.rows[i]["rows"] for i in self.windows[2][:8]], [129] * 7 + [121])
        footer = result["complete"]
        for key, expected in (("memory_snapshot_count", 1205), ("timeline_routes", 2027520),
                              ("finite_logit_values", 1048903680), ("compared_logit_values", 786677760),
                              ("rejection_preserved_fullspan_values", 1273136640),
                              ("memory_preserved_fullspan_values", 3595176960)):
            self.assertEqual(footer[key], expected)
        for phase_index, width in ((1, 128), (2, 129), (3, 1024)):
            proofs = self.rows[self.phases[phase_index]]["invalid_proofs"]
            self.assertEqual([p["prior_rows"] for p in proofs], [width] * 4 + [1])
            self.assertEqual([p["input_rows"] for p in proofs], [0, 1025, 1056 - width, 1056 - width, 2])
            self.assertEqual(proofs[-1]["continuation_offset"], 1055)
            self.assertEqual(proofs[-1]["continuation_rows"], 1)

    def test_source_provenance_ids_gate_geometry_and_evidence_are_frozen(self):
        for field, bad in (("revision", "g" * 40), ("revision", "f" * 39), ("dirty", 1),
                           ("model", "qwen38-keep1-Q4_0.gguf"), ("model", "/models/full-PLE.gguf"),
                           ("model_bytes", 0), ("model_bytes", 1 << 63), ("source_id_count", 1055),
                           ("session_instances", 2), ("group_gt128_proven", True)):
            self.change(0, (field,), bad)
        for path, bad in ((("teacher_ids", 1023), 1123), (("continuation_ids", 31), 1153),
                          (("gate", "absolute"), .021), (("gate", "relative"), .003),
                          (("gate", "allowed_violations"), 1), (("config", "trace"), True),
                          (("geometry", "projection_expert_chronological_microtile_limit"), 128),
                          (("geometry", "component_128_staging_repack_qualified"), True),
                          (("geometry", "individual_buffer_capacities_observable"), True)):
            self.change(0, path, bad)
        self.rows[0].update(revision="ABCDEF01" * 5, dirty=False)
        result = self.collect(self.rows)
        self.assertEqual(result["revision"], "ABCDEF01" * 5)
        self.assertIs(result["dirty"], False)

    def test_unknown_missing_fields_nested_stats_groups_memory_routes_metrics_proofs(self):
        representatives = [(0, ()), (0, ("geometry",)), (1, ("route_stats",)), (1, ("memory",)),
                           (1, ("memory", "devices", 1)), (self.windows[3][0], ()),
                           (self.windows[3][0], ("stats_after",)), (self.windows[3][0], ("route_stats_after",)),
                           (self.windows[3][0], ("route_delta",)), (self.windows[3][0], ("errors",)),
                           (self.windows[3][0], ("errors", "maxabs_coordinate")),
                           (self.phases[3], ()), (self.phases[3], ("invalid_proofs", 0)), (1180, ())]
        for row, path in representatives:
            obj = target(self.rows[row], path)
            obj["unknown"] = 1
            try:
                self.reject(self.rows)
            finally:
                del obj["unknown"]
            field = next(iter(obj))
            value = obj.pop(field)
            try:
                self.reject(self.rows)
            finally:
                obj[field] = value

    def test_all_public_stats_and_diagnostics_preserved_rejections_resets_memory(self):
        w = self.windows[3][0]
        for block in ("stats_before", "stats_after"):
            for field in zero_stats():
                self.change(w, (block, field), self.rows[w][block][field] + 1)
        for field in zero_groups():
            self.change(w, ("route_stats_before", field), 1)
            self.change(self.phases[3], ("invalid_proofs", 0, "route_stats_before_and_after", field), 0)
        for field in zero_stats():
            self.change(self.phases[3], ("invalid_proofs", 0, "stats_before_and_after", field), 0)
        for i, row in enumerate(self.rows):
            if row["kind"] == "prefill_wide_memory":
                self.change(i, ("stats", "last_completed_ms_bits"), 1)
                self.change(i, ("route_stats", "expert_groups_gt128"), 1)
        for field in ("all_public_stats_bitwise_preserved", "route_stats_preserved", "fullspan_bits_preserved",
                      "memory_ledger_preserved", "pinned_expert_staging_preserved", "fixture_guards_preserved",
                      "continued_without_reset"):
            self.change(self.phases[3], ("invalid_proofs", 0, field), False)
        for case, field, bad in ((0, "prior_rows", 1), (0, "preserved_fullspan_values", VOCAB),
                                 (0, "continuation_offset", 1025), (0, "continuation_rows", 32),
                                 (1, "input_fits_remaining_capacity", True), (2, "input_rows", 31),
                                 (4, "offset", 1054), (4, "prior_rows", 1024)):
            self.change(self.phases[3], ("invalid_proofs", case, field), bad)

    def test_groups_observed_per_call_bounds_and_max_counter_consistency(self):
        for phase_index, position, maximum, delta in ((0, 0, 1, 0), (1, 0, 128, 0),
                                                    (2, 0, 129, 480), (2, 1, 3, 0), (3, 0, 1024, 480)):
            row = self.rows[self.windows[phase_index][position]]
            self.assertEqual(row["max_expert_group_count"], maximum)
            self.assertEqual(row["expert_groups_gt128_delta"], delta)
        w = self.windows[3][0]
        for field, bad in (("max_expert_group_count", 1025), ("expert_groups_gt128_delta", 0),
                           ("group_gt128_proven", False)):
            self.change(w, (field,), bad)
        self.change(w, ("route_stats_after", "last_max_expert_group_assignments"), 1025)
        self.change(w, ("route_stats_after", "expert_groups_gt128"), 480 * 1024 // 129 + 1)
        self.change(self.windows[1][0], ("route_stats_after", "expert_groups_gt128"), 1)
        self.change(self.windows[0][0], ("route_stats_after", "last_max_expert_group_assignments"), 0)
        self.change(self.windows[2][1], ("group_gt128_proven",), True)  # N129 alone is NOT proof.
        self.change(self.phases[3], ("max_expert_group_count",), 1)  # Last N1 cannot erase phase max.
        self.change(1180, ("group_gt128_evidence",), "chunk_length")
        self.change(1180, ("single1024_expert_groups_gt128",), 0)

    def test_consistent_no_big_group_single1024_cannot_claim_qualification(self):
        # Even with otherwise valid counters and a big group in chunk129,
        # N1024 with distributed groups must fail its explicit required proof.
        for i in self.windows[3]:
            row = self.rows[i]
            row["route_stats_before"]["expert_groups_gt128"] = 0
            row["route_stats_after"]["expert_groups_gt128"] = 0
            row["expert_groups_gt128_delta"] = 0
            row["group_gt128_proven"] = False
        first = self.rows[self.windows[3][0]]
        first["max_expert_group_count"] = 128
        first["route_stats_after"]["last_max_expert_group_assignments"] = 128
        self.rows[self.windows[3][1]]["route_stats_before"]["last_max_expert_group_assignments"] = 128
        p = self.rows[self.phases[3]]
        p.update(max_expert_group_count=128, expert_groups_gt128=0, group_gt128_proven=False)
        p["teacher_route_stats"] = {"last_max_expert_group_assignments": 128, "expert_groups_gt128": 0}
        p["final_route_stats"]["expert_groups_gt128"] = 0
        for proof in p["invalid_proofs"]:
            proof["route_stats_before_and_after"]["expert_groups_gt128"] = 0
            if proof["offset"] == 1024:
                proof["route_stats_before_and_after"]["last_max_expert_group_assignments"] = 128
        self.raw.write_bytes(encoded(self.rows))
        with self.assertRaisesRegex(ValueError, "single1024: actual logical group"):
            self.collect()

    def test_memory_geometry_atomic_staging_capacity_floor_owner_counts_and_steady_growth(self):
        w = self.windows[3][0]
        for field, bad in (("pinned_expert_staging", 91750400), ("pinned_handoff", 40960),
                           ("host_logit_capacity", 1017118719), ("ram_expert_capacity", 68262297599),
                           ("expert_payload_reads", 145), ("expert_payload_bytes_read", 68262297599),
                           ("ownership_verified", False)):
            self.change(w, ("memory_snapshot", field), bad)
        for owner in (0, 1):
            for field, bad in (("device", 1 - owner), ("workspace", 2433482751), ("owned_buffers", 0),
                               ("owned_peak_bytes", 1), ("free_vram", 0), ("qsa_kv", 3649535),
                               ("qsa_index", 820223), ("expert_slots", 1), ("total_vram", 1)):
                self.change(w, ("memory_snapshot", "devices", owner, field), bad)
        d = self.rows[w]["memory_snapshot"]["devices"][0]
        d["workspace"] += 1
        d["owned_bytes"] += 1
        d["owned_peak_bytes"] += 1
        self.reject(self.rows)  # Internally consistent growth still violates steady ledger.

    def test_routes_upload_bounds_retained_n1_counters_and_numeric_types(self):
        w = self.windows[3][0]
        for block in ("route_delta", "retained_n1_route_delta"):
            for field in (*ROUTE_KEYS, "miss_payload_min_bytes", "miss_payload_max_bytes"):
                self.change(w, (block, field), self.rows[w][block][field] + 1)
        for field, bad in (("upload_bytes", 480 * 2764800 - 1), ("upload_bytes", 480 * 2867200 + 1),
                           ("hits", True), ("misses", 480.), ("upload_bytes", 1 << 64)):
            self.change(w, ("route_delta", field), bad)
        self.change(w, ("route_stats_after", "expert_groups_gt128"), 480.)
        self.change(w, ("group_gt128_proven",), 1)
        self.change(w, ("stats_after", "last_completed_ms_bits"), 1)

    def test_chronological_schedules_reordered_skipped_truncated_and_failure_records(self):
        for w in (self.windows[0][-1], self.windows[1][1], self.windows[2][7], self.windows[3][1]):
            for field in ("offset", "rows", "window_index", "phase_index", "expected_routes"):
                self.change(w, (field,), self.rows[w][field] + 1)
        for i in (0, 1, self.windows[2][7], self.phases[3], 1179, 1180):
            self.reject(self.rows[:i] + self.rows[i + 1:])
        w = self.windows[2][0]
        self.rows[w], self.rows[w + 1] = self.rows[w + 1], self.rows[w]
        self.reject(self.rows)
        self.rows[w], self.rows[w + 1] = self.rows[w + 1], self.rows[w]
        self.change(self.windows[3][0], ("passed",), False)
        self.change(self.windows[3][0], ("completed_call",), False)
        self.change(self.windows[3][0], ("failure_stage",), "frozen_full_logit_gate")
        self.rows[-1] = {"kind": "prefill_wide_failure", "protocol": 1, "passed": False,
                         "session_constructed": True, "raii_session_and_fixture_cleanup_completed": True, "error": "gate failed"}
        self.reject(self.rows)

    def test_finite_full_logit_gate_metric_consistency_and_coordinates(self):
        w = self.windows[3][0]
        for field, bad in (("finite_values", VOCAB), ("compared_values", VOCAB), ("finite_pairs", 0),
                           ("nonfinite_actual", 1), ("nonfinite_reference", 1), ("violations", 1),
                           ("maxboundratio", math.nextafter(1., math.inf)), ("maxabs", .053),
                           ("rms", .001), ("actual_rms", 0), ("actual_maxabs", 0), ("actual_rms", 17),
                           ("maxabs", None), ("actual_maxabs", 1e300), ("argmax_agree_rows", 1025)):
            self.change(w, ("errors", field), bad)
        for field in ("maxabs_coordinate", "maxboundratio_coordinate"):
            self.change(w, ("errors", field, "position"), 1024)
            self.change(w, ("errors", field, "vocabulary_index"), 1)  # Zero error selects first finite pair.
        self.change(self.windows[0][0], ("errors", "actual_rms"), 1e-8)

    def test_realizable_nonzero_bound_bit_argmax_diagnostics_and_zero_signed_bits(self):
        # Perturb a non-maximal zero reference value in ONE row by an exactly
        # representable FP32 .015625: bound=.02 at that value, not .052 from the
        # global norm. Reference norm stays16/RMS2, candidate RMS changes by the
        # actual one-element contribution. This is genuine within-bound error.
        w, p = self.windows[1][0], self.phases[1]
        count, error = 128 * VOCAB, .015625
        e = self.rows[w]["errors"]
        e.update(maxabs=error, rms=error / math.sqrt(count), maxboundratio=error / .02,
                 actual_rms=math.sqrt(4 + error * error / count), bit_mismatches=1,
                 maxabs_coordinate={"position": 0, "vocabulary_index": 1},
                 maxboundratio_coordinate={"position": 0, "vocabulary_index": 1})
        self.rows[p]["bit_mismatches_diagnostic"] = self.rows[-1]["bit_mismatches_diagnostic"] = 1
        self.assertIs(self.collect(self.rows)["passed"], True)
        e["maxboundratio"] = math.nextafter(1., math.inf)
        self.reject(self.rows)
        e["maxboundratio"] = error / .02
        # Argmax is diagnostic for nonzero errors; no universal exact gate.
        e["argmax_agree_rows"] = 127
        self.assertEqual(self.collect(self.rows)["window_summary"]["argmax_agree_rows_diagnostic"], 3167)
        e.update(maxabs=0, rms=0, maxboundratio=0, actual_rms=2., argmax_agree_rows=128,
                 maxabs_coordinate={"position": 0, "vocabulary_index": 0},
                 maxboundratio_coordinate={"position": 0, "vocabulary_index": 0})
        self.assertIs(self.collect(self.rows)["passed"], True)  # +/-0 bits may differ.

    def test_complete_phase_summary_cleanup_and_nonclaims_cannot_be_faked(self):
        for field in ("window_count", "finite_logit_values", "timeline_routes", "memory_snapshot_count",
                      "rejection_preserved_fullspan_values", "memory_preserved_fullspan_values", "expert_groups_gt128"):
            self.change(1180, (field,), self.rows[-1][field] + 1)
        for field in ("raii_session_and_fixture_cleanup_completed", "wide1024_full_logit_gate_passed", "group_gt128_proven"):
            self.change(1180, (field,), False)
        for field in ("individual_buffer_capacities_qualified", "component_128_staging_repack_qualified",
                      "owned_buffer_release_measured", "performance_claim", "R4_complete_claim"):
            self.change(1180, (field,), True)
        self.change(self.phases[2], ("teacher_route_stats", "expert_groups_gt128"), 0)
        self.change(self.phases[3], ("correctness_completed_call_wall_ms_sum",), 1.)
        self.change(1180, ("minimum_free_vram_bytes",), [0, 0])
        self.rows[-1]["minimum_free_vram_bytes"] = [1, 1]  # Unserialized observations may be lower.
        self.assertIs(self.collect(self.rows)["passed"], True)

    def test_duplicate_keys_nonfinite_json_malformed_utf8_and_unfinished_input(self):
        raw = self.original_raw
        for field in ("revision", "absolute", "device", "last_completed_ms_bits", "expert_groups_gt128",
                      "miss_payload_min_bytes", "actual_rms", "position", "case", "passed"):
            key = ('"' + field + '":').encode()
            self.raw.write_bytes(raw.replace(key, key + b"0," + key, 1))
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                self.collect()
        for value in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            self.raw.write_bytes(raw.replace(b'"actual_rms":2.0', b'"actual_rms":' + value, 1))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.collect()
        lines = raw.splitlines()
        for bad in (raw[:-1], raw[:-30], raw + b"\n", b"\xef\xbb\xbf" + raw,
                    raw.replace(b"prefill_wide_source", b"\xff", 1),
                    b"\n".join([*lines[:2], b"[]", *lines[3:]]) + b"\n"):
            self.raw.write_bytes(bad)
            with self.assertRaises(ValueError):
                self.collect()
        self.raw.write_bytes(raw.replace(b"\n", b"\r\n"))
        self.assertIs(self.collect()["passed"], True)

    def test_bounded_regular_and_growing_input_before_json_loading(self):
        self.assertLess(len(self.original_raw), MODULE.MAX_RAW_BYTES)
        self.raw.write_bytes(b" " * (MODULE.MAX_RAW_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "oversized"):
            self.collect()
        self.raw.write_bytes(self.original_raw)
        with mock.patch.object(Path, "open", return_value=mock.MagicMock()) as opened:
            opened.return_value.__enter__.return_value.read.return_value = b"x" * (MODULE.MAX_RAW_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "oversized"):
                self.collect()
            opened.return_value.__enter__.return_value.read.assert_called_once_with(MODULE.MAX_RAW_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "regular file"):
            MODULE.collect(self.root)

    def test_cli_valid_append_history_relative_explicit_path_and_shared_locking_helper(self):
        previous = b'{"kind":"previous","proof":"unchanged"}\n'
        self.results.write_bytes(previous)
        process = self.cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "")
        output = self.results.read_bytes()
        self.assertTrue(output.startswith(previous))
        self.assertEqual(len(output.splitlines()), 2)
        record = json.loads(output.splitlines()[1])
        self.assertEqual(record["kind"], "r4b_wide_prefill")
        self.assertEqual(record["source"], self.rows[0])
        self.assertEqual(record["raw_logs"], {"prefill_wide": str(self.raw.resolve())})
        self.assertEqual(self.raw.read_bytes(), self.original_raw)
        with mock.patch.object(MODULE, "append_record") as append:
            self.assertEqual(MODULE.main(["--raw", str(self.raw), "--results", str(self.results)]), 0)
            append.assert_called_once()
            self.assertEqual(append.call_args.args[0], self.results)
        self.assertEqual(self.cli(Path("caller-canonical.jsonl")).returncode, 0)

    def test_cli_alias_symlink_hardlink_unfinished_journal_and_errors_preserve_history(self):
        model = self.root / "qwen38-keep1-Q4_0.gguf"
        model.write_bytes(b"synthetic only; collector never opens weights")
        self.rows[0]["model"] = str(model)
        self.raw.write_bytes(encoded(self.rows))
        for source in (self.raw, model):
            before = source.read_bytes()
            symlink, hardlink = self.root / (source.name + ".symlink"), self.root / (source.name + ".hardlink")
            symlink.symlink_to(source)
            os.link(source, hardlink)
            for destination in (source, symlink, hardlink):
                self.assertEqual(self.cli(destination).returncode, 1)
                self.assertEqual(source.read_bytes(), before)
        previous = b'{"kind":"incomplete"}'
        self.results.write_bytes(previous)
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), previous)
        previous += b"\n"
        self.results.write_bytes(previous)
        self.raw.write_bytes(encoded(self.rows[:-1]))
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), previous)
        absent = self.root / "absent.jsonl"
        self.assertEqual(self.cli(absent).returncode, 1)
        self.assertFalse(absent.exists())

    def test_cli_requires_explicit_paths_no_tolerance_override_no_stdout_journal(self):
        for args in ([], ["--raw", str(self.raw)], ["--results", str(self.results)],
                     ["--raw", str(self.raw), "--results", str(self.results), "--absolute", "1"]):
            self.assertEqual(self.cli(args=args).returncode, 2)
            self.assertFalse(self.results.exists())
        for destination in (Path("/dev/stdout"), Path("/dev/stderr"), self.root):
            process = self.cli(destination)
            self.assertEqual(process.returncode, 1)
            self.assertEqual(process.stdout, "")
            self.assertEqual(self.raw.read_bytes(), self.original_raw)


if __name__ == "__main__":
    unittest.main()
