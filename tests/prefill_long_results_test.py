"""Synthetic exact long protocol evidence; no HIP, model run or real journal."""

import contextlib
import copy
import datetime
import io
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
from tools import record_memory
from tools import record_prefill_long as MODULE

SCRIPT = ROOT / "tools/record_prefill_long.py"
REVISION = "0123456789abcdef0123456789abcdef01234567"
VOCAB, Q40, Q41 = 248320, 2764800, 2867200
PAYLOAD, LOGITS, HANDOFF, STAGING, WORKSPACE = 68262297600, 1017118720, 41943040, 183500800, 2433482752
PHASES = ["reference_n1", "canonical1024", "occupied5_then997"]
COUNTERS = ("expert_hits", "expert_misses", "expert_upload_bytes")
CATEGORIES = ("weights", "expert_slots", "qsa_kv", "qsa_index", "gdn_state", "ple_state", "workspace")
COUNTS = ("values", "finite_values", "compared_values", "finite_pairs", "nonfinite_actual", "nonfinite_reference",
          "violations", "bit_mismatches_diagnostic", "argmax_rows_diagnostic", "argmax_agree_diagnostic")


def zero_stats():
    return {"consumed_tokens": 0, **dict.fromkeys(COUNTERS, 0), "last_completed_ms": 0, "last_completed_ms_bits": 0}


def zero_groups():
    return {"last_max_expert_group_assignments": 0, "expert_groups_gt128": 0}


def metrics(offset, rows, compared):
    return {"values": rows * VOCAB, "finite_values": rows * VOCAB,
            "compared_values": rows * VOCAB if compared else 0, "finite_pairs": rows * VOCAB if compared else 0,
            "nonfinite_actual": 0, "nonfinite_reference": 0, "violations": 0, "bit_mismatches_diagnostic": 0,
            "argmax_rows_diagnostic": rows if compared else 0, "argmax_agree_diagnostic": rows if compared else 0,
            "maxabs": 0 if compared else None, "rms": 0 if compared else None, "maxboundratio": 0 if compared else None,
            "maxabs_coordinate": {"position": offset, "vocabulary_index": 0} if compared else None,
            "maxboundratio_coordinate": {"position": offset, "vocabulary_index": 0} if compared else None,
            "first_violation_coordinate": None, "first_violation_actual": None,
            "first_violation_reference": None, "first_violation_bound": None}


def combined(components):
    result = metrics(0, 1, False)
    for key in COUNTS:
        result[key] = sum(e[key] for e in components)
    paired = [e for e in components if e["finite_pairs"]]
    if paired:
        for metric, coord in (("maxabs", "maxabs_coordinate"), ("maxboundratio", "maxboundratio_coordinate")):
            winner = max(paired, key=lambda e: e[metric])
            result[metric], result[coord] = winner[metric], copy.deepcopy(winner[coord])
        result["rms"] = math.hypot(*(e["rms"] * math.sqrt(e["finite_pairs"]) for e in paired)) / math.sqrt(result["finite_pairs"])
    return result


def fixture(teacher=4096, no_big_phase=None):
    """Independent spelling of driver records; baseline N1 evidence is compressed."""
    total, chunks = teacher + 32, (teacher - 5 + 996) // 997
    win_counts = [total, teacher // 1024 + 32, 5 + chunks + 32]
    record_count = 93 if teacher == 4096 else 129
    source = {
        "kind": "prefill_long_source", "protocol": 1, "revision": REVISION, "dirty": True,
        "model": '/models/данные "quoted"/qwen38-keep1-Q4_0.gguf', "model_bytes": 75399121792,
        "runtime": "own_48_layer_HIP", "session_instances": 1,
        "config": {"capacity": total, "expert_slots": 112, "max_batch_tokens": 1024, "trace": False},
        "large_RAM_opt_in": "explicit_ROWS_4096_or_16384",
        "teacher_family": "new_BOS_then_monotone_teacher_IDs_not_original_user_prompt_or_baseline_parity",
        "source_ID_formula": "id[p]=248044 if p==0 else 99+p; 0<=p<ROWS+32", "source_id_count": total,
        "teacher_id_count": teacher, "teacher_BOS": 248044, "teacher_non_BOS_first": 100, "teacher_last": teacher + 98,
        "continuation_id_count": 32, "continuation_first": teacher + 99, "continuation_last": total + 98,
        "every_source_ID_compile_time_and_runtime_checked": True, "expected_windows_per_phase": win_counts,
        "phase_order": PHASES.copy(),
        "teacher_schedules": ["ROWS_N1", "ROWS/1024_N1024", "5_N1_then_ceil((ROWS-5)/997)_chunks_final_remainder"],
        "mixed_final_remainder": teacher - 5 - (chunks - 1) * 997, "continuation_schedule": "32_N1_each_phase",
        "baseline_checkpoint_count": teacher // 1024 + 1, "expected_success_records": record_count,
        "expected_success_windows": sum(win_counts), "expected_finite_values": 3 * total * VOCAB,
        "expected_compared_values": 2 * total * VOCAB, "expected_memory_observations": sum(win_counts) + 21,
        "expected_invalid_rejections": 8,
        "gate": {"absolute": .02, "relative": .002, "formula": "abs(actual-reference)<=.02+.002*abs(reference)",
                 "all_finite_required": True, "allowed_violations": 0, "bit_equality_required": False,
                 "argmax_equality_required": False, "cli_adjustable": False},
        "reference_scope": "all_ROWS_plus32_full_vocabulary_logits_retained_sequential_N1_same_Session",
        "vocabulary": VOCAB, "weights": "unchanged_loaded_GGUF_values_and_tensor_types", "kv": "Q4_0_K_and_V",
        "sampling": "teacher_forced", "reset_retains_expert_cache": True, "cold_cache_equality_claim": False,
        "independent_HF_reference": False,
        "timing_scope": "diagnostic_correctness_completed_call_wall_excludes_comparison_memory_reset_rejection_not_PP_or_full_request_performance",
        "counter_scope": "480_assignments_per_row_hits_include_within_call_reuse_misses_count_uploaded_groups",
        "uploaded_group_payload_min_bytes": Q40, "uploaded_group_payload_max_bytes": Q41,
        "expert_payload_reads_required": 144, "expert_payload_bytes_read_required": PAYLOAD,
        "read_counter_scope": "constructor_payload_reads_not_physical_SSD_trace",
        "preallocated_reference_bytes": total * VOCAB * 4, "preallocated_preservation_bytes": LOGITS,
        "float_guard_elements_each_end": 16, "float_guard_total_bytes": 256, "guarded_input_workspace_bytes": 4228,
        "known_host_min_bytes_before_embedding_metadata_runtime": PAYLOAD + total * VOCAB * 4 + 2 * LOGITS + HANDOFF + STAGING + 256 + 4228,
        "host_accounting": "expert_RAM_plus_reference_plus_preservation_plus_Session_logits_plus_pinned_handoff_and_expert_staging_no_weight_copy",
        "workspace_aggregate_min_bytes_per_device": WORKSPACE, "individual_buffer_capacities_observable": False,
        "physical_128_column_tile_proven": False,
        "coverage_scope": "actual_teacher_rows_and_call_offsets_cover_visible2047..2056_each_mod4_not_selected_ID_trace_or_synthetic_QKV",
        "performance_claim": False, "peak_VRAM_qualification_claim": False, "R4_complete_claim": False,
        "remaining_R4_evidence": "separate_full_request_512_output_benchmark_and_peak_VRAM_qualification",
    }
    devices = []
    for i in (0, 1):
        d = {"device": i, "first_layer": 24 * i, "last_layer": 24 * i + 23, "gdn_layers": 18, "qsa_layers": 6,
             "weights": 1100000000 + i * 12345678, "expert_slots": [7500595200, 7431782400][i],
             "qsa_kv": 3456 * total, "qsa_index": 6 * (total // 4 * 128 + 384) * 4,
             "gdn_state": 58834944, "ple_state": 368640 if i == 0 else 0, "workspace": WORKSPACE + i * 43210,
             "owned_buffers": 1001 + i * 37, "total_vram": 17163091968}
        d["owned_bytes"] = sum(d[key] for key in CATEGORIES)
        d["owned_peak_bytes"] = d["owned_bytes"] + 4096
        d["free_vram"] = d["total_vram"] - d["owned_bytes"] - 64 * 1024 * 1024
        devices.append(d)
    loaded = {"capacity": total, "expert_slots": 112, "ownership_verified": True,
              "ram_expert_capacity": PAYLOAD + 4096, "ram_expert_payload": PAYLOAD, "host_embedding_capacity": 521472000,
              "host_logit_capacity": LOGITS + 4096, "pinned_handoff": HANDOFF, "pinned_expert_staging": STAGING,
              "expert_payload_reads": 144, "expert_payload_bytes_read": PAYLOAD, "devices": devices}
    known_host = sum(loaded[k] for k in ("ram_expert_capacity", "host_embedding_capacity", "host_logit_capacity",
                                        "pinned_handoff", "pinned_expert_staging")) + total * VOCAB * 4 + LOGITS + 256 + 4228
    records = [source]

    def snapshot():
        result = copy.deepcopy(loaded)
        for d in result["devices"]:
            d["free_vram"] -= len(records) * 1024
        return result

    def memory_row(event, index):
        return {"kind": "prefill_long_memory", "protocol": 1, "event": event, "phase_index": index,
                "stats": zero_stats(), "route_stats": zero_groups(), "memory": snapshot(),
                "reported_known_host_and_fixture_bytes": known_host,
                "host_accounting_excludes_metadata_allocator_runtime_overhead": True, "passed": True}

    records.append(memory_row("loaded", 0))
    schedules = [[1024] * (teacher // 1024) + [32], [1024] * (teacher // 1024) + [1] * 32,
                 [1] * 5 + [997] * (chunks - 1) + [teacher - 5 - (chunks - 1) * 997] + [1] * 32]
    summaries, rejection_values = [], 0
    for index, widths in enumerate(schedules):
        records.append(memory_row("reset", index))
        previous, groups_before, offset, current = zero_stats(), zero_groups(), 0, []
        for position, rows in enumerate(widths):
            # Baseline N1: cyclic disjoint ten-expert choices in each of48
            # layers miss480 each row. Candidate: ten repeated experts/layer
            # give480 misses then reuse; a deliberately distributed first
            # N1024 gives max20 and NO group>128 despite its logical width.
            distributed = index == no_big_phase or (index == 1 and position == 0)
            if index == 0:
                misses, maximum, group_delta = 480 * rows, 1, 0
            else:
                misses = 48 * 512 if distributed and rows > 128 else 480
                maximum = math.ceil(10 * rows / 512) if distributed else rows
                group_delta = 480 if maximum > 128 else 0
            hits = 480 * rows - misses
            upload = (misses - misses // 8) * Q40 + misses // 8 * Q41
            timing = 10.5 + index + (offset + rows) / 128
            after = {"consumed_tokens": offset + rows, "expert_hits": previous["expert_hits"] + hits,
                     "expert_misses": previous["expert_misses"] + misses,
                     "expert_upload_bytes": previous["expert_upload_bytes"] + upload, "last_completed_ms": timing,
                     "last_completed_ms_bits": struct.unpack("!Q", struct.pack("!d", timing))[0]}
            groups_after = {"last_max_expert_group_assignments": maximum,
                            "expert_groups_gt128": groups_before["expert_groups_gt128"] + group_delta}
            common = {"protocol": 1, "phase_index": index, "offset": offset, "rows": rows,
                      "segment": "teacher" if offset < teacher else "continuation", "stats_before": copy.deepcopy(previous),
                      "stats_after": after, "route_stats_before": copy.deepcopy(groups_before), "route_stats_after": groups_after,
                      "errors": metrics(offset, rows, index > 0), "memory": snapshot(), "passed": True}
            if index == 0:
                wall = rows * (10.5 + index + offset / 128) + rows * (rows + 1) / 256 + rows * .25
                w = {"kind": "prefill_long_baseline_checkpoint", **common, "checkpoint_index": position,
                     "windows": rows, "max_expert_group_count": 1, "expert_groups_gt128": 0,
                     "correctness_completed_call_wall_ms_sum": wall, "every_N1_call_finite_stats_routes_memory_guards_checked": True}
            else:
                w = {"kind": "prefill_long_window", **common, "phase": PHASES[index], "window_index": position,
                     "stage": "memory_and_full_live_span_preservation", "completed_call": True, "expected_routes": 480 * rows,
                     "expert_groups_gt128_delta": group_delta, "correctness_completed_call_wall_ms": timing + .25,
                     "completed_blocks_before": offset // 4, "completed_blocks_after": (offset + rows) // 4,
                     "tail_before": offset % 4, "tail_after": (offset + rows) % 4,
                     "stats_and_full_live_span_preserved": True}
            records.append(w)
            current.append(w)
            previous, groups_before, offset = after, groups_after, offset + rows
        proofs = []
        if index:
            first = 0 if index == 1 else 5
            cases = [("oversized1025", 1025, first), ("negative_last1024", 1024, first), ("oov_last1024", 1024, first),
                     ("capacity_remaining1_length2", 2, len(current) - 2)]
            for case, input_rows, prior_index in cases:
                prior, continued = current[prior_index:prior_index + 2]
                consumed = prior["offset"] + prior["rows"]
                proofs.append({"case": case, "input_rows": input_rows, "offset": consumed, "prior_rows": prior["rows"],
                               "input_fits_remaining_capacity": input_rows <= total - consumed,
                               "preserved_fullspan_values": prior["rows"] * VOCAB,
                               "stats_before_and_after": copy.deepcopy(prior["stats_after"]),
                               "route_stats_before_and_after": copy.deepcopy(prior["route_stats_after"]), "exception": "invalid_argument",
                               "all_public_stats_and_route_stats_bitwise_preserved": True, "fullspan_bits_preserved": True,
                               "memory_ledger_preserved": True, "guards_preserved": True, "continued_without_reset": True,
                               "continuation_offset": continued["offset"], "continuation_rows": continued["rows"],
                               "continuation_compared_values": continued["rows"] * VOCAB, "continuation_violations": 0})
        rejection_values += sum(p["preserved_fullspan_values"] for p in proofs)
        teacher_end = next(w for w in current if w["offset"] + w["rows"] == teacher)
        # Independent closed-form coverage for these exact schedules.
        cover = {"visible2047_2056_mask": 1023, "visible_mod4_mask": 15,
                 "multirow_visible2047_2056_mask": 1023 if index else 0,
                 "occupied_chunk_start_mod4_mask": [0, 1, 15][index],
                 "occupied_chunks": [0, teacher // 1024 - 1, chunks][index],
                 "completed_block_crossing_windows": [teacher // 4, teacher // 1024, 1 + chunks][index],
                 "occupied_budget2052_crossing": index > 0, "all_boundary_visibilities_observed": True,
                 "scope": "actual_accepted_teacher_rows_logical_causal_visibility_not_GPU_selected_ID_trace"}
        wall_key = "correctness_completed_call_wall_ms_sum" if index == 0 else "correctness_completed_call_wall_ms"
        summary = {"kind": "prefill_long_phase", "protocol": 1, "phase_index": index, "phase": PHASES[index],
                   "windows": win_counts[index], "teacher_rows": teacher, "continuation_rows": 32,
                   "errors": combined([w["errors"] for w in current]), "teacher_stats": copy.deepcopy(teacher_end["stats_after"]),
                   "final_stats": copy.deepcopy(previous), "teacher_route_stats": copy.deepcopy(teacher_end["route_stats_after"]),
                   "final_route_stats": copy.deepcopy(groups_before),
                   "max_expert_group_count": max(w["route_stats_after"]["last_max_expert_group_assignments"] for w in current),
                   "expert_groups_gt128": groups_before["expert_groups_gt128"],
                   "correctness_completed_call_wall_ms_sum": sum(w[wall_key] for w in current),
                   "coverage": cover, "invalid_proofs": proofs, "passed": True}
        summaries.append(summary)
        records.append(summary)
    records.append(memory_row("final_reset", 2))
    minima = [min(r["memory"]["devices"][i]["free_vram"] for r in records if "memory" in r) - 1024 for i in (0, 1)]
    records.append({"kind": "prefill_long_complete", "protocol": 1, "teacher_rows_per_phase": teacher,
                    "continuation_rows_per_phase": 32, "phase_count": 3, "timeline_rows": 3 * total,
                    "window_count": sum(win_counts), "record_count": record_count,
                    "errors": combined([p["errors"] for p in summaries]),
                    "max_expert_group_count": max(p["max_expert_group_count"] for p in summaries),
                    "expert_groups_gt128": sum(p["expert_groups_gt128"] for p in summaries),
                    "invalid_window_rejections": 8, "rejection_preserved_fullspan_values": rejection_values,
                    "memory_observation_count": sum(win_counts) + 21,
                    "memory_preserved_fullspan_values": 3 * total * VOCAB + 2 * rejection_values,
                    "minimum_free_vram_bytes": minima,
                    "correctness_completed_call_wall_ms_sum": sum(p["correctness_completed_call_wall_ms_sum"] for p in summaries),
                    "steady_owners_categories_Buffer_counts_peaks_and_host_read_ledger": True, "fixture_guards_preserved": True,
                    "raii_session_and_fixture_cleanup_completed": True, "owned_buffer_release_measured": False,
                    "full_prefill_self_parity_passed": True, "logical_group_gt128_proven": True,
                    "physical_128_column_tile_proven": False, "independent_HF_reference": False, "cold_cache_equality_claim": False,
                    "performance_claim": False, "peak_VRAM_qualification_claim": False, "R4_complete_claim": False, "passed": True})
    return records


def encoded(rows):
    return ("".join(json.dumps(r, allow_nan=False, separators=(",", ":")) + "\n" for r in rows)).encode()


def target(obj, path):
    for key in path:
        obj = obj[key]
    return obj


class PrefillLongResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = fixture()
        cls.original_raw = encoded(cls.original)

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="prefill-long-results-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.raw, self.results = self.root / "long.jsonl", self.root / "results.jsonl"
        self.rows = copy.deepcopy(self.original)
        self.raw.write_bytes(self.original_raw)
        self.windows = [[i for i, r in enumerate(self.rows) if r["kind"] == "prefill_long_window" and r["phase_index"] == p]
                        for p in range(3)]
        self.checkpoints = [i for i, r in enumerate(self.rows) if r["kind"] == "prefill_long_baseline_checkpoint"]
        self.phases = [i for i, r in enumerate(self.rows) if r["kind"] == "prefill_long_phase"]

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

    def change(self, row, path, bad):
        obj = target(self.rows[row], path[:-1])
        old = obj[path[-1]]
        obj[path[-1]] = bad
        try:
            self.reject(self.rows)
        finally:
            obj[path[-1]] = old

    def cli(self, destination=None, args=None):
        command = [sys.executable, "-B", str(SCRIPT)]
        command += args if args is not None else ["--raw", self.raw.name, "--results", str(destination or self.results)]
        return subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30, check=False)

    def recompute_errors(self):
        for index, p in enumerate(self.phases):
            positions = self.checkpoints if index == 0 else self.windows[index]
            self.rows[p]["errors"] = combined([self.rows[i]["errors"] for i in positions])
        self.rows[-1]["errors"] = combined([self.rows[i]["errors"] for i in self.phases])

    def test_positive_4096_and_16384_exact_counts_compact_scopes(self):
        for teacher, expected in ((4096, (93, 4206, 4227, 103)), (16384, (129, 16518, 16539, 427))):
            rows = fixture(teacher)
            result = self.collect(rows)
            self.assertEqual(len(rows), expected[0])
            self.assertEqual(result["kind"], "r4b_long_prefill")
            self.assertEqual(result["revision"], REVISION)
            self.assertIs(result["dirty"], True)
            self.assertEqual(result["source"], rows[0])
            self.assertEqual(result["complete"], rows[-1])
            self.assertEqual(result["complete"]["window_count"], expected[1])
            self.assertEqual(result["complete"]["memory_observation_count"], expected[2])
            self.assertEqual(result["source"]["mixed_final_remainder"], expected[3])
            self.assertEqual(result["window_summary"]["serialized_baseline_checkpoints"], teacher // 1024 + 1)
            self.assertEqual(result["window_summary"]["serialized_candidate_windows"], expected[1] - teacher - 32)
            self.assertEqual(result["raw_logs"], {"prefill_long": str(self.raw.resolve())})
            self.assertEqual(datetime.datetime.fromisoformat(result["timestamp"]).utcoffset(), datetime.timedelta(0))
            for field in ("records", "windows", "checkpoints"):
                self.assertNotIn(field, result)
            for field in ("GPU_selected_ID_trace_qualified", "physical_128_column_tile_proven", "individual_buffer_capacities_qualified",
                          "full_ram_accounting_qualified", "owned_buffer_release_measured", "independent_hf_claim", "R4_complete_claim",
                          "performance_claim", "peak_VRAM_qualification_claim", "speedup_claim", "cold_cache_equality_claim", "mtp"):
                self.assertIs(result[field], False)
            self.assertIn("not_PP_or_full_request_performance", result["timing_scope"])
            self.assertLess(len(encoded(result["phases"])), len(encoded(rows)))

    def test_exact_ids_schedule_checkpoint_and_live_span_lifetime(self):
        result = self.collect()
        self.assertEqual(result["source"]["expected_windows_per_phase"], [4128, 36, 42])
        self.assertEqual([self.rows[i]["rows"] for i in self.checkpoints], [1024] * 4 + [32])
        self.assertEqual([self.rows[i]["rows"] for i in self.windows[2]], [1] * 5 + [997] * 4 + [103] + [1] * 32)
        for index, width in ((1, 1024), (2, 997)):
            proofs = result["phases"][index]["invalid_proofs"]
            self.assertEqual([p["prior_rows"] for p in proofs], [width] * 3 + [1])
            self.assertEqual([p["input_rows"] for p in proofs], [1025, 1024, 1024, 2])
            self.assertEqual(proofs[-1]["continuation_offset"], 4127)
            self.assertEqual(proofs[-1]["continuation_rows"], 1)
        self.assertEqual(result["complete"]["rejection_preserved_fullspan_values"], 6065 * VOCAB)
        self.assertEqual(result["complete"]["memory_preserved_fullspan_values"], (3 * 4128 + 2 * 6065) * VOCAB)
        self.assertEqual(len(result["memory"]), 5)

    def test_source_geometry_provenance_formula_gate_counts_and_scopes_frozen(self):
        for field, bad in (("revision", "g" * 40), ("revision", "a" * 39), ("dirty", 1), ("dirty", "true"),
                           ("model", "qwen38-keep1-Q4_0.gguf"), ("model", "/models/full-PLE.gguf"),
                           ("model_bytes", 0), ("model_bytes", 1 << 63), ("teacher_id_count", 8192),
                           ("source_ID_formula", "original_prompt"), ("teacher_BOS", 100), ("teacher_non_BOS_first", 99),
                           ("teacher_last", 4195), ("continuation_first", 4196), ("continuation_last", 4225),
                           ("source_id_count", 4096), ("session_instances", 2), ("vocabulary", 248321),
                           ("kv", "F16"), ("weights", "requantized"), ("expected_success_records", 94),
                           ("expected_finite_values", 3 * 4128 * VOCAB - 1), ("preallocated_reference_bytes", 1),
                           ("expected_memory_observations", 4206), ("workspace_aggregate_min_bytes_per_device", WORKSPACE - 1),
                           ("mixed_final_remainder", 102), ("baseline_checkpoint_count", 4096),
                           ("reference_scope", "independent_HF"), ("performance_claim", True),
                           ("individual_buffer_capacities_observable", True)):
            self.change(0, (field,), bad)
        for path, bad in ((("config", "expert_slots"), 1), (("config", "capacity"), 16384),
                          (("config", "max_batch_tokens"), 997), (("config", "trace"), True),
                          (("gate", "absolute"), .021), (("gate", "relative"), .003),
                          (("gate", "allowed_violations"), 1), (("gate", "argmax_equality_required"), True),
                          (("expected_windows_per_phase", 0), 5), (("phase_order", 1), PHASES[2])):
            self.change(0, path, bad)
        self.rows[0].update(revision="ABCDEF01" * 5, dirty=False)
        self.assertIs(self.collect(self.rows)["dirty"], False)

    def test_unknown_and_missing_fields_in_every_nested_schema(self):
        w, c, p = self.windows[2][5], self.checkpoints[0], self.phases[2]
        representatives = [(0, ()), (0, ("config",)), (0, ("gate",)), (1, ()), (1, ("memory",)),
                           (1, ("memory", "devices", 0)), (1, ("stats",)), (1, ("route_stats",)),
                           (c, ()), (c, ("errors",)), (w, ()), (w, ("stats_after",)), (w, ("route_stats_after",)),
                           (w, ("errors",)), (w, ("errors", "maxabs_coordinate")), (p, ()), (p, ("coverage",)),
                           (p, ("invalid_proofs", 0)), (p, ("invalid_proofs", 0, "stats_before_and_after")),
                           (p, ("invalid_proofs", 0, "route_stats_before_and_after")), (92, ())]
        for row, path in representatives:
            obj = target(self.rows[row], path)
            obj["unknown"] = 1
            try:
                self.reject(self.rows)
            finally:
                del obj["unknown"]
            field = next(iter(obj))
            old = obj.pop(field)
            try:
                self.reject(self.rows)
            finally:
                obj[field] = old

    def test_chronological_checkpoints_candidate_windows_and_protocol_order(self):
        for row in (self.checkpoints[0], self.checkpoints[-1], self.windows[1][1], self.windows[2][5], self.windows[2][9]):
            for field in ("offset", "rows", "phase_index"):
                self.change(row, (field,), self.rows[row][field] + 1)
        self.change(self.checkpoints[1], ("checkpoint_index",), 0)
        self.change(self.checkpoints[0], ("windows",), 1)
        self.change(self.windows[2][5], ("window_index",), 0)
        self.change(self.windows[2][5], ("stage",), "API")
        self.change(self.windows[2][5], ("segment",), "continuation")
        for row in (0, 1, self.checkpoints[0], self.phases[0], self.phases[2], 91, 92):
            self.reject(self.rows[:row] + self.rows[row + 1:])
        for row in (self.checkpoints[0], self.windows[1][0], self.phases[1]):
            self.rows[row], self.rows[row + 1] = self.rows[row + 1], self.rows[row]
            try:
                self.reject(self.rows)
            finally:
                self.rows[row], self.rows[row + 1] = self.rows[row + 1], self.rows[row]
        self.reject(self.rows + [copy.deepcopy(self.rows[-1])])
        self.rows[0]["teacher_id_count"] = 16384
        self.reject(self.rows)  # A 93-record source cannot stand in for129.

    def test_failure_records_noncompletion_and_cleanup_are_rejected(self):
        w = self.windows[1][0]
        self.change(w, ("passed",), False)
        self.change(w, ("completed_call",), False)
        self.change(w, ("stats_and_full_live_span_preserved",), False)
        self.change(self.checkpoints[0], ("every_N1_call_finite_stats_routes_memory_guards_checked",), False)
        self.change(92, ("raii_session_and_fixture_cleanup_completed",), False)
        self.change(92, ("fixture_guards_preserved",), False)
        self.change(92, ("full_prefill_self_parity_passed",), False)
        for row in (w, 92):
            old = self.rows[row]
            self.rows[row] = {"kind": "prefill_long_failure", "protocol": 1, "passed": False,
                              "session_constructed": True, "raii_session_and_fixture_cleanup_completed": True, "error": "gate failed"}
            try:
                self.reject(self.rows)
            finally:
                self.rows[row] = old

    def test_oldstats_bits_unsigned_bounds_counters_and_reset_state(self):
        w, c = self.windows[2][5], self.checkpoints[0]
        for row in (w, c):
            for block in ("stats_before", "stats_after"):
                for field in zero_stats():
                    self.change(row, (block, field), self.rows[row][block][field] + 1)
            self.change(row, ("stats_after", "expert_hits"), True)
            self.change(row, ("stats_after", "expert_misses"), 480.)
            self.change(row, ("stats_after", "expert_upload_bytes"), 1 << 64)
            self.change(row, ("stats_after", "last_completed_ms_bits"), -1)
        for i, row in enumerate(self.rows):
            if row["kind"] == "prefill_long_memory":
                for field in ("consumed_tokens", "last_completed_ms", "last_completed_ms_bits"):
                    self.change(i, ("stats", field), 1)
                self.change(i, ("route_stats", "expert_groups_gt128"), 1)
                self.change(i, ("phase_index",), 3)
        for bad in (-0.0, 0, 1e300):
            self.change(w, ("correctness_completed_call_wall_ms",), bad) if bad <= 0 else self.change(w, ("stats_after", "last_completed_ms"), bad)

    def test_480_routes_payload_range_and_nonmonotonic_counters(self):
        w = self.windows[2][5]
        before, after = self.rows[w]["stats_before"], self.rows[w]["stats_after"]
        for bad in (before["expert_upload_bytes"] - 1, before["expert_upload_bytes"] + 480 * Q40 - 1,
                    before["expert_upload_bytes"] + 480 * Q41 + 1):
            self.change(w, ("stats_after", "expert_upload_bytes"), bad)
        self.change(w, ("stats_after", "expert_hits"), before["expert_hits"] - 1)
        self.change(w, ("expected_routes",), 480 * 997 - 1)
        self.assertEqual(after["expert_hits"] + after["expert_misses"], after["consumed_tokens"] * 480)

    def test_route_group_observation_bounds_not_logical_N_and_required_each_candidate(self):
        first = self.windows[1][0]
        self.assertEqual(self.rows[first]["rows"], 1024)
        self.assertEqual(self.rows[first]["route_stats_after"]["last_max_expert_group_assignments"], 20)
        self.assertEqual(self.rows[first]["expert_groups_gt128_delta"], 0)
        for path, bad in ((("route_stats_after", "last_max_expert_group_assignments"), 1025),
                          (("route_stats_after", "last_max_expert_group_assignments"), 129),
                          (("route_stats_after", "expert_groups_gt128"), 1), (("expert_groups_gt128_delta",), 1)):
            self.change(first, path, bad)
        w = self.windows[2][5]
        self.change(w, ("route_stats_after", "expert_groups_gt128"), 480 * 997 // 129 + 1)
        self.change(self.windows[2][0], ("route_stats_after", "last_max_expert_group_assignments"), 2)
        self.change(self.checkpoints[0], ("max_expert_group_count",), 1024)
        self.change(self.checkpoints[0], ("route_stats_after", "expert_groups_gt128"), 1)
        for index in (1, 2):
            self.raw.write_bytes(encoded(fixture(no_big_phase=index)))
            with self.assertRaisesRegex(ValueError, PHASES[index] + ": actual logical group>128"):
                self.collect()

    def test_memory_owners_ledger_geometry_categories_floor_and_steady_growth(self):
        w = self.windows[2][5]
        for field, bad in (("capacity", 4096), ("expert_slots", 1), ("ownership_verified", False),
                           ("ram_expert_capacity", PAYLOAD - 1), ("host_logit_capacity", LOGITS - 1),
                           ("pinned_handoff", 40960), ("pinned_expert_staging", STAGING // 2),
                           ("expert_payload_reads", 145), ("expert_payload_bytes_read", PAYLOAD - 1)):
            self.change(w, ("memory", field), bad)
        for owner in (0, 1):
            for field, bad in (("device", 1 - owner), ("last_layer", 47), ("gdn_layers", 0),
                               ("expert_slots", 1), ("qsa_kv", 1), ("qsa_index", 1), ("gdn_state", 1),
                               ("workspace", WORKSPACE - 1), ("owned_buffers", 0), ("owned_bytes", 1),
                               ("owned_peak_bytes", 1), ("free_vram", 0), ("total_vram", 1)):
                if field == "last_layer" and owner == 1:
                    bad = 23
                self.change(w, ("memory", "devices", owner, field), bad)
        self.change(1, ("reported_known_host_and_fixture_bytes",), 1)
        self.change(2, ("reported_known_host_and_fixture_bytes",), self.rows[2]["reported_known_host_and_fixture_bytes"] + 4)
        d = self.rows[w]["memory"]["devices"][0]
        d["workspace"] += 1
        d["owned_bytes"] += 1
        d["owned_peak_bytes"] += 1
        self.reject(self.rows)  # Consistent category arithmetic cannot hide allocation growth.

    def test_reported_host_fixture_capacity_slack_free_VRAM_variation_and_hidden_minima(self):
        for row in self.rows:
            if row["kind"] == "prefill_long_memory":
                row["reported_known_host_and_fixture_bytes"] += 4096
        self.rows[-1]["minimum_free_vram_bytes"] = [1, 1]
        self.assertIs(self.collect(self.rows)["passed"], True)
        self.change(92, ("minimum_free_vram_bytes",), [0, 1])
        self.change(92, ("minimum_free_vram_bytes",), [1 << 64, 1])
        self.change(92, ("minimum_free_vram_bytes",), [1])

    def test_causal_visible2047_to2056_and_mod4_evidence_recomputed_from_schedule(self):
        for index, phase in enumerate(self.phases):
            for field in ("visible2047_2056_mask", "visible_mod4_mask", "multirow_visible2047_2056_mask",
                          "occupied_chunk_start_mod4_mask", "occupied_chunks", "completed_block_crossing_windows"):
                self.change(phase, ("coverage", field), self.rows[phase]["coverage"][field] + 1)
            self.change(phase, ("coverage", "occupied_budget2052_crossing"), index == 0)
            self.change(phase, ("coverage", "scope"), "GPU_selected_ID_trace")
        for field in ("completed_blocks_before", "completed_blocks_after", "tail_before", "tail_after"):
            w = self.windows[2][5]
            self.change(w, (field,), self.rows[w][field] + 1)
        self.assertEqual(self.rows[self.phases[2]]["coverage"]["occupied_chunk_start_mod4_mask"], 15)

    def test_invalid_proof_case_order_states_guards_and_next_call_lifetime(self):
        p = self.phases[2]
        for field in zero_stats():
            self.change(p, ("invalid_proofs", 0, "stats_before_and_after", field), 0)
        for field in zero_groups():
            self.change(p, ("invalid_proofs", 0, "route_stats_before_and_after", field), 0)
        for field in ("all_public_stats_and_route_stats_bitwise_preserved", "fullspan_bits_preserved", "memory_ledger_preserved",
                      "guards_preserved", "continued_without_reset"):
            self.change(p, ("invalid_proofs", 0, field), False)
        for case, field, bad in ((0, "case", "empty"), (0, "prior_rows", 1), (0, "offset", 997),
                                 (0, "preserved_fullspan_values", VOCAB), (0, "continuation_offset", 1003),
                                 (0, "continuation_rows", 1), (0, "continuation_compared_values", VOCAB),
                                 (0, "continuation_violations", 1), (1, "exception", "out_of_range"),
                                 (3, "input_fits_remaining_capacity", True), (3, "prior_rows", 997)):
            self.change(p, ("invalid_proofs", case, field), bad)
        proofs = self.rows[p]["invalid_proofs"]
        proofs[0], proofs[1] = proofs[1], proofs[0]
        self.reject(self.rows)

    def test_metric_counts_coordinates_finiteness_and_strict_bound_ratio(self):
        w = self.windows[2][5]
        for field, bad in (("values", VOCAB), ("finite_values", VOCAB), ("compared_values", VOCAB), ("finite_pairs", 0),
                           ("nonfinite_actual", 1), ("nonfinite_reference", 1), ("violations", 1),
                           ("bit_mismatches_diagnostic", 997 * VOCAB + 1), ("argmax_rows_diagnostic", 0),
                           ("argmax_agree_diagnostic", 998), ("maxabs", None), ("maxabs", 1e300),
                           ("rms", .001), ("maxboundratio", math.nextafter(1., math.inf)),
                           ("first_violation_coordinate", {"position": 5, "vocabulary_index": 0}),
                           ("first_violation_actual", 0)):
            self.change(w, ("errors", field), bad)
        for field in ("maxabs_coordinate", "maxboundratio_coordinate"):
            for key, bad in (("position", 1002), ("vocabulary_index", VOCAB), ("vocabulary_index", 1)):
                self.change(w, ("errors", field, key), bad)
        self.change(self.checkpoints[0], ("errors", "maxabs"), 0)
        self.change(self.checkpoints[0], ("errors", "bit_mismatches_diagnostic"), 1)

    def test_realizable_nonzero_gate_bit_argmax_diagnostic_and_signed_zero(self):
        w = self.windows[2][5]
        e = self.rows[w]["errors"]
        # One FP32 perturbation of a zero reference by.015625, within.02.
        e.update(maxabs=.015625, rms=.015625 / math.sqrt(997 * VOCAB), maxboundratio=.015625 / .02,
                 bit_mismatches_diagnostic=1, argmax_agree_diagnostic=996,
                 maxabs_coordinate={"position": 5, "vocabulary_index": 1},
                 maxboundratio_coordinate={"position": 5, "vocabulary_index": 1})
        self.recompute_errors()
        self.assertIs(self.collect(self.rows)["passed"], True)
        self.change(w, ("errors", "maxboundratio"), math.nextafter(1., math.inf))
        self.change(w, ("errors", "rms"), .015625)
        self.change(w, ("errors", "rms"), 0)
        self.change(w, ("errors", "bit_mismatches_diagnostic"), 0)
        # The driver has NO reference max/norm; a .03125 error at reference16
        # satisfies.052 and must not be rejected by an invented absolute.02 gate.
        e.update(maxabs=.03125, rms=.03125 / math.sqrt(997 * VOCAB), maxboundratio=.03125 / .052)
        self.recompute_errors()
        self.assertIs(self.collect(self.rows)["passed"], True)
        e.update(maxabs=0, rms=0, maxboundratio=0, argmax_agree_diagnostic=997,
                 maxabs_coordinate={"position": 5, "vocabulary_index": 0},
                 maxboundratio_coordinate={"position": 5, "vocabulary_index": 0})
        self.recompute_errors()
        self.assertEqual(self.collect(self.rows)["complete"]["errors"]["bit_mismatches_diagnostic"], 1)

    def test_phase_complete_aggregates_maxima_RMS_ties_and_counts(self):
        w = self.windows[1][0]
        self.rows[w]["errors"].update(maxabs=.015625, rms=.015625 / math.sqrt(1024 * VOCAB),
                                       maxboundratio=.015625 / .02, bit_mismatches_diagnostic=1)
        second = self.windows[2][5]
        self.rows[second]["errors"].update(maxabs=.015625, rms=.015625 / math.sqrt(997 * VOCAB),
                                           maxboundratio=.015625 / .02, bit_mismatches_diagnostic=1)
        self.recompute_errors()
        self.assertIs(self.collect(self.rows)["passed"], True)
        self.change(92, ("errors", "maxabs_coordinate", "position"), 5)  # First tied max is phase1/offset0.
        self.change(self.phases[1], ("errors", "rms"), 0)
        self.change(92, ("errors", "bit_mismatches_diagnostic"), 1)
        for field in ("window_count", "record_count", "timeline_rows", "memory_observation_count",
                      "memory_preserved_fullspan_values", "rejection_preserved_fullspan_values", "expert_groups_gt128"):
            self.change(92, (field,), self.rows[-1][field] + 1)
        self.change(self.phases[2], ("teacher_route_stats", "expert_groups_gt128"), 0)
        self.change(self.phases[2], ("correctness_completed_call_wall_ms_sum",), 1.)
        for field in ("owned_buffer_release_measured", "physical_128_column_tile_proven", "independent_HF_reference",
                      "performance_claim", "peak_VRAM_qualification_claim", "R4_complete_claim"):
            self.change(92, (field,), True)

    def test_duplicate_keys_nonfinite_malformed_UTF8_no_empty_or_extra_rows(self):
        raw = self.original_raw
        for field in ("revision", "absolute", "device", "last_completed_ms_bits", "expert_groups_gt128", "position", "case", "passed"):
            key = ('"' + field + '":').encode()
            self.raw.write_bytes(raw.replace(key, key + b"0," + key, 1))
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                self.collect()
        for bad in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            self.raw.write_bytes(raw.replace(b'"last_completed_ms":0', b'"last_completed_ms":' + bad, 1))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.collect()
        lines = raw.splitlines()
        for bad in (b"", raw[:-1], raw[:-30], raw + b"\n", b"\xef\xbb\xbf" + raw,
                    raw.replace(b"prefill_long_source", b"\xff", 1),
                    b"\n".join([*lines[:2], b"[]", *lines[3:]]) + b"\n",
                    b"\n".join([*lines[:2], b"", *lines[3:]]) + b"\n"):
            self.raw.write_bytes(bad)
            with self.assertRaises(ValueError):
                self.collect()
        self.raw.write_bytes(raw.replace(b"\n", b"\r\n"))
        self.assertIs(self.collect()["passed"], True)

    def test_regular_bounded_and_growing_input_before_JSON_loading(self):
        self.assertEqual(MODULE.MAX_RAW_BYTES, 16 * 1024 * 1024)
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

    def test_negative_zero_bits_and_numeric_boolean_schema_types(self):
        raw = self.original_raw.replace(b'"last_completed_ms":0,"last_completed_ms_bits":0',
                                        b'"last_completed_ms":-0,"last_completed_ms_bits":9223372036854775808', 1)
        self.raw.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.collect()
        for row, path, bad in ((0, ("teacher_id_count",), 4096.), (0, ("protocol",), True),
                               (self.windows[2][5], ("route_stats_after", "expert_groups_gt128"), 480.),
                               (self.windows[2][5], ("errors", "rms"), False), (92, ("passed",), 1)):
            self.change(row, path, bad)

    def test_cli_append_one_compact_record_history_explicit_relative_results_and_helper(self):
        history = b'{"kind":"previous","proof":"unchanged"}\n'
        self.results.write_bytes(history)
        process = self.cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "")
        raw = self.results.read_bytes()
        self.assertTrue(raw.startswith(history))
        self.assertEqual(len(raw.splitlines()), 2)
        record = json.loads(raw.splitlines()[1])
        self.assertEqual(record["kind"], "r4b_long_prefill")
        self.assertEqual(record["source"], self.rows[0])
        self.assertEqual(record["raw_logs"], {"prefill_long": str(self.raw.resolve())})
        self.assertEqual(self.raw.read_bytes(), self.original_raw)
        self.assertIs(MODULE.append_record, record_memory.append_result)
        with mock.patch.object(MODULE, "append_record") as append:
            self.assertEqual(MODULE.main(["--raw", str(self.raw), "--results", str(self.results)]), 0)
            append.assert_called_once()
            self.assertEqual(append.call_args.args[0], self.results)
        self.assertEqual(self.cli(Path("caller-canonical.jsonl")).returncode, 0)

    def test_cli_alias_symlink_hardlink_model_and_incomplete_journal_preserve_history(self):
        model = self.root / "qwen38-keep1-Q4_0.gguf"
        model.write_bytes(b"synthetic only; no model inspection")
        self.rows[0]["model"] = str(model)
        self.raw.write_bytes(encoded(self.rows))
        for source in (self.raw, model):
            before = source.read_bytes()
            symlink, hardlink = self.root / (source.name + ".symlink"), self.root / (source.name + ".hardlink")
            symlink.symlink_to(source)
            os.link(source, hardlink)
            for destination in (source, symlink, hardlink):
                process = self.cli(destination)
                self.assertEqual(process.returncode, 1)
                self.assertEqual(process.stdout, "")
                self.assertEqual(source.read_bytes(), before)
        history = b'{"kind":"incomplete"}'
        self.results.write_bytes(history)
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)
        history += b"\n"
        self.results.write_bytes(history)
        self.raw.write_bytes(encoded(self.rows[:-1]))
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)
        absent = self.root / "absent.jsonl"
        self.assertEqual(self.cli(absent).returncode, 1)
        self.assertFalse(absent.exists())

    def test_shared_append_lock_and_short_write_or_flush_failure_rollback(self):
        record = self.collect()
        history = b'{"kind":"history"}\n'
        self.results.write_bytes(history)
        with mock.patch.object(record_memory.fcntl, "flock", wraps=record_memory.fcntl.flock) as lock:
            MODULE.append_record(self.results, record)
            lock.assert_called_once()
            self.assertEqual(lock.call_args.args[1], record_memory.fcntl.LOCK_EX)
        real_open = Path.open
        for short_write in (True, False):
            self.results.write_bytes(history)

            class BrokenAppend:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.stream.close()

                def __getattr__(self, key):
                    return getattr(self.stream, key)

                def write(self, value):
                    return self.stream.write(value[:len(value) // 2] if short_write else value)

                def flush(self):
                    raise OSError("synthetic flush failure")

            def open_file(path, *args, **kwargs):
                stream = real_open(path, *args, **kwargs)
                return BrokenAppend(stream) if path == self.results else stream

            with mock.patch.object(Path, "open", new=open_file), contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(MODULE.main(["--raw", str(self.raw), "--results", str(self.results)]), 1)
            self.assertIn("record_prefill_long:", error.getvalue())
            self.assertEqual(self.results.read_bytes(), history)

    def test_concurrent_cli_uses_shared_lock_for_complete_appends(self):
        history = b'{"kind":"history"}\n'
        self.results.write_bytes(history)
        command = [sys.executable, "-B", str(SCRIPT), "--raw", str(self.raw), "--results", str(self.results)]
        processes = [subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertEqual(stdout, b"")
        output = self.results.read_bytes()
        self.assertTrue(output.startswith(history))
        self.assertEqual(len(output.splitlines()), 4)
        for line in output.splitlines()[1:]:
            self.assertEqual(json.loads(line)["kind"], "r4b_long_prefill")

    def test_cli_requires_paths_no_gate_override_and_no_stdout_journal(self):
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
