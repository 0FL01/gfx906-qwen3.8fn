"""Standalone synthetic frozen core-prefill-test evidence; no model/HIP/raw dependency."""

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
from tools import record_prefill as MODULE

SCRIPT = ROOT / "tools/record_prefill.py"
REVISION = "0123456789abcdef0123456789abcdef01234567"
VOCAB = 248320
NAMES = ["reference_n1", "chunk4", "chunk32", "occupied_prefix5_chunk17_remainder10"]
COUNTERS = ("expert_hits", "expert_misses", "expert_upload_bytes")
ROUTE_KEYS = ("hits", "misses", "upload_bytes")
CATEGORIES = ("weights", "expert_slots", "qsa_kv", "qsa_index", "gdn_state", "ple_state", "workspace")


def timing_bits(value):
    return struct.unpack("!Q", struct.pack("!d", value))[0]


def zero_stats():
    return {"consumed_tokens": 0, "expert_hits": 0, "expert_misses": 0,
            "expert_upload_bytes": 0, "last_completed_ms": 0, "last_completed_ms_bits": 0}


def route(rows, hits):
    misses = 480 * rows - hits
    q41 = misses // 8
    return {"hits": hits, "misses": misses,
            "upload_bytes": (misses - q41) * 2764800 + q41 * 2867200}


def logit_metrics(offset, rows, compared):
    """Plausible full-vocabulary summaries: nonzero norms, exact self-replay."""
    return {
        "values": rows * VOCAB, "finite_values": rows * VOCAB,
        "nonfinite_actual": 0, "nonfinite_reference": 0, "actual_maxabs": 16., "actual_rms": 2.,
        "compared_values": rows * VOCAB if compared else 0,
        "finite_pairs": rows * VOCAB if compared else 0,
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
    """Spell the actual 92-row driver schema without importing collector constants."""
    source = {
        "kind": "prefill_source", "protocol": 1, "revision": REVISION, "dirty": True,
        "model": '/models/данные "quoted"/qwen38-keep1-Q4_0.gguf', "model_bytes": 75399121792,
        "model_source": "caller_supplied_GGUF_path_no_checksum_attestation", "runtime": "own_48_layer_HIP",
        "config": {"capacity": 40, "expert_slots": 1, "max_batch_tokens": 32, "trace": False},
        "teacher_ids": [248044, *range(100, 131)], "continuation_ids": list(range(131, 139)),
        "phase_order": NAMES.copy(), "continuation_schedule": "eight_N1_calls_each_phase",
        "occupied_prefix_schedule": "five_N1_calls", "chunk4_teacher_schedule": "eight_N4_calls",
        "gate": {"absolute": .02, "relative": .002,
                 "formula": "abs(actual-reference)<=.02+.002*abs(reference)",
                 "all_finite_required": True, "allowed_violations": 0, "bit_equality_required": False,
                 "argmax_equality_required": False, "cli_adjustable": False},
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
                     "expert_band": 16, "stages_per_device": 2, "pinned_expert_staging_bytes": 183500800,
                     "pinned_handoff_bytes": 1310720, "host_logit_min_bytes": 31784960,
                     "workspace_aggregate_min_bytes_per_device": 164929536,
                     "individual_workspace_geometry_observable": False, "free_vram_equality_required": False,
                     "memory_scope": "Session_owned_Buffer_ledger_and_reported_host_capacities_excludes_allocator_metadata"},
        "preallocated_reference_bytes": 39731200, "preallocated_preservation_snapshot_bytes": 31784960,
        "jsonl_after_session_cleanup": True,
    }
    devices = []
    for i in (0, 1):
        d = {"device": i, "first_layer": 24 * i, "last_layer": 24 * i + 23,
             "gdn_layers": 18, "qsa_layers": 6, "weights": 1100000000 + i * 12345678,
             "expert_slots": [66969600, 66355200][i], "qsa_kv": 138240, "qsa_index": 39936,
             "gdn_state": 58834944, "ple_state": [368640, 0][i], "workspace": 170000000 + i * 43210,
             "owned_buffers": 1001 + i * 37, "total_vram": 17163091968}
        d["owned_bytes"] = sum(d[key] for key in CATEGORIES)
        d["owned_peak_bytes"] = d["owned_bytes"] + 4096
        d["free_vram"] = d["total_vram"] - d["owned_bytes"] - 64 * 1024 * 1024
        devices.append(d)
    loaded = {"capacity": 40, "expert_slots": 1, "ownership_verified": True,
              "ram_expert_capacity": 68262297600 + 4096, "ram_expert_payload": 68262297600,
              "host_embedding_capacity": 521472000, "host_logit_capacity": 31784960 + 4096,
              "pinned_handoff": 1310720, "pinned_expert_staging": 183500800,
              "expert_payload_reads": 144, "expert_payload_bytes_read": 68262297600, "devices": devices}
    records = [source, {"kind": "prefill_loaded_memory", "protocol": 1, "stats": zero_stats(),
                        "memory": copy.deepcopy(loaded), "passed": True}]

    def snapshot():
        memory = copy.deepcopy(loaded)
        for d in memory["devices"]:
            d["free_vram"] -= len(records) * 1024
        return memory

    schedules = [[1] * 40, [4] * 8 + [1] * 8, [32] + [1] * 8, [1] * 5 + [17, 10] + [1] * 8]
    reference, preserved, wall_sum = None, 0, 0
    for index, (name, widths) in enumerate(zip(NAMES, schedules)):
        records.append({"kind": "prefill_reset", "protocol": 1, "phase_index": index, "phase": name,
                        "stats": zero_stats(), "expert_cache_retained": True, "memory": snapshot(), "passed": True})
        offset, previous, windows = 0, zero_stats(), []
        for window_index, rows in enumerate(widths):
            delta = route(rows, (0 if offset == 0 else 10) if index == 0 else (15 if rows == 1 else 80 * rows))
            timing = 50.25 + index / 4 + offset / 8 + rows
            after = {"consumed_tokens": offset + rows,
                     **{c: previous[c] + delta[k] for c, k in zip(COUNTERS, ROUTE_KEYS)},
                     "last_completed_ms": timing, "last_completed_ms_bits": timing_bits(timing)}
            retained = copy.deepcopy(delta) if index == 0 else {
                key: sum(w["route_delta"][key] for w in reference[offset:offset + rows]) for key in ROUTE_KEYS}
            window = {"kind": "prefill_window", "protocol": 1, "phase_index": index, "phase": name,
                      "window_index": window_index, "offset": offset, "rows": rows,
                      "segment": "teacher" if offset < 32 else "continuation", "reference": index == 0,
                      "completed_call": True, "completed_window_wall_ms": timing + 1.5,
                      "stats_before": copy.deepcopy(previous), "stats_after": after,
                      "expected_routes": rows * 480, "route_delta": delta, "retained_n1_route_delta": retained,
                      "errors": logit_metrics(offset, rows, index > 0), "memory_snapshot": snapshot(),
                      "memory_stats_and_full_active_span_preserved": True,
                      "first_input_n1_smoke": index == 0 and offset == 0, "passed": True, "failure": None}
            records.append(window)
            windows.append(window)
            previous = after
            offset += rows
        if index == 0:
            reference = windows
        proofs = []
        if index:
            ordinary = [0, 0, 0, 5][index]
            cases = [("empty", 0, ordinary), ("length33", 33, ordinary),
                     ("negative_last", 3, ordinary), ("oov_last", 3, ordinary),
                     ("capacity39_length2", 2, len(windows) - 2)]
            for case, input_rows, prior_index in cases:
                prior, continuation = windows[prior_index:prior_index + 2]
                proofs.append({"case": case, "input_rows": input_rows, "offset": prior["offset"] + prior["rows"],
                               "prior_rows": prior["rows"], "preserved_fullspan_values": prior["rows"] * VOCAB,
                               "stats_before_and_after": copy.deepcopy(prior["stats_after"]),
                               "exception": "invalid_argument", "all_public_stats_bitwise_preserved": True,
                               "fullspan_bits_preserved": True, "memory_ledger_preserved": True,
                               "pinned_expert_staging_preserved": True, "free_vram_equality_required": False,
                               "consumed_increment": 0, "continued_without_reset": True,
                               "continuation_offset": continuation["offset"], "continuation_rows": continuation["rows"],
                               "continuation_compared_values": continuation["rows"] * VOCAB, "continuation_violations": 0})
        preserved += sum(p["preserved_fullspan_values"] for p in proofs)
        phase_wall = sum(w["completed_window_wall_ms"] for w in windows)
        wall_sum += phase_wall
        records.append({"kind": "prefill_phase", "protocol": 1, "phase_index": index, "phase": name,
                        "windows": len(windows), "teacher_rows": 32, "continuation_rows": 8,
                        "finite_logit_values": 9932800, "compared_logit_values": 9932800 if index else 0,
                        "bit_mismatches_diagnostic": 0, "violations": 0, "completed_window_wall_ms_sum": phase_wall,
                        "timing_includes_rejection_reset_comparison_memory": False,
                        "teacher_stats": copy.deepcopy(next(w["stats_after"] for w in windows if w["offset"] + w["rows"] == 32)),
                        "final_stats": copy.deepcopy(windows[-1]["stats_after"]), "invalid_proofs": proofs, "passed": True})
    records.append({"kind": "prefill_final_reset", "protocol": 1, "stats": zero_stats(),
                    "memory": snapshot(), "passed": True})
    minima = [min(row.get("memory", row.get("memory_snapshot"))["devices"][i]["free_vram"]
                  for row in records[1:] if "memory" in row or "memory_snapshot" in row) - 1024 for i in (0, 1)]
    records.append({"kind": "prefill_complete", "protocol": 1, "phase_count": 4, "record_count": 92,
                    "timeline_rows": 160, "teacher_rows": 128, "continuation_rows": 32, "window_count": 80,
                    "finite_logit_values": 39731200, "compared_logit_values": 29798400,
                    "violations": 0, "bit_mismatches_diagnostic": 0, "bit_equality_required": False,
                    "completed_window_wall_ms_sum": wall_sum, "invalid_window_rejections": 15,
                    "rejection_preserved_fullspan_values": preserved, "memory_snapshot_count": 116,
                    "memory_preserved_fullspan_values": 39731200 + 2 * preserved, "minimum_free_vram_bytes": minima,
                    "steady_categories_allocated_bytes_counts_peak_host_pinned_and_payload_reads": True,
                    "session_instances": 1, "raii_session_cleanup_completed": True, "owned_buffer_release_measured": False,
                    "reference_scope": "same_session_N1_self_parity", "first_PP_gate_passed": True,
                    "R4_complete_claim": False, "qualification_4K_16K_claim": False,
                    "independent_HF_reference": False, "performance_claim": False, "passed": True})
    return records


def encoded(rows):
    return ("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)).encode("utf-8")


def target(value, path):
    for key in path:
        value = value[key]
    return value


def objects(value, path=()):
    if type(value) is dict:
        yield path, value
        for key, child in value.items():
            yield from objects(child, path + (key,))
    elif type(value) is list and value:
        for i in sorted({0, len(value) - 1}):
            yield from objects(value[i], path + (i,))


def leaves(value, path=()):
    if type(value) is dict:
        for key, child in value.items():
            yield from leaves(child, path + (key,))
    elif type(value) is list:
        for i, child in enumerate(value):
            yield from leaves(child, path + (i,))
    else:
        yield path, value


class PrefillResultsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="prefill-results-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.raw = self.root / "prefill.jsonl"
        self.results = self.root / "results.jsonl"
        self.rows = fixture()
        self.raw.write_bytes(encoded(self.rows))
        self.windows = [i for i, row in enumerate(self.rows) if row["kind"] == "prefill_window"]
        self.phases = [i for i, row in enumerate(self.rows) if row["kind"] == "prefill_phase"]
        # Cover every distinct schema, both device owners, proof types and phase
        # nullability without repeating the same large-memory schema 86 times.
        self.representatives = [0, 1, 2, 3, 43, 44, 45, 61, 62, 63, 72, 73, 74, 89, 90, 91]

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

    def reject_change(self, row, path, value):
        changed = copy.deepcopy(self.rows)
        target(changed[row], path[:-1])[path[-1]] = value
        self.reject(changed)

    def cli(self, destination=None, args=None):
        command = [sys.executable, "-B", str(SCRIPT)]
        command += args if args is not None else ["--raw", self.raw.name, "--results", str(destination or self.results)]
        return subprocess.run(command, cwd=self.root, capture_output=True, text=True, check=False, timeout=30)

    def test_success_preserves_every_original_row_path_provenance_and_scope(self):
        before = self.raw.read_bytes()
        result = self.collect()
        self.assertEqual(result["kind"], "r4b_short_prefill")
        self.assertIs(result["passed"], True)
        self.assertIs(result["short_pp_qualified"], True)
        self.assertEqual(result["records"], self.rows)
        self.assertEqual(result["source"], self.rows[0])
        self.assertEqual(result["loaded_memory"], self.rows[1])
        self.assertEqual(result["windows"], [self.rows[i] for i in self.windows])
        self.assertEqual(result["phases"], [self.rows[i] for i in self.phases])
        self.assertEqual(result["resets"], [r for r in self.rows if r["kind"] == "prefill_reset"])
        self.assertEqual(result["final_reset"], self.rows[-2])
        self.assertEqual(result["complete"], self.rows[-1])
        self.assertEqual(result["revision"], REVISION)
        self.assertIs(result["dirty"], True)
        self.assertEqual(result["model"], self.rows[0]["model"])
        self.assertEqual(result["raw_logs"], {"prefill": str(self.raw.resolve())})
        self.assertEqual(datetime.datetime.fromisoformat(result["timestamp"]).utcoffset(), datetime.timedelta(0))
        for key in ("R4_complete_claim", "qualification_4K_16K_claim", "group_over_128_qualified",
                    "independent_hf_claim", "performance_claim", "mtp"):
            self.assertIs(result[key], False)
        self.assertIn("self-parity", result["scope"])
        self.assertIn("not performance evidence", result["timing_scope"])
        self.assertIn("no individual-buffer-size proof", result["memory_scope"])
        self.assertIn("no measured owned-buffer release/recovery", result["memory_scope"])
        self.assertIn("not a syscall trace", result["expert_read_scope"])
        self.assertEqual(self.raw.read_bytes(), before)
        json.dumps(result, allow_nan=False)

    def test_exact_driver_counts_and_schedules(self):
        footer = self.collect()["complete"]
        self.assertEqual(len(self.rows), 92)
        for field, value in (("timeline_rows", 160), ("window_count", 80), ("finite_logit_values", 39731200),
                             ("compared_logit_values", 29798400), ("invalid_window_rejections", 15),
                             ("memory_snapshot_count", 116), ("rejection_preserved_fullspan_values", 53388800),
                             ("memory_preserved_fullspan_values", 146508800)):
            self.assertEqual(footer[field], value)
        self.assertEqual([self.rows[i]["windows"] for i in self.phases], [40, 16, 9, 15])
        self.assertEqual([self.rows[i]["invalid_proofs"][0]["offset"] for i in self.phases[1:]], [4, 32, 22])
        self.assertEqual([self.rows[i]["invalid_proofs"][0]["continuation_rows"] for i in self.phases[1:]], [4, 1, 10])

    def test_arbitrary_emitted_40hex_revision_boolean_dirty_and_absolute_model_preserved(self):
        for revision, dirty in (("f" * 40, False), ("ABCDEF01" * 5, True)):
            rows = copy.deepcopy(self.rows)
            rows[0].update(revision=revision, dirty=dirty, model="/other/location/qwen38-keep1-Q4_0.gguf")
            result = self.collect(rows)
            self.assertEqual(result["revision"], revision)
            self.assertIs(result["dirty"], dirty)
            self.assertEqual(result["model"], rows[0]["model"])

    def test_invalid_provenance_model_variant_paths_and_size(self):
        for bad in (None, 0, True, "", "a" * 39, "a" * 41, "g" * 40, "a" * 39 + "\n"):
            self.reject_change(0, ("revision",), bad)
        for bad in (None, 0, 1, 1., "false"):
            self.reject_change(0, ("dirty",), bad)
        for bad in (None, 1, "", "qwen38-keep1-Q4_0.gguf", "/models/full-PLE.gguf",
                    "/models/qwen38-keep1-Q8_0.gguf", "/models/\ud800/qwen38-keep1-Q4_0.gguf",
                    "/models/\n/qwen38-keep1-Q4_0.gguf", "/models/\x00/qwen38-keep1-Q4_0.gguf",
                    "/" + "x" * 4096 + "/qwen38-keep1-Q4_0.gguf"):
            self.reject_change(0, ("model",), bad)
        for bad in (0, -1, 1 << 63, True, 1., "1"):
            self.reject_change(0, ("model_bytes",), bad)

    def test_fixed_source_contract_every_gate_scope_and_geometry(self):
        for path, value in leaves(self.rows[0]):
            if path[0] in ("revision", "dirty", "model", "model_bytes"):
                continue
            bad = value + 1 if type(value) in (int, float) else (not value if type(value) is bool else "changed")
            self.reject_change(0, path, bad)

    def test_all_object_schemas_require_exact_fields(self):
        for i in self.representatives:
            for path, obj in objects(self.rows[i]):
                for field in (*obj, "unexpected"):
                    with self.subTest(row=i, path=path, field=field):
                        rows = copy.deepcopy(self.rows)
                        changed = target(rows[i], path)
                        if field == "unexpected":
                            changed[field] = True
                        else:
                            del changed[field]
                        self.reject(rows)

    def test_all_numeric_leaves_reject_boolean_and_counter_floats(self):
        double_fields = {"last_completed_ms", "completed_window_wall_ms", "completed_window_wall_ms_sum",
                         "actual_maxabs", "actual_rms", "maxabs", "rms", "maxboundratio"}
        for i in self.representatives:
            for path, value in leaves(self.rows[i]):
                if type(value) in (int, float):
                    with self.subTest(row=i, path=path):
                        self.reject_change(i, path, True)
                        if type(value) is int and path[-1] not in double_fields:
                            self.reject_change(i, path, float(value))

    def test_all_boolean_proofs_and_nullability_are_strict(self):
        for i in self.representatives:
            for path, value in leaves(self.rows[i]):
                if type(value) is bool:
                    self.reject_change(i, path, int(value))
                    if path[-1] != "dirty":
                        self.reject_change(i, path, not value)
                elif value is None:
                    self.reject_change(i, path, False)

    def test_each_row_missing_duplicate_replaced_reordered_and_every_truncated_protocol(self):
        for i in range(92):
            for operation in ("missing", "duplicate", "replaced"):
                with self.subTest(row=i, operation=operation):
                    rows = copy.deepcopy(self.rows)
                    if operation == "missing":
                        rows.pop(i)
                    elif operation == "duplicate":
                        rows.insert(i, copy.deepcopy(rows[i]))
                    else:
                        rows[i] = copy.deepcopy(rows[(i + 1) % 92])
                    self.reject(rows)
        for end in range(92):
            self.reject(self.rows[:end])
        for i in range(91):
            rows = copy.deepcopy(self.rows)
            rows[i], rows[i + 1] = rows[i + 1], rows[i]
            self.reject(rows)

    def test_malformed_nonobject_utf8_bom_blank_unfinished_and_free_text(self):
        raw = encoded(self.rows)
        bads = [raw[:-1], raw[:-20], raw + b"\n", raw + b"failure\n", raw + b" ",
                b"\xef\xbb\xbf" + raw, raw.replace(b"prefill_source", b"\xff", 1)]
        lines = raw.split(b"\n")[:-1]
        for line in (b"", b" ", b"[]", b"null", b"true", b"1", b'"passed"', b"{", b"{} {}",
                     b"core-prefill-test: failed", b'{"kind":1,}',
                     b'{"x":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}"):
            bads.append(b"\n".join([*lines[:3], line, *lines[4:]]) + b"\n")
        for bad in bads:
            self.raw.write_bytes(bad)
            with self.assertRaises(ValueError):
                self.collect()
            self.assertEqual(self.raw.read_bytes(), bad)

    def test_duplicate_keys_at_every_schema_level(self):
        raw = encoded(self.rows)
        for field in ("revision", "capacity", "absolute", "logical_candidate_limit", "device", "weights",
                      "last_completed_ms_bits", "hits", "actual_rms", "position", "case", "continuation_rows",
                      "invalid_window_rejections", "passed"):
            key = ('"' + field + '":').encode()
            self.raw.write_bytes(raw.replace(key, key + b"0," + key, 1))
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                self.collect()

    def test_nonfinite_constants_exponents_and_nonfinite_null_metrics(self):
        raw = encoded(self.rows)
        for literal in (b"NaN", b"Infinity", b"-Infinity", b"1e999", b"-1e999"):
            for old in (b'"last_completed_ms": 51.25', b'"actual_rms": 2.0', b'"weights": 1100000000',
                        b'"model_bytes": 75399121792'):
                bad = raw.replace(old, old.split(b":")[0] + b": " + literal, 1)
                self.assertNotEqual(bad, raw)
                self.raw.write_bytes(bad)
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.collect()
        for field in ("actual_rms", "actual_maxabs", "maxabs", "rms", "maxboundratio"):
            self.reject_change(45, ("errors", field), None)

    def test_bounded_regular_input_and_growing_read(self):
        raw = encoded(self.rows)
        limit = 2 * 1024 * 1024
        self.assertEqual(MODULE.MAX_RAW_BYTES, limit)
        self.raw.write_bytes(b" " * (limit - len(raw)) + raw)
        self.assertIs(self.collect()["passed"], True)
        self.raw.write_bytes(b" " * (limit + 1 - len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, "oversized"):
            self.collect()
        self.raw.write_bytes(raw)
        with mock.patch.object(Path, "open", return_value=mock.MagicMock()) as opened:
            opened.return_value.__enter__.return_value.read.return_value = b"x" * (limit + 1)
            with self.assertRaisesRegex(ValueError, "oversized"):
                self.collect()
            opened.return_value.__enter__.return_value.read.assert_called_once_with(limit + 1)
        with self.assertRaisesRegex(ValueError, "regular file"):
            MODULE.collect(self.root)
        self.raw.write_bytes(raw.replace(b"\n", b"\r\n"))
        self.assertIs(self.collect()["passed"], True)

    def test_every_window_counts_order_extents_completion_and_frozen_gate(self):
        for i in self.windows:
            w = self.rows[i]
            for field in ("offset", "rows", "window_index", "phase_index", "expected_routes"):
                self.reject_change(i, (field,), w[field] + 1)
            for field in ("values", "finite_values", "compared_values", "finite_pairs", "argmax_compared_rows"):
                self.reject_change(i, ("errors", field), w["errors"][field] + 1)
            for field in ("nonfinite_actual", "nonfinite_reference", "violations"):
                self.reject_change(i, ("errors", field), 1)
            self.reject_change(i, ("passed",), False)
            self.reject_change(i, ("completed_call",), False)
            self.reject_change(i, ("memory_stats_and_full_active_span_preserved",), False)

    def test_completed_times_positive_finite_and_all_double_bits_match(self):
        for row in (3, 45, 63, 80):
            for bad in (0, -1, "1", True, None, 10**400):
                self.reject_change(row, ("completed_window_wall_ms",), bad)
                self.reject_change(row, ("stats_after", "last_completed_ms"), bad)
            self.reject_change(row, ("stats_after", "last_completed_ms_bits"),
                               self.rows[row]["stats_after"]["last_completed_ms_bits"] + 1)
        rows = copy.deepcopy(self.rows)
        rows[1]["stats"]["last_completed_ms"] = -0.0
        self.reject(rows)
        raw = encoded(rows).replace(b'"last_completed_ms": -0.0', b'"last_completed_ms": -0')
        self.raw.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "timing_bits"):
            self.collect()

    def test_every_window_route_upload_and_stats_accounting(self):
        for i in self.windows:
            w = self.rows[i]
            for block in ("route_delta", "retained_n1_route_delta"):
                for field in ROUTE_KEYS:
                    self.reject_change(i, (block, field), w[block][field] + 1)
            for block in ("stats_before", "stats_after"):
                for field in ("consumed_tokens", *COUNTERS, "last_completed_ms_bits"):
                    self.reject_change(i, (block, field), w[block][field] + 1)
            for bad in (w["route_delta"]["misses"] * 2764800 - 1,
                        w["route_delta"]["misses"] * 2867200 + 1):
                self.reject_change(i, ("route_delta", "upload_bytes"), bad)

    def test_matching_retained_n1_sums_cannot_be_plausible_other_routes(self):
        for i in self.windows[40:]:
            changed = copy.deepcopy(self.rows[i]["retained_n1_route_delta"])
            changed["hits"] += 1
            changed["misses"] -= 1
            changed["upload_bytes"] -= 2764800
            self.reject_change(i, ("retained_n1_route_delta",), changed)

    def error_candidate(self, maximum=.01, rms=.002, ratio=.5, mismatches=VOCAB):
        rows = copy.deepcopy(self.rows)
        rows[45]["errors"].update(maxabs=maximum, rms=rms, maxboundratio=ratio,
                                  bit_mismatches=mismatches, argmax_agree_rows=3,
                                  maxabs_coordinate={"position": 2, "vocabulary_index": 17},
                                  maxboundratio_coordinate={"position": 1, "vocabulary_index": 9})
        rows[61]["bit_mismatches_diagnostic"] = mismatches
        rows[-1]["bit_mismatches_diagnostic"] = mismatches
        return rows

    def test_bit_and_argmax_mismatches_are_diagnostics_not_universal_exact_gate(self):
        rows = self.error_candidate()
        result = self.collect(rows)
        self.assertEqual(result["complete"]["bit_mismatches_diagnostic"], VOCAB)
        self.assertEqual(result["windows"][40]["errors"]["argmax_agree_rows"], 3)
        self.assertIs(result["fixture_gates"]["bit_equality_required"], False)
        self.assertIs(result["fixture_gates"]["argmax_equality_required"], False)
        # Signed-zero mismatches can have zero numerical error and identical argmax.
        rows = copy.deepcopy(self.rows)
        rows[45]["errors"]["bit_mismatches"] = 1
        rows[61]["bit_mismatches_diagnostic"] = 1
        rows[-1]["bit_mismatches_diagnostic"] = 1
        self.assertIs(self.collect(rows)["passed"], True)

    def test_frozen_gate_does_not_trust_zero_violations_or_passed(self):
        for maximum, rms, ratio in ((.01, .002, 1.0000000000000002), (.053, .002, .9), (.1, .01, 0)):
            self.reject(self.error_candidate(maximum, rms, ratio))
        # A bound exactly reached is permitted. At reference magnitude16 the
        # maximum admissible error is .052, independent of caller-adjusted gates.
        rows = self.error_candidate(.02 + .002 * 16, .002, 1.)
        self.assertIs(self.collect(rows)["passed"], True)
        rows[45]["errors"]["maxabs"] = math.nextafter(.02 + .002 * 16, math.inf)
        self.reject(rows)

    def test_quantitative_error_mean_norm_counts_and_ratios_prevent_false_pass(self):
        for field, bad in (("maxabs", -.1), ("rms", -.1), ("maxboundratio", -.1),
                           ("actual_rms", -1), ("actual_maxabs", -1), ("actual_rms", 17),
                           ("actual_maxabs", 0), ("actual_rms", 0), ("actual_rms", 10),
                           ("actual_maxabs", 17), ("maxabs", .01), ("rms", .001),
                           ("maxboundratio", .1), ("bit_mismatches", -1), ("bit_mismatches", 4 * VOCAB + 1),
                           ("argmax_agree_rows", -1), ("argmax_agree_rows", 5)):
            self.reject_change(45, ("errors", field), bad)
        for maximum, rms, ratio, mismatches in ((.01, 0, .5, VOCAB), (.01, .02, .5, VOCAB),
                                               (.01, 1e-8, .5, VOCAB), (.01, .002, .001, VOCAB),
                                               (.01, .002, .9, VOCAB), (.01, .002, .5, 1),
                                               (.01, .002, .5, 0), (0, .002, 0, VOCAB), (0, 0, .01, VOCAB)):
            self.reject(self.error_candidate(maximum, rms, ratio, mismatches))
        for field in ("maxabs_coordinate", "maxboundratio_coordinate"):
            for bad in ({"position": 4, "vocabulary_index": 0}, {"position": -1, "vocabulary_index": 0},
                        {"position": 0, "vocabulary_index": VOCAB}, {"position": 0, "vocabulary_index": -1}):
                self.reject_change(45, ("errors", field), bad)
            self.reject_change(45, ("errors", field, "vocabulary_index"), 1)  # zero-error default coordinate
        self.reject_change(3, ("errors", "actual_rms"), 1e-8)
        rows = copy.deepcopy(self.rows)
        rows[3]["errors"].update(actual_maxabs=0, actual_rms=0)
        self.reject(rows)
        rows = copy.deepcopy(self.rows)
        for i in self.windows:
            rows[i]["errors"].update(actual_maxabs=1e300, actual_rms=1e300)
        self.reject(rows)

    def test_unequal_retained_n1_norms_are_aggregated_not_averaged_or_last_row_only(self):
        rows = copy.deepcopy(self.rows)
        reference = [rows[i] for i in self.windows[:40]]
        for w in reference:
            w["errors"].update(actual_maxabs=16 + w["offset"] / 8, actual_rms=1 + w["offset"] / 64)
        for i in self.windows[40:]:
            w = rows[i]
            retained = reference[w["offset"]:w["offset"] + w["rows"]]
            w["errors"].update(actual_maxabs=max(r["errors"]["actual_maxabs"] for r in retained),
                              actual_rms=math.sqrt(sum(r["errors"]["actual_rms"] ** 2 for r in retained) / w["rows"]))
        self.assertIs(self.collect(rows)["passed"], True)
        rows[63]["errors"]["actual_rms"] = sum(r["errors"]["actual_rms"] for r in reference[:32]) / 32
        self.reject(rows)

    def test_double_rms_accumulation_roundoff_does_not_impose_bitwise_summary_equality(self):
        rows = copy.deepcopy(self.rows)
        rows[63]["errors"]["actual_rms"] += 4e-10
        self.assertIs(self.collect(rows)["passed"], True)
        rows[63]["errors"]["actual_rms"] += 1e-6
        self.reject(rows)

    def test_all_invalid_cases_stats_fullspan_memory_pinned_and_continuations(self):
        for row in self.phases[1:]:
            for i, proof in enumerate(self.rows[row]["invalid_proofs"]):
                for field, value in proof.items():
                    if field == "stats_before_and_after":
                        for key, number in value.items():
                            self.reject_change(row, ("invalid_proofs", i, field, key), number + 1)
                    else:
                        bad = value + 1 if type(value) is int else (not value if type(value) is bool else "changed")
                        self.reject_change(row, ("invalid_proofs", i, field), bad)
            self.reject_change(row, ("invalid_proofs",), self.rows[row]["invalid_proofs"][::-1])
            self.reject_change(row, ("invalid_proofs",), self.rows[row]["invalid_proofs"][:-1])
            self.reject_change(row, ("invalid_proofs",), self.rows[row]["invalid_proofs"] + [self.rows[row]["invalid_proofs"][-1]])

    def test_phase_teacher_final_stats_sums_and_complete_all_exact_counts(self):
        for row in self.phases:
            for field, value in self.rows[row].items():
                if type(value) is int:
                    self.reject_change(row, (field,), value + 1)
            for field in ("teacher_stats", "final_stats"):
                for key, value in self.rows[row][field].items():
                    self.reject_change(row, (field, key), value + 1)
            self.reject_change(row, ("completed_window_wall_ms_sum",), self.rows[row]["completed_window_wall_ms_sum"] + 1)
        for field, value in self.rows[-1].items():
            if type(value) in (int, float):
                self.reject_change(91, (field,), value + 1)
            elif type(value) is bool:
                self.reject_change(91, (field,), not value)

    def test_loaded_reset_and_final_reset_stats_are_zero_including_timing_bits(self):
        for row in (1, 2, 44, 62, 73, 90):
            for field in zero_stats():
                self.reject_change(row, ("stats", field), 1)

    def test_each_visible_memory_snapshot_has_exact_steady_owners_categories_and_host_reads(self):
        for i, row in enumerate(self.rows):
            field = "memory" if "memory" in row else "memory_snapshot" if "memory_snapshot" in row else None
            if field is None:
                continue
            self.reject_change(i, (field, "ownership_verified"), False)
            for key in ("ram_expert_capacity", "host_embedding_capacity", "host_logit_capacity", "pinned_handoff",
                        "pinned_expert_staging", "expert_payload_reads", "expert_payload_bytes_read"):
                self.reject_change(i, (field, key), row[field][key] + 1)
            for owner in (0, 1):
                self.reject_change(i, (field, "devices", owner, "device"), 1 - owner)
                for key in ("weights", "workspace", "owned_peak_bytes", "owned_buffers"):
                    self.reject_change(i, (field, "devices", owner, key), 0)

    def test_memory_geometry_floors_category_sum_peak_and_observed_vram(self):
        for i in (0, 1):
            d = self.rows[1]["memory"]["devices"][i]
            for field in ("first_layer", "last_layer", "gdn_layers", "qsa_layers", "expert_slots",
                          "qsa_kv", "qsa_index", "gdn_state", "ple_state", "owned_bytes"):
                self.reject_change(1, ("memory", "devices", i, field), d[field] + 1)
            for field, bad in (("workspace", 164929535), ("owned_peak_bytes", d["owned_bytes"] - 1),
                               ("free_vram", 0), ("free_vram", d["total_vram"] + 1),
                               ("free_vram", d["total_vram"] - d["owned_bytes"] + 1), ("total_vram", 0)):
                self.reject_change(1, ("memory", "devices", i, field), bad)
        for field, bad in (("ram_expert_capacity", 68262297599), ("host_embedding_capacity", 0),
                           ("host_logit_capacity", 31784959), ("pinned_expert_staging", 183500799)):
            self.reject_change(1, ("memory", field), bad)
        rows = copy.deepcopy(self.rows)
        d = rows[1]["memory"]["devices"][0]
        d.update(weights=(1 << 64) - 1, workspace=(1 << 64) - 1, owned_bytes=(1 << 64) - 1)
        self.reject(rows)

    def test_consistent_category_growth_cannot_evade_steady_memory(self):
        for field in (*CATEGORIES, "owned_buffers", "owned_peak_bytes", "total_vram"):
            for i in (0, 1):
                rows = copy.deepcopy(self.rows)
                d = rows[45]["memory_snapshot"]["devices"][i]
                d[field] += 1
                if field in CATEGORIES:
                    d["owned_bytes"] += 1
                    d["owned_peak_bytes"] += 1
                self.reject(rows)

    def test_observed_capacities_not_guessed_individual_buffer_geometry(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            m = row.get("memory", row.get("memory_snapshot"))
            if m is None:
                continue
            m["ram_expert_capacity"] += 4096
            m["host_embedding_capacity"] = 1
            m["host_logit_capacity"] += 4096
            for d in m["devices"]:
                d["workspace"] += 31
                d["weights"] += 17
                d["owned_bytes"] += 48
                d["owned_peak_bytes"] += 48
                d["owned_buffers"] += 2
        result = self.collect(rows)
        self.assertEqual(result["records"], rows)
        self.assertIs(result["source"]["geometry"]["individual_workspace_geometry_observable"], False)
        self.reject_change(1, ("memory", "individual_q8_bytes"), 368640)
        self.reject_change(91, ("after_destruction",), {"released": True})

    def test_free_vram_minima_include_unserialized_observations_and_allow_variation(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]["minimum_free_vram_bytes"] = [1, 1]
        self.assertIs(self.collect(rows)["passed"], True)
        for i in (0, 1):
            visible = min(row.get("memory", row.get("memory_snapshot"))["devices"][i]["free_vram"]
                          for row in self.rows if "memory" in row or "memory_snapshot" in row)
            self.reject_change(91, ("minimum_free_vram_bytes", i), visible + 1)
            self.reject_change(91, ("minimum_free_vram_bytes", i), 0)
        self.reject_change(91, ("minimum_free_vram_bytes",), [1])

    def test_failed_and_incomplete_logs_cannot_append_despite_claimed_footer(self):
        for i in (3, 45, 80):
            rows = copy.deepcopy(self.rows)
            rows[i].update(passed=False, failure="frozen full-logit gate failed")
            self.reject(rows)
        rows = copy.deepcopy(self.rows)
        rows[-1] = {"kind": "prefill_failure", "protocol": 1, "passed": False,
                    "session_constructed": True, "raii_session_cleanup_completed": True,
                    "error": "free-form text is not evidence"}
        self.reject(rows)

    def test_cli_appends_one_record_to_explicit_caller_journal_without_stdout_or_input_changes(self):
        previous = b'{"kind":"previous"}\n'
        self.results.write_bytes(previous)
        before = self.raw.read_bytes()
        process = self.cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "")
        output = self.results.read_bytes()
        self.assertTrue(output.startswith(previous))
        self.assertEqual(len(output.splitlines()), 2)
        record = json.loads(output.splitlines()[1])
        self.assertEqual(record["kind"], "r4b_short_prefill")
        self.assertEqual(record["records"], self.rows)
        self.assertEqual(record["raw_logs"], {"prefill": str(self.raw.resolve())})
        self.assertEqual(self.raw.read_bytes(), before)

    def test_cli_relative_explicit_path_and_no_default_or_stdout_journal(self):
        process = self.cli(Path("caller-canonical.jsonl"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "")
        self.assertEqual(len((self.root / "caller-canonical.jsonl").read_bytes().splitlines()), 1)
        self.assertFalse(self.results.exists())
        for args in ([], ["--raw", str(self.raw)], ["--results", str(self.results)]):
            self.assertEqual(self.cli(args=args).returncode, 2)
            self.assertFalse(self.results.exists())
        for destination in (Path("/dev/stdout"), Path("/dev/stderr")):
            before = self.raw.read_bytes()
            process = self.cli(destination)
            self.assertEqual(process.returncode, 1)
            self.assertEqual(process.stdout, "")
            self.assertEqual(self.raw.read_bytes(), before)
            self.assertFalse(self.results.exists())
        self.assertEqual(self.cli(args=["--raw", str(self.raw), "--results", str(self.results),
                                        "--absolute", "1"]).returncode, 2)
        self.assertFalse(self.results.exists())

    def test_cli_invalid_input_never_creates_appends_or_mutates_any_input(self):
        previous = b'{"kind":"previous"}\n'
        self.results.write_bytes(previous)
        bads = [encoded(self.rows[:-1]), encoded(self.rows)[:-1], b"not JSON\n"]
        for row, path, bad in ((0, ("revision",), "g" * 40), (45, ("errors", "maxboundratio"), 1.01),
                               (61, ("invalid_proofs", 0, "continued_without_reset"), False),
                               (80, ("memory_snapshot", "pinned_expert_staging"), 0),
                               (91, ("raii_session_cleanup_completed",), False)):
            rows = copy.deepcopy(self.rows)
            target(rows[row], path[:-1])[path[-1]] = bad
            bads.append(encoded(rows))
        for raw in bads:
            self.raw.write_bytes(raw)
            process = self.cli()
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertEqual(process.stdout, "")
            self.assertIn("record_prefill:", process.stderr)
            self.assertEqual(self.results.read_bytes(), previous)
            absent = self.root / "absent-results.jsonl"
            self.assertEqual(self.cli(absent).returncode, 1)
            self.assertFalse(absent.exists())
            self.assertEqual(self.raw.read_bytes(), raw)

    def test_cli_aliases_hardlinks_symlinks_and_incomplete_journal_are_immutable(self):
        model = self.root / "qwen38-keep1-Q4_0.gguf"
        model.write_bytes(b"synthetic model; collector must not inspect model weights")
        self.rows[0]["model"] = str(model)
        self.raw.write_bytes(encoded(self.rows))
        for source in (self.raw, model):
            before = source.read_bytes()
            symlink = self.root / (source.name + ".symlink")
            symlink.symlink_to(source)
            hardlink = self.root / (source.name + ".hardlink")
            os.link(source, hardlink)
            for destination in (source, symlink, hardlink):
                process = self.cli(destination)
                self.assertEqual(process.returncode, 1)
                self.assertEqual(process.stdout, "")
                self.assertEqual(source.read_bytes(), before)
        self.results.write_bytes(b'{"kind":"incomplete"}')
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), b'{"kind":"incomplete"}')
        self.assertEqual(self.cli(self.root).returncode, 1)

    def test_cli_uses_shared_append_result_with_explicit_path(self):
        with mock.patch.object(MODULE, "append_result") as append:
            self.assertEqual(MODULE.main(["--raw", str(self.raw), "--results", str(self.results)]), 0)
            append.assert_called_once()
            self.assertEqual(append.call_args.args[0], self.results)
            self.assertEqual(append.call_args.args[1]["records"], self.rows)


if __name__ == "__main__":
    unittest.main()
