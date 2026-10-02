"""Independent complete batch protocol fixtures; stdlib only, no HIP/model."""

import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import record_batch as MODULE

SCRIPT = ROOT / "tools/record_batch.py"
REVISION = "0123456789abcdef0123456789abcdef01234567"
VOCAB = 248320
NAMES = ["reference_n1", "replay_n1", "replay_n2", "replay_n3", "replay_mixed_123_prefix5"]
COUNTERS = ("expert_hits", "expert_misses", "expert_upload_bytes")
ROUTE_KEYS = ("hits", "misses", "upload_bytes")
CATEGORIES = ("weights", "expert_slots", "qsa_kv", "qsa_index", "gdn_state", "ple_state", "workspace")


def zero_stats():
    return {"consumed_tokens": 0, "expert_hits": 0, "expert_misses": 0,
            "expert_upload_bytes": 0, "last_completed_ms": 0}


def route(rows, hits):
    misses = 480 * rows - hits
    # Both actual Q4_0 and Q4_1 payload sizes occur, not just the lower bound.
    q41 = misses // 8
    return {"hits": hits, "misses": misses, "upload_bytes": (misses - q41) * 2764800 + q41 * 2867200}


def summed(windows):
    return {key: sum(w["delta"][key] for w in windows) for key in ROUTE_KEYS}


def fill_accounting(schedule, reference):
    """Independent cumulative stats, matching suffix, and invalid continuation data."""
    windows = schedule["windows"]
    previous = zero_stats()
    for w in windows:
        w["n1_reference_delta"] = summed(reference[w["offset"]:w["offset"] + w["rows"]])
        w["stats"] = {"consumed_tokens": w["offset"] + w["rows"],
                      **{counter: previous[counter] + w["delta"][key] for counter, key in zip(COUNTERS, ROUTE_KEYS)},
                      "last_completed_ms": 0.25 + w["offset"] / 8 + schedule["index"] / 16}
        previous = w["stats"]
    schedule["teacher_stats"] = copy.deepcopy(next(w["stats"] for w in windows if w["offset"] + w["rows"] == 32))
    schedule["timeline_stats"] = copy.deepcopy(windows[-1]["stats"])
    index = schedule["index"]
    empty_routes = {"hits": 0, "misses": 0, "upload_bytes": 0}
    reuse = {"applicable": index >= 2, "first_row": 0, "rows": 0,
             "n1_reference": copy.deepcopy(empty_routes), "candidate": copy.deepcopy(empty_routes),
             "hits_gained": 0, "misses_saved": 0, "teacher_within_window_hit_lower_bound": 0,
             "initial_cache_contents_matched": False, "performance_claim": False}
    if index >= 2:
        first = next(w["offset"] + w["rows"] for w in windows if w["rows"] > 1)
        baseline = summed(reference[first:32])
        candidate = summed([w for w in windows if first <= w["offset"] < 32])
        reuse.update(first_row=first, rows=32 - first, n1_reference=baseline, candidate=candidate,
                     hits_gained=candidate["hits"] - baseline["hits"],
                     misses_saved=baseline["misses"] - candidate["misses"],
                     teacher_within_window_hit_lower_bound=sum(max(0, w["delta"]["hits"] - 48)
                                                               for w in windows if w["offset"] < 32))
    schedule["reuse"] = reuse
    if index == 0:
        schedule["invalid_proofs"] = []
        schedule["reset_probe"] = {"rows": 0, "finite_logit_values": 0, "bitwise_compared_logit_values": 0, "window": None}
        return
    reset_delta = route(1, 4 + index)
    reset_stats = {"consumed_tokens": 1, **dict(zip(COUNTERS, reset_delta.values())), "last_completed_ms": 0}
    reset_window = {"offset": 0, "rows": 1, "requested_rows": 1, "routes": 480,
                    "delta": reset_delta, "n1_reference_delta": copy.deepcopy(reference[0]["delta"]), "stats": reset_stats}
    schedule["reset_probe"] = {"rows": 1, "finite_logit_values": 248320,
                               "bitwise_compared_logit_values": 248320, "window": reset_window}
    ordinary_index = 6 if index == 4 else 0  # mixed: five N1 rows, then N2, then N3.
    cases = [("empty", 0, ordinary_index, False), ("length4", 4, ordinary_index, False),
             ("negative_last", 3, ordinary_index, False), ("vocabulary_last", 3, ordinary_index, False),
             ("capacity39_length2", 2, len(windows) - 2, False),
             ("capacity39_length3", 3, len(windows) - 2, False),
             ("capacity40_length2", 2, len(windows) - 1, True)]
    proofs = []
    for name, rows, prior_index, reset in cases:
        prior = windows[prior_index]
        continuation = reset_window if reset else windows[prior_index + 1]
        proofs.append({"case": name, "rows": rows, "prior_rows": prior["rows"],
                       "preserved_logit_values": prior["rows"] * VOCAB,
                       "stats_before_and_after": copy.deepcopy(prior["stats"]),
                       "exception": "invalid_argument", "stats_bitwise_preserved": True,
                       "logits_bitwise_preserved": True, "memory_ledger_preserved": True,
                       "consumed_advance": 0, "continued": True, "reset_before_continuation": reset,
                       "continuation_rows": continuation["rows"],
                       "continuation_bitwise_logit_values": continuation["rows"] * VOCAB})
    schedule["invalid_proofs"] = proofs


def fixture():
    """Spell all seven rows of the 748-line C++ fixture, without collector constants."""
    source = {
        "kind": "session_batch_source", "protocol": 1, "revision": REVISION, "dirty": True,
        "model": '/models/данные "quoted"/qwen38-keep1-Q4_0.gguf', "model_bytes": 75399121792,
        "runtime": "own_48_layer_HIP",
        "config": {"capacity": 40, "expert_slots": 1, "max_batch_tokens": 3, "trace": False},
        "teacher_ids": [248044, *range(100, 131)], "capacity_tail_ids": list(range(131, 139)),
        "schedule_order": NAMES.copy(), "scope": "exact_causal_short_window_replay_before_PP",
        "comparison": "all_FP32_bits_no_tolerance",
        "reference_scope": "retained_sequential_N1_same_Session_unchanged_weights_and_gates",
        "sampling": "teacher_forced", "performance_claim": False, "session_instances": 1,
        "reset_clears_weight_slots": False, "cache_start_equality_claim": False,
        "reuse_scope": "matching_teacher_suffix_after_first_grouped_window_initial_slots_overwritten",
        "grouping_scope": "within_call_only_all_48_layers_10_routes_per_row",
        "counter_definitions": {
            "routes_per_row": 480, "max_inherited_slot_hits_per_window": 48, "timeline_rows_per_schedule": 40,
            "teacher_rows_per_schedule": 32, "tail_rows_per_schedule": 8, "logit_values_per_row": 248320,
            "invalid_windows_per_replay": 7, "bitwise_count_scope": "timeline_and_reset_probe_only_excludes_preservation_rechecks",
            "route_stats_scope": "timeline_cumulative_reset_probe_separate",
            "within_window_hit_lower_bound_scope": "sum_max_0_window_hits_minus_48_all_teacher_windows",
        },
        "geometry": {
            "min_q8_blocks_per_device": 960, "q8_block_bytes": 36, "scratch_buffers_per_device": 24,
            "min_scratch_floats_per_buffer": 36864, "min_host_logit_bytes": 2979840,
            "pinned_handoff_bytes": 122880, "min_workspace_aggregate_bytes_per_device": 6553344,
            "individual_q8_scratch_geometry_observable": False, "workspace_check": "aggregate_floor_only",
            "free_vram_equality_required": False,
            "ledger_scope": "Session_owned_buffers_and_reported_host_capacities_excludes_allocator_metadata",
        },
        "preallocated_reference_bytes": 31784960, "preallocated_tail_reference_bytes": 7946240,
        "preallocated_snapshot_bytes": 2979840,
    }
    devices = []
    for i in (0, 1):
        device = {"device": i, "first_layer": 24 * i, "last_layer": 24 * i + 23,
                  "gdn_layers": 18, "qsa_layers": 6, "weights": 1100000000 + i * 12345678,
                  "expert_slots": [66969600, 66355200][i], "qsa_kv": 138240,
                  "qsa_index": 39936, "gdn_state": 58834944, "ple_state": [368640, 0][i],
                  "workspace": 6553344 + i * 43210, "owned_buffers": 901 + i * 37,
                  "total_vram": 17163091968}
        device["owned_bytes"] = sum(device[key] for key in CATEGORIES)
        device["owned_peak_bytes"] = device["owned_bytes"] + 4096
        device["free_vram"] = device["total_vram"] - device["owned_bytes"] - 64 * 1024 * 1024
        devices.append(device)
    loaded = {"capacity": 40, "expert_slots": 1, "ownership_verified": True,
              "ram_expert_capacity": 68262297600 + 4096, "ram_expert_payload": 68262297600,
              "host_embedding_capacity": 521472000, "host_logit_capacity": 2979840,
              "pinned_handoff": 122880, "expert_payload_reads": 144, "expert_payload_bytes_read": 68262297600,
              "devices": devices}
    # Literal widths independently expose teacher32, capacity39 and final40
    # partials, and retain the mixed cycle across those boundaries.
    widths = [
        [(1, 1)] * 40,
        [(1, 1)] * 40,
        [(2, 2)] * 16 + [(2, 2)] * 3 + [(1, 2), (1, 2)],
        [(3, 3)] * 10 + [(2, 3)] + [(3, 3)] * 2 + [(1, 3), (1, 3)],
        [(1, 1)] * 5 + [(2, 2), (3, 3), (1, 1)] * 4 + [(2, 2), (1, 3)] +
        [(1, 1), (2, 2), (3, 3), (1, 1), (1, 2)],
    ]
    schedules = []
    for index, (name, shapes) in enumerate(zip(NAMES, widths)):
        windows, offset = [], 0
        for rows, requested in shapes:
            hits = (0 if offset == 0 else 3 + offset % 11) if index == 0 else (
                9 + offset % 7 if rows == 1 else 80 * (rows - 1) + 12)
            windows.append({"offset": offset, "rows": rows, "requested_rows": requested, "routes": 480 * rows,
                            "delta": route(rows, hits)})
            offset += rows
        schedule = {"kind": "session_batch_schedule", "protocol": 1, "index": index, "schedule": name,
                    "reference": index == 0, "occupied_prefix_n1_rows": 5 if index == 4 else 0,
                    "initial_stats": zero_stats(), "final_reset_stats": zero_stats(), "windows": windows, "passed": True}
        fill_accounting(schedule, windows if index == 0 else schedules[0]["windows"])
        preserved = sum(p["preserved_logit_values"] for p in schedule["invalid_proofs"])
        teacher_windows = [32, 32, 16, 11, 19][index]
        schedule["counts"] = {
            "teacher_rows": 32, "tail_rows": 8, "rows": 40, "finite_rows": 40,
            "bitwise_compared_rows": 40 if index else 0,
            "teacher_finite_logit_values": 7946240, "teacher_bitwise_compared_logit_values": 7946240 if index else 0,
            "tail_finite_logit_values": 1986560, "tail_bitwise_compared_logit_values": 1986560 if index else 0,
            "finite_logit_values": 9932800, "bitwise_compared_logit_values": 9932800 if index else 0,
            "windows": len(windows), "teacher_windows": teacher_windows, "tail_windows": len(windows) - teacher_windows,
            "partial_windows": [0, 0, 2, 3, 2][index], "routes": 19200, "invalid_windows": 7 if index else 0,
            "rejection_preserved_logit_values": preserved,
        }
        final = copy.deepcopy(loaded)
        for d in final["devices"]:
            d["free_vram"] -= 100 + index
        snapshots = [42, 58, 39, 33, 42][index]
        schedule["memory"] = {"snapshot_count": snapshots, "steady_comparisons_to_loaded": snapshots,
                              "stats_preserved_comparisons": snapshots,
                              "bitwise_preserved_logit_values": 9932800 + 2 * preserved + (248320 if index else 0),
                              "minimum_free_vram_bytes": [d["free_vram"] - 200 for d in final["devices"]],
                              "steady_categories_counts_peak_host_and_payload_reads": True, "final_snapshot": final,
                              "loaded_anchor_snapshot": copy.deepcopy(loaded) if index == 0 else None}
        schedules.append(schedule)
    footer = {
        "kind": "session_batch_complete", "protocol": 1, "schedule_count": 5, "record_count": 7,
        "timeline_rows": 200, "teacher_rows": 160, "tail_rows": 40, "finite_rows": 200, "bitwise_compared_rows": 160,
        "timeline_routes": 96000, "window_count": 140, "finite_logit_values": 49664000,
        "bitwise_compared_logit_values": 39731200, "reset_probe_rows": 4, "reset_probe_routes": 1920,
        "reset_probe_finite_logit_values": 993280, "reset_probe_bitwise_compared_logit_values": 993280,
        "invalid_window_rejections": 28, "rejection_preserved_logit_values": 11919360,
        "loaded_memory_snapshot_count": 1, "schedule_memory_snapshot_count": 214, "memory_snapshot_count": 215,
        "memory_bitwise_preserved_logit_values": 74496000, "positive_grouped_reuse_schedules": 3,
        "session_instances": 1, "raii_session_cleanup_completed": True, "owned_buffer_release_measured": False,
        "performance_claim": False, "passed": True,
    }
    return [source, *schedules, footer]


def encoded(rows):
    return "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode("utf-8")


def target(value, path):
    for key in path:
        value = value[key]
    return value


def objects(value, path=()):
    """Visit every object schema; first/last array entries cover repeated windows."""
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


class BatchResultsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="batch-results-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.raw = self.root / "batch.jsonl"
        self.results = self.root / "results.jsonl"
        self.rows = fixture()
        self.raw.write_bytes(encoded(self.rows))

    def collect(self, rows=None):
        if rows is not None:
            self.raw.write_bytes(encoded(rows))
        return MODULE.collect(self.raw)

    def reject(self, rows):
        self.raw.write_bytes(encoded(rows))
        original = self.raw.read_bytes()
        with self.assertRaises(ValueError):
            MODULE.collect(self.raw)
        self.assertEqual(self.raw.read_bytes(), original)
        self.assertFalse(self.results.exists())

    def reject_change(self, row, path, value):
        rows = copy.deepcopy(self.rows)
        target(rows[row], path[:-1])[path[-1]] = value
        self.reject(rows)

    def cli(self, destination=None):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", self.raw.name,
                               "--results", str(destination if destination is not None else self.results)],
                              cwd=self.root, capture_output=True, text=True, check=False, timeout=30)

    def test_complete_synthetic_protocol_preserves_every_row_original_path_and_frozen_scope(self):
        original = self.raw.read_bytes()
        result = self.collect()
        self.assertEqual(result["kind"], "r4a_grouped_session")
        self.assertIs(result["passed"], True)
        self.assertEqual(result["source"], self.rows[0])
        self.assertEqual(result["schedules"], self.rows[1:6])
        self.assertEqual(result["complete"], self.rows[6])
        self.assertEqual(result["revision"], REVISION)
        self.assertIs(result["dirty"], True)
        self.assertEqual(result["model"], self.rows[0]["model"])
        self.assertEqual(result["model_variant"], "qwen38-keep1-Q4_0")
        self.assertEqual(result["raw_logs"], {"batch": str(self.raw.resolve())})
        self.assertEqual(datetime.datetime.fromisoformat(result["timestamp"]).utcoffset(), datetime.timedelta(0))
        for key in ("performance_claim", "pp_qualified", "independent_hf_claim", "mtp"):
            self.assertIs(result[key], False)
        self.assertIn("self-parity", result["scope"])
        self.assertIn("reuse diagnostics", result["scope"])
        self.assertIn("not PP, performance, or independent HF", result["scope"])
        self.assertIn("no individual-buffer-size proof", result["memory_scope"])
        self.assertIn("no after-destruction owned-buffer release measurement", result["memory_scope"])
        self.assertIn("not a syscall trace", result["expert_read_scope"])
        self.assertIn("not performance evidence", result["timing_scope"])
        self.assertEqual(self.raw.read_bytes(), original)
        json.dumps(result, allow_nan=False)

    def test_geometry_derived_exact_timeline_rejection_reset_and_audit_totals(self):
        result = self.collect()
        footer = result["complete"]
        for key, expected in (("timeline_rows", 200), ("window_count", 140), ("finite_logit_values", 49664000),
                              ("bitwise_compared_logit_values", 39731200), ("timeline_routes", 96000),
                              ("invalid_window_rejections", 28), ("reset_probe_rows", 4),
                              ("schedule_memory_snapshot_count", 214), ("memory_snapshot_count", 215),
                              ("memory_bitwise_preserved_logit_values", 74496000)):
            self.assertEqual(footer[key], expected)
        self.assertEqual([s["counts"]["windows"] for s in result["schedules"]], [40, 40, 21, 15, 24])
        self.assertEqual([s["counts"]["teacher_windows"] for s in result["schedules"]], [32, 32, 16, 11, 19])
        self.assertEqual([s["counts"]["partial_windows"] for s in result["schedules"]], [0, 0, 2, 3, 2])
        self.assertEqual([s["reuse"]["first_row"] for s in result["schedules"]], [0, 0, 2, 3, 7])
        self.assertEqual([s["counts"]["rejection_preserved_logit_values"] // VOCAB for s in result["schedules"]],
                         [0, 7, 11, 15, 15])

    def test_actual_revision_dirty_and_relative_model_path_are_preserved(self):
        for revision, dirty, model in (("f" * 40, False, "qwen38-keep1-Q4_0.gguf"),
                                       ("ABCDEF01" * 5, True, "/different/qwen38-keep1-Q4_0.gguf")):
            rows = copy.deepcopy(self.rows)
            rows[0].update(revision=revision, dirty=dirty, model=model)
            result = self.collect(rows)
            self.assertEqual(result["revision"], revision)
            self.assertIs(result["dirty"], dirty)
            self.assertEqual(result["model"], model)

    def test_missing_extra_duplicate_reordered_and_all_truncated_protocols(self):
        for i in range(7):
            for operation in ("missing", "duplicate", "replaced"):
                with self.subTest(row=i, operation=operation):
                    rows = copy.deepcopy(self.rows)
                    if operation == "missing":
                        rows.pop(i)
                    elif operation == "duplicate":
                        rows.insert(i, copy.deepcopy(rows[i]))
                    else:
                        rows[i] = copy.deepcopy(rows[(i + 1) % 7])
                    self.reject(rows)
        for end in range(7):
            self.reject(self.rows[:end])
        for i in range(6):
            rows = copy.deepcopy(self.rows)
            rows[i], rows[i + 1] = rows[i + 1], rows[i]
            self.reject(rows)

    def test_malformed_nonobject_utf8_bom_blank_or_unfinished_lines(self):
        raw = encoded(self.rows)
        bads = [raw[:-1], raw[:-20], raw + b"\n", raw + b"failure\n", raw + b" ",
                b"\xef\xbb\xbf" + raw, raw.replace(b"session_batch_source", b"\xff", 1)]
        lines = raw.split(b"\n")[:-1]
        for line in (b"", b" ", b"[]", b"null", b"true", b"1", b'"passed"', b"{", b"{} {}",
                     b'{"kind":1,}', b'{"x":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}"):
            bads.append(b"\n".join([*lines[:3], line, *lines[4:]]) + b"\n")
        for bad in bads:
            self.raw.write_bytes(bad)
            with self.assertRaises(ValueError):
                self.collect()
            self.assertEqual(self.raw.read_bytes(), bad)

    def test_duplicate_keys_rejected_at_every_protocol_level(self):
        raw = encoded(self.rows)
        for field in ("revision", "capacity", "min_q8_blocks_per_device", "windows", "offset", "hits",
                      "expert_hits", "case", "stats_bitwise_preserved", "applicable", "workspace", "snapshot_count", "passed"):
            key = ('"' + field + '":').encode()
            self.raw.write_bytes(raw.replace(key, key + b"0," + key, 1))
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                self.collect()

    def test_nonfinite_constants_and_overflowing_exponents_rejected(self):
        raw = encoded(self.rows)
        for literal in (b"NaN", b"Infinity", b"-Infinity", b"1e999", b"-1e999"):
            for field in (b'"last_completed_ms": 0.25', b'"weights": 1100000000', b'"model_bytes": 75399121792'):
                with self.subTest(literal=literal, field=field):
                    bad = raw.replace(field, field.split(b":")[0] + b": " + literal, 1)
                    self.assertNotEqual(raw, bad)
                    self.raw.write_bytes(bad)
                    with self.assertRaisesRegex(ValueError, "nonfinite"):
                        self.collect()

    def test_bounded_input_and_growing_read(self):
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

    def test_crlf_and_nonregular_input(self):
        self.raw.write_bytes(encoded(self.rows).replace(b"\n", b"\r\n"))
        self.assertIs(self.collect()["passed"], True)
        with self.assertRaisesRegex(ValueError, "regular file"):
            MODULE.collect(self.root)

    def test_all_object_schemas_require_exact_fields(self):
        for i, row in enumerate(self.rows):
            for path, obj in objects(row):
                for field in (*obj, "unexpected"):
                    with self.subTest(row=i, path=path, field=field):
                        rows = copy.deepcopy(self.rows)
                        changed = target(rows[i], path)
                        if field == "unexpected":
                            changed[field] = True
                        else:
                            del changed[field]
                        self.reject(rows)

    def test_numeric_schema_rejects_booleans_floating_counters_and_out_of_range(self):
        # Every numeric field, with repeated windows sampled at both boundaries.
        seen = set()
        for i, row in enumerate(self.rows):
            for path, obj in objects(row):
                for key, value in obj.items():
                    if type(value) not in (int, float):
                        continue
                    identity = (i, path + (key,))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    with self.subTest(row=i, path=path + (key,)):
                        self.reject_change(i, path + (key,), True)
                        if type(value) is int and key != "last_completed_ms":
                            self.reject_change(i, path + (key,), float(value))
        for i in range(1, 6):
            for key in (*COUNTERS, "consumed_tokens", "last_completed_ms"):
                for bad in (-1, "1", None, 10**400):
                    self.reject_change(i, ("windows", 0, "stats", key), bad)
        for path in (("teacher_ids", 0), ("capacity_tail_ids", 0)):
            self.reject_change(0, path, True)
        self.reject_change(1, ("memory", "minimum_free_vram_bytes", 0), True)

    def test_false_proofs_boolean_types_and_phase_nullability(self):
        for i, row in enumerate(self.rows):
            for path, obj in objects(row):
                for key, value in obj.items():
                    if type(value) is bool:
                        self.reject_change(i, path + (key,), int(value))
                        if key != "dirty":
                            self.reject_change(i, path + (key,), not value)
                    elif value is None:
                        self.reject_change(i, path + (key,), False)

    def test_all_fixed_source_scopes_counts_and_geometry_cannot_change(self):
        for path, value in leaves(self.rows[0]):
            if path[0] in ("revision", "dirty", "model", "model_bytes"):
                continue
            bad = value + 1 if type(value) is int else (not value if type(value) is bool else "changed")
            self.reject_change(0, path, bad)

    def test_invalid_compiled_revision_dirty_model_path_and_size(self):
        for bad in (None, 0, True, "", "a" * 39, "a" * 41, "g" * 40, "a" * 39 + "\n"):
            self.reject_change(0, ("revision",), bad)
        for bad in (None, 0, 1, "false", 1.0):
            self.reject_change(0, ("dirty",), bad)
        for bad in (None, 1, "", "/models/full-PLE.gguf", "/models/qwen38-keep1-Q8_0.gguf",
                    "/models/\ud800/qwen38-keep1-Q4_0.gguf", "/models/\n/qwen38-keep1-Q4_0.gguf",
                    "/models/\x00/qwen38-keep1-Q4_0.gguf", "/" + "x" * 4096 + "/qwen38-keep1-Q4_0.gguf"):
            self.reject_change(0, ("model",), bad)
        for bad in (0, -1, 1 << 63, True, 1.0, "1"):
            self.reject_change(0, ("model_bytes",), bad)

    def test_exact_schedule_window_order_extents_and_requested_partials(self):
        for row in range(1, 6):
            schedule = self.rows[row]
            self.reject_change(row, ("windows",), schedule["windows"][:-1])
            self.reject_change(row, ("windows",), schedule["windows"] + [schedule["windows"][-1]])
            swapped = copy.deepcopy(schedule["windows"])
            swapped[0], swapped[1] = swapped[1], swapped[0]
            self.reject_change(row, ("windows",), swapped)
            for position in (0, len(schedule["windows"]) - 2, len(schedule["windows"]) - 1):
                for field in ("offset", "rows", "requested_rows", "routes"):
                    self.reject_change(row, ("windows", position, field), schedule["windows"][position][field] + 1)
            for field, value in schedule["counts"].items():
                self.reject_change(row, ("counts", field), value + 1)

    def test_every_window_route_delta_and_cumulative_accounting(self):
        for row in range(1, 6):
            for pos, w in enumerate(self.rows[row]["windows"]):
                for field in ROUTE_KEYS:
                    self.reject_change(row, ("windows", pos, "delta", field), w["delta"][field] + 1)
                for counter in COUNTERS:
                    self.reject_change(row, ("windows", pos, "stats", counter), w["stats"][counter] + 1)
        rows = copy.deepcopy(self.rows)
        # Correct total but a decreased individual cumulative counter.
        rows[2]["windows"][1]["stats"].update(expert_hits=0, expert_misses=960)
        self.reject(rows)

    def test_upload_payload_bounds_and_n1_one_slot_bound_are_not_relaxed(self):
        for row in range(1, 6):
            for pos in (0, len(self.rows[row]["windows"]) - 1):
                delta = self.rows[row]["windows"][pos]["delta"]
                for bad in (delta["misses"] * 2764800 - 1, delta["misses"] * 2867200 + 1):
                    self.reject_change(row, ("windows", pos, "delta", "upload_bytes"), bad)
        rows = copy.deepcopy(self.rows)
        rows[2]["windows"][0]["delta"] = route(1, 49)
        fill_accounting(rows[2], rows[1]["windows"])
        self.reject(rows)
        rows = copy.deepcopy(self.rows)
        rows[3]["reset_probe"]["window"]["delta"] = route(1, 49)
        self.reject(rows)

    def test_matching_full_n1_deltas_are_required_for_every_window_and_reset(self):
        for row in range(1, 6):
            for pos, w in enumerate(self.rows[row]["windows"]):
                reference = copy.deepcopy(w["n1_reference_delta"])
                reference["hits"] += 1
                reference["misses"] -= 1
                reference["upload_bytes"] -= 2764800
                self.reject_change(row, ("windows", pos, "n1_reference_delta"), reference)
            if row > 1:
                self.reject_change(row, ("reset_probe", "window", "n1_reference_delta"), route(1, 1))

    def test_positive_matching_suffix_reuse_and_correct_arithmetic(self):
        for row in (3, 4, 5):
            for field, value in self.rows[row]["reuse"].items():
                if type(value) is int:
                    self.reject_change(row, ("reuse", field), value + 1)
            for field in ("hits_gained", "misses_saved", "teacher_within_window_hit_lower_bound"):
                self.reject_change(row, ("reuse", field), 0)
            for block in ("candidate", "n1_reference"):
                for field in ROUTE_KEYS:
                    self.reject_change(row, ("reuse", block, field), self.rows[row]["reuse"][block][field] + 1)

    def test_within_window_lower_bound_is_required_even_with_positive_suffix_gains(self):
        for row in (3, 4, 5):
            rows = copy.deepcopy(self.rows)
            for w in rows[row]["windows"]:
                if w["offset"] < 32:
                    w["delta"] = route(w["rows"], 16 * w["rows"])
            fill_accounting(rows[row], rows[1]["windows"])
            self.assertGreater(rows[row]["reuse"]["hits_gained"], 0)
            self.assertGreater(rows[row]["reuse"]["misses_saved"], 0)
            self.assertEqual(rows[row]["reuse"]["teacher_within_window_hit_lower_bound"], 0)
            self.reject(rows)

    def test_positive_within_window_hits_cannot_substitute_for_matching_suffix_gain(self):
        for row in (3, 4, 5):
            rows = copy.deepcopy(self.rows)
            first = rows[row]["reuse"]["first_row"]
            for w in rows[row]["windows"]:
                if first <= w["offset"] < 32:
                    w["delta"] = route(w["rows"], 0)
            fill_accounting(rows[row], rows[1]["windows"])
            self.assertGreater(rows[row]["reuse"]["teacher_within_window_hit_lower_bound"], 0)
            self.assertLess(rows[row]["reuse"]["hits_gained"], 0)
            self.reject(rows)

    def test_reuse_scope_is_matching_suffix_not_all_teacher_or_tail(self):
        for row in (3, 4, 5):
            rows = copy.deepcopy(self.rows)
            all_teacher = [w for w in rows[row]["windows"] if w["offset"] < 32]
            rows[row]["reuse"].update(first_row=0, rows=32, candidate=summed(all_teacher),
                                      n1_reference=summed(rows[1]["windows"][:32]))
            self.reject(rows)

    def test_all_invalid_cases_preserve_stats_timing_logits_and_ledger_then_continue(self):
        for row in range(2, 6):
            for i, proof in enumerate(self.rows[row]["invalid_proofs"]):
                for field, bad in (("case", "other"), ("rows", proof["rows"] + 1),
                                   ("prior_rows", proof["prior_rows"] + 1),
                                   ("preserved_logit_values", proof["preserved_logit_values"] - 1),
                                   ("exception", "runtime_error"), ("stats_bitwise_preserved", False),
                                   ("logits_bitwise_preserved", False), ("memory_ledger_preserved", False),
                                   ("consumed_advance", 1), ("continued", False),
                                   ("reset_before_continuation", not proof["reset_before_continuation"]),
                                   ("continuation_rows", proof["continuation_rows"] + 1),
                                   ("continuation_bitwise_logit_values", proof["continuation_bitwise_logit_values"] - 1)):
                    self.reject_change(row, ("invalid_proofs", i, field), bad)
                for field, value in proof["stats_before_and_after"].items():
                    self.reject_change(row, ("invalid_proofs", i, "stats_before_and_after", field), value + 1)
            self.reject_change(row, ("invalid_proofs",), self.rows[row]["invalid_proofs"][::-1])
            self.reject_change(row, ("invalid_proofs",), self.rows[row]["invalid_proofs"][:-1])

    def test_invalid_timing_preserves_double_bits_including_signed_zero(self):
        rows = copy.deepcopy(self.rows)
        prior = rows[2]["windows"][0]["stats"]
        prior["last_completed_ms"] = 0
        for proof in rows[2]["invalid_proofs"][:4]:
            proof["stats_before_and_after"]["last_completed_ms"] = 0.0
        self.assertIs(self.collect(rows)["passed"], True)
        rows[2]["invalid_proofs"][0]["stats_before_and_after"]["last_completed_ms"] = -0.0
        self.reject(rows)

    def test_integer_looking_negative_zero_timing_retains_its_sign(self):
        rows = copy.deepcopy(self.rows)
        rows[2]["windows"][0]["stats"]["last_completed_ms"] = -0.0
        for proof in rows[2]["invalid_proofs"][:4]:
            proof["stats_before_and_after"]["last_completed_ms"] = -0.0
        raw = encoded(rows).replace(b'"last_completed_ms": -0.0', b'"last_completed_ms": -0')
        self.raw.write_bytes(raw)
        self.assertIs(self.collect()["passed"], True)
        bad = raw.replace(b'"last_completed_ms": -0', b'"last_completed_ms": 0', 1)
        self.raw.write_bytes(bad)
        with self.assertRaisesRegex(ValueError, "timing bits changed"):
            self.collect()

    def test_zero_reset_stats_exact_continuation_counts_and_end_stats(self):
        for row in range(1, 6):
            for block in ("initial_stats", "final_reset_stats"):
                for field in zero_stats():
                    self.reject_change(row, (block, field), 1)
            for block in ("teacher_stats", "timeline_stats"):
                for field, value in self.rows[row][block].items():
                    self.reject_change(row, (block, field), value + 1)
            if row > 1:
                for field in ("rows", "finite_logit_values", "bitwise_compared_logit_values"):
                    self.reject_change(row, ("reset_probe", field), self.rows[row]["reset_probe"][field] - 1)
                self.reject_change(row, ("reset_probe", "window", "offset"), 1)

    def test_memory_owners_category_sum_workspace_host_capacities_and_exact_reads(self):
        for row in range(1, 6):
            for i, d in enumerate(self.rows[row]["memory"]["final_snapshot"]["devices"]):
                for field in ("device", "first_layer", "last_layer", "gdn_layers", "qsa_layers", "owned_bytes"):
                    self.reject_change(row, ("memory", "final_snapshot", "devices", i, field), d[field] + 1)
                for field in ("weights", "expert_slots", "qsa_kv", "qsa_index", "gdn_state", "workspace", "owned_buffers"):
                    self.reject_change(row, ("memory", "final_snapshot", "devices", i, field), 0)
                for field, bad in (("owned_peak_bytes", d["owned_bytes"] - 1), ("total_vram", d["owned_bytes"] - 1),
                                   ("free_vram", 0), ("free_vram", d["total_vram"] + 1), ("ple_state", 0 if i == 0 else 1)):
                    self.reject_change(row, ("memory", "final_snapshot", "devices", i, field), bad)
            host = self.rows[row]["memory"]["final_snapshot"]
            for field in ("capacity", "expert_slots", "ram_expert_payload", "pinned_handoff",
                          "expert_payload_reads", "expert_payload_bytes_read"):
                self.reject_change(row, ("memory", "final_snapshot", field), host[field] + 1)
            for field, bad in (("ram_expert_capacity", 68262297599), ("host_embedding_capacity", 0),
                               ("host_logit_capacity", 2979839)):
                self.reject_change(row, ("memory", "final_snapshot", field), bad)
        for field, bad in (("workspace", 6553343), ("weights", 0), ("owned_buffers", 0)):
            self.reject_change(1, ("memory", "loaded_anchor_snapshot", "devices", 0, field), bad)

    def test_steady_ledger_cannot_hide_growth_with_a_consistent_category_sum(self):
        for row in range(1, 6):
            for i in (0, 1):
                for field in (*CATEGORIES, "owned_buffers", "owned_peak_bytes", "total_vram"):
                    rows = copy.deepcopy(self.rows)
                    d = rows[row]["memory"]["final_snapshot"]["devices"][i]
                    d[field] += 1
                    if field in CATEGORIES:
                        d["owned_bytes"] += 1
                        d["owned_peak_bytes"] += 1
                    self.reject(rows)
            for field in ("ram_expert_capacity", "host_embedding_capacity", "host_logit_capacity"):
                self.reject_change(row, ("memory", "final_snapshot", field),
                                   self.rows[row]["memory"]["final_snapshot"][field] + 1)

    def test_observed_memory_is_not_guessed_individual_buffer_or_release_geometry(self):
        rows = copy.deepcopy(self.rows)
        memories = [rows[1]["memory"]["loaded_anchor_snapshot"],
                    *(s["memory"]["final_snapshot"] for s in rows[1:6])]
        for m in memories:
            m["ram_expert_capacity"] += 65536
            m["host_embedding_capacity"] = 1
            m["host_logit_capacity"] += 4096
            for d in m["devices"]:
                d["weights"] += 17
                d["workspace"] += 31
                d["owned_bytes"] += 48
                d["owned_peak_bytes"] += 48
                d["owned_buffers"] += 2
        result = self.collect(rows)
        self.assertEqual(result["schedules"], rows[1:6])
        self.assertIs(result["source"]["geometry"]["individual_q8_scratch_geometry_observable"], False)
        self.assertIs(result["complete"]["owned_buffer_release_measured"], False)
        self.reject_change(1, ("memory", "final_snapshot", "q8_buffer_bytes"), 34560)
        self.reject_change(6, ("after_destruction",), {"released": True})

    def test_minimum_free_vram_is_bounded_by_loaded_and_final_but_need_not_equal_either(self):
        for row in range(1, 6):
            for i in (0, 1):
                final = self.rows[row]["memory"]["final_snapshot"]["devices"][i]["free_vram"]
                self.reject_change(row, ("memory", "minimum_free_vram_bytes", i), final + 1)
                self.reject_change(row, ("memory", "minimum_free_vram_bytes", i), 0)
            self.reject_change(row, ("memory", "minimum_free_vram_bytes"), [1])
        rows = copy.deepcopy(self.rows)
        for s in rows[1:6]:
            s["memory"]["minimum_free_vram_bytes"] = [1, 1]
            for d in s["memory"]["final_snapshot"]["devices"]:
                d["free_vram"] += 1000  # Final free may exceed the anchor; minima still cannot.
        self.assertIs(self.collect(rows)["passed"], True)
        rows[2]["memory"]["minimum_free_vram_bytes"][0] = rows[1]["memory"]["loaded_anchor_snapshot"]["devices"][0]["free_vram"] + 1
        self.reject(rows)

    def test_memory_audit_counts_include_both_invalid_snapshots_and_reset_probe(self):
        for row in range(1, 6):
            for field in ("snapshot_count", "steady_comparisons_to_loaded", "stats_preserved_comparisons",
                          "bitwise_preserved_logit_values"):
                self.reject_change(row, ("memory", field), self.rows[row]["memory"][field] - 1)

    def test_complete_footer_exact_counts_cleanup_and_nonclaims(self):
        for field, value in self.rows[6].items():
            if type(value) is int:
                self.reject_change(6, (field,), value + 1)
            elif type(value) is bool:
                self.reject_change(6, (field,), not value)

    def test_cli_appends_exactly_one_record_to_caller_journal_preserving_old_bytes_and_raw(self):
        before = b'{"kind":"previous"}\n'
        self.results.write_bytes(before)
        original = self.raw.read_bytes()
        process = self.cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        output = self.results.read_bytes()
        self.assertTrue(output.startswith(before))
        self.assertEqual(len(output.splitlines()), 2)
        record = json.loads(output.splitlines()[1])
        self.assertEqual(record["kind"], "r4a_grouped_session")
        self.assertEqual(record["source"], self.rows[0])
        self.assertEqual(record["schedules"], self.rows[1:6])
        self.assertEqual(record["complete"], self.rows[6])
        self.assertEqual(record["raw_logs"], {"batch": str(self.raw.resolve())})
        self.assertEqual(self.raw.read_bytes(), original)

    def test_cli_creates_one_row_at_explicit_relative_destination(self):
        process = self.cli(Path("results.jsonl"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(self.results.read_bytes().splitlines()), 1)
        self.assertEqual(json.loads(self.results.read_bytes())["kind"], "r4a_grouped_session")

    def test_cli_rejection_never_appends_creates_journal_or_mutates_raw(self):
        previous = b'{"kind":"previous"}\n'
        self.results.write_bytes(previous)
        bads = [encoded(self.rows[:-1]), encoded(self.rows)[:-1], b"not JSON\n"]
        for row, path, bad in ((0, ("revision",), "g" * 40), (3, ("reuse", "hits_gained"), 0),
                               (4, ("invalid_proofs", 0, "stats_bitwise_preserved"), False),
                               (5, ("memory", "final_snapshot", "pinned_handoff"), 40960),
                               (6, ("raii_session_cleanup_completed",), False)):
            rows = copy.deepcopy(self.rows)
            target(rows[row], path[:-1])[path[-1]] = bad
            bads.append(encoded(rows))
        for raw in bads:
            self.raw.write_bytes(raw)
            process = self.cli()
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertIn("record_batch:", process.stderr)
            self.assertEqual(self.results.read_bytes(), previous)
            absent = self.root / "absent-results.jsonl"
            self.assertEqual(self.cli(absent).returncode, 1)
            self.assertFalse(absent.exists())
            self.assertEqual(self.raw.read_bytes(), raw)

    def test_cli_raw_model_symlink_hardlink_and_incomplete_destinations_are_immutable(self):
        model = self.root / "qwen38-keep1-Q4_0.gguf"
        model.write_bytes(b"synthetic model path; collector must not read model weights")
        self.rows[0]["model"] = str(model)
        self.raw.write_bytes(encoded(self.rows))
        for source in (self.raw, model):
            original = source.read_bytes()
            symlink = self.root / (source.name + ".symlink")
            symlink.symlink_to(source)
            hardlink = self.root / (source.name + ".hardlink")
            os.link(source, hardlink)
            for destination in (source, symlink, hardlink):
                self.assertEqual(self.cli(destination).returncode, 1)
                self.assertEqual(source.read_bytes(), original)
        self.results.write_bytes(b'{"kind":"incomplete"}')
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), b'{"kind":"incomplete"}')
        self.assertEqual(self.cli(self.root).returncode, 1)

    def test_cli_requires_both_explicit_paths(self):
        for args in ([], ["--raw", str(self.raw)], ["--results", str(self.results)]):
            process = subprocess.run([sys.executable, "-B", str(SCRIPT), *args], cwd=self.root,
                                     capture_output=True, text=True, check=False, timeout=30)
            self.assertEqual(process.returncode, 2)
            self.assertFalse(self.results.exists())

    def test_concurrent_cli_calls_use_existing_locked_whole_record_append(self):
        previous = b'{"kind":"previous"}\n'
        self.results.write_bytes(previous)
        command = [sys.executable, "-B", str(SCRIPT), "--raw", str(self.raw), "--results", str(self.results)]
        processes = []
        try:
            for _ in range(2):
                processes.append(subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE, text=True))
            for process in processes:
                _, errors = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, errors)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        lines = self.results.read_bytes().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0] + b"\n", previous)
        for line in lines[1:]:
            self.assertEqual(json.loads(line)["schedules"], self.rows[1:6])


if __name__ == "__main__":
    unittest.main()
