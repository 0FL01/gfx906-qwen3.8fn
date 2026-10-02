"""Independent full core-memory protocol fixtures; stdlib only, no HIP/model."""

import copy
import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/record_memory.py"
SPEC = importlib.util.spec_from_file_location("record_memory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
REVISION = "0123456789abcdef0123456789abcdef01234567"


def fixture():
    """Spell the 569-line driver's schema independently of collector constants."""
    total = 17163091968
    before = [{"device": i, "name": f"synthetic GPU {i}", "arch": "gfx906:sramecc+:xnack-",
               "wave_size": 64, "compute_units": 60, "async_engines": 8,
               "properties_total_vram": total, "total_vram": total,
               "free_vram": total - 64 * 1024 * 1024} for i in (0, 1)]
    source = {
        "kind": "memory_source", "protocol": 1, "revision": REVISION, "dirty": True,
        "build_provenance_complete": True, "model_path": "/models/qwen38-keep1-Q4_0.gguf",
        "runtime": "own_48_layer_HIP", "scope": "r3_allocation_owner_headroom_reset_replay",
        "capacity": 131072, "expert_slots": 112, "session_instances": 1, "trace": False,
        "sampling": "teacher_forced", "token_ids": [248044, 100], "vocabulary": 248320,
        "min_free_headroom_bytes": 1073741824, "gates_frozen_before_execution": True,
        "occupied_128k_qualified": False, "performance_claim": False,
        "limits": {
            "gpu_categories": "live_Session_Buffer_allocations_only",
            "gpu_private_allocations": "HIP_context_rocBLAS_visible_only_in_total_free_VRAM",
            "workspace": "includes_staged_recurrent_conv_PLE_state",
            "host_capacities": "reported_payload_buffers_only_excludes_metadata_allocator_overhead",
            "host_embedding_payload": "not_exposed_by_SessionMemory_capacity_checked_nonzero_and_steady",
            "rss": "getrusage_RUSAGE_SELF_ru_maxrss_Linux_KiB_times_1024",
            "context": "capacity_allocation_only_two_consumed_tokens_per_pass", "mtp": False,
        },
        "expected_geometry": {
            "qsa_kv_bytes_per_gpu": 452984832, "qsa_pooled_index_bytes_per_gpu": 100663296,
            "qsa_index_tail_bytes_per_gpu": 9216, "qsa_index_bytes_per_gpu": 100672512,
            "gdn_state_bytes_per_gpu": 58834944, "ple_state_bytes": [368640, 0],
            "q4_1_down_expert_bytes": 2867200, "q4_0_down_expert_bytes": 2764800,
            "expert_slots_bytes": [7500595200, 7431782400], "ram_expert_payload_bytes": 68262297600,
            "expert_payload_reads": 144, "host_logit_payload_bytes": 993280, "pinned_handoff_bytes": 40960,
        }, "before_load": before, "caller_device": 1, "peak_rss_bytes": 8388608,
    }
    devices = []
    for i in (0, 1):
        device = {
            "device": i, "first_layer": i * 24, "last_layer": i * 24 + 23,
            "gdn_layers": 18, "qsa_layers": 6, "weights": 1100000000 + i * 12345678,
            "expert_slots": [7500595200, 7431782400][i], "qsa_kv": 452984832,
            "qsa_index": 100672512, "gdn_state": 58834944, "ple_state": [368640, 0][i],
            "workspace": 170000000 + i * 43210, "owned_buffers": 901 + i * 37,
            "total_vram": total,
        }
        device["owned_bytes"] = sum(device[k] for k in
                                    ("weights", "expert_slots", "qsa_kv", "qsa_index", "gdn_state", "ple_state", "workspace"))
        device["owned_peak_bytes"] = device["owned_bytes"] + 4096
        device["free_vram"] = before[i]["free_vram"] - device["owned_bytes"] - 128 * 1024 * 1024
        devices.append(device)
    host = {"ram_expert_capacity": 68262297600 + 4096, "ram_expert_payload": 68262297600,
            "host_embedding_capacity": 521472000, "host_logit_capacity": 993280 + 4096,
            "pinned_handoff": 40960, "expert_payload_reads": 144, "expert_payload_bytes_read": 68262297600}
    zero = {"consumed_tokens": 0, "expert_hits": 0, "expert_misses": 0,
            "expert_upload_bytes": 0, "last_completed_ms": 0}
    stats = [zero,
             {"consumed_tokens": 1, "expert_hits": 0, "expert_misses": 480,
              "expert_upload_bytes": 480 * 2764800, "last_completed_ms": 12.5},
             {"consumed_tokens": 2, "expert_hits": 120, "expert_misses": 840,
              "expert_upload_bytes": 480 * 2764800 + 360 * 2867200, "last_completed_ms": 11.0},
             zero,
             {"consumed_tokens": 1, "expert_hits": 480, "expert_misses": 0,
              "expert_upload_bytes": 0, "last_completed_ms": 0},
             {"consumed_tokens": 2, "expert_hits": 880, "expert_misses": 80,
              "expert_upload_bytes": 80 * 2800000, "last_completed_ms": 10.75}]
    snapshots = []
    for i, (phase, consumed) in enumerate(zip(("loaded", "step0", "step1", "reset", "replay0", "replay1"),
                                              (0, 1, 2, 0, 1, 2))):
        has_logits, replay = consumed > 0, i >= 4
        snapshots.append({
            "kind": "memory_snapshot", "phase": phase, "capacity": 131072, "expert_slots": 112,
            "consumed_tokens": consumed, "token_id": [248044, 100][consumed - 1] if has_logits else None,
            "stats": copy.deepcopy(stats[i]), "ownership_verified": True, "devices": copy.deepcopy(devices),
            "host": copy.deepcopy(host), "peak_rss_bytes": 69 * 1024**3 + i * 1024,
            "checks": {"geometry": True, "headroom": True, "steady_allocations": True if i else None,
                       "zero_stats": None if has_logits else True, "finite_logits": True if has_logits else None,
                       "finite_logit_values": 248320 if has_logits else 0,
                       "bitwise_replay": True if replay else None,
                       "bitwise_replay_compared_values": 248320 if replay else 0,
                       "snapshot_stats_preserved": True, "snapshot_logits_preserved": True if has_logits else None,
                       "snapshot_logit_compared_values": 248320 if has_logits else 0,
                       "snapshot_caller_device_preserved": True, "snapshot_caller_device": i % 2},
        })
        # Free VRAM is observed, not part of the steady live-buffer ledger.
        for device in snapshots[-1]["devices"]:
            device["free_vram"] -= i * 1024
    after = copy.deepcopy(before)
    recovery = []
    for i in (0, 1):
        after[i]["free_vram"] -= 32 * 1024 * 1024
        last = snapshots[-1]["devices"][i]
        recovered = after[i]["free_vram"] - last["free_vram"]
        recovery.append({
            "device": i, "before_load_free_vram": before[i]["free_vram"],
            "last_snapshot_free_vram": last["free_vram"], "after_destroy_free_vram": after[i]["free_vram"],
            "required_owned_bytes": last["owned_bytes"], "free_recovered_bytes": recovered,
            "recovered_minus_owned_bytes": recovered - last["owned_bytes"],
            "min_snapshot_free_vram": min(s["devices"][i]["free_vram"] for s in snapshots),
            "owned_bytes_recovered": True,
        })
    footer = {
        "kind": "memory_complete", "protocol": 1, "scope": "r3_allocation_owner_headroom_reset_replay",
        "capacity": 131072, "expert_slots": 112, "session_instances": 1, "consumed_tokens": 2,
        "baseline_consumed_tokens": 2, "replay_consumed_tokens": 2,
        "occupied_128k_qualified": False, "performance_claim": False, "after_destruction": after,
        "cleanup": {"both_devices_synchronized": True, "device_reset_used": False, "devices": recovery,
                    "caller_device_before": 1, "caller_device_after": 1},
        "peak_rss_bytes": 69 * 1024**3 + 8192,
        "checks": {"completed_steps": 4, "snapshot_count": 6, "steady_memory_comparisons": 5,
                   "initial_stats_zero": True, "reset_consumed_and_stats_zero": True, "all_outputs_finite": True,
                   "finite_logit_values": 993280, "bitwise_replay": True, "bitwise_replay_compared_values": 496640,
                   "snapshot_stats_checks": 6, "snapshot_device_checks": 6, "snapshot_logit_compared_values": 993280,
                   "ownership_verified": True, "steady_categories_counts_peak_host_and_payload_reads": True,
                   "min_free_headroom_bytes": 1073741824, "expert_payload_reads": 144,
                   "expert_payload_bytes_read": 68262297600, "caller_device_restored": True,
                   "owned_bytes_recovered_both_devices": True}, "passed": True,
    }
    return [source, *snapshots, footer]


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
    elif type(value) is list:
        for i, child in enumerate(value):
            yield from objects(child, path + (i,))


def leaves(value, path=()):
    if type(value) is dict:
        for key, child in value.items():
            yield from leaves(child, path + (key,))
    elif type(value) is list:
        for i, child in enumerate(value):
            yield from leaves(child, path + (i,))
    else:
        yield path, value


class MemoryResultsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="memory-results-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.raw = self.root / "memory.jsonl"
        self.results = self.root / "results.jsonl"
        self.rows = fixture()
        self.raw.write_bytes(encoded(self.rows))

    def collect(self, rows=None):
        if rows is not None:
            self.raw.write_bytes(encoded(rows))
        return MODULE.collect(self.raw)

    def reject_change(self, row, path, value):
        changed = copy.deepcopy(self.rows)
        target(changed[row], path[:-1])[path[-1]] = value
        with self.assertRaises(ValueError):
            self.collect(changed)

    def cli(self, results=None):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", self.raw.name,
                               "--results", str(results or self.results)], cwd=self.root,
                              capture_output=True, text=True, check=False)

    def test_complete_proof_preserves_all_rows_provenance_absolute_paths_and_scope(self):
        original = self.raw.read_bytes()
        result = self.collect()
        self.assertEqual(result["kind"], "r3c_memory")
        self.assertIs(result["passed"], True)
        self.assertEqual(result["source"], self.rows[0])
        self.assertEqual(result["snapshots"], self.rows[1:7])
        self.assertEqual(result["complete"], self.rows[7])
        self.assertEqual(result["revision"], REVISION)
        self.assertIs(result["dirty"], True)
        self.assertEqual(result["model"], self.rows[0]["model_path"])
        self.assertEqual(result["raw_logs"], {"memory": str(self.raw.resolve())})
        self.assertEqual(datetime.datetime.fromisoformat(result["timestamp"]).utcoffset(), datetime.timedelta(0))
        for key in ("performance_claim", "occupied_128k_qualified", "mtp"):
            self.assertIs(result[key], False)
        self.assertIn("capacity allocation only", result["scope"])
        self.assertIn("two consumed teacher tokens per pass", result["scope"])
        self.assertIn("not occupied128K", result["scope"])
        self.assertIn("not a syscall trace or physical SSD-read trace", result["expert_read_scope"])
        self.assertIn("no full-RAM accounting", result["ram_scope"])
        self.assertIn("not performance evidence", result["timing_scope"])
        self.assertEqual(result["fixture_gates"]["min_free_headroom_bytes"], 1073741824)
        self.assertEqual(self.raw.read_bytes(), original)
        json.dumps(result, allow_nan=False)

    def test_other_compiled_revision_dirty_and_caller_are_not_hardcoded(self):
        for revision, dirty, caller in (("f" * 40, False, 0), ("ABCDEF01" * 5, True, 1)):
            rows = copy.deepcopy(self.rows)
            rows[0].update(revision=revision, dirty=dirty, caller_device=caller)
            rows[7]["cleanup"].update(caller_device_before=caller, caller_device_after=caller)
            result = self.collect(rows)
            self.assertEqual(result["revision"], revision)
            self.assertIs(result["dirty"], dirty)
            self.assertEqual(result["source"], rows[0])

    def test_all_schema_objects_require_exact_fields(self):
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
                        with self.assertRaises(ValueError):
                            self.collect(rows)

    def test_all_numeric_leaves_reject_booleans_and_integer_leaves_reject_floats(self):
        for i, row in enumerate(self.rows):
            for path, value in leaves(row):
                if type(value) in (int, float):
                    with self.subTest(row=i, path=path, replacement="bool"):
                        self.reject_change(i, path, True)
                if type(value) is int and path[-1] != "last_completed_ms":
                    with self.subTest(row=i, path=path, replacement="float"):
                        self.reject_change(i, path, float(value))

    def test_flags_require_exact_booleans_and_phase_nullability(self):
        for i, row in enumerate(self.rows):
            for path, value in leaves(row):
                if type(value) is bool:
                    for bad in (1, 0, None, "true"):
                        with self.subTest(row=i, path=path, value=bad):
                            self.reject_change(i, path, bad)
                    if path[-1] != "dirty":
                        self.reject_change(i, path, not value)
                elif value is None:
                    for bad in (True, False, 0):
                        self.reject_change(i, path, bad)

    def test_fixed_source_contract_and_every_geometry_leaf_cannot_change(self):
        for path, value in leaves(self.rows[0]):
            if path[0] in ("expected_geometry", "limits"):
                bad = value + 1 if type(value) is int else (not value if type(value) is bool else "changed")
                self.reject_change(0, path, bad)
        for key, bad in (("runtime", "llama_decode"), ("scope", "performance"), ("protocol", 2),
                         ("capacity", 131071), ("expert_slots", 111), ("session_instances", 2),
                         ("sampling", "greedy"), ("vocabulary", 248319), ("min_free_headroom_bytes", 1073741823)):
            self.reject_change(0, (key,), bad)
        for bad in ([248044], [248044, 101], [100, 248044], [248044, 100, 101]):
            self.reject_change(0, ("token_ids",), bad)

    def test_invalid_compiled_provenance_and_model_paths(self):
        for value in (None, 0, True, "", "a" * 39, "a" * 41, "g" * 40, "a" * 39 + "\n"):
            self.reject_change(0, ("revision",), value)
        for value in (None, 0, 1, 1.0, "false"):
            self.reject_change(0, ("dirty",), value)
        for value in (None, 1, "", "qwen38-keep1-Q4_0.gguf", "/models/full-PLE.gguf",
                      "/models/qwen38-keep1-Q8_0.gguf", "/models/\ud800/qwen38-keep1-Q4_0.gguf",
                      "/models/\n/qwen38-keep1-Q4_0.gguf", "/models/\x00/qwen38-keep1-Q4_0.gguf"):
            self.reject_change(0, ("model_path",), value)
        rows = copy.deepcopy(self.rows)
        rows[0]["model_path"] = '/models/данные "quoted"/qwen38-keep1-Q4_0.gguf'
        self.assertEqual(self.collect(rows)["model"], rows[0]["model_path"])

    def test_missing_extra_duplicate_replaced_and_reordered_rows(self):
        for i in range(8):
            with self.subTest(missing=i):
                rows = copy.deepcopy(self.rows)
                rows.pop(i)
                with self.assertRaises(ValueError):
                    self.collect(rows)
            rows = copy.deepcopy(self.rows)
            rows.insert(i, copy.deepcopy(rows[i]))
            with self.assertRaises(ValueError):
                self.collect(rows)
            if i:
                rows = copy.deepcopy(self.rows)
                rows[i] = copy.deepcopy(rows[i - 1])
                with self.assertRaises(ValueError):
                    self.collect(rows)
        for i in range(7):
            rows = copy.deepcopy(self.rows)
            rows[i], rows[i + 1] = rows[i + 1], rows[i]
            with self.assertRaises(ValueError):
                self.collect(rows)
        for end in range(8):
            with self.assertRaises(ValueError):
                self.collect(self.rows[:end])

    def test_malformed_nonobject_utf8_bom_blank_and_unfinished_lines(self):
        raw = encoded(self.rows)
        bads = [raw[:-1], raw[:-12], raw + b"\n", raw + b"failure\n", raw + b" ",
                b"\xef\xbb\xbf" + raw, raw.replace(b"memory_source", b"\xff", 1)]
        lines = raw.split(b"\n")[:-1]
        for line in (b"", b" ", b"[]", b"null", b"true", b"1", b'"passed"', b"failed", b"{", b"{} {}",
                     b'{"kind":1,}', b'{"x":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}"):
            bads.append(b"\n".join([*lines[:2], line, *lines[3:]]) + b"\n")
        for i, bad in enumerate(bads):
            with self.subTest(bad=i):
                self.raw.write_bytes(bad)
                with self.assertRaises(ValueError):
                    self.collect()

    def test_duplicate_keys_and_nonfinite_constants_at_nested_schema_levels(self):
        raw = encoded(self.rows)
        for field in ("revision", "context", "qsa_kv_bytes_per_gpu", "device", "weights",
                      "ram_expert_payload", "last_completed_ms", "snapshot_count", "passed"):
            key = ('"' + field + '":').encode()
            bad = raw.replace(key, key + b"0," + key, 1)
            self.assertNotEqual(bad, raw)
            self.raw.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                self.collect()
        for literal in (b"NaN", b"Infinity", b"-Infinity", b"1e999", b"-1e999"):
            for before in (b'"last_completed_ms": 12.5', b'"weights": 1100000000', b'"peak_rss_bytes": 8388608'):
                with self.subTest(literal=literal, field=before):
                    bad = raw.replace(before, before.split(b":")[0] + b": " + literal, 1)
                    self.assertNotEqual(bad, raw)
                    self.raw.write_bytes(bad)
                    with self.assertRaisesRegex(ValueError, "nonfinite"):
                        self.collect()

    def test_two_mib_exact_bound_and_growing_read_are_bounded(self):
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

    def test_every_snapshot_geometry_owner_categories_counts_and_bounds(self):
        for row in range(1, 7):
            for i in (0, 1):
                for key in ("device", "first_layer", "last_layer", "gdn_layers", "qsa_layers", "expert_slots",
                            "qsa_kv", "qsa_index", "gdn_state", "ple_state", "owned_bytes"):
                    self.reject_change(row, ("devices", i, key), self.rows[row]["devices"][i][key] + 1)
                for key in ("weights", "workspace", "owned_buffers"):
                    self.reject_change(row, ("devices", i, key), 0)
                self.reject_change(row, ("devices", i, "owned_peak_bytes"), self.rows[row]["devices"][i]["owned_bytes"] - 1)
                for value in (-1, (1 << 64), None, "123"):
                    self.reject_change(row, ("devices", i, "weights"), value)
                self.reject_change(row, ("devices", i, "free_vram"), 1073741823)
                self.reject_change(row, ("devices", i, "free_vram"), self.rows[row]["devices"][i]["total_vram"] + 1)
                device = self.rows[row]["devices"][i]
                self.reject_change(row, ("devices", i, "free_vram"), device["total_vram"] - device["owned_bytes"] + 1)
            self.reject_change(row, ("devices",), self.rows[row]["devices"][::-1])
            self.reject_change(row, ("devices",), self.rows[row]["devices"][:1])

    def test_source_and_footer_properties_arch_total_free_agree(self):
        for row, key in ((0, "before_load"), (7, "after_destruction")):
            for i in (0, 1):
                for field, value in (("device", 1 - i), ("arch", "gfx908"), ("wave_size", 32),
                                     ("compute_units", 0), ("async_engines", -1), ("name", ""),
                                     ("properties_total_vram", 17163091967), ("total_vram", 0),
                                     ("free_vram", 17163091969), ("free_vram", 1073741823)):
                    self.reject_change(row, (key, i, field), value)
            self.reject_change(row, (key,), self.rows[row][key][::-1])
            self.reject_change(row, (key,), self.rows[row][key][:1])
        for i in (0, 1):
            for field, value in (("arch", "gfx906"), ("name", "changed"), ("compute_units", 59), ("async_engines", 7)):
                self.reject_change(7, ("after_destruction", i, field), value)
            for row in range(1, 7):
                self.reject_change(row, ("devices", i, "total_vram"), 17163091967)
        rows = copy.deepcopy(self.rows)
        rows[7]["after_destruction"][0].update(total_vram=17163091967, properties_total_vram=17163091967)
        with self.assertRaises(ValueError):
            self.collect(rows)

    def test_host_payload_capacities_reads_and_no_invented_embedding_payload(self):
        for row in range(1, 7):
            host = self.rows[row]["host"]
            for key in ("ram_expert_payload", "pinned_handoff", "expert_payload_reads", "expert_payload_bytes_read"):
                self.reject_change(row, ("host", key), host[key] + 1)
            for key, bad in (("ram_expert_capacity", 68262297599), ("host_logit_capacity", 993279),
                             ("host_embedding_capacity", 0), ("host_embedding_payload", 521472000)):
                self.reject_change(row, ("host", key), bad)
        # A mutually changed logical read count or geometry is still rejected.
        rows = copy.deepcopy(self.rows)
        rows[0]["expected_geometry"]["expert_payload_reads"] = 145
        for row in rows[1:7]:
            row["host"]["expert_payload_reads"] = 145
        rows[7]["checks"]["expert_payload_reads"] = 145
        with self.assertRaises(ValueError):
            self.collect(rows)

    def test_each_of_five_steady_comparisons_covers_all_ledger_and_host_fields(self):
        for row in range(2, 7):
            for i in (0, 1):
                for key in ("weights", "workspace", "owned_bytes", "owned_peak_bytes", "owned_buffers"):
                    rows = copy.deepcopy(self.rows)
                    device = rows[row]["devices"][i]
                    device[key] += 1
                    if key in ("weights", "workspace"):
                        device["owned_bytes"] += 1
                        device["free_vram"] -= 1
                    with self.subTest(row=row, owner=i, key=key), self.assertRaises(ValueError):
                        self.collect(rows)
            for key, value in self.rows[row]["host"].items():
                self.reject_change(row, ("host", key), value + 1)

    def test_observed_weights_buffer_counts_capacities_rss_are_not_compiletime_constants(self):
        rows = copy.deepcopy(self.rows)
        for row in rows[1:7]:
            for device in row["devices"]:
                device["weights"] += 12345
                device["workspace"] += 54321
                device["owned_bytes"] += 66666
                device["owned_peak_bytes"] += 66666
                device["owned_buffers"] += 17
                device["free_vram"] -= 66666
            row["host"]["ram_expert_capacity"] += 65536
            row["host"]["host_logit_capacity"] += 4096
            row["host"]["host_embedding_capacity"] = 1
            row["peak_rss_bytes"] += 1024**3
        rows[7]["peak_rss_bytes"] += 1024**3
        for i, proof in enumerate(rows[7]["cleanup"]["devices"]):
            proof["last_snapshot_free_vram"] -= 66666
            proof["required_owned_bytes"] += 66666
            proof["free_recovered_bytes"] += 66666
            proof["min_snapshot_free_vram"] -= 66666
        result = self.collect(rows)
        self.assertEqual(result["snapshots"], rows[1:7])

    def test_rss_is_positive_integer_nondecreasing_not_full_ram_accounting(self):
        for row in range(8):
            for value in (0, -1, None, True, "123", 1.0, 1 << 64):
                self.reject_change(row, ("peak_rss_bytes",), value)
            if row:
                self.reject_change(row, ("peak_rss_bytes",), self.rows[row - 1]["peak_rss_bytes"] - 1)
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["peak_rss_bytes"] = 1024  # No unjustified RSS == payload/capacity sum requirement.
        self.assertIs(self.collect(rows)["passed"], True)

    def test_consumed_ids_zero_stats_and_snapshot_proof_counts(self):
        for row in range(1, 7):
            self.reject_change(row, ("consumed_tokens",), self.rows[row]["consumed_tokens"] + 1)
            self.reject_change(row, ("stats", "consumed_tokens"), self.rows[row]["consumed_tokens"] + 1)
            self.reject_change(row, ("token_id",), 101)
            self.reject_change(row, ("phase",), "step0" if row != 2 else "replay0")
            self.reject_change(row, ("checks", "snapshot_caller_device"), 1 - self.rows[row]["checks"]["snapshot_caller_device"])
            for field in ("finite_logit_values", "bitwise_replay_compared_values", "snapshot_logit_compared_values"):
                self.reject_change(row, ("checks", field), self.rows[row]["checks"][field] + 1)
        for row in (1, 4):
            for key in ("expert_hits", "expert_misses", "expert_upload_bytes", "last_completed_ms"):
                self.reject_change(row, ("stats", key), 1)

    def test_cumulative_routes_upload_miss_bounds_and_completed_duration(self):
        for row in (2, 3, 5, 6):
            for key in ("expert_hits", "expert_misses", "expert_upload_bytes"):
                for value in (-1, 1 << 64, None, "1"):
                    self.reject_change(row, ("stats", key), value)
            self.reject_change(row, ("stats", "expert_hits"), self.rows[row]["stats"]["expert_hits"] + 1)
            self.reject_change(row, ("stats", "expert_misses"), self.rows[row]["stats"]["expert_misses"] + 1)
            for bad in (-1, True, None, "0", 10**400):
                self.reject_change(row, ("stats", "last_completed_ms"), bad)
            rows = copy.deepcopy(self.rows)
            previous = rows[row - 1]["stats"]
            misses = rows[row]["stats"]["expert_misses"] - previous["expert_misses"]
            for bad in (previous["expert_upload_bytes"] + misses * 2764800 - 1,
                        previous["expert_upload_bytes"] + misses * 2867200 + 1):
                rows[row]["stats"]["expert_upload_bytes"] = bad
                with self.assertRaises(ValueError):
                    self.collect(rows)
        # Decreasing individual counters cannot hide behind the correct total.
        rows = copy.deepcopy(self.rows)
        rows[3]["stats"].update(expert_hits=481, expert_misses=479)
        with self.assertRaisesRegex(ValueError, "decreased"):
            self.collect(rows)
        # Zero is valid correctness timing, and replay cache statistics need not match baseline.
        rows = copy.deepcopy(self.rows)
        for row in rows[1:7]:
            row["stats"]["last_completed_ms"] = 0.0
        self.assertIs(self.collect(rows)["passed"], True)

    def test_footer_fixed_completion_counts_and_quantitative_cleanup_fields(self):
        for field, value in self.rows[7]["checks"].items():
            if type(value) is int:
                self.reject_change(7, ("checks", field), value + 1)
        for key in ("capacity", "expert_slots", "session_instances", "consumed_tokens",
                    "baseline_consumed_tokens", "replay_consumed_tokens", "protocol"):
            self.reject_change(7, (key,), self.rows[7][key] + 1)
        self.reject_change(7, ("scope",), "occupied128K")
        for i in (0, 1):
            for field, value in self.rows[7]["cleanup"]["devices"][i].items():
                if type(value) is int:
                    self.reject_change(7, ("cleanup", "devices", i, field), value + 1)
        self.reject_change(7, ("cleanup", "caller_device_before"), 0)
        self.reject_change(7, ("cleanup", "caller_device_after"), 0)
        self.reject_change(7, ("cleanup", "devices"), self.rows[7]["cleanup"]["devices"][::-1])
        self.reject_change(7, ("cleanup", "devices"), self.rows[7]["cleanup"]["devices"][:1])

    def test_cleanup_exact_owned_recovery_passes_and_one_byte_leak_fails_without_tolerance(self):
        for shortage in (0, 1, 4096):
            rows = copy.deepcopy(self.rows)
            for i in (0, 1):
                last = rows[6]["devices"][i]
                free = last["free_vram"] + last["owned_bytes"] - shortage
                rows[7]["after_destruction"][i]["free_vram"] = free
                rows[7]["cleanup"]["devices"][i].update(
                    after_destroy_free_vram=free, free_recovered_bytes=last["owned_bytes"] - shortage,
                    recovered_minus_owned_bytes=-shortage)
            if shortage:
                with self.assertRaisesRegex(ValueError, "recovery"):
                    self.collect(rows)
            else:
                self.assertIs(self.collect(rows)["passed"], True)
        for i in (0, 1):
            self.reject_change(7, ("after_destruction", i, "free_vram"), self.rows[6]["devices"][i]["free_vram"] - 1)

    def test_minimum_snapshot_free_is_not_assumed_to_be_last(self):
        rows = copy.deepcopy(self.rows)
        for i in (0, 1):
            rows[2]["devices"][i]["free_vram"] -= 123456
            rows[7]["cleanup"]["devices"][i]["min_snapshot_free_vram"] = rows[2]["devices"][i]["free_vram"]
        self.assertIs(self.collect(rows)["passed"], True)

    def test_one_gib_headroom_exact_boundary_passes(self):
        rows = copy.deepcopy(self.rows)
        for row in rows[1:7]:
            for device in row["devices"]:
                device["free_vram"] = 1073741824
        for i, proof in enumerate(rows[7]["cleanup"]["devices"]):
            last = rows[6]["devices"][i]
            recovered = rows[7]["after_destruction"][i]["free_vram"] - last["free_vram"]
            proof.update(last_snapshot_free_vram=1073741824, min_snapshot_free_vram=1073741824,
                         free_recovered_bytes=recovered, recovered_minus_owned_bytes=recovered - last["owned_bytes"])
        self.assertIs(self.collect(rows)["passed"], True)

    def test_cli_appends_one_record_preserves_existing_journal_and_input(self):
        original = self.raw.read_bytes()
        previous = b'{"kind":"previous"}\n'
        self.results.write_bytes(previous)
        process = self.cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        output = self.results.read_bytes()
        self.assertTrue(output.startswith(previous))
        self.assertEqual(len(output.splitlines()), 2)
        result = json.loads(output.splitlines()[1])
        self.assertEqual(result["kind"], "r3c_memory")
        self.assertEqual(result["source"], self.rows[0])
        self.assertEqual(result["snapshots"], self.rows[1:7])
        self.assertEqual(result["complete"], self.rows[7])
        self.assertEqual(result["raw_logs"], {"memory": str(self.raw.resolve())})
        self.assertEqual(self.raw.read_bytes(), original)

    def test_cli_failed_protocol_never_appends_or_creates_journal_or_mutates_raw(self):
        previous = b'{"kind":"previous"}\n'
        self.results.write_bytes(previous)
        failures = [encoded(self.rows[:-1]), encoded(self.rows)[:-1], b"not JSON\n",
                    encoded([{**self.rows[0], "build_provenance_complete": False}, *self.rows[1:]]),
                    encoded([*self.rows[:7], {**self.rows[7], "passed": False}])]
        leaked = copy.deepcopy(self.rows)
        leaked[7]["after_destruction"][1]["free_vram"] = leaked[6]["devices"][1]["free_vram"] + leaked[6]["devices"][1]["owned_bytes"] - 1
        failures.append(encoded(leaked))
        for i, raw in enumerate(failures):
            with self.subTest(failure=i):
                self.raw.write_bytes(raw)
                process = self.cli()
                self.assertEqual(process.returncode, 1, process.stderr)
                self.assertIn("record_memory:", process.stderr)
                self.assertEqual(self.results.read_bytes(), previous)
                absent = self.root / "absent-results.jsonl"
                self.assertEqual(self.cli(absent).returncode, 1)
                self.assertFalse(absent.exists())
                self.assertEqual(self.raw.read_bytes(), raw)

    def test_cli_raw_model_aliases_and_incomplete_nonregular_destination_are_immutable(self):
        model = self.root / "qwen38-keep1-Q4_0.gguf"
        model.write_bytes(b"synthetic input model, never read by collector")
        self.rows[0]["model_path"] = str(model)
        self.raw.write_bytes(encoded(self.rows))
        for source in (self.raw, model):
            original = source.read_bytes()
            symlink = self.root / (source.name + ".symlink")
            symlink.symlink_to(source)
            hardlink = self.root / (source.name + ".hardlink")
            os.link(source, hardlink)
            for destination in (source, symlink, hardlink):
                with self.subTest(destination=destination.name):
                    self.assertEqual(self.cli(destination).returncode, 1)
                    self.assertEqual(source.read_bytes(), original)
        self.results.write_bytes(b'{"kind":"incomplete"}')
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), b'{"kind":"incomplete"}')
        self.assertEqual(self.cli(self.root).returncode, 1)

    def test_cli_new_journal_and_relative_destination(self):
        process = self.cli(Path("results.jsonl"))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(self.results.read_bytes().splitlines()), 1)
        self.assertEqual(json.loads(self.results.read_bytes())["kind"], "r3c_memory")

    def test_serialization_failure_precedes_destination_open(self):
        record = self.collect()
        with mock.patch.object(MODULE.json, "dumps", side_effect=ValueError("serialization failure")):
            with self.assertRaises(ValueError):
                MODULE.append_result(self.results, record)
        self.assertFalse(self.results.exists())

    def test_short_write_and_io_failure_roll_back_to_original_journal(self):
        record = self.collect()
        previous = b'{"kind":"previous"}\n'
        real_open = Path.open
        for failure in ("short", "write", "flush"):
            with self.subTest(failure=failure):
                self.results.write_bytes(previous)

                def faulty_open(path, *args, **kwargs):
                    output = real_open(path, *args, **kwargs)
                    proxy = mock.MagicMock(wraps=output)
                    proxy.__enter__.return_value = proxy
                    proxy.__exit__.side_effect = lambda *_: output.close()

                    def write(payload):
                        if failure == "flush":
                            return output.write(payload)
                        output.write(payload[:17])
                        if failure == "write":
                            raise OSError("injected write failure")
                        return 17

                    proxy.write.side_effect = write
                    if failure == "flush":
                        proxy.flush.side_effect = OSError("injected flush failure")
                    return proxy

                with mock.patch.object(Path, "open", side_effect=faulty_open):
                    with self.assertRaises((ValueError, OSError)):
                        MODULE.append_result(self.results, record)
                self.assertEqual(self.results.read_bytes(), previous)

    def test_concurrent_cli_calls_append_whole_individual_records(self):
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
            result = json.loads(line)
            self.assertEqual(result["kind"], "r3c_memory")
            self.assertEqual(result["snapshots"], self.rows[1:7])


if __name__ == "__main__":
    unittest.main()
