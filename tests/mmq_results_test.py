"""Independent core-mmq v1 fixtures; stdlib only, no model or GPU execution."""

import copy
import datetime
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import record_mmq as MODULE

SCRIPT = ROOT / "tools/record_mmq.py"
REVISION = "0123456789abcdef0123456789abcdef01234567"
REPRESENTATIVE = (0, 1, 2, 3, 4, 17, 92, 93)


def counters():
    # Independently frozen from emit_expected(), not imported from the collector.
    return {
        "grid_cases": 936, "actual_cases": 39, "corner_cases": 3, "reuse_cases": 1,
        "quantizer_edges": 8, "matrix_cases": 979, "quantization_cases": 984,
        "packing_cases": 987, "poison_cases": 316, "poisoned_subblocks": 27540,
        "compared_elements": 2568652, "reference_slices": 9904,
        "quantizer_rejects": 30, "packer_rejects": 23, "mmq_rejects": 36,
        "host_rejects": 89, "manual_cases": 3, "int8_min_cases": 3,
        "raw_sum_difference_cases": 3, "signed_q8_scale_cases": 3, "half_product_pairs": 21,
    }


def fixture():
    """All 94 driver rows, real tensor geometry, HIP-float times and mixed A/B wins."""
    actual = [
        {"tensor_index": 0, "tensor": "blk.0.ffn_gate_exps.weight", "type": "Q4_0", "rank": 3,
         "dimensions": [2560, 640, 512], "strides_bytes": [18, 1440, 921600],
         "elements": 838860800, "tensor_bytes": 471859200, "file_offset": 4096,
         "read_api": "Model.read_expert", "expert": 0, "slice_offset": 0,
         "selected_bytes": 921600, "width": 2560, "rows": 640, "unchanged": True},
        {"tensor_index": 1, "tensor": "blk.0.ffn_down_exps.weight", "type": "Q4_1", "rank": 3,
         "dimensions": [640, 2560, 512], "strides_bytes": [20, 400, 1024000],
         "elements": 838860800, "tensor_bytes": 524288000, "file_offset": 471863296,
         "read_api": "Model.read_expert", "expert": 0, "slice_offset": 0,
         "selected_bytes": 1024000, "width": 640, "rows": 2560, "unchanged": True},
        {"tensor_index": 2, "tensor": "blk.0.hc_attn_down.weight", "type": "Q4_0", "rank": 2,
         "dimensions": [10240, 320], "strides_bytes": [18, 5760],
         "elements": 3276800, "tensor_bytes": 1843200, "file_offset": 996151296,
         "read_api": "Model.read_slice", "expert": None, "slice_offset": 0,
         "selected_bytes": 1843200, "width": 10240, "rows": 320, "unchanged": True},
    ]
    origin = {
        "kind": "mmq_source", "protocol": 1, "revision": REVISION, "dirty": 1,
        "model": '/models/данные "quoted"/qwen38-keep1-Q4_0.gguf',
        "model_variant": "qwen38-keep1-Q4_0", "architecture": "qwen4exp",
        "absolute_gate": .0002, "relative_gate": .00002,
        "scope": "component common-Q8 quantization/DS4 packing/canonical Q4 MMQ; no PP or inference speed claim",
        "numeric_contract": "finite static weight halves and valid finite Q8 halves/codes; numeric validity is caller-owned",
        "oracle": "unchanged matmul_q8_reference, ordered 1..3-column CPU slices, full-N finite comparison",
        "baseline": "diagnostic sliced current qualified launch_quantized_linear Columns<=3; benchmark only",
        "timing_scope": "20 individually completed resident HIP-event intervals per phase; all output/read-only/redzones validated each interval",
        "timing_excludes": ["allocation", "model_reads", "H2D", "quantization", "packing", "CPU_reference", "output_reset", "readback"],
        "ordering": "source; correctness device0,device1 BEFORE performance; measurements device,tensor,column,phase A1/B/A2; complete after all cleanup",
        "stream": "explicit hipStreamNonBlocking",
        "q8_1": {"bytes": 36, "qs_offset": 4, "sum": "original half raw-input sum, not d*sum(codes)"},
        "ds4": {"bytes": 144, "alignment": 16, "ds_offset": 0, "qs_offset": 16,
                "layout": "[K128][column]", "ds_order": "d0,s0,d1,s1,d2,s2,d3,s3",
                "missing_subblocks": "zero halves/codes; unused final subblocks alone poisoned before qualification MMQ"},
        "tiles": {"rows": 64, "k": 256, "block": [64, 4], "j": [8, 16, 32, 64],
                  "selection": "smallest J>=N capped64; N65..128 two J64 tiles"},
        "buffers": {"exact_payloads": True, "weight_pointer_offset": 2, "weight_alignment": 2,
                    "q8_alignment": 4, "packed_alignment": 16, "output_alignment": 4,
                    "redzones": "prefix/suffix bytes weights2/2, raw4/4, Q8_1 4/4, DS4 16/16, output4/4, flag4/4; half-qNaN poison"},
        "coverage": {
            "types": ["Q4_0", "Q4_1"], "cartesian": True,
            "widths": [32, 160, 640, 2560, 10240, 16384], "rows": [1, 7, 63, 64, 65, 67],
            "columns": [1, 2, 3, 4, 8, 9, 16, 17, 32, 33, 64, 65, 128],
            "quantizer_edges": {"width": 256, "columns": 1, "cases": [
                "zero", "negative_zero", "half_rounding", "minimum_half_scale", "signed_alternating",
                "small_sine", "unrounded_scale_codes", "large_cancellation"]},
            "manual_corners": [
                {"type": "Q4_0", "width": 160, "rows": 7, "columns": 3},
                {"type": "Q4_1", "width": 160, "rows": 7, "columns": 3},
                {"type": "Q4_1", "width": 32, "rows": 7, "columns": 3}],
            "reuse": {"type": "Q4_1", "width": 160, "rows": 7, "columns": 3},
            "reject_categories": ["null", "shape", "type", "alignment", "range_wrap", "write_alias"],
        },
        "benchmark_columns": [1, 3, 8, 32, 128], "benchmark_phases": ["A1", "B", "A2"], "repeats": 20,
        "expected_per_device": counters(),
        "expected_protocol": {"devices": 2, "source_records": 1, "correctness_records": 2,
                              "measurements_per_device": 45, "measurement_records": 90,
                              "validated_intervals": 1800, "complete_records": 1, "jsonl_records": 94},
        "actual": actual,
        "devices": [{"device": 0, "arch": "gfx906:sramecc+:xnack-", "wave": 64},
                    {"device": 1, "arch": "gfx906:sramecc+:xnack-", "wave": 64}],
    }
    rows = [origin]
    for device in (0, 1):
        rows.append({
            "kind": "mmq_correctness", "protocol": 1, "device": device, "counters": counters(),
            "max_abs": 1e-5, "max_bound_ratio": .04,
            "q8_bytes_exact": True, "packed_bytes_exact": True, "missing_subblocks_zero": True,
            "poison_masked": True, "active_subblocks_unchanged": True, "half_RNE_corners": True,
            "rejects_unchanged": True, "valid_reuse": True, "finite": True, "canary": True,
            "readonly": True, "cleanup": True, "passed": True,
        })
    for device in (0, 1):
        for tensor_index, tensor in enumerate(actual):
            for column_index, n in enumerate((1, 3, 8, 32, 128)):
                for phase_index, phase in enumerate(("A1", "B", "A2")):
                    sequence = len(rows) - 3
                    # HIP elapsed time is float, then promoted to a double and
                    # serialized at precision17. Include a B regression at N1.
                    base = .012 + n * .0003 + device * .001 + tensor_index * .003
                    if phase == "B":
                        base *= 1.15 if n == 1 else .65
                    elif phase == "A2":
                        base *= 1.03
                    times = [struct.unpack("f", struct.pack("f", base * (1 + (i % 5 - 2) * .003)))[0]
                             for i in range(20)]
                    tile = 8 if n <= 8 else 32 if n <= 32 else 64
                    rows.append({
                        "kind": "mmq_measurement", "protocol": 1, "sequence": sequence, "device": device,
                        "tensor_index": tensor_index, "column_index": column_index, "phase_index": phase_index,
                        "phase": phase, "path": "mmq" if phase == "B" else "diagnostic_sliced_linear",
                        "tensor": tensor["tensor"], "type": tensor["type"], "width": tensor["width"],
                        "rows": tensor["rows"], "columns": n, "tile_j": tile,
                        "column_tiles": 2 if n == 128 else 1,
                        "launches_per_repeat": 1 if phase == "B" else (n + 2) // 3,
                        "repeats": 20, "validated_intervals": 20, "elements_per_interval": n * tensor["rows"],
                        "completed_ms": times, "completed_ms_total": sum(times), "resident_ms": sum(times) / 20,
                        "max_abs": 1e-5, "max_bound_ratio": .04,
                        "warmup_max_abs": 8e-6, "warmup_max_bound_ratio": .03,
                        "finite": True, "canary": True, "readonly": True, "q8_bytes_exact": True,
                        "packed_bytes_exact": True, "passed": True,
                    })
    rows.append({"kind": "mmq_complete", "protocol": 1, "devices": 2, "correctness_records": 2,
                 "measurement_records": 90, "validated_intervals": 1800, "jsonl_records": 94,
                 "cleanup": True, "passed": True})
    # Literal schema ordering follows the emitter. Derive the fixture's lists
    # from independently spelled rows, never from collector SCHEMAS.
    origin["schemas"] = {r["kind"]: list(r) for r in (origin, rows[1], rows[3], rows[-1])}
    origin["schemas"]["mmq_source"].append("schemas")
    return rows


def encoded(rows):
    return "".join(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n" for row in rows).encode()


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


class MmqResultsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mmq-results-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.raw = self.root / "mmq.jsonl"
        self.results = self.root / "results.jsonl"
        self.rows = fixture()
        self.raw.write_bytes(encoded(self.rows))

    def collect(self, rows=None):
        if rows is not None:
            self.raw.write_bytes(encoded(rows))
        return MODULE.collect(self.raw)

    def reject(self, rows):
        raw = encoded(rows)
        self.raw.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.collect()
        self.assertEqual(self.raw.read_bytes(), raw)

    def reject_change(self, row, path, value):
        rows = self.rows.copy()
        rows[row] = copy.deepcopy(rows[row])
        target(rows[row], path[:-1])[path[-1]] = value
        with self.subTest(row=row, path=path, value=value):
            self.reject(rows)

    def cli(self, destination=None, extra=()):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", self.raw.name,
                               "--results", str(destination if destination is not None else self.results), *extra],
                              cwd=self.root, capture_output=True, text=True, timeout=30, check=False)

    def test_complete_fixture_preserves_all_94_rows_and_absolute_raw_path(self):
        before = self.raw.read_bytes()
        result = self.collect()
        self.assertEqual(len(self.rows), 94)
        self.assertEqual(result["kind"], "r4b_mmq_primitives")
        self.assertEqual(result["source"], self.rows[0])
        self.assertEqual(result["correctness"], self.rows[1:3])
        self.assertEqual(result["measurements"], self.rows[3:93])
        self.assertEqual(result["complete"], self.rows[-1])
        self.assertEqual(result["raw_logs"], {"mmq": str(self.raw.resolve())})
        self.assertEqual(result["revision"], REVISION)
        self.assertEqual(result["dirty"], 1)
        self.assertIs(type(result["dirty"]), int)
        self.assertEqual(result["model"], self.rows[0]["model"])
        self.assertIs(result["passed"], True)
        self.assertEqual(datetime.datetime.fromisoformat(result["timestamp"]).utcoffset(), datetime.timedelta(0))
        self.assertEqual(sum(r["validated_intervals"] for r in result["measurements"]), 1800)
        self.assertEqual(self.raw.read_bytes(), before)
        json.dumps(result, allow_nan=False)

    def test_frozen_driver_schema_and_static_counter_emitter_match_fixture(self):
        # Catch fixture/collector agreeing on a typo against the real protocol.
        driver = (ROOT / "src/mmq_main.hip").read_text(encoding="utf-8")
        start = driver.index('],\\"schemas\\":')
        fragment = driver[driver.rfind("\n", 0, start):driver.index("void correctness(")]
        literals = re.findall(r'"(?:\\.|[^"\\])*"', fragment)
        decoded = "".join(json.loads(literal) for literal in literals)
        schemas = json.loads(decoded[decoded.index('"schemas":') + len('"schemas":'):].rstrip("\n")[:-1])
        self.assertEqual(schemas, self.rows[0]["schemas"])
        emitter = driver[driver.index("void emit_expected()"):driver.index("void source(")]
        decoded = "".join(json.loads(literal) for literal in re.findall(r'"(?:\\.|[^"\\])*"', emitter))
        self.assertEqual(json.loads(decoded), counters())

    def test_component_scope_and_timing_exclusions_make_no_speedup_claim(self):
        result = self.collect()
        for key in ("performance_claim", "universal_speedup_claim", "pp_qualified", "large_prompt_qualified",
                    "end_to_end_qualified", "independent_hf_claim", "mtp"):
            self.assertIs(result[key], False)
        for phrase in ("component A/B/A", "ONLY a fixture baseline", "not production PP", "large-prompt",
                       "end-to-end", "independent HF", "MTP", "performance-win", "no universal speedup"):
            self.assertIn(phrase, result["scope"])
        self.assertEqual(result["timing_scope"], self.rows[0]["timing_scope"])
        self.assertEqual(result["timing_excludes"], self.rows[0]["timing_excludes"])
        self.assertEqual(result["fixture_gates"], {
            "absolute": .0002, "relative_cpu_magnitude": .00002,
            "bound": "absolute+relative_cpu_magnitude*abs(CPU_reference)",
            "max_bound_ratio_limit": 1, "frozen_before_execution": True})
        # Both a B regression and a B win are accepted: no speedup success gate.
        self.assertGreater(result["measurements"][1]["resident_ms"], result["measurements"][0]["resident_ms"])
        self.assertLess(result["measurements"][4]["resident_ms"], result["measurements"][3]["resident_ms"])
        self.assertNotIn("donors", result)
        self.assertNotIn("hf_revision", result)

    def test_all_missing_extra_duplicate_replaced_and_truncated_rows(self):
        for index in range(94):
            for operation in ("missing", "extra", "replacement"):
                rows = self.rows.copy()
                if operation == "missing":
                    rows.pop(index)
                elif operation == "extra":
                    rows.insert(index, rows[index])
                else:
                    rows[index] = rows[(index + 1) % 94]
                with self.subTest(index=index, operation=operation):
                    self.reject(rows)
        for end in range(94):
            self.reject(self.rows[:end])

    def test_every_adjacent_swap_and_performance_before_second_correctness(self):
        for i in range(93):
            rows = self.rows.copy()
            rows[i], rows[i + 1] = rows[i + 1], rows[i]
            with self.subTest(index=i):
                self.reject(rows)
        rows = self.rows.copy()
        rows.insert(48, rows.pop(2))
        self.reject(rows)
        # Spoofing sequence alone cannot hide a reordered A/B or tensor order.
        rows = copy.deepcopy(self.rows)
        rows[3], rows[4] = rows[4], rows[3]
        rows[3]["sequence"], rows[4]["sequence"] = 0, 1
        self.reject(rows)

    def test_missing_extra_fields_at_all_nested_schema_levels(self):
        for index in REPRESENTATIVE:
            for path, obj in objects(self.rows[index]):
                for field in (*obj, "unexpected"):
                    rows = self.rows.copy()
                    rows[index] = copy.deepcopy(rows[index])
                    changed = target(rows[index], path)
                    if field == "unexpected":
                        changed[field] = True
                    else:
                        del changed[field]
                    with self.subTest(row=index, path=path, field=field):
                        self.reject(rows)

    def test_schema_declarations_are_frozen_including_field_order(self):
        for kind, fields in self.rows[0]["schemas"].items():
            for bad in (fields[:-1], fields + ["unexpected"], fields[::-1], fields + fields[:1], None):
                self.reject_change(0, ("schemas", kind), bad)
        rows = copy.deepcopy(self.rows)
        # Unknown fields remain invalid even if a self-declared schema adds them.
        rows[0]["schemas"]["mmq_measurement"].append("speedup")
        for row in rows[3:93]:
            row["speedup"] = 2
        self.reject(rows)

    def test_boolean_numeric_substitutions_and_integer_float_types(self):
        for index in REPRESENTATIVE:
            for path, value in leaves(self.rows[index]):
                if type(value) in (int, float):
                    self.reject_change(index, path, True)
                if type(value) is int:
                    self.reject_change(index, path, float(value))
        raw = encoded(self.rows)
        for key in (b'"device":0', b'"sequence":0', b'"slice_offset":0'):
            self.raw.write_bytes(raw.replace(key, key[:-1] + b"-0", 1))
            with self.assertRaises(ValueError):
                self.collect()

    def test_all_proof_flags_require_true_json_booleans(self):
        for index in REPRESENTATIVE:
            for path, value in leaves(self.rows[index]):
                if type(value) is bool:
                    for bad in (False, 1, None, "true"):
                        self.reject_change(index, path, bad)
        # Every timed repetition row is checked, not only one representative.
        for index in range(3, 93):
            self.reject_change(index, ("canary",), False)

    def test_malformed_nonobject_utf8_bom_deep_or_unfinished_input(self):
        raw = encoded(self.rows)
        bads = [raw[:-1], raw[:-20], raw + b"\n", raw + b"failure\n", raw + b" ",
                b"\xef\xbb\xbf" + raw, raw.replace(b"mmq_source", b"\xff", 1)]
        lines = raw.split(b"\n")[:-1]
        for line in (b"", b" ", b"null", b"[]", b"true", b"1", b'"passed"', b"{", b"{} {}",
                     b'{"kind":1,}', b'{"x":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}"):
            bads.append(b"\n".join([*lines[:3], line, *lines[4:]]) + b"\n")
        for bad in bads:
            self.raw.write_bytes(bad)
            with self.assertRaises(ValueError):
                self.collect()
            self.assertEqual(self.raw.read_bytes(), bad)

    def test_duplicate_keys_at_all_protocol_levels(self):
        raw = encoded(self.rows)
        for field in ("kind", "revision", "ds_order", "exact_payloads", "cases", "grid_cases",
                      "mmq_source", "dimensions", "arch", "max_abs", "completed_ms", "passed"):
            key = ('"' + field + '":').encode()
            bad = raw.replace(key, key + b"0," + key, 1)
            self.assertNotEqual(raw, bad)
            self.raw.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                self.collect()

    def test_nonfinite_constants_exponents_and_huge_numeric_values(self):
        raw = encoded(self.rows)
        for literal in (b"NaN", b"Infinity", b"-Infinity", b"1e999", b"-1e999"):
            for token in (b'"absolute_gate":0.0002', b'"max_abs":1e-05', b'"dirty":1'):
                self.raw.write_bytes(raw.replace(token, token.split(b":")[0] + b":" + literal, 1))
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.collect()
        for field in ("resident_ms", "completed_ms_total", "max_abs", "max_bound_ratio"):
            self.reject_change(3, (field,), 10**400)

    def test_bounded_input_including_growing_read_and_nonregular_input(self):
        raw = encoded(self.rows)
        limit = 2 * 1024 * 1024
        self.assertEqual(MODULE.MAX_RAW_BYTES, limit)
        self.raw.write_bytes(b" " * (limit - len(raw)) + raw)
        self.assertTrue(self.collect()["passed"])
        self.raw.write_bytes(b" " * (limit + 1 - len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, "oversized"):
            self.collect()
        self.raw.write_bytes(raw)
        with mock.patch.object(Path, "open", return_value=mock.MagicMock()) as opened:
            opened.return_value.__enter__.return_value.read.return_value = b"x" * (limit + 1)
            with self.assertRaisesRegex(ValueError, "oversized"):
                self.collect()
            opened.return_value.__enter__.return_value.read.assert_called_once_with(limit + 1)
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            MODULE.collect(self.root)

    def test_crlf_and_actual_clean_dirty_revision_model_provenance(self):
        for dirty in (0, 1):
            rows = copy.deepcopy(self.rows)
            rows[0].update(revision="f" * 40, dirty=dirty, model='relative/модель "quoted".gguf')
            self.raw.write_bytes(encoded(rows).replace(b"\n", b"\r\n"))
            result = self.collect()
            self.assertEqual(result["revision"], "f" * 40)
            self.assertEqual(result["dirty"], dirty)
            self.assertIs(type(result["dirty"]), int)
            self.assertEqual(result["model"], rows[0]["model"])
            self.assertEqual(result["source"], rows[0])

    def test_invalid_revision_dirty_model_and_device_provenance(self):
        for bad in (None, True, 1, "", "unknown", "a" * 39, "a" * 41, "g" * 40, "A" * 40):
            self.reject_change(0, ("revision",), bad)
        for bad in (True, False, None, -1, 2, 1.0, "1"):
            self.reject_change(0, ("dirty",), bad)
        for bad in (None, True, 0, "", "path\n", "\ud800", "path\x00", "x" * 4097):
            self.reject_change(0, ("model",), bad)
        for i in (0, 1):
            for bad in ("gfx908", "gfx9060", "gfx906\n", "", None, True):
                self.reject_change(0, ("devices", i, "arch"), bad)
            self.reject_change(0, ("devices", i, "wave"), 32)
            self.reject_change(0, ("devices", i, "device"), 1 - i)
        rows = copy.deepcopy(self.rows)
        rows[0]["devices"][0]["arch"] = "gfx906"
        self.assertTrue(self.collect(rows)["passed"])

    def test_source_contracts_cpu_reference_baseline_and_timing_are_frozen(self):
        for field in ("model_variant", "architecture", "scope", "numeric_contract", "oracle", "baseline",
                      "timing_scope", "ordering", "stream"):
            self.reject_change(0, (field,), "changed contract")
        for field in ("absolute_gate", "relative_gate"):
            for bad in (0, -1, True, None, .001, "0.0002", self.rows[0][field] * 1.00001):
                self.reject_change(0, (field,), bad)
        exclusions = self.rows[0]["timing_excludes"]
        for bad in (exclusions[:-1], exclusions[::-1], exclusions + ["warmup"], None):
            self.reject_change(0, ("timing_excludes",), bad)
        self.reject_change(0, ("benchmark_columns",), [1, 3, 8, 32, 64])
        self.reject_change(0, ("benchmark_phases",), ["A1", "A2", "B"])
        self.reject_change(0, ("repeats",), 19)

    def test_q8_ds4_bit_packing_raw_sum_zero_tail_and_tile_contracts(self):
        changes = [
            (("q8_1", "bytes"), 40), (("q8_1", "qs_offset"), 8),
            (("q8_1", "sum"), "d*sum(codes)"), (("ds4", "bytes"), 4 * 36 + 16),
            (("ds4", "alignment"), 4), (("ds4", "ds_offset"), 128), (("ds4", "qs_offset"), 0),
            (("ds4", "layout"), "[column][K128]"), (("ds4", "ds_order"), "d0,d1,d2,d3,s0,s1,s2,s3"),
            (("ds4", "missing_subblocks"), "read padding from adjacent slot"),
            (("tiles", "k"), 128), (("tiles", "rows"), 32), (("tiles", "block"), [32, 8]),
            (("tiles", "j"), [8, 16, 32, 128]), (("tiles", "selection"), "always J64"),
        ]
        for path, value in changes:
            self.reject_change(0, path, value)
        for field in self.rows[0]["buffers"]:
            value = self.rows[0]["buffers"][field]
            self.reject_change(0, ("buffers", field), value + 1 if type(value) is int else "unchecked")
        for index in (1, 2):
            for flag in ("q8_bytes_exact", "packed_bytes_exact", "missing_subblocks_zero", "poison_masked",
                         "active_subblocks_unchanged", "half_RNE_corners", "rejects_unchanged", "valid_reuse",
                         "finite", "canary", "readonly", "cleanup"):
                self.reject_change(index, (flag,), False)

    def test_exact_cartesian_tails_manual_corners_and_quantizer_edge_coverage(self):
        coverage = self.rows[0]["coverage"]
        for field in ("types", "widths", "rows", "columns", "manual_corners", "reject_categories"):
            self.reject_change(0, ("coverage", field), coverage[field][:-1])
            self.reject_change(0, ("coverage", field), coverage[field][::-1])
        self.reject_change(0, ("coverage", "cartesian"), False)
        self.reject_change(0, ("coverage", "quantizer_edges", "cases"), coverage["quantizer_edges"]["cases"][:-1])
        self.reject_change(0, ("coverage", "quantizer_edges", "width"), 128)
        for i in range(3):
            self.reject_change(0, ("coverage", "manual_corners", i, "width"), 128)
        self.reject_change(0, ("coverage", "reuse", "columns"), 1)

    def test_actual_unchanged_geometry_types_payloads_and_read_bindings(self):
        for i, tensor in enumerate(self.rows[0]["actual"]):
            for field, value in tensor.items():
                if field == "file_offset":
                    continue
                if type(value) is int:
                    bad = value + 1
                elif type(value) is list:
                    bad = value[:-1]
                elif type(value) is bool:
                    bad = False
                elif value is None:
                    bad = 0
                else:
                    bad = "wrong"
                self.reject_change(0, ("actual", i, field), bad)
            for field in ("dimensions", "strides_bytes"):
                for axis, value in enumerate(tensor[field]):
                    self.reject_change(0, ("actual", i, field, axis), value + 1)
            self.reject_change(0, ("actual", i, "type"), 2 if i != 1 else 3)
        actual = self.rows[0]["actual"]
        for bad in (None, actual[:-1], actual[::-1], actual + actual[:1]):
            self.reject_change(0, ("actual",), bad)
        rows = copy.deepcopy(self.rows)
        rows[0]["actual"][1]["dimensions"][-1] = 1
        rows[0]["actual"][1]["tensor_bytes"] = 1024000
        rows[0]["actual"][1]["elements"] = 1638400
        self.reject(rows)

    def test_uint64_offsets_nonoverlap_not_hardcoded_and_no_invented_alignment(self):
        for i in range(3):
            for bad in (-1, 0, 1.0, True, 2**64, None):
                self.reject_change(0, ("actual", i, "file_offset"), bad)
            self.reject_change(0, ("actual", i, "file_offset"), 2**64 - self.rows[0]["actual"][i]["tensor_bytes"])
        self.reject_change(0, ("actual", 1, "file_offset"), 471863295)
        self.reject_change(0, ("actual", 2, "file_offset"), 4096)
        rows = copy.deepcopy(self.rows)
        for tensor in rows[0]["actual"]:
            tensor["file_offset"] += 1000000000
        self.assertTrue(self.collect(rows)["passed"])
        # GGUF alignment is dynamic and absent in this protocol. A valid GGUF
        # can have alignment1; the collector must not guess alignment32.
        for tensor in rows[0]["actual"]:
            tensor["file_offset"] += 1
        self.assertTrue(self.collect(rows)["passed"])

    def test_frozen_counters_every_device_and_mutually_edited_claims(self):
        for field, value in counters().items():
            self.reject_change(0, ("expected_per_device", field), value - 1)
            for index in (1, 2):
                self.reject_change(index, ("counters", field), value - 1)
        for field, value in self.rows[0]["expected_protocol"].items():
            self.reject_change(0, ("expected_protocol", field), value - 1)
        rows = copy.deepcopy(self.rows)
        rows[0]["expected_per_device"]["half_product_pairs"] = 0
        for index in (1, 2):
            rows[index]["counters"]["half_product_pairs"] = 0
        self.reject(rows)

    def test_all_measurement_coordinates_geometry_and_per_interval_coverage(self):
        for index in range(3, 93):
            row = self.rows[index]
            for field in ("sequence", "device", "tensor_index", "column_index", "phase_index",
                          "width", "rows", "columns", "tile_j", "column_tiles", "launches_per_repeat",
                          "repeats", "validated_intervals", "elements_per_interval"):
                self.reject_change(index, (field,), row[field] + 1)
        for index in (3, 4, 5, 15, 16, 47, 92):
            for field in ("phase", "path", "tensor", "type"):
                self.reject_change(index, (field,), "wrong")
        self.reject_change(4, ("path",), "production_pp")
        self.reject_change(3, ("path",), "mmq")
        self.reject_change(3, ("elements_per_interval",), 640 * 20)
        self.reject_change(16, ("column_tiles",), 1)

    def test_frozen_numeric_error_gates_including_warmup_and_all_rows(self):
        for index in range(1, 93):
            fields = ("max_bound_ratio", "warmup_max_bound_ratio") if index >= 3 else ("max_bound_ratio",)
            for field in fields:
                self.reject_change(index, (field,), 1.0000000000000002)
        for index in (1, 2, 3, 4, 92):
            fields = ("max_abs", "max_bound_ratio", "warmup_max_abs", "warmup_max_bound_ratio") if index >= 3 else ("max_abs", "max_bound_ratio")
            for field in fields:
                for bad in (-1, True, None, "0"):
                    self.reject_change(index, (field,), bad)
        for ratio in (0, 1):
            rows = copy.deepcopy(self.rows)
            for row in rows[1:93]:
                row.update(max_abs=0, max_bound_ratio=ratio)
                if "warmup_max_abs" in row:
                    row.update(warmup_max_abs=0, warmup_max_bound_ratio=ratio)
            self.assertTrue(self.collect(rows)["passed"])
        # Relative reference term permits max_abs>absolute_gate; no false cap.
        rows[4].update(max_abs=.001, max_bound_ratio=1)
        self.assertTrue(self.collect(rows)["passed"])

    def test_every_interval_must_be_positive_finite_and_array_complete(self):
        for i in range(20):
            for bad in (0, -0.0, -1, True, None, "0.01"):
                self.reject_change(4, ("completed_ms", i), bad)
        times = self.rows[4]["completed_ms"]
        for bad in (None, {}, times[:-1], times + times[:1]):
            self.reject_change(4, ("completed_ms",), bad)
        for field in ("completed_ms_total", "resident_ms"):
            for bad in (0, -0.0, -1, True, None, "0.01"):
                self.reject_change(4, (field,), bad)
        raw = encoded(self.rows)
        token = json.dumps(times[0]).encode()
        for literal in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            self.raw.write_bytes(raw.replace(b'"completed_ms":[' + token, b'"completed_ms":[' + literal, 1))
            with self.assertRaises(ValueError):
                self.collect()

    def test_mean_total_and_interval_array_consistency_every_measurement(self):
        for index in range(3, 93):
            for field in ("resident_ms", "completed_ms_total"):
                self.reject_change(index, (field,), self.rows[index][field] * 1.00000001)
        self.reject_change(4, ("completed_ms", 19), self.rows[4]["completed_ms"][19] * 1.01)
        rows = copy.deepcopy(self.rows)
        rows[4]["resident_ms"] *= 2
        rows[4]["completed_ms_total"] *= 2
        self.reject(rows)
        rows = copy.deepcopy(self.rows)
        rows[4]["completed_ms"] = [1e308] * 20
        self.reject(rows)

    def test_rounding_aware_sum_and_mean_tolerance_is_tight_and_scale_aware(self):
        for scale in (1, 1e-20, 1e20):
            rows = copy.deepcopy(self.rows)
            # Positive finite representable HIP float times, with a summation
            # order that loses small terms when a large interval comes first.
            times = [struct.unpack("f", struct.pack("f", scale))[0]]
            times += [struct.unpack("f", struct.pack("f", scale * 1e-16))[0]] * 19
            # Recent Python sum() uses compensation. Explicitly reproduce the
            # driver's sequential double additions to exercise its rounding.
            total = 0.0
            for value in times:
                total += value
            self.assertNotEqual(total, math.fsum(times))
            rows[4].update(completed_ms=times, completed_ms_total=total, resident_ms=total / 20)
            self.assertTrue(self.collect(rows)["passed"])
            for field in ("completed_ms_total", "resident_ms"):
                changed = copy.deepcopy(rows)
                changed[4][field] = math.nextafter(changed[4][field], math.inf)
                self.assertTrue(self.collect(changed)["passed"])
                changed[4][field] *= 1.000000000001
                self.reject(changed)
        rows = copy.deepcopy(self.rows)
        rows[4].update(completed_ms=[1] * 20, completed_ms_total=20, resident_ms=1)
        self.assertTrue(self.collect(rows)["passed"])

    def test_complete_footer_requires_every_frozen_count_and_cleanup(self):
        for field, value in self.rows[-1].items():
            bad = value + 1 if type(value) is int else False if type(value) is bool else "wrong"
            self.reject_change(93, (field,), bad)
        rows = self.rows.copy()
        rows[-1] = {"kind": "mmq_complete", "passed": True}
        self.reject(rows)

    def test_cli_appends_one_record_to_explicit_journal_and_preserves_inputs(self):
        original = self.raw.read_bytes()
        old = b'{"kind":"existing"}\n'
        self.results.write_bytes(old)
        process = self.cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        saved = self.results.read_bytes()
        self.assertTrue(saved.startswith(old))
        lines = saved.splitlines()
        self.assertEqual(len(lines), 2)
        result = json.loads(lines[1])
        self.assertEqual(result["kind"], "r4b_mmq_primitives")
        self.assertEqual(result["source"], self.rows[0])
        self.assertEqual(result["correctness"], self.rows[1:3])
        self.assertEqual(result["measurements"], self.rows[3:93])
        self.assertEqual(result["complete"], self.rows[-1])
        self.assertEqual(result["raw_logs"], {"mmq": str(self.raw.resolve())})
        self.assertEqual(self.raw.read_bytes(), original)
        self.assertFalse((self.root / "src/results.jsonl").exists())

    def test_cli_invalid_logs_do_not_modify_existing_or_create_new_journal(self):
        existing = b'{"kind":"existing"}\n'
        cases = [(0, ("revision",), "unknown"), (0, ("ds4", "bytes"), 40),
                 (1, ("half_RNE_corners",), False), (2, ("poison_masked",), False),
                 (4, ("max_bound_ratio",), 1.1), (4, ("warmup_max_bound_ratio",), 1.1),
                 (92, ("resident_ms",), 1), (93, ("cleanup",), False)]
        for index, path, value in cases:
            rows = copy.deepcopy(self.rows)
            target(rows[index], path[:-1])[path[-1]] = value
            self.raw.write_bytes(encoded(rows))
            before = self.raw.read_bytes()
            self.results.write_bytes(existing)
            process = self.cli()
            self.assertEqual(process.returncode, 1)
            self.assertIn("record_mmq:", process.stderr)
            self.assertEqual(self.results.read_bytes(), existing)
            self.assertEqual(self.raw.read_bytes(), before)
            absent = self.root / "new-results.jsonl"
            self.assertEqual(self.cli(absent).returncode, 1)
            self.assertFalse(absent.exists())
        self.raw.write_bytes(encoded(self.rows[:-1]))
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), existing)

    def test_cli_cannot_append_to_raw_model_or_alias_artifacts(self):
        original = self.raw.read_bytes()
        self.assertEqual(self.cli(self.raw).returncode, 1)
        self.assertEqual(self.raw.read_bytes(), original)
        for kind in ("symlink", "hardlink"):
            alias = self.root / kind
            if kind == "symlink":
                alias.symlink_to(self.raw)
            else:
                os.link(self.raw, alias)
            self.assertEqual(self.cli(alias).returncode, 1)
            self.assertEqual(self.raw.read_bytes(), original)
        model = self.root / "model.gguf"
        model.write_bytes(b"model payload\n")
        rows = copy.deepcopy(self.rows)
        rows[0]["model"] = str(model)
        self.raw.write_bytes(encoded(rows))
        self.assertEqual(self.cli(model).returncode, 1)
        self.assertEqual(model.read_bytes(), b"model payload\n")

    def test_cli_rejects_incomplete_or_nonregular_journal_and_requires_explicit_results(self):
        incomplete = b'{"kind":"existing"}'
        self.results.write_bytes(incomplete)
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), incomplete)
        self.assertEqual(self.cli(self.root).returncode, 1)
        process = subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", str(self.raw)],
                                 cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(self.results.read_bytes(), incomplete)

    def test_main_uses_existing_append_result_only_after_validation(self):
        with mock.patch.object(MODULE, "append_result") as append:
            self.assertEqual(MODULE.main(["--raw", str(self.raw), "--results", str(self.results)]), 0)
            append.assert_called_once()
            destination, result = append.call_args.args
            self.assertEqual(destination, self.results)
            self.assertEqual(result["measurements"], self.rows[3:93])
        self.raw.write_bytes(encoded(self.rows[:-1]))
        with mock.patch.object(MODULE, "append_result") as append, mock.patch.object(sys, "stderr"):
            self.assertEqual(MODULE.main(["--raw", str(self.raw), "--results", str(self.results)]), 1)
            append.assert_not_called()


if __name__ == "__main__":
    unittest.main()
