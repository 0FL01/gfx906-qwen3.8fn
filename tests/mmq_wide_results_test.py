"""Independent frozen wide-MMQ fixture/recorder guards; stdlib, no GPU/model."""

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
from tools import record_mmq_wide as MODULE
from tools.record_memory import append_result

SCRIPT = ROOT / "tools/record_mmq_wide.py"
REVISION = "0123456789abcdef0123456789abcdef01234567"
PROTOCOL = "qwen.mmq-wide.v1"
HEAD_ROWS = [0, 1, 7, 31, 63, 127, 255, 511, 1023, 4095, 8191, 16383,
             32767, 65535, 131071, 200003, 248043, 248044, 248045, 248046,
             248047, 248127, 248318, 248319]
REPRESENTATIVE = (0, 1, 2, 3, 4, 33, 47, 92, 93)


def counters():
    # Independently frozen from expected_counters(), not collector constants.
    return {
        "grid_cases": 588, "q4_forward_cases": 48, "limit_cases": 5,
        "manual_cases": 3, "reuse_cases": 5, "actual_cases": 15,
        "matrices": 664, "quantization": 669, "packing": 672,
        "poison_cases": 202, "poison_subblocks": 21018,
        "finite_elements": 45769172, "reference_elements": 3062260,
        "reference_slices": 9497, "readonly_validations": 664,
        "quantizer_rejects": 30, "packer_rejects": 23, "mmq_rejects": 153,
        "int8_min_cases": 3, "int8_max_cases": 3, "raw_sum_difference_cases": 3,
        "signed_scale_cases": 3, "subnormal_scale_cases": 3, "extreme_scale_cases": 3,
        "q6_subscale_cases": 1, "q6_activation_scale_cases": 1, "q6_integer_grouping_cases": 1,
    }


def fixture():
    """All 94 emitted rows, actual canonical geometry and promoted HIP-float times."""
    actual = [
        {"tensor_index": 0, "tensor": "blk.6.ffn_down_shexp.weight", "type": "Q5_0", "rank": 2,
         "dimensions": [640, 2560], "strides_bytes": [22, 440], "elements": 1638400,
         "bytes": 1126400, "file_offset": 4096, "read_api": "Model.read_tensor", "payload_reads": 1,
         "unchanged": True, "reference_coverage": "all_rows", "reference_rows": 2560,
         "sample_rows": [], "finite_rows": 2560, "finite_halfscales": 51200, "all_halfscales_finite": True},
        {"tensor_index": 1, "tensor": "blk.0.ffn_down_shexp.weight", "type": "Q8_0", "rank": 2,
         "dimensions": [640, 2560], "strides_bytes": [34, 680], "elements": 1638400,
         "bytes": 1740800, "file_offset": 1130496, "read_api": "Model.read_tensor", "payload_reads": 1,
         "unchanged": True, "reference_coverage": "all_rows", "reference_rows": 2560,
         "sample_rows": [], "finite_rows": 2560, "finite_halfscales": 51200, "all_halfscales_finite": True},
        {"tensor_index": 2, "tensor": "output.weight", "type": "Q6_K", "rank": 2,
         "dimensions": [2560, 248320], "strides_bytes": [210, 2100], "elements": 635699200,
         "bytes": 521472000, "file_offset": 2871296, "read_api": "Model.read_tensor", "payload_reads": 1,
         "unchanged": True, "reference_coverage": "sampled_rows", "reference_rows": 24,
         "sample_rows": HEAD_ROWS.copy(), "finite_rows": 248320, "finite_halfscales": 2483200,
         "all_halfscales_finite": True},
    ]
    origin = {
        "kind": "mmq_wide_source", "protocol": PROTOCOL, "revision": REVISION, "dirty": 1,
        "model": '/models/данные "quoted"/qwen38-keep1-Q4_0.gguf',
        "model_variant": "qwen38-keep1-Q4_0", "architecture": "qwen4exp",
        "donor_revision": "dcd685463d597d31f5ca759d32c94592a2740fa4", "donor_license": "MIT",
        "absolute_gate": .0002, "relative_gate": .00002,
        "scope": "component qualification; no PP-production/full-model/inference speed claim",
        "oracle": "unchanged matmul_q8_reference in ordered <=3-column slices; full synthetic/shared-expert outputs, 24 original head rows ONLY",
        "baseline": "diagnostic_sliced_linear",
        "timing": {
            "clock": "individually completed HIP events", "repeats": 20, "phases": ["A1", "B", "A2"],
            "columns": [1, 3, 8, 32, 128],
            "excludes": ["allocation", "Model", "H2D", "quantization", "DS4_pack", "CPU_reference", "output_reset", "D2H"],
            "validation": "all GPU outputs finite/redzones; all applicable CPU references and complete readonly payloads EVERY warmup/repeat",
            "element_counters": "sum over 20 validated intervals; excludes warmup",
        },
        "ordering": "source; correctness0/1 after device cleanup BEFORE any timing; device,tensor,column,A1/B/A2 metrics after device cleanup; complete after Model/host/device cleanup",
        "abi": {
            "q8_bytes": 36, "q8_qs_offset": 4, "ds4_bytes": 144, "ds4_alignment": 16, "ds4_qs_offset": 16,
            "layout": "[K128][column]", "ds_order": "d0,s0,d1,s1,d2,s2,d3,s3",
            "sum": "original raw-input half sum, not d8*sum(codes)",
            "padding": "missing halves/codes zero; unused K128 subblocks alone poisoned",
        },
        "resources": {
            "stream": "borrowed explicit nonblocking", "exact_payloads": True, "weight_alignment": 2,
            "weight_pointer_mod4": 2, "q8_alignment": 4, "output_alignment": 4,
            "redzones_bytes": {"weights": [2, 2], "raw": [4, 4], "q8": [4, 4], "ds4": [16, 16],
                               "flag": [4, 4], "output": [4, 4]},
            "redzone_bits": "half qNaN 0x7e00",
            "weight_readonly": "full original bytes in <=4MiB D2H chunks, no weight backup", "head_host_reads": 1,
        },
        "coverage": {
            "types": ["Q4_0", "Q4_1", "Q5_0", "Q8_0", "Q6_K"], "new_format_cartesian": True,
            "q5_q8_widths": [32, 160, 640, 2560, 16384], "q6_widths": [256, 768, 2560, 16384],
            "rows": [1, 63, 64, 65, 127, 128, 129], "columns": [1, 3, 4, 17, 65, 128],
            "q4_forward": {"widths": [160, 2560], "rows": [65, 129], "columns": [1, 3, 4, 17, 65, 128]},
            "limits": "all five types M16384 N3; new formats K16384 grid; sole large exception Q6_K[2560,248320]",
            "manual": {"q5_q8": [160, 65, 3], "q6": [768, 65, 3], "checks": [
                "int8 -128/127", "rawsum != d8*sumcodes", "signed/subnormal/extreme halves",
                "16 distinct Q6 subscales", "all 8 distinct Q6 activation scales", "integer dot*subscale before FP32"]},
            "quantizer_edges": {"shape": [256, 1], "cases": ["zero", "negative_zero", "half_rounding",
                "minimum_half_scale", "signed_alternating", "small_sine", "unrounded_scale_codes", "large_cancellation"]},
            "rejects": ["null", "shape", "type", "alignment", "range_wrap", "write_alias", "exact_head_exception"],
            "reuse": "all five formats; same producer/pack/linear allocations after rejection",
        },
        "expected_per_device": counters(),
        "expected_protocol": {"source": 1, "correctness": 2, "metrics": 90, "intervals": 1800, "complete": 1, "rows": 94},
        "actual": actual,
        "devices": [{"device": 0, "arch": "gfx906:sramecc+:xnack-", "wave": 64},
                    {"device": 1, "arch": "gfx906:sramecc+:xnack-", "wave": 64}],
    }
    proofs = {"q8_bytes_exact": True, "ds4_bytes_exact": True, "finite": True, "redzones": True, "readonly": True}
    rows = [origin]
    for device in (0, 1):
        rows.append({"kind": "mmq_wide_correctness", "protocol": PROTOCOL, "device": device,
                     "counters": counters(), "error": {"max_abs": .0001, "max_bound_ratio": .125},
                     "proofs": proofs.copy(), "cleanup": True, "passed": True})
    for device in (0, 1):
        for tensor_index, tensor in enumerate(actual):
            k, m = tensor["dimensions"]
            for column_index, n in enumerate((1, 3, 8, 32, 128)):
                for phase_index, phase in enumerate(("A1", "B", "A2")):
                    # HIP events produce float times promoted to double. Retain
                    # mixed A/B outcomes, not a fixture which presumes a win.
                    base = .018 + n * .001 + device * .002 + tensor_index * .009
                    if phase == "B":
                        base *= 1.15 if n == 1 else .65
                    elif phase == "A2":
                        base *= 1.03
                    times = [struct.unpack("f", struct.pack("f", base * (1 + (rep % 5 - 2) * .003)))[0]
                             for rep in range(20)]
                    total = 0.0
                    for ms in times:
                        total += ms
                    rows.append({
                        "kind": "mmq_wide_metric", "protocol": PROTOCOL, "sequence": len(rows) - 3,
                        "device": device, "tensor_index": tensor_index, "column_index": column_index,
                        "phase_index": phase_index, "phase": phase,
                        "path": "mmq_wide" if phase == "B" else "diagnostic_sliced_linear",
                        "tensor": tensor["tensor"], "type": tensor["type"], "width": k, "rows": m,
                        "columns": n, "weight_bytes": tensor["bytes"],
                        "reference_coverage": tensor["reference_coverage"], "reference_rows": tensor["reference_rows"],
                        "sample_rows": tensor["sample_rows"].copy(),
                        "launches_per_repeat": 1 if phase == "B" else (n + 2) // 3,
                        "repeats": 20, "validated_intervals": 20, "finite_elements": m * n * 20,
                        "reference_elements": tensor["reference_rows"] * n * 20,
                        "completed_ms": times, "completed_ms_total": total, "resident_ms": total / 20,
                        "warmup_error": {"max_abs": .00004, "max_bound_ratio": .0625},
                        "error": {"max_abs": .0001, "max_bound_ratio": .125},
                        "proofs": proofs.copy(), "cleanup": True, "passed": True,
                    })
    rows.append({"kind": "mmq_wide_complete", "protocol": PROTOCOL, "devices": 2, "source_records": 1,
                 "correctness_records": 2, "metric_records": 90, "validated_intervals": 1800,
                 "finite_elements": 5231001600, "reference_elements": 106172160, "jsonl_records": 94,
                 "live_owners": {"buffers": 0, "streams": 0, "events": 0}, "cleanup": True, "passed": True})
    # Ordered schema declarations from independently spelled fixture fields,
    # never from collector SCHEMAS. JSON object key order itself is not gated.
    origin["schemas"] = {r["kind"]: list(r) for r in (origin, rows[1], rows[3], rows[-1])}
    origin["schemas"]["mmq_wide_source"].append("schemas")
    origin["schemas"].update({
        "counters": list(counters()), "error": list(rows[1]["error"]), "proofs": list(proofs),
        "actual": list(actual[0]), "devices": list(origin["devices"][0]), "timing": list(origin["timing"]),
        "abi": list(origin["abi"]), "resources": list(origin["resources"]),
        "redzones_bytes": list(origin["resources"]["redzones_bytes"]), "coverage": list(origin["coverage"]),
        "q4_forward": list(origin["coverage"]["q4_forward"]), "manual": list(origin["coverage"]["manual"]),
        "quantizer_edges": list(origin["coverage"]["quantizer_edges"]),
        "expected_protocol": list(origin["expected_protocol"]), "live_owners": list(rows[-1]["live_owners"]),
    })
    return rows


def encoded(rows):
    return b"".join((json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n").encode() for row in rows)


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


class MmqWideResultsTest(unittest.TestCase):
    def setUp(self):
        # tempfile honors the caller's approved TMPDIR; no workspace artifacts.
        directory = tempfile.TemporaryDirectory(prefix="mmq-wide-results-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.raw = self.root / "mmq-wide.jsonl"
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

    def reject_change(self, index, path, value):
        rows = self.rows.copy()
        rows[index] = copy.deepcopy(rows[index])
        target(rows[index], path[:-1])[path[-1]] = value
        with self.subTest(row=index, path=path, value=value):
            self.reject(rows)

    def cli(self, destination=None):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", self.raw.name,
                               "--results", str(self.results if destination is None else destination)],
                              cwd=self.root, capture_output=True, text=True, timeout=30, check=False)

    def test_full_fixture_preserves_source_all_94_rows_and_compiled_provenance(self):
        before = self.raw.read_bytes()
        result = self.collect()
        self.assertEqual(len(self.rows), 94)
        self.assertEqual(result["kind"], "r4b_mmq_wide")
        for name, expected in (("source", self.rows[0]), ("correctness", self.rows[1:3]),
                               ("measurements", self.rows[3:93]), ("complete", self.rows[-1])):
            self.assertEqual(result[name], expected)
        self.assertEqual(result["raw_logs"], {"mmq_wide": str(self.raw.resolve())})
        self.assertEqual(result["revision"], REVISION)
        self.assertEqual(result["dirty"], 1)
        self.assertIs(type(result["dirty"]), int)
        self.assertEqual(result["model"], self.rows[0]["model"])
        self.assertIs(result["passed"], True)
        self.assertEqual(datetime.datetime.fromisoformat(result["timestamp"]).utcoffset(), datetime.timedelta(0))
        self.assertEqual(sum(r["validated_intervals"] for r in result["measurements"]), 1800)
        self.assertEqual(sum(r["finite_elements"] for r in result["measurements"]), 5231001600)
        self.assertEqual(sum(r["reference_elements"] for r in result["measurements"]), 106172160)
        self.assertEqual(self.raw.read_bytes(), before)
        json.dumps(result, allow_nan=False)

    def test_schema_declarations_and_arrays_match_actual_frozen_driver(self):
        # Catch fixture/collector agreement on a typo without executing HIP.
        driver = (ROOT / "src/mmq_wide_main.hip").read_text(encoding="utf-8")
        macro = driver[driver.index("#define WIDE_COUNTERS(X)"):driver.index("struct Counters")]
        names = re.findall(r"X\((\w+)\)", macro)
        self.assertEqual(names, list(counters()))
        start = driver.index('std::cout << "],\\"schemas\\":')
        middle = driver.index("bool first=true;", start)
        end_start = driver.index('std::cout << "],\\"error\\":', middle)
        end = driver.index("void proofs()", end_start)

        def strings(code):
            return "".join(json.loads(s) for s in re.findall(r'"(?:\\.|[^"\\])*"', code))

        emitted = strings(driver[start:middle]) + ",".join(json.dumps(s) for s in names) + strings(driver[end_start:end])
        schemas = json.loads(emitted[emitted.index('"schemas":') + len('"schemas":'):].rstrip()[:-1])
        self.assertEqual(schemas, self.rows[0]["schemas"])
        for name, expected in (("widths32", [32, 160, 640, 2560, 16384]),
                               ("widths256", [256, 768, 2560, 16384]),
                               ("row_counts", [1, 63, 64, 65, 127, 128, 129]),
                               ("column_counts", [1, 3, 4, 17, 65, 128]),
                               ("benchmark_columns", [1, 3, 8, 32, 128]), ("head_rows", HEAD_ROWS)):
            contents = re.search(r"constexpr std::array " + name + r"\{([^}]+)\}", driver).group(1)
            self.assertEqual([int(s) for s in contents.split(",")], expected)
        self.assertIn('protocol = "qwen.mmq-wide.v1";', driver)
        self.assertIn('absolute_gate = 2e-4, relative_gate = 2e-5;', driver)
        self.assertIn('c.mmq_rejects=153;', driver)
        self.assertIn('sizes{1126400, 1740800, 521472000}', driver)

    def test_independent_combinatorial_correctness_counts(self):
        observed = dict.fromkeys(counters(), 0)

        def add(k, m, n, category, manual=False, head=False):
            observed[category] += 1
            for key in ("matrices", "readonly_validations", "packing"):
                observed[key] += 1
            observed["quantization"] += not manual
            observed["finite_elements"] += m * n
            observed["reference_elements"] += (24 if head else m) * n
            observed["reference_slices"] += ((n + 2) // 3) * (24 if head else 1)
            used = (k // 32) % 4
            if used:
                observed["poison_cases"] += 1
                observed["poison_subblocks"] += (4 - used) * n

        for widths in ([32, 160, 640, 2560, 16384], [32, 160, 640, 2560, 16384], [256, 768, 2560, 16384]):
            for k in widths:
                for m in (1, 63, 64, 65, 127, 128, 129):
                    for n in (1, 3, 4, 17, 65, 128):
                        add(k, m, n, "grid_cases")
        for _ in range(2):
            for k in (160, 2560):
                for m in (65, 129):
                    for n in (1, 3, 4, 17, 65, 128):
                        add(k, m, n, "q4_forward_cases")
        for fmt in range(5):
            add(256 if fmt == 4 else 160, 65, 3, "reuse_cases")
            add(256 if fmt == 4 else 32, 16384, 3, "limit_cases")
        for k in (160, 160, 768):
            add(k, 65, 3, "manual_cases", manual=True)
        for k, m, head in ((640, 2560, False), (640, 2560, False), (2560, 248320, True)):
            for n in (1, 3, 8, 32, 128):
                add(k, m, n, "actual_cases", head=head)
        observed["quantization"] += 8
        observed["packing"] += 8
        observed.update(quantizer_rejects=30, packer_rejects=23, mmq_rejects=153)
        for key in ("int8_min_cases", "int8_max_cases", "raw_sum_difference_cases", "signed_scale_cases",
                    "subnormal_scale_cases", "extreme_scale_cases"):
            observed[key] = 3
        for key in ("q6_subscale_cases", "q6_activation_scale_cases", "q6_integer_grouping_cases"):
            observed[key] = 1
        self.assertEqual(observed, counters())
        self.assertEqual(MODULE.COUNTERS, observed)

    def test_component_scope_head_sampling_and_no_speedup_inference(self):
        result = self.collect()
        for key in ("performance_claim", "universal_speedup_claim", "pp_qualified", "large_prompt_qualified",
                    "end_to_end_qualified", "independent_hf_claim", "mtp"):
            self.assertIs(result[key], False)
        for phrase in ("component A/B/A", "ONLY a fixture baseline", "not production PP", "large-prompt",
                       "end-to-end", "independent HF", "MTP", "no universal speedup", "24", "248319"):
            self.assertIn(phrase, result["scope"])
        self.assertEqual(result["timing_scope"], self.rows[0]["timing"]["clock"])
        self.assertEqual(result["timing_excludes"], self.rows[0]["timing"]["excludes"])
        self.assertEqual(result["fixture_gates"], {"absolute": .0002, "relative_cpu_magnitude": .00002,
            "bound": "absolute+relative_cpu_magnitude*abs(CPU_reference)", "max_bound_ratio_limit": 1,
            "frozen_before_execution": True})
        self.assertIn("only 24 sampled original rows", result["provenance_limits"])
        self.assertNotIn("hf_revision", result)
        self.assertGreater(result["measurements"][1]["resident_ms"], result["measurements"][0]["resident_ms"])
        self.assertLess(result["measurements"][4]["resident_ms"], result["measurements"][3]["resident_ms"])

    def test_protocol_string_is_separate_frozen_and_required_at_every_row(self):
        for index in range(94):
            self.reject_change(index, ("protocol",), "qwen.mmq-wide.v2")
        for index in REPRESENTATIVE:
            for bad in (None, True, False, 1, 2, 1.0, "1", "qwen.mmq.v1", PROTOCOL + " ", {}, []):
                self.reject_change(index, ("protocol",), bad)

    def test_every_missing_extra_duplicate_replaced_truncated_and_swapped_row(self):
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
            self.reject(self.rows[:index])
        for index in range(93):
            rows = self.rows.copy()
            rows[index], rows[index + 1] = rows[index + 1], rows[index]
            self.reject(rows)
        rows = self.rows.copy()
        rows.insert(48, rows.pop(2))
        self.reject(rows)
        rows = copy.deepcopy(self.rows)
        rows[3], rows[4] = rows[4], rows[3]
        rows[3]["sequence"], rows[4]["sequence"] = 0, 1
        self.reject(rows)

    def test_missing_extra_fields_at_every_nested_schema_level(self):
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
                    with self.subTest(index=index, path=path, field=field):
                        self.reject(rows)

    def test_schema_whitelist_cannot_self_authorize_new_fields_or_reordered_declarations(self):
        for kind, fields in self.rows[0]["schemas"].items():
            for bad in (fields[:-1], fields + ["unexpected"], fields[::-1], fields + fields[:1], None):
                self.reject_change(0, ("schemas", kind), bad)
        rows = copy.deepcopy(self.rows)
        rows[0]["schemas"]["mmq_wide_metric"].append("speedup")
        for row in rows[3:93]:
            row["speedup"] = 2
        self.reject(rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["schemas"]["actual"].append("full_cpu_oracle")
        rows[0]["actual"][2]["full_cpu_oracle"] = True
        self.reject(rows)

    def test_nonboolean_numerics_integer_types_and_negative_zero_spelling(self):
        for index in REPRESENTATIVE:
            for path, value in leaves(self.rows[index]):
                if type(value) in (int, float):
                    self.reject_change(index, path, True)
                if type(value) is int:
                    self.reject_change(index, path, float(value))
        raw = encoded(self.rows)
        for token in (b'"device":0', b'"sequence":0', b'"tensor_index":0', b'"buffers":0'):
            self.raw.write_bytes(raw.replace(token, token[:-1] + b"-0", 1))
            with self.assertRaises(ValueError):
                self.collect()

    def test_every_proof_flag_requires_true_and_cleanup_owners_are_zero(self):
        for index in REPRESENTATIVE:
            for path, value in leaves(self.rows[index]):
                if type(value) is bool:
                    for bad in (False, 1, None, "true"):
                        self.reject_change(index, path, bad)
        for index in range(1, 94):
            self.reject_change(index, ("passed",), False)
            self.reject_change(index, ("cleanup",), False)
            if index < 93:
                for flag in ("q8_bytes_exact", "ds4_bytes_exact", "finite", "redzones", "readonly"):
                    self.reject_change(index, ("proofs", flag), False)
        for owner in ("buffers", "streams", "events"):
            for bad in (1, -1, False, None):
                self.reject_change(93, ("live_owners", owner), bad)

    def test_malformed_nonobject_utf8_bom_deep_and_unfinished_input(self):
        raw = encoded(self.rows)
        bads = [raw[:-1], raw[:-20], raw + b"\n", raw + b"failure\n", raw + b" ", b"\xef\xbb\xbf" + raw,
                raw.replace(b"mmq_wide_source", b"\xff", 1)]
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
        for field in ("kind", "revision", "ds_order", "exact_payloads", "cases", "grid_cases", "mmq_wide_source",
                      "dimensions", "arch", "max_abs", "q8_bytes_exact", "completed_ms", "passed", "buffers"):
            key = ('"' + field + '":').encode()
            bad = raw.replace(key, key + b"0," + key, 1)
            self.assertNotEqual(raw, bad)
            self.raw.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                self.collect()

    def test_nonfinite_constants_overflow_exponents_and_huge_numeric_values(self):
        raw = encoded(self.rows)
        for literal in (b"NaN", b"Infinity", b"-Infinity", b"1e999", b"-1e999"):
            for token in (b'"absolute_gate":0.0002', b'"max_abs":0.0001', b'"dirty":1', b'"buffers":0'):
                self.raw.write_bytes(raw.replace(token, token.split(b":")[0] + b":" + literal, 1))
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.collect()
        for field in ("resident_ms", "completed_ms_total"):
            self.reject_change(3, (field,), 10**400)
        for field in ("max_abs", "max_bound_ratio"):
            self.reject_change(3, ("error", field), 10**400)

    def test_bounded_read_including_growing_log_and_nonregular_input(self):
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

    def test_crlf_clean_dirty_lowerhex_revision_relative_model_and_object_key_order(self):
        for dirty in (0, 1):
            rows = copy.deepcopy(self.rows)
            rows[0].update(revision="f" * 40, dirty=dirty, model='relative/модель "quoted".gguf')
            rows = [dict(reversed(list(row.items()))) for row in rows]
            self.raw.write_bytes(encoded(rows).replace(b"\n", b"\r\n"))
            result = self.collect()
            self.assertEqual(result["revision"], "f" * 40)
            self.assertEqual(result["dirty"], dirty)
            self.assertIs(type(result["dirty"]), int)
            self.assertEqual(result["model"], rows[0]["model"])
            self.assertEqual(result["source"], rows[0])

    def test_invalid_own_revision_dirty_model_and_device_provenance(self):
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

    def test_pinned_donor_and_frozen_scope_oracle_baseline_timing_error_gates(self):
        for field in ("donor_revision", "donor_license", "model_variant", "architecture", "scope", "oracle",
                      "baseline", "ordering"):
            self.reject_change(0, (field,), "changed contract")
        for bad in (REVISION, "DCD685463D597D31F5CA759D32C94592A2740FA4", None):
            self.reject_change(0, ("donor_revision",), bad)
        for field in ("absolute_gate", "relative_gate"):
            for bad in (0, -1, True, None, .001, "0.0002", self.rows[0][field] * 1.00001):
                self.reject_change(0, (field,), bad)
        for field, value in self.rows[0]["timing"].items():
            bad = value[::-1] if type(value) is list else value + 1 if type(value) is int else "changed"
            self.reject_change(0, ("timing", field), bad)
        self.reject_change(0, ("baseline",), "production_pp")
        self.reject_change(0, ("oracle",), "full LM-head CPU oracle")

    def test_q8_ds4_resources_readonly_poison_manual_reuse_contracts(self):
        for parent in ("abi", "resources", "coverage"):
            for path, value in leaves(self.rows[0][parent]):
                if type(value) is int:
                    bad = value + 1
                elif type(value) is bool:
                    bad = False
                else:
                    bad = "changed contract"
                self.reject_change(0, (parent, *path), bad)
        for field in ("types", "q5_q8_widths", "q6_widths", "rows", "columns", "rejects"):
            value = self.rows[0]["coverage"][field]
            self.reject_change(0, ("coverage", field), value[:-1])
            self.reject_change(0, ("coverage", field), value[::-1])

    def test_actual_source_geometry_types_bytes_strides_sample_read_bindings(self):
        for i, tensor in enumerate(self.rows[0]["actual"]):
            for field, value in tensor.items():
                if field == "file_offset":
                    continue
                if type(value) is int:
                    bad = value + 1
                elif type(value) is list:
                    bad = value[:-1] if value else [0]
                elif type(value) is bool:
                    bad = False
                else:
                    bad = "wrong"
                self.reject_change(0, ("actual", i, field), bad)
            for field in ("dimensions", "strides_bytes"):
                for axis, value in enumerate(tensor[field]):
                    self.reject_change(0, ("actual", i, field, axis), value + 1)
            self.reject_change(0, ("actual", i, "type"), {0: 6, 1: 8, 2: 14}[i])
        actual = self.rows[0]["actual"]
        for bad in (None, actual[:-1], actual[::-1], actual + actual[:1]):
            self.reject_change(0, ("actual",), bad)

    def test_uint64_nonoverlapping_offsets_not_hardcoded_and_alignment_not_invented(self):
        for i, tensor in enumerate(self.rows[0]["actual"]):
            for bad in (-1, 0, 1.0, True, 2**64, None, 2**64 - tensor["bytes"]):
                self.reject_change(0, ("actual", i, "file_offset"), bad)
        self.reject_change(0, ("actual", 1, "file_offset"), 1130495)
        self.reject_change(0, ("actual", 2, "file_offset"), 4096)
        rows = copy.deepcopy(self.rows)
        for tensor in rows[0]["actual"]:
            tensor["file_offset"] += 1000000001
        # Dynamic GGUF alignment is not emitted; alignment1 is a valid format.
        self.assertTrue(self.collect(rows)["passed"])

    def test_live_correctness_counts_match_frozen_expectations_not_mutually_edited_claims(self):
        for field, value in counters().items():
            self.reject_change(0, ("expected_per_device", field), value - 1)
            for index in (1, 2):
                self.reject_change(index, ("counters", field), value - 1)
        for field, value in self.rows[0]["expected_protocol"].items():
            self.reject_change(0, ("expected_protocol", field), value - 1)
        for field in ("mmq_rejects", "poison_subblocks", "reference_elements", "finite_elements",
                      "q6_integer_grouping_cases"):
            rows = copy.deepcopy(self.rows)
            rows[0]["expected_per_device"][field] = 0
            for index in (1, 2):
                rows[index]["counters"][field] = 0
            self.reject(rows)

    def test_all_metric_coordinates_geometry_and_completed_validation_counts(self):
        for index in range(3, 93):
            row = self.rows[index]
            for field in ("sequence", "device", "tensor_index", "column_index", "phase_index", "width", "rows",
                          "columns", "weight_bytes", "reference_rows", "launches_per_repeat", "repeats",
                          "validated_intervals", "finite_elements", "reference_elements"):
                self.reject_change(index, (field,), row[field] + 1)
        for index in (3, 4, 5, 16, 33, 47, 78, 92):
            for field in ("phase", "path", "tensor", "type", "reference_coverage"):
                self.reject_change(index, (field,), "wrong")
        self.reject_change(4, ("path",), "production_pp")
        self.reject_change(3, ("path",), "mmq_wide")

    def test_honest_head_sample_only_cpu_coverage_full_gpu_finite_and_final_row(self):
        for index, path in [(0, ("actual", 2)), *((i, ()) for i in range(3, 93) if self.rows[i]["tensor_index"] == 2)]:
            self.reject_change(index, (*path, "reference_coverage"), "all_rows")
            self.reject_change(index, (*path, "reference_rows"), 248320)
            sample = target(self.rows[index], path)["sample_rows"]
            for bad in (sample[:-1], sample[::-1], sample + [248319], sample[:-1] + [248318]):
                self.reject_change(index, (*path, "sample_rows"), bad)
        rows = copy.deepcopy(self.rows)
        rows[0]["actual"][2].update(reference_coverage="all_rows", reference_rows=248320, sample_rows=[])
        for row in rows[3:93]:
            if row["tensor_index"] == 2:
                row.update(reference_coverage="all_rows", reference_rows=248320, sample_rows=[],
                           reference_elements=row["finite_elements"])
        rows[-1]["reference_elements"] = rows[-1]["finite_elements"]
        self.reject(rows)
        for index in (33, 34, 47, 78, 92):
            row = self.rows[index]
            self.reject_change(index, ("finite_elements",), row["reference_elements"])
            self.reject_change(index, ("reference_elements",), row["finite_elements"])
            self.reject_change(index, ("finite_elements",), row["rows"] * row["columns"])
            self.reject_change(index, ("reference_elements",), 24 * row["columns"])
        result = self.collect(self.rows)
        for row in result["measurements"]:
            if row["tensor_index"] == 2:
                self.assertEqual(row["sample_rows"], HEAD_ROWS)
                self.assertEqual(row["sample_rows"][-1], 248319)
                self.assertGreater(row["finite_elements"], row["reference_elements"])
            else:
                self.assertEqual(row["finite_elements"], row["reference_elements"])

    def test_frozen_numeric_error_gates_every_row_and_warmup(self):
        for index in range(1, 93):
            for error_name in (("error", "warmup_error") if index >= 3 else ("error",)):
                self.reject_change(index, (error_name, "max_bound_ratio"), 1.0000000000000002)
        for index in (1, 2, 3, 4, 33, 92):
            for error_name in (("error", "warmup_error") if index >= 3 else ("error",)):
                for field in ("max_abs", "max_bound_ratio"):
                    for bad in (-1, True, None, "0"):
                        self.reject_change(index, (error_name, field), bad)
        for ratio in (0, 1):
            rows = copy.deepcopy(self.rows)
            for row in rows[1:93]:
                row["error"].update(max_abs=0, max_bound_ratio=ratio)
                if "warmup_error" in row:
                    row["warmup_error"].update(max_abs=0, max_bound_ratio=ratio)
            self.assertTrue(self.collect(rows)["passed"])
        # The relative term can allow max_abs above the absolute gate; no fake cap.
        rows[4]["error"].update(max_abs=.001, max_bound_ratio=1)
        self.assertTrue(self.collect(rows)["passed"])

    def test_every_interval_positive_finite_and_exactly20(self):
        for index in range(3, 93):
            self.reject_change(index, ("completed_ms", 19), 0)
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
        first = json.dumps(self.rows[3]["completed_ms"][0]).encode()
        for literal in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            self.raw.write_bytes(raw.replace(b'"completed_ms":[' + first, b'"completed_ms":[' + literal, 1))
            with self.assertRaises(ValueError):
                self.collect()

    def test_mean_sum_and_array_consistency_all_metrics(self):
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

    def test_32_double_ulp_aggregate_tolerance_is_rounding_aware_and_scale_tight(self):
        for scale in (1, 1e-20, 1e20):
            rows = copy.deepcopy(self.rows)
            times = [struct.unpack("f", struct.pack("f", scale))[0]]
            times += [struct.unpack("f", struct.pack("f", scale * 1e-16))[0]] * 19
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

    def test_complete_requires_all_frozen_counts_owners_cleanup_and_live_totals(self):
        for field, value in self.rows[-1].items():
            bad = value + 1 if type(value) is int else False if type(value) is bool else "wrong"
            self.reject_change(93, (field,), bad)
        rows = self.rows.copy()
        rows[-1] = {"kind": "mmq_wide_complete", "protocol": PROTOCOL, "passed": True}
        self.reject(rows)
        rows = copy.deepcopy(self.rows)
        rows[33]["finite_elements"] -= 20
        rows[-1]["finite_elements"] -= 20
        self.reject(rows)

    def test_cli_one_locked_append_to_explicit_root_journal_preserves_existing_and_raw(self):
        original = self.raw.read_bytes()
        old = b'{"kind":"existing"}\n'
        self.results.write_bytes(old)
        process = self.cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        saved = self.results.read_bytes()
        self.assertTrue(saved.startswith(old))
        self.assertEqual(len(saved.splitlines()), 2)
        result = json.loads(saved.splitlines()[1])
        self.assertEqual(result["kind"], "r4b_mmq_wide")
        self.assertEqual(result["source"], self.rows[0])
        self.assertEqual(result["correctness"], self.rows[1:3])
        self.assertEqual(result["measurements"], self.rows[3:93])
        self.assertEqual(result["complete"], self.rows[-1])
        self.assertEqual(result["raw_logs"], {"mmq_wide": str(self.raw.resolve())})
        self.assertEqual(self.raw.read_bytes(), original)
        self.assertFalse((self.root / "src/results.jsonl").exists())

    def test_cli_atomic_rejection_before_append_or_creation(self):
        old = b'{"kind":"existing"}\n'
        cases = [(0, ("revision",), "unknown"), (0, ("donor_revision",), REVISION),
                 (0, ("abi", "ds4_bytes"), 40), (1, ("counters", "mmq_rejects"), 0),
                 (2, ("proofs", "readonly"), False), (4, ("error", "max_bound_ratio"), 1.1),
                 (4, ("warmup_error", "max_bound_ratio"), 1.1), (33, ("reference_rows",), 248320),
                 (92, ("resident_ms",), 1), (93, ("cleanup",), False)]
        for index, path, value in cases:
            rows = copy.deepcopy(self.rows)
            target(rows[index], path[:-1])[path[-1]] = value
            self.raw.write_bytes(encoded(rows))
            before = self.raw.read_bytes()
            self.results.write_bytes(old)
            process = self.cli()
            self.assertEqual(process.returncode, 1)
            self.assertIn("record_mmq_wide:", process.stderr)
            self.assertEqual(self.results.read_bytes(), old)
            self.assertEqual(self.raw.read_bytes(), before)
            absent = self.root / "new-results.jsonl"
            self.assertEqual(self.cli(absent).returncode, 1)
            self.assertFalse(absent.exists())
        self.raw.write_bytes(encoded(self.rows[:-1]))
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), old)

    def test_cli_cannot_append_to_raw_model_or_aliases(self):
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

    def test_cli_incomplete_nonregular_journal_and_required_explicit_results(self):
        incomplete = b'{"kind":"existing"}'
        self.results.write_bytes(incomplete)
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), incomplete)
        self.assertEqual(self.cli(self.root).returncode, 1)
        process = subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", str(self.raw)],
                                 cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(self.results.read_bytes(), incomplete)

    def test_shared_append_function_only_called_after_full_validation(self):
        self.assertIs(MODULE.append_result, append_result)
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
