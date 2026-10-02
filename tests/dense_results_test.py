import copy
import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "tools/record_dense.py"
SPEC = importlib.util.spec_from_file_location("record_dense", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REVISION = "ce05879e0238c354134d35742f63fdecbfef4fa7"
MODEL = "/models/qwen38-keep1-Q4_0.gguf"


def error(elements):
    return {"max_abs": 1e-5, "max_bound_ratio": .05, "elements": elements}


def dense_fixture():
    # Independent spelling of the frozen driver schema, including full sizes.
    actual = [
        {"type": "F32", "tensor": "blk.0.ssm_alpha.weight", "rank": 2,
         "dimensions": [2560, 48], "strides_bytes": [4, 10240],
         "elements": 122880, "bytes": 491520, "file_offset": 8192,
         "relative_offset": 4096, "converted_type": "F32", "converted_bytes": 491520},
        {"type": "BF16", "tensor": "blk.3.indexer.k_proj.weight", "rank": 2,
         "dimensions": [2560, 128], "strides_bytes": [2, 5120],
         "elements": 327680, "bytes": 655360, "file_offset": 499712,
         "relative_offset": 495616, "converted_type": "F32", "converted_bytes": 1310720}]
    rows = [{
        "kind": "dense_source", "revision": REVISION, "dirty": 1, "model": MODEL,
        "absolute_gate": .0002, "relative_gate": .0002,
        "scope": "full unchanged F32 and BF16 tensors converted once to F32; deterministic raw-FP32 inputs; ascending FP32 CPU oracle",
        "timing_scope": "completed HIP events around resident dense linear only; excludes allocation, conversion, upload, CPU reference, quantization, readback and warmup",
        "tokens": [1, 2, 3, 128], "repeats": 20, "measurements": 16, "protocol_rows": 20,
        "correctness_per_device": {"matrix_cases": 28, "synthetic_cases": 20, "actual_cases": 8,
                                   "host_rejects": 26, "prefix_cases": 21,
                                   "matrix_elements": 2220648, "prefix_elements": 66288},
        "synthetic_shapes": [[1, 3], [7, 5], [3, 1], [16384, 3], [3, 16384]], "actual": actual}]
    for device in (0, 1):
        rows.append({"kind": "dense_correctness", "device": device, "matrix_cases": 28,
                     "synthetic_cases": 20, "actual_cases": 8, "host_rejects": 26, "prefix_cases": 21,
                     "error": error(2220648), "prefix_error": error(66288),
                     "all_finite": True, "canaries": True, "immutable": True, "passed": True})
    for device in (0, 1):
        for tensor in actual:
            width, count = tensor["dimensions"]
            for n in (1, 2, 3, 128):
                rows.append({"kind": "dense_measurement", "device": device, "type": tensor["type"],
                             "tensor": tensor["tensor"], "width": width, "rows": count,
                             "tokens": n, "repeats": 20, "resident_ms": .012345,
                             "error": error(count * n * 20),
                             "all_finite": True, "canaries": True, "immutable": True})
    rows.append({"kind": "dense_complete", "measurements": 16, "rows": 20, "passed": True})
    return rows


def head_fixture():
    rows = [{
        "kind": "head_source", "revision": REVISION, "dirty": 1,
        "model_variant": "qwen38-keep1-Q4_0", "type": "Q6_K",
        "input": 2560, "output": 248320, "bytes": 521472000,
        "absolute_gate": .0002, "relative_gate": .00002,
        "sample_rows": [0, 1, 7, 31, 63, 127, 255, 511, 1023, 4095, 8191, 16383,
                        32767, 65535, 131071, 200003, 248043, 248044, 248045, 248046,
                        248047, 248127, 248318, 248319],
        "scope": "full actual LM head with deterministic common-Q8 input; sampled CPU oracle, full finite and prefix checks",
        "timing_scope": "completed resident linear events; excludes upload, quantization, reset, CPU oracle and readback"}]
    for device in (0, 1):
        for n in (1, 2, 3):
            rows.append({"kind": "head_measurement", "device": device, "tokens": n,
                         "rows": 248320, "repeats": 20, "resident_ms": .012345,
                         "error": error(24 * n * 20), "all_finite": True,
                         "canaries": True, "q8_bytes_exact": True})
        rows.append({"kind": "head_correctness", "device": device, "host_rejects": 11,
                     "prefix_elements": 744960, "prefix_exact": True, "passed": True})
    rows.append({"kind": "head_complete", "measurements": 6, "rows": 10, "passed": True})
    return rows


def jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


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
        for index, child in enumerate(value):
            yield from objects(child, path + (index,))


def leaves(value, path=()):
    if type(value) is dict:
        for key, child in value.items():
            yield from leaves(child, path + (key,))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from leaves(child, path + (index,))
    else:
        yield path, value


class DenseResultsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.paths = {name: self.directory / (name + ".jsonl") for name in ("dense", "head")}
        self.results = self.directory / "results.jsonl"
        self.rows = {"dense": dense_fixture(), "head": head_fixture()}
        self.write(self.rows)

    def write(self, rows):
        for name, path in self.paths.items():
            path.write_text(jsonl(rows[name]), encoding="utf-8")

    def collect(self, rows=None):
        if rows is not None:
            self.write(rows)
        return MODULE.collect(self.paths["dense"], self.paths["head"])

    def reject_change(self, name, index, path, value):
        rows = copy.deepcopy(self.rows)
        target(rows[name][index], path[:-1])[path[-1]] = value
        with self.assertRaises(ValueError):
            self.collect(rows)

    def run_cli(self):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--dense", self.paths["dense"].name,
                               "--head", self.paths["head"].name, "--results", self.results.name],
                              cwd=self.directory, capture_output=True, text=True)

    def test_one_complete_record_preserves_every_raw_row(self):
        result = self.collect()
        self.assertEqual(set(result), {"kind", "timestamp", "revision", "dirty", "raw_logs", "scope",
                                      "timing_scope", "dense_contract", "head_contract", "fixture_gates",
                                      "donors", "dense", "head", "passed"})
        self.assertEqual(result["kind"], "r3a_dense_head")
        self.assertEqual(result["revision"], REVISION)
        self.assertEqual(result["dirty"], 1)
        self.assertIs(type(result["dirty"]), int)
        self.assertIs(result["passed"], True)
        timestamp = datetime.datetime.fromisoformat(result["timestamp"])
        self.assertEqual(timestamp.utcoffset(), datetime.timedelta(0))
        self.assertEqual(result["raw_logs"], {name: str(path.resolve()) for name, path in self.paths.items()})
        for name, count, metrics in (("dense", 20, 16), ("head", 10, 6)):
            saved = result[name]
            self.assertEqual(len(self.rows[name]), count)
            self.assertEqual(set(saved), {"source", "correctness", "measurements", "complete"})
            self.assertEqual(saved["source"], self.rows[name][0])
            self.assertEqual(saved["complete"], self.rows[name][-1])
            self.assertEqual(saved["correctness"], [r for r in self.rows[name] if r["kind"] == name + "_correctness"])
            self.assertEqual(saved["measurements"], [r for r in self.rows[name] if r["kind"] == name + "_measurement"])
            self.assertEqual(len(saved["measurements"]), metrics)
            for device in (0, 1):
                self.assertEqual(sum(r["device"] == device for r in saved["measurements"]), metrics // 2)
        json.dumps(result, allow_nan=False)

    def test_compiled_clean_dirty_provenance_is_preserved(self):
        for dirty in (0, 1):
            rows = copy.deepcopy(self.rows)
            for name in rows:
                rows[name][0].update(revision="b" * 40, dirty=dirty)
            result = self.collect(rows)
            self.assertEqual(result["revision"], "b" * 40)
            self.assertEqual(result["dirty"], dirty)
            for source in (result, result["dense"]["source"], result["head"]["source"]):
                self.assertIs(type(source["dirty"]), int)

    def test_scope_contracts_and_donor_pins(self):
        result = self.collect()
        self.assertIn("full unchanged dense matmuls", result["scope"])
        self.assertIn("sampled CPU common-Q8", result["scope"])
        self.assertIn("not full-network teacher-forced logits, A/B speedup", result["scope"])
        self.assertIn("20 individually completed", result["timing_scope"])
        self.assertIn("every repetition validated", result["timing_scope"])
        for excluded in ("conversion", "quantization", "transfers", "CPU oracle", "output reset", "warmup"):
            self.assertIn(excluded, result["timing_scope"])
        dc, hc = result["dense_contract"], result["head_contract"]
        self.assertIn("exactly once", dc["conversion"])
        self.assertIn("F32 bits preserved", dc["conversion"])
        self.assertIn("BF16 bits shifted left16", dc["conversion"])
        self.assertIn("transpose-A / no-transpose-B", dc["layout"])
        self.assertIn("lda=ldb=K, ldc=M", dc["layout"])
        self.assertEqual((dc["alpha"], dc["beta"]), (1, 0))
        self.assertIn("unfused FP32", dc["reference"])
        self.assertIn("bounded", dc["prefix"])
        self.assertEqual(hc["dimensions"], [2560, 248320])
        self.assertEqual(hc["bytes"], 521472000)
        self.assertIn("block256/210 bytes", hc["storage"])
        self.assertIn("36 bytes", hc["activation_abi"])
        self.assertIn("raw input sum", hc["activation_abi"])
        self.assertIn("FP32", hc["codes"])
        self.assertIn("not sum of quantized codes", hc["raw_sum"])
        self.assertIn("int32 scaled dot4", hc["arithmetic"])
        self.assertIn("24 exact sampled", hc["reference"])
        self.assertIn("not a full-output CPU oracle", hc["reference"])
        self.assertIn("744960", hc["prefix"])
        self.assertIn("not model path/offsets", hc["provenance_limits"])
        self.assertEqual(result["donors"], {
            "mx": "dcd685463d597d31f5ca759d32c94592a2740fa4",
            "furnace": "905021dbad71c5056ef51f9fd45d545403fc989c",
            "reinstinct": "0b79e326351d90d4554a1c18df92da5d0ab692e8"})

    def test_frozen_gates_and_source_scope_cannot_change(self):
        result = self.collect()
        self.assertEqual(result["fixture_gates"], {
            "dense": {"absolute": .0002, "relative_cpu_magnitude": .0002,
                      "frozen_before_gpu": True, "max_bound_ratio_limit": 1},
            "head": {"absolute": .0002, "relative_cpu_magnitude": .00002,
                     "frozen_before_gpu": True, "max_bound_ratio_limit": 1},
            "bound": "absolute+relative_cpu_magnitude*abs(CPU_reference)"})
        for name in self.rows:
            for field in ("absolute_gate", "relative_gate"):
                for bad in (0, -1, True, None, .001, "0.0002"):
                    self.reject_change(name, 0, (field,), bad)
            for field in ("scope", "timing_scope"):
                self.reject_change(name, 0, (field,), "full-network logits or host enqueue time")

    def test_all_missing_rows_and_truncated_prefixes(self):
        for name in self.rows:
            for index in range(len(self.rows[name])):
                rows = copy.deepcopy(self.rows)
                rows[name].pop(index)
                with self.subTest(driver=name, missing=index), self.assertRaises(ValueError):
                    self.collect(rows)
            for length in range(len(self.rows[name])):
                rows = copy.deepcopy(self.rows)
                rows[name] = rows[name][:length]
                with self.assertRaises(ValueError):
                    self.collect(rows)

    def test_all_adjacent_reorderings(self):
        for name in self.rows:
            for index in range(len(self.rows[name]) - 1):
                rows = copy.deepcopy(self.rows)
                rows[name][index], rows[name][index + 1] = rows[name][index + 1], rows[name][index]
                with self.subTest(driver=name, index=index), self.assertRaises(ValueError):
                    self.collect(rows)

    def test_extra_duplicate_and_replaced_rows(self):
        for name in self.rows:
            for index in range(len(self.rows[name])):
                rows = copy.deepcopy(self.rows)
                rows[name].insert(index, copy.deepcopy(rows[name][index]))
                with self.assertRaises(ValueError):
                    self.collect(rows)
            for index in range(1, len(self.rows[name])):
                rows = copy.deepcopy(self.rows)
                rows[name][index] = copy.deepcopy(rows[name][index - 1])
                with self.assertRaises(ValueError):
                    self.collect(rows)

    def test_missing_and_extra_fields_at_every_object(self):
        for name in self.rows:
            for index, row in enumerate(self.rows[name]):
                for path, obj in objects(row):
                    for field in (*obj, "unexpected"):
                        rows = copy.deepcopy(self.rows)
                        changed = target(rows[name][index], path)
                        if field == "unexpected":
                            changed[field] = True
                        else:
                            del changed[field]
                        with self.subTest(driver=name, index=index, path=path, field=field):
                            with self.assertRaises(ValueError):
                                self.collect(rows)

    def test_nonobjects_blank_malformed_and_nonjson_rows(self):
        for name in self.rows:
            for value in (None, [], "passed", 1, True):
                rows = copy.deepcopy(self.rows)
                rows[name][1] = value
                with self.assertRaises(ValueError):
                    self.collect(rows)
            for line in ("", " ", "diagnostic text", "{", "{} {}", "{\"kind\":1,}"):
                self.write(self.rows)
                lines = jsonl(self.rows[name]).splitlines()
                lines[1] = line
                self.paths[name].write_text("\n".join(lines) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.collect()

    def test_invalid_utf8_bom_and_unterminated_logs(self):
        for name in self.rows:
            encoded = jsonl(self.rows[name]).encode()
            for raw in (b"\xef\xbb\xbf" + encoded, encoded.replace(b"source", b"\xff", 1),
                        encoded[:-1], encoded[:-8], encoded + b"\n", encoded + b" "):
                self.write(self.rows)
                self.paths[name].write_bytes(raw)
                with self.assertRaises(ValueError):
                    self.collect()

    def test_duplicate_json_keys_at_all_schema_levels(self):
        for name in self.rows:
            changes = [("\"revision\":", "\"revision\":\"ignored\",\"revision\":"),
                       ("\"max_abs\":", "\"max_abs\":0,\"max_abs\":")]
            if name == "dense":
                changes.extend([("\"matrix_cases\":", "\"matrix_cases\":0,\"matrix_cases\":"),
                                ("\"converted_bytes\":", "\"converted_bytes\":0,\"converted_bytes\":")])
            for before, after in changes:
                self.write(self.rows)
                self.paths[name].write_text(jsonl(self.rows[name]).replace(before, after, 1), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.collect()

    def test_nonfinite_and_overflow_json_literals(self):
        for name in self.rows:
            for field, original in (("resident_ms", "0.012345"), ("max_abs", "1e-05"),
                                    ("max_bound_ratio", "0.05"), ("absolute_gate", "0.0002")):
                for literal in ("NaN", "Infinity", "-Infinity", "1e309", "-1e309"):
                    self.write(self.rows)
                    raw = jsonl(self.rows[name]).replace(f'"{field}": {original}', f'"{field}": {literal}', 1)
                    self.paths[name].write_text(raw, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.collect()

    def test_oversized_and_exact_boundary_logs(self):
        for name in self.rows:
            self.write(self.rows)
            raw = jsonl(self.rows[name]).encode()
            limit = 4 * 1024 * 1024
            self.paths[name].write_bytes(b" " * (limit - len(raw)) + raw)
            self.assertTrue(self.collect()["passed"])
            self.paths[name].write_bytes(b" " * (limit + 1 - len(raw)) + raw)
            with self.assertRaises(ValueError):
                self.collect()

    def test_deeply_nested_json_is_rejected(self):
        for name in self.rows:
            self.write(self.rows)
            lines = jsonl(self.rows[name]).splitlines()
            lines[1] = '{"kind":' + "[" * 2000 + "0" + "]" * 2000 + "}"
            self.paths[name].write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.collect()

    def test_booleans_never_replace_any_numeric_leaf(self):
        for name in self.rows:
            for index, row in enumerate(self.rows[name]):
                for path, value in leaves(row):
                    if type(value) in (int, float):
                        with self.subTest(driver=name, index=index, path=path):
                            self.reject_change(name, index, path, True)

    def test_floats_never_replace_any_integer_leaf(self):
        for name in self.rows:
            for index, row in enumerate(self.rows[name]):
                for path, value in leaves(row):
                    if type(value) is int:
                        with self.subTest(driver=name, index=index, path=path):
                            self.reject_change(name, index, path, float(value))

    def test_proof_flags_are_true_booleans(self):
        for name in self.rows:
            for index, row in enumerate(self.rows[name]):
                for path, value in leaves(row):
                    if type(value) is bool:
                        for bad in (False, 1, None, "true"):
                            self.reject_change(name, index, path, bad)

    def test_invalid_and_mismatched_compiled_provenance(self):
        for name in self.rows:
            for bad in (None, True, 1, "", "dirty", "a" * 39, "g" * 40, "A" * 40):
                self.reject_change(name, 0, ("revision",), bad)
            for bad in (True, False, None, -1, 2, 1.0, "1"):
                self.reject_change(name, 0, ("dirty",), bad)
            self.reject_change(name, 0, ("revision",), "b" * 40)
            self.reject_change(name, 0, ("dirty",), 0)

    def test_source_model_and_variant(self):
        for bad in (None, True, 0, "", "path\n", "\ud800", "path\x00"):
            self.reject_change("dense", 0, ("model",), bad)
        for bad in (None, True, "qwen38-full-PLE-Q4_0", "qwen38-keep1-Q8_0", ""):
            self.reject_change("head", 0, ("model_variant",), bad)
        # No head model path is emitted: do not invent a cross-log path check.
        rows = copy.deepcopy(self.rows)
        rows["dense"][0]["model"] = "relative-model.gguf"
        self.assertEqual(self.collect(rows)["dense"]["source"]["model"], "relative-model.gguf")

    def test_source_counts_tokens_and_synthetic_shapes(self):
        source = self.rows["dense"][0]
        for field in ("repeats", "measurements", "protocol_rows"):
            self.reject_change("dense", 0, (field,), source[field] + 1)
        for bad in ([1, 2, 3], [1, 3, 2, 128], [1, 2, 3, 127], None):
            self.reject_change("dense", 0, ("tokens",), bad)
        for index, shape in enumerate(source["synthetic_shapes"]):
            for dim in (0, 1):
                self.reject_change("dense", 0, ("synthetic_shapes", index, dim), shape[dim] + 1)
        self.reject_change("dense", 0, ("synthetic_shapes",), source["synthetic_shapes"][:-1])

    def test_dense_tensor_types_shapes_storage_and_conversion_sizes(self):
        for index, tensor in enumerate(self.rows["dense"][0]["actual"]):
            for field, bad in (("type", "F16"), ("tensor", "wrong"), ("rank", 3),
                               ("dimensions", list(reversed(tensor["dimensions"]))),
                               ("strides_bytes", [4, 4]), ("elements", tensor["elements"] + 1),
                               ("bytes", tensor["bytes"] + 1), ("converted_type", "BF16"),
                               ("converted_bytes", tensor["converted_bytes"] + 1)):
                self.reject_change("dense", 0, ("actual", index, field), bad)
            for field in ("dimensions", "strides_bytes"):
                for dim in (0, 1):
                    self.reject_change("dense", 0, ("actual", index, field, dim), tensor[field][dim] + 1)
        actual = self.rows["dense"][0]["actual"]
        for bad in (None, actual[:1], actual + [actual[0]], actual[::-1]):
            self.reject_change("dense", 0, ("actual",), bad)
        # Reduced actual tensors cannot qualify even with mutually edited rows.
        rows = copy.deepcopy(self.rows)
        tensor = rows["dense"][0]["actual"][0]
        tensor.update(dimensions=[2560, 1], elements=2560, bytes=10240, converted_bytes=10240)
        for row in rows["dense"]:
            if row["kind"] == "dense_measurement" and row["type"] == "F32":
                row.update(rows=1, error=error(row["tokens"] * 20))
        with self.assertRaises(ValueError):
            self.collect(rows)

    def test_dense_tensor_uint64_offsets_common_origin_and_nonoverlap(self):
        for index in (0, 1):
            for field in ("relative_offset", "file_offset"):
                for bad in (-1, 1.0, True, 2**64, None):
                    self.reject_change("dense", 0, ("actual", index, field), bad)
        first = self.rows["dense"][0]["actual"][0]
        self.reject_change("dense", 0, ("actual", 0, "file_offset"), first["relative_offset"])
        self.reject_change("dense", 0, ("actual", 0, "file_offset"), first["relative_offset"] - 1)
        self.reject_change("dense", 0, ("actual", 1, "file_offset"), 499713)
        rows = copy.deepcopy(self.rows)
        a, b = rows["dense"][0]["actual"]
        b.update(file_offset=a["file_offset"] + a["bytes"] - 1,
                 relative_offset=a["relative_offset"] + a["bytes"] - 1)
        with self.assertRaises(ValueError):
            self.collect(rows)
        rows = copy.deepcopy(self.rows)
        for tensor in rows["dense"][0]["actual"]:
            tensor["file_offset"] += 2**64 - 500000
            tensor["relative_offset"] += 2**64 - 500000
        with self.assertRaises(ValueError):
            self.collect(rows)

    def test_dense_offsets_are_not_hardcoded_model_positions(self):
        rows = copy.deepcopy(self.rows)
        for tensor in rows["dense"][0]["actual"]:
            tensor["file_offset"] += 100000000
            tensor["relative_offset"] += 99999000
        self.assertTrue(self.collect(rows)["passed"])
        rows["dense"][0]["actual"][0].update(file_offset=4096, relative_offset=0)
        rows["dense"][0]["actual"][1].update(file_offset=495616, relative_offset=491520)
        self.assertTrue(self.collect(rows)["passed"])

    def test_dense_frozen_correctness_counts_and_comparison_coverage(self):
        for field, count in self.rows["dense"][0]["correctness_per_device"].items():
            self.reject_change("dense", 0, ("correctness_per_device", field), count - 1)
        for index in (1, 2):
            row = self.rows["dense"][index]
            for field in ("matrix_cases", "synthetic_cases", "actual_cases", "host_rejects", "prefix_cases"):
                self.reject_change("dense", index, (field,), row[field] - 1)
            for field in ("error", "prefix_error"):
                self.reject_change("dense", index, (field, "elements"), row[field]["elements"] - 1)
        rows = copy.deepcopy(self.rows)
        rows["dense"][0]["correctness_per_device"]["matrix_cases"] = 27
        for index in (1, 2):
            rows["dense"][index]["matrix_cases"] = 27
        with self.assertRaises(ValueError):
            self.collect(rows)

    def test_dense_measurement_geometry_order_and_every_repeat(self):
        for index in range(3, 19):
            row = self.rows["dense"][index]
            for field, bad in (("device", 1 - row["device"]), ("type", "F16"), ("tensor", "wrong"),
                               ("width", 2559), ("rows", row["rows"] + 1),
                               ("tokens", 4), ("repeats", 19)):
                self.reject_change("dense", index, (field,), bad)
            self.reject_change("dense", index, ("error", "elements"), row["rows"] * row["tokens"])
            self.reject_change("dense", index, ("error", "elements"), row["error"]["elements"] + 1)

    def test_head_full_geometry_type_and_byte_size(self):
        for field, bad in (("type", "Q8_0"), ("input", 2304), ("output", 248319),
                           ("bytes", 521472000 - 2100)):
            self.reject_change("head", 0, (field,), bad)
        rows = copy.deepcopy(self.rows)
        rows["head"][0].update(output=24, bytes=50400)
        for row in rows["head"]:
            if row["kind"] == "head_measurement":
                row["rows"] = 24
        with self.assertRaises(ValueError):
            self.collect(rows)

    def test_head_exact_sample_row_ids(self):
        sample = self.rows["head"][0]["sample_rows"]
        for index, row_id in enumerate(sample):
            self.reject_change("head", 0, ("sample_rows", index), row_id + 1)
        for bad in (sample[:-1], sample + [0], sample[::-1], None):
            self.reject_change("head", 0, ("sample_rows",), bad)
        self.reject_change("head", 0, ("sample_rows", 1), 0)

    def test_head_sampled_error_coverage_not_full_output_claim(self):
        for index in (1, 2, 3, 5, 6, 7):
            row = self.rows["head"][index]
            for bad in (24 * row["tokens"], 248320 * row["tokens"] * 20,
                        row["error"]["elements"] + 1):
                self.reject_change("head", index, ("error", "elements"), bad)
            for field, bad in (("rows", 248319), ("tokens", 128), ("repeats", 19),
                               ("device", 1 - row["device"])):
                self.reject_change("head", index, (field,), bad)

    def test_head_per_device_rejects_and_full_exact_prefix(self):
        for index in (4, 8):
            for field, bad in (("host_rejects", 10), ("prefix_elements", 744959),
                               ("prefix_exact", False), ("device", 2)):
                self.reject_change("head", index, (field,), bad)

    def test_positive_finite_completed_times(self):
        for name in self.rows:
            for index, row in enumerate(self.rows[name]):
                if "resident_ms" in row:
                    for bad in (0, -1, True, None, "0.01", float("nan"), float("inf"),
                                -float("inf"), 10**400):
                        self.reject_change(name, index, ("resident_ms",), bad)

    def test_all_error_groups_bounds_and_elements_types(self):
        for name in self.rows:
            for index, row in enumerate(self.rows[name]):
                for group in ("error", "prefix_error"):
                    if group not in row:
                        continue
                    for field in ("max_abs", "max_bound_ratio"):
                        for bad in (-1, True, None, "0", float("nan"), float("inf"), 10**400):
                            self.reject_change(name, index, (group, field), bad)
                    self.reject_change(name, index, (group, "max_bound_ratio"), 1.0000001)
                    for bad in (0, -1, True, 1.0, None):
                        self.reject_change(name, index, (group, "elements"), bad)

    def test_error_boundary_values_and_integer_times(self):
        for ratio in (0, 1):
            rows = copy.deepcopy(self.rows)
            for name in rows:
                for row in rows[name]:
                    if "resident_ms" in row:
                        row["resident_ms"] = 1
                    for group in ("error", "prefix_error"):
                        if group in row:
                            row[group].update(max_abs=0, max_bound_ratio=ratio)
            self.assertTrue(self.collect(rows)["passed"])
        # The gate includes reference magnitude: max_abs is not capped at 2e-4.
        rows["dense"][3]["error"].update(max_abs=.001, max_bound_ratio=1)
        self.assertTrue(self.collect(rows)["passed"])

    def test_footers_require_exact_full_completion(self):
        for name in self.rows:
            for field in self.rows[name][-1]:
                for bad in (None, False, "complete", 0, 999):
                    self.reject_change(name, -1, (field,), bad)
            rows = copy.deepcopy(self.rows)
            rows[name][-1] = {"kind": name + "_complete", "passed": True}
            with self.assertRaises(ValueError):
                self.collect(rows)

    def test_crlf_and_escaped_unicode_model_path(self):
        rows = copy.deepcopy(self.rows)
        model = '/models/модель "quoted".gguf'
        rows["dense"][0]["model"] = model
        for name in self.paths:
            self.paths[name].write_bytes(jsonl(rows[name]).replace("\n", "\r\n").encode())
        self.assertEqual(self.collect()["dense"]["source"]["model"], model)

    def test_cli_appends_one_record_with_absolute_raw_paths(self):
        self.results.write_text('{"kind":"existing"}\n', encoding="utf-8")
        process = self.run_cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        saved = [json.loads(line) for line in self.results.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0], {"kind": "existing"})
        result = saved[1]
        self.assertEqual(result["kind"], "r3a_dense_head")
        self.assertEqual(result["revision"], REVISION)
        self.assertEqual(result["raw_logs"], {name: str(path.resolve()) for name, path in self.paths.items()})
        self.assertEqual(len(result["dense"]["measurements"]), 16)
        self.assertEqual(len(result["head"]["measurements"]), 6)
        for name in self.rows:
            self.assertEqual(result[name]["source"], self.rows[name][0])

    def test_cli_invalid_logs_preserve_existing_destination(self):
        existing = b'{"kind":"existing"}\n'
        self.results.write_bytes(existing)
        cases = [("dense", -1, ("passed",), False), ("head", -1, ("measurements",), 5),
                 ("head", 0, ("revision",), "b" * 40), ("dense", 3, ("canaries",), False),
                 ("head", 4, ("prefix_exact",), False)]
        for name, index, path, value in cases:
            rows = copy.deepcopy(self.rows)
            target(rows[name][index], path[:-1])[path[-1]] = value
            self.write(rows)
            self.assertNotEqual(self.run_cli().returncode, 0)
            self.assertEqual(self.results.read_bytes(), existing)
        for name in self.rows:
            self.write(self.rows)
            self.paths[name].write_text(jsonl(self.rows[name][:-1]), encoding="utf-8")
            self.assertNotEqual(self.run_cli().returncode, 0)
            self.assertEqual(self.results.read_bytes(), existing)

    def test_cli_invalid_logs_do_not_create_destination(self):
        for name in self.rows:
            rows = copy.deepcopy(self.rows)
            rows[name][-1] = {"kind": name + "_complete", "passed": True}
            self.write(rows)
            self.assertNotEqual(self.run_cli().returncode, 0)
            self.assertFalse(self.results.exists())


if __name__ == "__main__":
    unittest.main()
