import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).parents[1] / "tools/record_gdn.py"
spec = importlib.util.spec_from_file_location("record_gdn", SCRIPT)
record_gdn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record_gdn)


def fixture():
    rows = [{"kind": "gdn_source", "revision": "0123456789abcdef" * 2 + "01234567", "dirty": 1,
             "model": "qwen38-keep1-Q4_0.gguf", "layer": 0,
             "projection": "CPU raw FP32 oracle", "rms_epsilon": 1e-6,
             "scope": "loaded layer0 weights; eight synthetic projected tokens repeated for N128; not full inference"}]
    gates = {"max_output_error": 1e-5, "max_state_error": 1e-6,
             "output_gate_ratio": .05, "state_gate_ratio": .05, "history_byte_parity": True}
    for device in (0, 1):
        for tokens in (1, 2, 3, 128):
            rows.append({"kind": "gdn_gpu", "device": device, "tokens": tokens, "repeats": 20,
                         "reset_and_resident_ms": .01 * tokens, "cpu_oracle_ms": 1.5 * tokens, **gates})
            if tokens != 1:
                for path in ("decode_steps", "resident_chunk", "decode_steps"):
                    rows.append({"kind": "gdn_dispatch", "device": device, "tokens": tokens,
                                 "path": path, "repeats": 20, "reset_and_resident_ms": .02 * tokens, **gates})
        rows.append({"kind": "gdn_restore", "device": device, "cases": 8, "windows": 2,
                     **gates, "split_chunk_parity": True})
        rows.append({"kind": "gdn_invalid", "device": device, "rejections": 11,
                     "chunk_atomic": True, "reuse": True})
    rows.append({"kind": "gdn_complete", "passed": True})
    return rows


def jsonl(rows):
    return "".join(json.dumps(item) + "\n" for item in rows)


class GdnResultsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.raw = self.directory / "gdn.jsonl"
        self.results = self.directory / "results.jsonl"

    def collect(self, rows):
        self.raw.write_text(jsonl(rows), encoding="utf-8")
        return record_gdn.collect(self.raw)

    def reject_change(self, index, field, value):
        rows = fixture()
        rows[index][field] = value
        with self.assertRaises(ValueError):
            self.collect(rows)

    def run_cli(self):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", self.raw.name,
                               "--results", self.results.name], cwd=self.directory,
                              capture_output=True, text=True)

    def test_complete_exact_protocol_and_measurements(self):
        rows = fixture()
        self.assertEqual(len(rows), 32)
        result = self.collect(rows)
        self.assertEqual(result["kind"], "r2b_gdn")
        self.assertEqual(result["raw_log"], str(self.raw.resolve()))
        self.assertEqual(result["source"], rows[0])
        self.assertEqual(result["revision"], rows[0]["revision"])
        self.assertEqual(result["dirty"], 1)
        self.assertIs(type(result["dirty"]), int)
        measurements = [row for row in rows if row["kind"] in ("gdn_gpu", "gdn_dispatch")]
        self.assertEqual(len(result["measurements"]), 26)
        self.assertEqual(result["measurements"], measurements)
        self.assertEqual(result["restore"], [rows[14], rows[29]])
        self.assertEqual(result["rejections"], [rows[15], rows[30]])
        json.dumps(result, allow_nan=False)

    def test_scope_abi_and_frozen_gates(self):
        rows = fixture()
        rows[0]["rms_epsilon"] = 3e-6
        result = self.collect(rows)
        for scope in ("actual layer0 weights", "CPU raw FP32 projections", "GPU recurrence",
                      "before out_proj", "not whole-model inference or end-to-end speed"):
            self.assertIn(scope, result["scope"])
        self.assertIn("completed HIP events", result["timing_scope"])
        self.assertIn("excluding projections/host transfers/readback", result["timing_scope"])
        self.assertIn("FP32 [V-head][V-component][K-component]", result["state_abi"])
        self.assertIn("[48][128][128]", result["state_abi"])
        self.assertIn("K contiguous", result["state_abi"])
        self.assertIn("[10240][3]", result["history_abi"])
        self.assertIn("oldest -> newest RAW projections", result["history_abi"])
        self.assertIn("h%16", result["head_mapping"])
        self.assertEqual(result["prefix_abi"], {
            "slot_0": "pre-chunk state", "slot_n": "after n consumed inputs, not emitted tokens",
            "verify_inputs": "one pending input + two drafts", "accept_slot": "1 + a",
            "accepted_drafts_a": [0, 1, 2]})
        self.assertEqual(result["normalization"], {"qk_l2_epsilon": 1e-6, "rms_epsilon": 3e-6,
                                                  "convolution_activation": "SiLU", "output_gate": "sigmoid"})
        self.assertEqual(result["fixture_gates"], {
            "output_abs": 2e-4, "output_rel": 2e-4, "state_abs": 2e-5, "state_rel": 2e-4,
            "rule": "error <= abs + rel * |ref|", "gate_ratio_limit": 1,
            "frozen_before_first_gpu_run": True})
        self.assertEqual(result["donors"], {
            "mx": "dcd685463d597d31f5ca759d32c94592a2740fa4",
            "furnace": "905021dbad71c5056ef51f9fd45d545403fc989c",
            "reinstinct": "0b79e326351d90d4554a1c18df92da5d0ab692e8"})
        self.assertEqual(result["reference"], "transformers a005fc82babfe8871d87746decad2dbee100a125")

    def test_clean_compiled_provenance_is_unchanged(self):
        rows = fixture()
        rows[0].update(revision="b" * 40, dirty=0)
        result = self.collect(rows)
        self.assertEqual(result["revision"], "b" * 40)
        self.assertEqual(result["dirty"], 0)
        self.assertIs(type(result["dirty"]), int)
        self.assertEqual(result["source"], rows[0])

    def test_numeric_boundaries_are_valid(self):
        rows = fixture()
        for row in rows:
            for field in ("max_output_error", "max_state_error"):
                if field in row:
                    row[field] = 0
            for field in ("output_gate_ratio", "state_gate_ratio"):
                if field in row:
                    row[field] = 0
        self.collect(rows)
        for row in rows:
            for field in ("output_gate_ratio", "state_gate_ratio"):
                if field in row:
                    row[field] = 1
            for field in ("max_output_error", "max_state_error", "reset_and_resident_ms", "cpu_oracle_ms"):
                if field in row:
                    row[field] = 1
        self.collect(rows)

    def test_missing_rows_and_missing_completion(self):
        for index in range(32):
            with self.subTest(index=index):
                rows = fixture()
                rows.pop(index)
                with self.assertRaises(ValueError):
                    self.collect(rows)
        for length in (0, 1, 15, 16, 30):
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.collect(fixture()[:length])

    def test_extra_rows_and_duplicate_rows(self):
        for index in range(32):
            with self.subTest(index=index):
                rows = fixture()
                rows.insert(index, copy.deepcopy(rows[index]))
                with self.assertRaises(ValueError):
                    self.collect(rows)
        with self.assertRaises(ValueError):
            self.collect(fixture() * 2)

    def test_reordered_protocol(self):
        for index in range(31):
            with self.subTest(index=index):
                rows = fixture()
                rows[index], rows[index + 1] = rows[index + 1], rows[index]
                with self.assertRaises(ValueError):
                    self.collect(rows)
        rows = fixture()
        rows[1:16], rows[16:31] = rows[16:31], rows[1:16]
        with self.assertRaises(ValueError):
            self.collect(rows)

    def test_duplicate_measurement_replacing_expected_row(self):
        for target, duplicate in ((2, 1), (4, 3), (6, 2), (14, 15), (16, 1), (29, 14), (31, 0)):
            with self.subTest(target=target, duplicate=duplicate):
                rows = fixture()
                rows[target] = copy.deepcopy(rows[duplicate])
                with self.assertRaises(ValueError):
                    self.collect(rows)

    def test_missing_fields(self):
        for index, row in enumerate(fixture()):
            for field in row:
                with self.subTest(index=index, field=field):
                    rows = fixture()
                    del rows[index][field]
                    with self.assertRaises(ValueError):
                        self.collect(rows)

    def test_extra_fields_and_spurious_prefix_passes(self):
        for index in range(32):
            for field in ("prefix_passes", "state_elements", "fixture_gates", "unexpected"):
                with self.subTest(index=index, field=field):
                    self.reject_change(index, field, True)

    def test_nonobject_rows(self):
        for index in range(32):
            for value in (None, [], "gdn_complete passed", 1, True):
                with self.subTest(index=index, value=value):
                    rows = fixture()
                    rows[index] = value
                    with self.assertRaises(ValueError):
                        self.collect(rows)

    def test_unknown_kinds(self):
        for index in range(32):
            with self.subTest(index=index):
                self.reject_change(index, "kind", "gdn_unknown")

    def test_source_fields(self):
        mutations = {"model": ("other.gguf", "qwen38-full-PLE-Q4_0.gguf", None, True),
                     "layer": (1, False, 0.0, "0", None),
                     "projection": ("GPU", "CPU quantized oracle", "", None),
                     "scope": ("full inference", "", True, None)}
        for field, values in mutations.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.reject_change(0, field, value)

    def test_revision_format(self):
        for value in (None, 1, True, "", "a" * 39, "a" * 41, "g" * 40,
                      "A" * 40, "a" * 40 + "\n", " " + "a" * 39):
            with self.subTest(value=value):
                self.reject_change(0, "revision", value)

    def test_dirty_type_and_value(self):
        for value in (False, True, 0.0, 1.0, -1, 2, "0", "1", None):
            with self.subTest(value=value):
                self.reject_change(0, "dirty", value)

    def test_rms_epsilon(self):
        for value in (0, -1e-6, True, False, "1e-6", None, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.reject_change(0, "rms_epsilon", value)

    def test_device_and_token_geometry(self):
        for index, row in enumerate(fixture()):
            for field in ("device", "tokens"):
                if field not in row:
                    continue
                values = (True, False, float(row[field]), str(row[field]), None, -1, 4, 129,
                          1 - row[field] if field == "device" else 0)
                for value in values:
                    with self.subTest(index=index, field=field, value=value):
                        self.reject_change(index, field, value)

    def test_wrong_supported_token_at_each_position(self):
        for index, row in enumerate(fixture()):
            if "tokens" in row:
                for tokens in (1, 2, 3, 128):
                    if tokens != row["tokens"]:
                        with self.subTest(index=index, tokens=tokens):
                            self.reject_change(index, "tokens", tokens)

    def test_dispatch_a_b_a_paths(self):
        for index, row in enumerate(fixture()):
            if row["kind"] == "gdn_dispatch":
                other = "resident_chunk" if row["path"] == "decode_steps" else "decode_steps"
                for value in (other, "decode", "", 1, True, None):
                    with self.subTest(index=index, value=value):
                        self.reject_change(index, "path", value)

    def test_repeats(self):
        for index, row in enumerate(fixture()):
            if "repeats" in row:
                for value in (19, 21, 0, -20, 20.0, True, "20", None):
                    with self.subTest(index=index, value=value):
                        self.reject_change(index, "repeats", value)

    def test_positive_finite_nonboolean_timings(self):
        for index, row in enumerate(fixture()):
            for field in ("reset_and_resident_ms", "cpu_oracle_ms"):
                if field in row:
                    for value in (0, -1, True, False, "0.1", None, float("nan"), float("inf"), float("-inf")):
                        with self.subTest(index=index, field=field, value=value):
                            self.reject_change(index, field, value)

    def test_nonnegative_finite_nonboolean_errors(self):
        for index, row in enumerate(fixture()):
            for field in ("max_output_error", "max_state_error"):
                if field in row:
                    for value in (-1e-12, True, False, "0", None, float("nan"), float("inf"), float("-inf")):
                        with self.subTest(index=index, field=field, value=value):
                            self.reject_change(index, field, value)

    def test_frozen_gate_ratio_ranges_and_types(self):
        for index, row in enumerate(fixture()):
            for field in ("output_gate_ratio", "state_gate_ratio"):
                if field in row:
                    for value in (-1e-12, 1.000000001, 2, True, False, "0.5", None,
                                  float("nan"), float("inf"), float("-inf")):
                        with self.subTest(index=index, field=field, value=value):
                            self.reject_change(index, field, value)

    def test_history_byte_parity(self):
        for index, row in enumerate(fixture()):
            if "history_byte_parity" in row:
                for value in (False, 0, 1, "true", None):
                    with self.subTest(index=index, value=value):
                        self.reject_change(index, "history_byte_parity", value)

    def test_restore_prefix_cases_windows_and_split_parity(self):
        for index in (14, 29):
            for field, values in (("cases", (0, 4, 7, 9, 8.0, True, "8", None)),
                                  ("windows", (0, 1, 3, 2.0, True, "2", None)),
                                  ("split_chunk_parity", (False, 0, 1, "true", None))):
                for value in values:
                    with self.subTest(index=index, field=field, value=value):
                        self.reject_change(index, field, value)

    def test_rejection_count_atomicity_and_reuse(self):
        for index in (15, 30):
            for field, values in (("rejections", (0, 10, 12, 11.0, True, "11", None)),
                                  ("chunk_atomic", (False, 0, 1, "true", None)),
                                  ("reuse", (False, 0, 1, "true", None))):
                for value in values:
                    with self.subTest(index=index, field=field, value=value):
                        self.reject_change(index, field, value)

    def test_completion_is_exact_true(self):
        for value in (False, 0, 1, "true", None):
            with self.subTest(value=value):
                self.reject_change(31, "passed", value)

    def test_malformed_jsonl_is_not_silently_skipped(self):
        valid = jsonl(fixture())
        for text in ("", "\n" + valid, valid + "\n", valid.replace("\n", "\n \n", 1),
                     "progress: passed\n" + valid, valid + "not JSON\n",
                     valid.replace("\n", " trailing\n", 1),
                     json.dumps(fixture()), "\ufeff" + valid,
                     valid.replace('"dirty": 1', '"dirty": 01', 1)):
            with self.subTest(text=text[:60]):
                self.raw.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    record_gdn.collect(self.raw)

    def test_only_lf_or_crlf_delimits_jsonl_rows(self):
        valid = jsonl(fixture())
        for text in (valid, valid.rstrip("\n"), valid.replace("\n", "\r\n")):
            with self.subTest(valid=text[-40:]):
                with self.raw.open("w", encoding="utf-8", newline="") as output:
                    output.write(text)
                self.assertEqual(record_gdn.collect(self.raw)["kind"], "r2b_gdn")
        for separator in ("\r", "\v", "\f", "\x85", "\u2028", "\u2029"):
            with self.subTest(separator=separator):
                with self.raw.open("w", encoding="utf-8", newline="") as output:
                    output.write(valid.replace("\n", separator))
                with self.assertRaises(ValueError):
                    record_gdn.collect(self.raw)

    def test_json_nonfinite_constants_and_exponent_overflow(self):
        valid = jsonl(fixture())
        for token in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            with self.subTest(token=token):
                text = valid.replace('"rms_epsilon": 1e-06', '"rms_epsilon": ' + token, 1)
                self.assertNotEqual(text, valid)
                self.raw.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    record_gdn.collect(self.raw)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(parse_constant=value), self.assertRaises(ValueError):
                record_gdn.invalid_constant(str(value))

    def test_large_integer_cannot_escape_finiteness_validation(self):
        for index, field in ((0, "rms_epsilon"), (1, "reset_and_resident_ms"),
                             (3, "max_output_error"), (14, "state_gate_ratio")):
            with self.subTest(index=index, field=field):
                self.reject_change(index, field, 10 ** 1000)

    def test_duplicate_json_keys_are_rejected(self):
        valid = jsonl(fixture())
        for original, duplicate in (('"dirty": 1', '"dirty": 0, "dirty": 1'),
                                    ('"dirty": 1', '"dirty": 1, "dirty": 1'),
                                    ('"output_gate_ratio": 0.05', '"output_gate_ratio": 2, "output_gate_ratio": 0.05'),
                                    ('"history_byte_parity": true', '"history_byte_parity": false, "history_byte_parity": true'),
                                    ('"passed": true', '"passed": false, "passed": true')):
            with self.subTest(duplicate=duplicate):
                text = valid.replace(original, duplicate, 1)
                self.assertNotEqual(text, valid)
                self.raw.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    record_gdn.collect(self.raw)

    def test_cli_appends_exactly_one_scoped_record(self):
        self.raw.write_text(jsonl(fixture()), encoding="utf-8")
        prior = '{"kind":"prior","value":7}\n'
        self.results.write_text(prior, encoding="utf-8")
        process = self.run_cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "")
        text = self.results.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(prior))
        lines = text.splitlines()
        self.assertEqual(len(lines), 2)
        result = json.loads(lines[1])
        self.assertEqual(result["kind"], "r2b_gdn")
        self.assertEqual(result["raw_log"], str(self.raw.resolve()))
        self.assertEqual(result["revision"], fixture()[0]["revision"])
        self.assertEqual(result["dirty"], 1)
        self.assertIs(type(result["dirty"]), int)
        self.assertEqual(len(result["measurements"]), 26)
        self.assertEqual(len(result["restore"]), 2)
        self.assertEqual(len(result["rejections"]), 2)
        self.assertNotIn("speedup", result)
        self.assertNotIn("improvement", result)

    def test_cli_creates_one_record(self):
        self.raw.write_text(jsonl(fixture()), encoding="utf-8")
        process = self.run_cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        lines = self.results.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["kind"], "r2b_gdn")

    def test_cli_rejection_does_not_append_or_create_results(self):
        prior = '{"kind":"prior"}\n'
        self.raw.write_text(jsonl(fixture()[:-1]), encoding="utf-8")
        self.results.write_text(prior, encoding="utf-8")
        process = self.run_cli()
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(self.results.read_text(encoding="utf-8"), prior)
        self.results.unlink()
        process = self.run_cli()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(self.results.exists())


if __name__ == "__main__":
    unittest.main()
