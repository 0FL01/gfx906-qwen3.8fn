import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "tools/record_helpers.py"
SPEC = importlib.util.spec_from_file_location("record_helpers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REVISION = "0123456789abcdef0123456789abcdef01234567"
MODEL = "/models/qwen38-keep1-Q4_0.gguf"
TYPE_ORDER = ("Q4_0", "Q4_1", "Q5_0", "Q8_0", "Q6_K")
COUNTERS = {
    "norm_cases": 86, "activation_cases": 5, "mix_cases": 5, "injection_cases": 5,
    "combine_cases": 5, "gate_cases": 8, "conv_cases": 19, "head_cases": 5,
    "broadcast_cases": 56, "signed_zero_cases": 2, "large_finite_cases": 1,
    "prefix_slots": 21, "restore_cases": 10, "repeated_accept0_cases": 4,
    "mask_cases": 2, "readonly_active_checks": 65, "publication_checks": 73,
    "rejected_publications": 46, "numeric_rejects": 46, "host_rejects": 224,
    "readonly_alias_cases": 2, "valid_reuses": 46, "numeric_invalid_cases": 39,
    "numeric_arithmetic_cases": 6, "numeric_combined_cases": 1}
BASE_PARITY = {name: True for name in (
    "cpu_equations", "same_projections", "finite", "byte_canaries", "immutable_inputs")}


def gate(elements):
    return {"max_abs": 1e-5 if elements else 0,
            "max_bound_ratio": .05 if elements else 0, "elements": elements}


def linear_fixture():
    # Independently spell the emitted source/measurement/summary/footer fields.
    actual = [
        {"type": "Q4_0", "tensor": "token_embd.weight", "width": 2560, "rows": 9},
        {"type": "Q4_1", "tensor": "blk.0.ffn_down_exps.weight", "width": 640, "rows": 9},
        {"type": "Q5_0", "tensor": "blk.0.ssm_out.weight", "width": 6144, "rows": 9},
        {"type": "Q8_0", "tensor": "blk.0.hc_attn_down.weight", "width": 10240, "rows": 9},
        {"type": "Q6_K", "tensor": "blk.3.attn_output.weight", "width": 6144, "rows": 9}]
    rows = [{"kind": "linear_source", "revision": REVISION, "dirty": 1, "model": MODEL,
             "absolute_gate": .0002, "relative_gate": .00002,
             "scope": "resident common-Q8 linear; synthetic matrices and first up-to-nine unchanged actual weight rows; not full projection/inference",
             "actual": actual}]
    for device in (0, 1):
        matrices = []
        for kind in TYPE_ORDER:
            widths = (256, 512, 1024, 1280, 2560, 16384) if kind == "Q6_K" else (32, 64, 2048, 2080, 2560, 16384)
            for width, count in zip(widths, (1, 3, 9, 7, 9, 9)):
                matrices.append({"type": kind, "tensor": "synthetic", "width": width, "rows": count})
        matrices.extend(actual)
        for matrix in matrices:
            for n in (1, 2, 3):
                rows.append({"kind": "linear_measurement", "device": device, **matrix,
                             "columns": n, "repeats": 20, "resident_ms": .012345,
                             "max_abs": 1e-5, "max_bound_ratio": .05,
                             "q8_byte_parity": True, "canary": True, "immutable": True})
        rows.append({"kind": "linear_correctness", "device": device, "matrix_cases": 110,
                     "quantization_cases": 123, "numeric_rejects": 5, "host_rejects": 32,
                     "corner_cases": 5, "passed": True})
    rows.append({"kind": "linear_complete", "measurements": 210, "rows": 214, "passed": True})
    return rows


def blocks_fixture():
    tensors = []
    offset = 1024
    specs = [
        ("blk.0.hc_attn_norm.weight", "F32", [10240]),
        ("blk.0.hc_ffn_norm.weight", "F32", [10240]),
        ("output_hc_norm.weight", "F32", [10240]),
        ("blk.1.ple_norm_key.weight", "F32", [10240]),
        ("blk.1.ple_norm_query.weight", "F32", [10240]),
        ("blk.1.ple_norm_conv.weight", "F32", [10240]),
        ("blk.1.ple_conv1d.weight", "F16", [4, 10240]),
        ("blk.0.ssm_norm.weight", "F32", [128]),
        ("blk.3.indexer.q_norm.weight", "F32", [128]),
        ("blk.3.indexer.k_norm.weight", "F32", [128]),
        ("blk.3.attn_q_norm.weight", "F32", [256]),
        ("blk.3.attn_k_norm.weight", "F32", [256])]
    for name, kind, dims in specs:
        size, elements, strides = (4 if kind == "F32" else 2), 1, []
        for dim in dims:
            strides.append(size)
            size *= dim
            elements *= dim
        tensors.append({"name": name, "type": kind, "rank": len(dims), "dimensions": dims,
                        "strides_bytes": strides, "elements": elements, "bytes": size,
                        "relative_offset": offset, "file_offset": 4096 + offset})
        offset += size
    rows = [{
        "kind": "blocks_source", "protocol": 1,
        "scope": "isolated_resident_fp32_primitives_same_synthetic_projections_actual_gamma_conv",
        "revision": REVISION, "dirty": True, "build_provenance_complete": True,
        "model_path": MODEL, "model_file_bytes": 75399121792,
        "actual_metadata": {"architecture": "qwen4exp", "hidden": 2560, "branches": 4,
                            "rank": 320, "rms_epsilon": 9.999999974752427e-7,
                            "rms_epsilon_type": "FLOAT32", "ple_layer": 1, "ple_embedding_dim": 2560,
                            "ple_conv_kernel": 4, "ple_ngram_size": 3, "ple_history_length": 9},
        "actual_tensors": tensors,
        "devices": [{"device": d, "arch": "gfx906:sramecc+:xnack-", "wave_size": 64} for d in (0, 1)],
        "gates": {"frozen_before_gpu": True, "absolute": .0002, "relative_cpu_magnitude": .0002,
                  "groups": ["rms", "hc", "ple", "history"], "max_bound_ratio_limit": 1,
                  "finite_required": True, "exact_history": "identical_norm_gated_inputs_only"},
        "projection_inputs": {"source": "fixed_finite_synthetic_same_CPU_GPU_arrays", "immutable": True,
                              "names": ["raw_down", "raw_up", "raw_inject", "raw_key", "raw_query", "shared_value"],
                              "actual_matrix_multiplication": False, "block_output": "synthetic"},
        "cpu_equations": "independent_ascending_FP32_direct_stored_gamma_unfused",
        "history_modes": {"isolated_conv": "exact_common_norm_gated", "PLE_pipeline": "bounded_own_parallel_RMS"},
        "transaction_scope": "conv_only_owner_publication_marker_no_hash_or_EOS_reset",
        "token_counts": [1, 2, 3, 128], "repeats": 20,
        "row_counts": {"source": 1, "correctness": 2, "measurement": 16, "complete": 1, "total": 20},
        "order": {"correctness_devices": [0, 1], "measurement_outer": "device",
                  "measurement_middle": "tokens", "measurement_inner": ["HC", "PLE"]},
        "correctness_exact_counters": copy.deepcopy(COUNTERS),
        "correctness_positive_counters": ["immutable_checks", "canary_checks"],
        "host_validation": {"expected_status": "hipErrorInvalidValue",
                            "checks": ["geometry", "epsilon", "null_required", "all_writable_overlap", "misalignment", "wrapping"],
                            "invalid_addresses_enqueued": False},
        "timing_scope": "individual_completed_HIP_event_intervals_resident_primitives_only",
        "launch_counts": {"HC": 5, "PLE": 7},
        "full_pipeline_qualification": False, "request_performance_claim": False}]
    for device in (0, 1):
        rows.append({
            "kind": "blocks_correctness", "protocol": 1, "device": device,
            "counters": {**COUNTERS, "immutable_checks": 1024, "canary_checks": 2048},
            "error": {"rms": gate(19297536), "hc": gate(3967500),
                      "ple": gate(6041600), "history": gate(12165120)},
            "parity": {**BASE_PARITY, **{name: True for name in (
                "root_head_no_injection", "original_widened_tap_unchanged", "history_exact_common_inputs",
                "history_bounded_own_parallel_rms", "all_four_prefix_tails", "restore_1_plus_accepted",
                "repeated_accept0", "keep_mask_0_1_no_EOS_reset", "active_readonly_until_publish",
                "rejected_entire_active_and_marker_unchanged", "valid_reuse")}}, "passed": True})
    for device in (0, 1):
        for n in (1, 2, 3, 128):
            for pipeline in ("HC", "PLE"):
                ple = pipeline == "PLE"
                rows.append({
                    "kind": "blocks_measurement", "protocol": 1, "device": device, "tokens": n,
                    "pipeline": pipeline, "repeats": 20, "validated_repetitions": 20,
                    "resident_ms": .012345, "launchcount": 7 if ple else 5,
                    "timing_excludes": ["model_load", "fixture_allocation", "CPU_reference", "H2D", "D2H",
                                        "history_restore", "output_reset", "error_clear", "validation", "warmup"],
                    "error": {"rms": gate(20 * n * (30720 if ple else 10240)),
                              "hc": gate(0 if ple else 20 * n * 13124),
                              "ple": gate(20 * n * 20480 if ple else 0),
                              "history": gate(3686400 if ple else 0)},
                    "counters": {**dict.fromkeys(COUNTERS, 0), "publication_checks": 20 if ple else 0,
                                 "immutable_checks": 260, "canary_checks": 1220},
                    "history_comparison": "bounded_own_parallel_RMS" if ple else "not_applicable",
                    "parity": {**BASE_PARITY, "history_bounded_own_parallel_rms": True if ple else None,
                               "passed": True}})
    rows.append({"kind": "blocks_complete", "protocol": 1, "correctness_rows": 2,
                 "measurement_rows": 16, "measurement_repetitions": 320,
                 "jsonl_rows": 20, "passed": True})
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


class HelpersResultsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.paths = {"linear": self.directory / "linear.jsonl", "blocks": self.directory / "blocks.jsonl"}
        self.results = self.directory / "results.jsonl"
        self.rows = {"linear": linear_fixture(), "blocks": blocks_fixture()}
        self.write(self.rows)

    def write(self, rows):
        for name in self.paths:
            self.paths[name].write_text(jsonl(rows[name]), encoding="utf-8")

    def collect(self, rows=None):
        if rows is not None:
            self.write(rows)
        return MODULE.collect(self.paths["linear"], self.paths["blocks"])

    def reject_change(self, driver, index, path, value):
        rows = copy.deepcopy(self.rows)
        target(rows[driver][index], path[:-1])[path[-1]] = value
        with self.assertRaises(ValueError):
            self.collect(rows)

    def run_cli(self):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--linear", self.paths["linear"].name,
                               "--blocks", self.paths["blocks"].name, "--results", self.results.name],
                              cwd=self.directory, capture_output=True, text=True)

    def test_complete_record_exact_schema_and_all_raw_records(self):
        result = self.collect()
        self.assertEqual(set(result), {"kind", "timestamp", "revision", "dirty", "raw_logs", "scope",
                                      "timing_scope", "linear_contract", "blocks_contract", "fixture_gates",
                                      "donors", "linear", "blocks", "passed"})
        self.assertEqual(result["kind"], "r2e_helpers")
        self.assertEqual(result["revision"], REVISION)
        self.assertEqual(result["dirty"], 1)
        self.assertIs(type(result["dirty"]), int)
        self.assertIs(result["passed"], True)
        self.assertEqual(len(self.rows["linear"]), 214)
        self.assertEqual(len(self.rows["blocks"]), 20)
        self.assertEqual(result["raw_logs"], {name: str(path.resolve()) for name, path in self.paths.items()})
        for name in ("linear", "blocks"):
            saved = result[name]
            self.assertEqual(set(saved), {"source", "correctness", "measurements", "complete"})
            self.assertEqual(saved["source"], self.rows[name][0])
            self.assertEqual(saved["complete"], self.rows[name][-1])
            self.assertEqual(saved["correctness"], [row for row in self.rows[name] if row["kind"] == name + "_correctness"])
            self.assertEqual(saved["measurements"], [row for row in self.rows[name] if row["kind"] == name + "_measurement"])
        for device in (0, 1):
            self.assertEqual(sum(r["device"] == device for r in result["linear"]["measurements"]), 105)
            self.assertEqual(sum(r["device"] == device for r in result["blocks"]["measurements"]), 8)
        json.dumps(result, allow_nan=False)

    def test_compiled_clean_dirty_provenance_preserved(self):
        for dirty in (0, 1):
            rows = copy.deepcopy(self.rows)
            rows["linear"][0].update(revision="b" * 40, dirty=dirty)
            rows["blocks"][0].update(revision="b" * 40, dirty=bool(dirty))
            result = self.collect(rows)
            self.assertEqual(result["revision"], "b" * 40)
            self.assertIs(type(result["dirty"]), int)
            self.assertEqual(result["dirty"], dirty)
            self.assertIs(type(result["linear"]["source"]["dirty"]), int)
            self.assertIs(type(result["blocks"]["source"]["dirty"]), bool)

    def test_scopes_gates_contracts_and_source_pins(self):
        result = self.collect()
        self.assertIn("diagnostic component", result["scope"])
        self.assertIn("not whole projections", result["scope"])
        self.assertIn("inference performance", result["scope"])
        self.assertIn("20 individually completed", result["timing_scope"])
        self.assertIn("every repetition validated", result["timing_scope"])
        lc, bc = result["linear_contract"], result["blocks_contract"]
        self.assertIn("36 bytes", lc["activation_abi"])
        self.assertIn("raw input sum", lc["activation_abi"])
        self.assertIn("positive-zero", lc["zero"])
        self.assertIn("half zero is rejected", lc["tiny"])
        self.assertIn("FP32", lc["codes"])
        self.assertIn("not dequantized-F32", lc["reference"])
        self.assertEqual(set(lc["arithmetic"]), set(TYPE_ORDER))
        self.assertIn("half_RNE", lc["arithmetic"]["Q4_1"])
        self.assertIn("ONLY identical", bc["cpu_history_byte_parity"])
        self.assertIn("bounded", bc["cpu_history_byte_parity"])
        self.assertIn("byte-identical", bc["readonly_tap"])
        self.assertEqual(bc["prefix_abi"]["restore_slot"], "1+accepted")
        self.assertEqual(bc["prefix_abi"]["accepted_drafts"], [0, 1, 2])
        self.assertEqual(result["fixture_gates"]["linear"], {
            "absolute": .0002, "relative_cpu_magnitude": .00002,
            "frozen_before_gpu": True, "max_bound_ratio_limit": 1})
        self.assertEqual(result["fixture_gates"]["blocks"], self.rows["blocks"][0]["gates"])
        self.assertEqual(result["donors"], {
            "mx": "dcd685463d597d31f5ca759d32c94592a2740fa4",
            "furnace": "905021dbad71c5056ef51f9fd45d545403fc989c",
            "reinstinct": "0b79e326351d90d4554a1c18df92da5d0ab692e8"})

    def test_all_missing_rows_and_truncated_prefixes(self):
        for name in self.rows:
            for index in range(len(self.rows[name])):
                with self.subTest(driver=name, missing=index):
                    rows = copy.deepcopy(self.rows)
                    rows[name].pop(index)
                    with self.assertRaises(ValueError):
                        self.collect(rows)
            for length in (0, 1, 2, 3, len(self.rows[name]) // 2):
                rows = copy.deepcopy(self.rows)
                rows[name] = rows[name][:length]
                with self.assertRaises(ValueError):
                    self.collect(rows)

    def test_all_adjacent_reorderings(self):
        for name in self.rows:
            for index in range(len(self.rows[name]) - 1):
                with self.subTest(driver=name, index=index):
                    rows = copy.deepcopy(self.rows)
                    rows[name][index], rows[name][index + 1] = rows[name][index + 1], rows[name][index]
                    with self.assertRaises(ValueError):
                        self.collect(rows)

    def test_extra_duplicate_and_replaced_rows(self):
        for name in self.rows:
            for index in (0, 1, len(self.rows[name]) // 2, len(self.rows[name]) - 1):
                rows = copy.deepcopy(self.rows)
                rows[name].insert(index, copy.deepcopy(rows[name][index]))
                with self.assertRaises(ValueError):
                    self.collect(rows)
            for index in range(1, len(self.rows[name])):
                rows = copy.deepcopy(self.rows)
                rows[name][index] = copy.deepcopy(rows[name][index - 1])
                with self.assertRaises(ValueError):
                    self.collect(rows)

    def test_footer_completion_and_partial_complete(self):
        for name in self.rows:
            for field, value in self.rows[name][-1].items():
                for bad in (None, False, "complete", 0, 999):
                    with self.subTest(driver=name, field=field, bad=bad):
                        self.reject_change(name, -1, (field,), bad)
            rows = copy.deepcopy(self.rows)
            rows[name][-1] = {"kind": name + "_complete", "passed": True}
            with self.assertRaises(ValueError):
                self.collect(rows)

    def test_missing_and_extra_nested_schema_fields(self):
        for name, indices in (("linear", (0, 1, 91, 106, 213)), ("blocks", (0, 1, 3, 4, 19))):
            for index in indices:
                for path, obj in objects(self.rows[name][index]):
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

    def test_nonobject_blank_and_nonjson_rows(self):
        for name in self.rows:
            for value in (None, [], "passed", 1, True):
                rows = copy.deepcopy(self.rows)
                rows[name][1] = value
                with self.assertRaises(ValueError):
                    self.collect(rows)
            for line in ("", " ", "diagnostic text", "{", "{} {}"):
                self.write(self.rows)
                lines = jsonl(self.rows[name]).splitlines()
                lines[1] = line
                self.paths[name].write_text("\n".join(lines) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.collect()

    def test_invalid_utf8_bom_and_unterminated_final_row(self):
        for name in self.rows:
            encoded = jsonl(self.rows[name]).encode()
            for raw in (b"\xef\xbb\xbf" + encoded, encoded.replace(b"synthetic", b"\xff", 1),
                        encoded[:-1], encoded[:-8], encoded + b"\n"):
                self.write(self.rows)
                self.paths[name].write_bytes(raw)
                with self.assertRaises(ValueError):
                    self.collect()

    def test_duplicate_json_keys_top_level_and_nested(self):
        for name in self.rows:
            raw = jsonl(self.rows[name])
            replacements = [('"revision":', '"revision":"ignored","revision":'),
                            ('"max_abs":', '"max_abs":0,"max_abs":')]
            for before, after in replacements:
                self.write(self.rows)
                self.paths[name].write_text(raw.replace(before, after, 1), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.collect()

    def test_nan_inf_and_overflow_json_literals(self):
        for name in self.rows:
            for literal in ("NaN", "Infinity", "-Infinity", "1e309", "-1e309"):
                self.write(self.rows)
                raw = jsonl(self.rows[name]).replace('"resident_ms": 0.012345', '"resident_ms": ' + literal, 1)
                self.paths[name].write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.collect()

    def test_raw_size_limit_including_exact_boundary(self):
        for name in self.rows:
            raw = jsonl(self.rows[name]).encode()
            bound = 4 * 1024 * 1024
            self.write(self.rows)
            self.paths[name].write_bytes(b" " * (bound - len(raw)) + raw)
            self.assertTrue(self.collect()["passed"])
            self.paths[name].write_bytes(b" " * (bound + 1 - len(raw)) + raw)
            with self.assertRaises(ValueError):
                self.collect()

    def test_boolean_never_a_numeric_field(self):
        for name, indices in (("linear", (0, 1, 106, 213)), ("blocks", (0, 1, 3, 4, 19))):
            for index in indices:
                for path, value in leaves(self.rows[name][index]):
                    if type(value) in (int, float):
                        with self.subTest(driver=name, index=index, path=path):
                            self.reject_change(name, index, path, True)

    def test_boolean_flags_must_be_booleans(self):
        for name, indices in (("linear", (1, 106, 213)), ("blocks", (0, 1, 3, 4, 19))):
            for index in indices:
                for path, value in leaves(self.rows[name][index]):
                    if type(value) is bool:
                        self.reject_change(name, index, path, int(value))

    def test_compiled_source_invalid_and_mismatched(self):
        for name in self.rows:
            for value in (None, True, 1, "", "dirty", "a" * 39, "g" * 40, "A" * 40):
                self.reject_change(name, 0, ("revision",), value)
        for value in (True, False, None, -1, 2, 1.0, "1"):
            self.reject_change("linear", 0, ("dirty",), value)
        for value in (None, 0, 1, "true"):
            self.reject_change("blocks", 0, ("dirty",), value)
        self.reject_change("blocks", 0, ("build_provenance_complete",), False)
        self.reject_change("blocks", 0, ("revision",), "b" * 40)
        self.reject_change("blocks", 0, ("dirty",), False)
        self.reject_change("linear", 0, ("dirty",), 0)

    def test_model_paths_and_source_scopes(self):
        for name, field in (("linear", "model"), ("blocks", "model_path")):
            for value in (None, True, "", "path\n", "\ud800", "/models/other.gguf"):
                self.reject_change(name, 0, (field,), value)
            self.reject_change(name, 0, ("scope",), "whole inference")

    def test_fixed_source_gates_and_proofs(self):
        for field in ("absolute_gate", "relative_gate"):
            for value in (0, .001, True, None):
                self.reject_change("linear", 0, (field,), value)
        for path, value in leaves(self.rows["blocks"][0]["gates"]):
            self.reject_change("blocks", 0, ("gates",) + path, False if type(value) is bool else "changed")
        for path in (("projection_inputs", "actual_matrix_multiplication"),
                     ("full_pipeline_qualification",), ("request_performance_claim",)):
            self.reject_change("blocks", 0, path, True)
        for path in (("history_modes", "PLE_pipeline"), ("transaction_scope",), ("cpu_equations",)):
            self.reject_change("blocks", 0, path, "exact_history_with_different_inputs")

    def test_two_gfx906_wave64_device_proof(self):
        for value in ([], self.rows["blocks"][0]["devices"][:1], None):
            self.reject_change("blocks", 0, ("devices",), value)
        for device in (0, 1):
            for arch in ("gfx908", "gfx9060", "gfx906_fake", "gfx906suffix", None):
                self.reject_change("blocks", 0, ("devices", device, "arch"), arch)
            self.reject_change("blocks", 0, ("devices", device, "wave_size"), 32)
            self.reject_change("blocks", 0, ("devices", device, "device"), 1 - device)
        rows = copy.deepcopy(self.rows)
        for proof in rows["blocks"][0]["devices"]:
            proof["arch"] = "gfx906"
        self.assertTrue(self.collect(rows)["passed"])

    def test_linear_type_width_rows_column_and_actual_match(self):
        for index in (1, 19, 73, 91, 107, 197):
            for field, value in (("type", "F32"), ("width", 33), ("rows", 8),
                                 ("columns", 4), ("device", 2), ("tensor", "wrong"), ("repeats", 19)):
                self.reject_change("linear", index, (field,), value)
        for index in range(5):
            for field, value in (("width", 0), ("width", 16416), ("width", 33), ("rows", 0),
                                 ("rows", 10), ("tensor", "synthetic"), ("tensor", ""), ("type", "F32")):
                self.reject_change("linear", 0, ("actual", index, field), value)
        self.reject_change("linear", 0, ("actual", 4, "width"), 32)
        self.reject_change("linear", 0, ("actual",), self.rows["linear"][0]["actual"][:4])
        self.reject_change("linear", 0, ("actual", 1, "tensor"), "token_embd.weight")
        # Source itself is shape-valid, but actual measurement rows must match it.
        self.reject_change("linear", 0, ("actual", 0, "width"), 2048)
        self.reject_change("linear", 0, ("actual", 0, "rows"), 3)

    def test_actual_linear_geometry_is_source_driven(self):
        rows = copy.deepcopy(self.rows)
        rows["linear"][0]["actual"][0].update(width=32, rows=1, tensor="actual.changed.weight")
        for index in (91, 92, 93, 197, 198, 199):
            rows["linear"][index].update(width=32, rows=1, tensor="actual.changed.weight")
        self.assertTrue(self.collect(rows)["passed"])

    def test_actual_blocks_metadata_and_tensor_geometry(self):
        for field in ("hidden", "branches", "rank", "ple_layer", "ple_embedding_dim",
                      "ple_conv_kernel", "ple_ngram_size", "ple_history_length"):
            self.reject_change("blocks", 0, ("actual_metadata", field),
                               self.rows["blocks"][0]["actual_metadata"][field] + 1)
        for field in ("architecture", "rms_epsilon_type"):
            self.reject_change("blocks", 0, ("actual_metadata", field), "wrong")
        for value in (0, -1, True, None, float("nan"), float("inf")):
            self.reject_change("blocks", 0, ("actual_metadata", "rms_epsilon"), value)
        for index, tensor in enumerate(self.rows["blocks"][0]["actual_tensors"]):
            for field, value in (("name", "wrong"), ("type", "BF16"), ("rank", 3),
                                 ("dimensions", [1]), ("strides_bytes", [1]),
                                 ("elements", tensor["elements"] + 1), ("bytes", tensor["bytes"] + 1)):
                self.reject_change("blocks", 0, ("actual_tensors", index, field), value)
        self.reject_change("blocks", 0, ("actual_tensors",), self.rows["blocks"][0]["actual_tensors"][:11])

    def test_actual_tensor_offsets_file_bounds_origin_and_overlap(self):
        for field in ("relative_offset", "file_offset"):
            for value in (-1, 1.0, True, 2**64, None):
                self.reject_change("blocks", 0, ("actual_tensors", 0, field), value)
        self.reject_change("blocks", 0, ("actual_tensors", 0, "file_offset"), 512)
        self.reject_change("blocks", 0, ("actual_tensors", 0, "file_offset"), 5121)
        for value in (0, True, -1, 5000, 2**64):
            self.reject_change("blocks", 0, ("model_file_bytes",), value)
        rows = copy.deepcopy(self.rows)
        first, second = rows["blocks"][0]["actual_tensors"][:2]
        second.update(relative_offset=first["relative_offset"], file_offset=first["file_offset"])
        with self.assertRaises(ValueError):
            self.collect(rows)

    def test_linear_exact_correctness_counters(self):
        for index in (106, 212):
            for field in ("matrix_cases", "quantization_cases", "numeric_rejects", "host_rejects", "corner_cases"):
                self.reject_change("linear", index, (field,), self.rows["linear"][index][field] + 1)

    def test_blocks_exact_counters_in_source_and_both_correctness_records(self):
        for field, expected in COUNTERS.items():
            for index, path in ((0, "correctness_exact_counters"), (1, "counters"), (2, "counters")):
                self.reject_change("blocks", index, (path, field), expected - 1)
        # A mutually edited source and summary must not redefine the frozen gate.
        rows = copy.deepcopy(self.rows)
        rows["blocks"][0]["correctness_exact_counters"]["prefix_slots"] = 20
        for index in (1, 2):
            rows["blocks"][index]["counters"]["prefix_slots"] = 20
        with self.assertRaises(ValueError):
            self.collect(rows)

    def test_blocks_positive_immutable_and_canary_counters(self):
        for index in (1, 2, 3, 4, 17, 18):
            for field in ("immutable_checks", "canary_checks"):
                for value in (0, -1, False, 1.0, None, float("inf")):
                    self.reject_change("blocks", index, ("counters", field), value)

    def test_blocks_measurement_counters_launches_repetitions_and_history_mode(self):
        for index in range(3, 19):
            row = self.rows["blocks"][index]
            for field in ("tokens", "launchcount", "repeats", "validated_repetitions"):
                self.reject_change("blocks", index, (field,), row[field] + 1)
            self.reject_change("blocks", index, ("history_comparison",), "exact_CPU_bytes")
            for field, value in row["counters"].items():
                if field not in ("immutable_checks", "canary_checks"):
                    self.reject_change("blocks", index, ("counters", field), value + 1)

    def test_all_applicable_parities_and_prefix_publication_proof(self):
        for index in (1, 2, 3, 4):
            for field, expected in self.rows["blocks"][index]["parity"].items():
                values = (True, False) if expected is None else (False, None, 1)
                for value in values:
                    self.reject_change("blocks", index, ("parity", field), value)
        for index in (1, 91, 107, 197):
            for field in ("q8_byte_parity", "canary", "immutable"):
                for value in (False, None, 1):
                    self.reject_change("linear", index, (field,), value)

    def test_finite_positive_completed_timings(self):
        for name, indices in (("linear", (1, 91, 107, 211)), ("blocks", (3, 4, 17, 18))):
            for index in indices:
                for value in (0, -1, True, None, "0.01", float("nan"), float("inf"), -float("inf")):
                    self.reject_change(name, index, ("resident_ms",), value)

    def test_structured_error_bounds_elements_and_unused_groups(self):
        for name, indices in (("linear", (1, 91)), ("blocks", (1, 2, 3, 4))):
            for index in indices:
                paths = [()] if name == "linear" else [("error", group) for group in ("rms", "hc", "ple", "history")]
                for path in paths:
                    for field in ("max_abs", "max_bound_ratio"):
                        for value in (-1, True, None, float("nan"), float("inf")):
                            self.reject_change(name, index, path + (field,), value)
                    self.reject_change(name, index, path + ("max_bound_ratio",), 1.0000001)
                    if name == "blocks":
                        expected = target(self.rows[name][index], path)["elements"]
                        for value in (expected + 1, -1, True, 1.0, float("inf")):
                            self.reject_change(name, index, path + ("elements",), value)
        self.reject_change("blocks", 3, ("error", "history", "max_abs"), .001)
        self.reject_change("blocks", 4, ("error", "hc", "max_bound_ratio"), .1)

    def test_error_ratio_boundaries_and_integer_times(self):
        for ratio in (0, 1):
            rows = copy.deepcopy(self.rows)
            for name in rows:
                for row in rows[name]:
                    if "resident_ms" in row:
                        row["resident_ms"] = 1
                    if name == "linear" and row["kind"] == "linear_measurement":
                        row.update(max_abs=0, max_bound_ratio=ratio)
                    elif "error" in row:
                        for error in row["error"].values():
                            error.update(max_abs=0, max_bound_ratio=ratio if error["elements"] else 0)
            self.assertTrue(self.collect(rows)["passed"])

    def test_crlf_and_unicode_paths_are_preserved(self):
        rows = copy.deepcopy(self.rows)
        model = '/models/модель "quoted".gguf'
        rows["linear"][0]["model"] = rows["blocks"][0]["model_path"] = model
        for name in self.paths:
            self.paths[name].write_bytes(jsonl(rows[name]).replace("\n", "\r\n").encode())
        self.assertEqual(self.collect()["blocks"]["source"]["model_path"], model)

    def test_cli_appends_exactly_one_complete_record(self):
        self.results.write_text('{"kind":"existing"}\n', encoding="utf-8")
        process = self.run_cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        saved = [json.loads(line) for line in self.results.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0], {"kind": "existing"})
        self.assertEqual(saved[1]["kind"], "r2e_helpers")
        self.assertEqual(saved[1]["revision"], REVISION)
        self.assertEqual(saved[1]["raw_logs"], {name: str(path.resolve()) for name, path in self.paths.items()})
        self.assertEqual(len(saved[1]["linear"]["measurements"]), 210)
        self.assertEqual(len(saved[1]["blocks"]["measurements"]), 16)

    def test_cli_invalid_logs_preserve_destination_bytes(self):
        existing = b'{"kind":"existing"}\n'
        self.results.write_bytes(existing)
        cases = [("linear", -1, ("passed",), False), ("blocks", -1, ("measurement_repetitions",), 319),
                 ("blocks", 0, ("revision",), "b" * 40), ("linear", 1, ("canary",), False),
                 ("blocks", 1, ("parity", "all_four_prefix_tails"), False)]
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

    def test_cli_invalid_does_not_create_results_file(self):
        rows = copy.deepcopy(self.rows)
        rows["blocks"][-1] = {"kind": "blocks_complete", "passed": True}
        self.write(rows)
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertFalse(self.results.exists())


if __name__ == "__main__":
    unittest.main()
