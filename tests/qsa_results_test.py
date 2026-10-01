import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "tools/record_qsa.py"
SPEC = importlib.util.spec_from_file_location("record_qsa", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def error(elements):
    return {"max_abs": 1e-5, "max_bound_ratio": .05, "elements": elements}


def fixture():
    # Exact emitted fields in qsa_main.hip source_json/summary_json/*_measurements.
    # FLOAT32 epsilon is promoted to double for setprecision(17) JSON output.
    rows = [{
        "kind": "qsa_source", "protocol": 1,
        "revision": "0123456789abcdef0123456789abcdef01234567", "dirty": True,
        "model_path": "/models/qwen38-keep1-Q4_0.gguf", "model_variant": "qwen38-keep1-Q4_0",
        "layer": 3,
        "scope": "fixture_only_loaded_indexer_weights_CPU_raw_FP32_projection_of_eight_synthetic_columns_repeated_in_cache_timeline; synthetic_attention_KV; not_prompt_or_full_QSA_block_or_inference",
        "synthetic_x": "FP32_0.3*sin(i*0.137)+0.1*cos(i*0.071); eight_2560_element_columns",
        "projection_types": {"q_proj": "BF16[2560,512]", "k_proj": "BF16[2560,128]",
                             "q_norm": "F32[128]", "k_norm": "F32[128]"},
        "metadata_types": {"rope.dimension_count": "UINT32", "rope.freq_base": "FLOAT32",
                           "attention.layer_norm_rms_epsilon": "FLOAT32",
                            "rope.dimension_sections": "ARRAY_INT32"},
        "config": {"capacity": 131072, "index_D": 128, "index_Q": 4, "index_KV": 1,
                   "compress": 4, "budget": 2048, "rotary_dim": 64, "rope_base": 10000000,
                   "rms_epsilon": 9.9999999747524271e-7, "rope_scale": 1,
                   "sections": [11, 11, 10, 0], "text_position_axes": "identical_split_half",
                   "frequencies": "FP32_1/pow(base,2*i/64); bit_identical_CPU_and_uploaded_GPU",
                   "pooled_cache_bytes_per_device": 16777216,
                   "Q4_cache_bytes_each_per_device": 37748736},
        "gates": {"key": {"abs": .0002, "rel": .0002}, "score": {"abs": .0002, "rel": .0002},
                  "attention": {"abs": .0002, "rel": .0002},
                  "bound": "abs+rel*abs(CPU_reference)", "max_bound_ratio_limit": 1},
        "correctness_before_any_metrics": True, "validate_after_each_timed_repeat": True,
        "event_protocol": "20_individual_completed_GPU_event_intervals; arithmetic_mean_ms; reset_restore_and_flag_clear_outside_events",
        "rows": {"qsa_source": 1, "qsa_correctness": 2, "qsa_append": 32, "qsa_score": 6,
                 "qsa_select": 8, "qsa_attention": 12, "qsa_complete": 1, "total": 62}}]
    for device in (0, 1):
        rows.append({
            "kind": "qsa_correctness", "device": device, "cache_cases": 32,
            "prefix_checks": 648, "restore_cases": 24, "reject_windows": 8, "q4_blocks": 8,
            "append_rejects": 6, "selection_checks": 200, "count_checks": 200,
            "selection_rejects": 3, "score_checks": 60, "score_rejects": 2,
            "attention_checks": 26, "attention_rejects": 11,
            "key_error": error(8451328), "common_score_error": error(47157),
            "actual_score_error": error(96362), "attention_error": error(233472),
            "q4_byte_parity": True, "q4_dequant_bit_parity": True, "tail_byte_parity": True,
            "unused_id_canaries": True, "cache_readonly_byte_parity": True, "chunk_atomic": True,
            "sticky_reuse": True, "future_poison_unread": True, "passed": True})
    for device in (0, 1):
        for phase in range(4):
            for n in (1, 2, 3, 128):
                base = 4096 + phase
                rows.append({
                    "kind": "qsa_append", "device": device, "N": n, "base": base,
                    "phase": phase, "visible": base + n, "repeats": 20,
                    "launches": 4 if phase + n >= 4 else 3, "resident_ms": .03,
                    "scope": "resident_raw_K_q4_roundtrip_new_pool_norm_rope_publish; no_prefix_snapshots",
                    "timing_excludes": "projection,H2D,D2H,reset,tail_restore,error_clear,validation",
                    "key_error": error(20 * ((base + n) // 4) * 128),
                    "completed_blocks": (base + n) // 4, "new_blocks": (phase + n) // 4,
                    "tail_byte_parity": True, "prefix_byte_parity": True})
        for visible in (4096, 32768, 131072):
            rows.append({
                "kind": "qsa_score", "device": device, "visible": visible, "prepared": 131072,
                "repeats": 20, "launches": 2, "resident_ms": .04,
                "scope": "resident_raw_Q_norm_rope_and_causal_score; two_launches; actual_GPU_keys",
                "timing_excludes": "projection,H2D,D2H,append,error_clear,validation",
                "completed_blocks": visible // 4, "score_error": error(20 * (visible // 4)),
                "future_output_canary": True})
        for visible in (2052, 4096, 32768, 131072):
            rows.append({
                "kind": "qsa_select", "device": device, "visible": visible, "repeats": 20,
                "launches": 19, "resident_ms": .05,
                "scope": "resident_selection_only; nineteen_launches; random_fixed_same_score_floats",
                "timing_excludes": "projection,H2D,D2H,index_score,error_clear,validation",
                "token_count": 2048, "block_count": 512, "ids_exact": True,
                "counts_exact": True, "unused_capacity_byte_parity": True})
        for n in (1, 63, 64, 65, 2048, 2051):
            rows.append({
                "kind": "qsa_attention", "device": device, "visible": 4096, "capacity": 131072,
                "selected": n, "repeats": 20, "launches": 6, "resident_ms": .06,
                "scope": "resident_selected_Q4_gather_FP16_and_attention_publish; Q24_KV2_D256",
                "cache_source": "synthetic_canonical_Q4; all_unselected_scales_Inf",
                "reference": "CPU_gathered_fp16_same_ID_order_and_repeats",
                "timing_excludes": "projection,Hadamard,RoPE,gate,index_select,H2D,D2H,error_clear,validation",
                "token_count": n, "block_count": min(512, n // 4),
                "attention_error": error(20 * 6144),
                "cache_readonly_byte_parity": True, "output_canaries": True})
    rows.append({"kind": "qsa_complete", "protocol": 1, "rows": 62, "devices": 2, "passed": True})
    return rows


def jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


def leaves(value, path=()):
    if type(value) is dict:
        for key, child in value.items():
            yield from leaves(child, path + (key,))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from leaves(child, path + (index,))
    else:
        yield path, value


def containers(value, path=()):
    if type(value) is dict:
        yield path, value
        for key, child in value.items():
            yield from containers(child, path + (key,))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from containers(child, path + (index,))


def target(value, path):
    for key in path:
        value = value[key]
    return value


class QsaResultsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.raw = self.directory / "qsa.jsonl"
        self.results = self.directory / "results.jsonl"
        self.rows = fixture()

    def collect(self, rows):
        self.raw.write_text(jsonl(rows), encoding="utf-8")
        return MODULE.collect(self.raw)

    def reject_change(self, index, path, value):
        rows = copy.deepcopy(self.rows)
        target(rows[index], path[:-1])[path[-1]] = value
        with self.assertRaises(ValueError):
            self.collect(rows)

    def run_cli(self):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", self.raw.name,
                               "--results", self.results.name], cwd=self.directory,
                              capture_output=True, text=True)

    def test_complete_protocol_preserves_all_raw_fields(self):
        result = self.collect(self.rows)
        self.assertEqual(len(self.rows), 62)
        self.assertEqual(result["kind"], "r2d_qsa")
        self.assertEqual(result["raw_log"], str(self.raw.resolve()))
        self.assertEqual(result["source"], self.rows[0])
        self.assertEqual(result["correctness"], self.rows[1:3])
        self.assertEqual(result["measurements"], self.rows[3:61])
        self.assertEqual(result["complete"], self.rows[61])
        self.assertIs(result["passed"], True)
        self.assertEqual([r["device"] for r in result["correctness"]], [0, 1])
        for device in (0, 1):
            metrics = [r for r in result["measurements"] if r["device"] == device]
            self.assertEqual(len(metrics), 29)
            for kind, count in (("qsa_append", 16), ("qsa_score", 3),
                                ("qsa_select", 4), ("qsa_attention", 6)):
                self.assertEqual(sum(r["kind"] == kind for r in metrics), count)
        json.dumps(result, allow_nan=False)

    def test_compiled_provenance_is_preserved_for_clean_and_dirty(self):
        for dirty in (False, True):
            with self.subTest(dirty=dirty):
                rows = copy.deepcopy(self.rows)
                rows[0].update(revision="b" * 40, dirty=dirty)
                result = self.collect(rows)
                self.assertEqual(result["revision"], "b" * 40)
                self.assertEqual(result["dirty"], int(dirty))
                self.assertIs(type(result["dirty"]), int)
                self.assertEqual(result["source"], rows[0])
                self.assertIs(result["source"]["dirty"], dirty)

    def test_scope_state_contract_gates_and_donor_provenance(self):
        result = self.collect(self.rows)
        for phrase in ("actual layer3 F32 indexer norms", "BF16 Q/K weights", "CPU raw FP32",
                       "eight synthetic columns repeated", "numeric cache timeline",
                       "synthetic read-only Q4", "not GPU projections", "full-model inference",
                       "inference speed"):
            self.assertIn(phrase, result["scope"])
        for phrase in ("completed GPU events", "20 individually completed", "validated after every repeat"):
            self.assertIn(phrase, result["timing_scope"])
        self.assertIn("[32768][128]", result["state_abi"]["pooled_keys"])
        self.assertIn("[3][128]", result["state_abi"]["raw_tail"])
        self.assertIn("sticky error==0", result["state_abi"]["publication"])
        self.assertIn("384 tail floats", result["state_abi"]["restore"])
        self.assertEqual(result["prefix_abi"]["accepted_drafts_a"], [0, 1, 2])
        self.assertEqual(result["prefix_abi"]["accept_slot"], "1 + a")
        self.assertIn("canonical Q4_0", result["indexer_contract"]["key"])
        self.assertEqual(result["selection_contract"]["capacity"], 2051)
        self.assertIn("actual visible%4", result["selection_contract"]["tail"])
        self.assertIn("identical score floats", result["selection_contract"]["ids_reference"])
        self.assertIn("identical selected ID order", result["attention_contract"]["reference"])
        self.assertEqual(result["fixture_gates"], {
            **self.rows[0]["gates"], "frozen_before_first_gpu_run": True})
        self.assertEqual(result["donors"], {
            "mx": "dcd685463d597d31f5ca759d32c94592a2740fa4",
            "furnace": "905021dbad71c5056ef51f9fd45d545403fc989c",
            "reinstinct": "0b79e326351d90d4554a1c18df92da5d0ab692e8"})
        self.assertEqual(result["reference"], "transformers a005fc82babfe8871d87746decad2dbee100a125")

    def test_error_boundaries_and_positive_integer_timing(self):
        for ratio in (0, 1):
            rows = copy.deepcopy(self.rows)
            for row in rows:
                for _, obj in containers(row):
                    if "max_abs" in obj:
                        obj.update(max_abs=0, max_bound_ratio=ratio)
                if "resident_ms" in row:
                    row["resident_ms"] = 1
            self.assertTrue(self.collect(rows)["passed"])

    def test_every_missing_row_and_truncated_prefix(self):
        for index in range(62):
            with self.subTest(missing=index):
                rows = copy.deepcopy(self.rows)
                rows.pop(index)
                with self.assertRaises(ValueError):
                    self.collect(rows)
        for length in (0, 1, 2, 3, 31, 32, 60, 61):
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.collect(self.rows[:length])

    def test_extra_or_duplicate_rows(self):
        for index in range(62):
            with self.subTest(index=index):
                rows = copy.deepcopy(self.rows)
                rows.insert(index, copy.deepcopy(rows[index]))
                with self.assertRaises(ValueError):
                    self.collect(rows)
        with self.assertRaises(ValueError):
            self.collect(self.rows * 2)

    def test_every_adjacent_reordering(self):
        for index in range(61):
            with self.subTest(index=index):
                rows = copy.deepcopy(self.rows)
                rows[index], rows[index + 1] = rows[index + 1], rows[index]
                with self.assertRaises(ValueError):
                    self.collect(rows)

    def test_summary_and_device_group_order(self):
        for first, second in ((slice(1, 3), slice(3, 5)),
                              (slice(3, 32), slice(32, 61)),
                              (slice(3, 19), slice(32, 48))):
            with self.subTest(first=first, second=second):
                rows = copy.deepcopy(self.rows)
                rows[first], rows[second] = rows[second], rows[first]
                with self.assertRaises(ValueError):
                    self.collect(rows)

    def test_duplicate_replacing_expected_coordinate(self):
        for index in range(1, 62):
            with self.subTest(index=index):
                rows = copy.deepcopy(self.rows)
                rows[index] = copy.deepcopy(rows[index - 1])
                with self.assertRaises(ValueError):
                    self.collect(rows)

    def test_missing_fields_in_every_nested_object(self):
        for index, row in enumerate(self.rows):
            for path, obj in containers(row):
                for field in obj:
                    with self.subTest(index=index, path=path, field=field):
                        rows = copy.deepcopy(self.rows)
                        del target(rows[index], path)[field]
                        with self.assertRaises(ValueError):
                            self.collect(rows)

    def test_extra_fields_in_every_nested_object(self):
        for index, row in enumerate(self.rows):
            for path, _ in containers(row):
                with self.subTest(index=index, path=path):
                    rows = copy.deepcopy(self.rows)
                    target(rows[index], path)["unexpected"] = "passed"
                    with self.assertRaises(ValueError):
                        self.collect(rows)

    def test_nonobject_rows(self):
        for index in range(62):
            for value in (None, [], "qsa_complete passed", 1, True):
                with self.subTest(index=index, value=value):
                    rows = copy.deepcopy(self.rows)
                    rows[index] = value
                    with self.assertRaises(ValueError):
                        self.collect(rows)

    def test_malformed_nested_object_types(self):
        for index, row in enumerate(self.rows):
            for path, _ in containers(row):
                if path:
                    for value in (None, [], True, "passed"):
                        with self.subTest(index=index, path=path, value=value):
                            self.reject_change(index, path, value)

    def test_all_boolean_proofs_require_literal_true(self):
        for index, row in enumerate(self.rows):
            for path, value in leaves(row):
                if type(value) is bool and path != ("dirty",):
                    for wrong in (False, 0, 1, "true", None):
                        with self.subTest(index=index, path=path, value=wrong):
                            self.reject_change(index, path, wrong)

    def test_every_numeric_field_rejects_boolean_and_string(self):
        for index, row in enumerate(self.rows):
            for path, value in leaves(row):
                if type(value) in (int, float):
                    for wrong in (False, True, str(value), None):
                        with self.subTest(index=index, path=path, value=wrong):
                            self.reject_change(index, path, wrong)

    def test_integer_counters_and_geometry_reject_float_types(self):
        for index, row in enumerate(self.rows):
            for path, value in leaves(row):
                if type(value) is int and path != ("config", "rope_base"):
                    with self.subTest(index=index, path=path):
                        self.reject_change(index, path, float(value))

    def test_exact_summary_coverage_and_element_counts(self):
        for index in (1, 2):
            for path, value in leaves(self.rows[index]):
                if type(value) is int and path != ("device",):
                    for wrong in (0, value - 1, value + 1):
                        with self.subTest(index=index, path=path, value=wrong):
                            self.reject_change(index, path, wrong)

    def test_every_measurement_coordinate_geometry_and_launch_count(self):
        fields = ("device", "N", "base", "phase", "visible", "prepared", "capacity", "selected",
                  "repeats", "launches", "completed_blocks", "new_blocks", "token_count", "block_count")
        for index, row in enumerate(self.rows[3:61], 3):
            for field in fields:
                if field in row:
                    for wrong in (row[field] - 1, row[field] + 1):
                        with self.subTest(index=index, field=field, value=wrong):
                            self.reject_change(index, (field,), wrong)
            for name, obj in row.items():
                if type(obj) is dict and "elements" in obj:
                    # One repeat, newly completed keys only, or wrong head width
                    # cannot masquerade as validation of all 20 full outputs.
                    for wrong in (0, 1, obj["elements"] // 20, obj["elements"] - 1):
                        with self.subTest(index=index, name=name, value=wrong):
                            self.reject_change(index, (name, "elements"), wrong)

    def test_supported_coordinates_in_wrong_slots(self):
        alternatives = {"N": (1, 2, 3, 128), "phase": (0, 1, 2, 3), "device": (0, 1),
                        "visible": (2052, 4096, 32768, 131072),
                        "selected": (1, 63, 64, 65, 2048, 2051)}
        for index, row in enumerate(self.rows[3:61], 3):
            for field, values in alternatives.items():
                if field in row:
                    for wrong in values:
                        if wrong != row[field]:
                            with self.subTest(index=index, field=field, value=wrong):
                                self.reject_change(index, (field,), wrong)

    def test_positive_finite_timing(self):
        for index in range(3, 61):
            for wrong in (0, -1, float("nan"), float("inf"), float("-inf"), 10**1000):
                with self.subTest(index=index, value=wrong):
                    self.reject_change(index, ("resident_ms",), wrong)

    def test_nonnegative_finite_errors_and_frozen_ratio(self):
        for index, row in enumerate(self.rows):
            for path, obj in containers(row):
                if "max_abs" not in obj:
                    continue
                for field, values in (("max_abs", (-1e-12,)),
                                      ("max_bound_ratio", (-1e-12, 1.000000001, 2))):
                    for wrong in (*values, float("nan"), float("inf"), float("-inf"), 10**1000):
                        with self.subTest(index=index, path=path, field=field, value=wrong):
                            self.reject_change(index, path + (field,), wrong)

    def test_changed_source_geometry_metadata_and_gates(self):
        for path, value in leaves(self.rows[0]):
            if path[0] in ("config", "projection_types", "metadata_types", "gates", "rows"):
                wrong = value + 1 if type(value) in (int, float) else "changed"
                with self.subTest(path=path):
                    self.reject_change(0, path, wrong)
        for sections in ([11, 11, 10], [11, 11, 10, 0, 0], [10, 11, 11, 0], None):
            with self.subTest(sections=sections):
                self.reject_change(0, ("config", "sections"), sections)
        for epsilon in (1e-6, 0, -1e-6, float("nan"), float("inf"), 10**1000):
            with self.subTest(epsilon=epsilon):
                self.reject_change(0, ("config", "rms_epsilon"), epsilon)

    def test_source_and_measurement_text_is_exact_not_pass_matching(self):
        for index, row in enumerate(self.rows):
            for path, value in leaves(row):
                if type(value) is str and path != ("revision",) and path != ("model_path",):
                    with self.subTest(index=index, path=path):
                        self.reject_change(index, path, value + "; passed true")

    def test_revision_and_raw_boolean_dirty_provenance(self):
        for revision in (None, 1, True, "", "a" * 39, "a" * 41, "g" * 40,
                         "A" * 40, "a" * 40 + "\n", " " + "a" * 39, "unknown"):
            with self.subTest(revision=revision):
                self.reject_change(0, ("revision",), revision)
        for dirty in (0, 1, 0.0, 1.0, -1, 2, "true", None):
            with self.subTest(dirty=dirty):
                self.reject_change(0, ("dirty",), dirty)
        for path in ("", None, True, 1, "model\npassed", "model\x00.gguf"):
            with self.subTest(model_path=path):
                self.reject_change(0, ("model_path",), path)

    def test_footer_protocol_count_devices_and_passed(self):
        for field, value in self.rows[61].items():
            wrong = value + 1 if type(value) is int else False if type(value) is bool else "wrong"
            with self.subTest(field=field):
                self.reject_change(61, (field,), wrong)

    def test_malformed_json_blank_or_unstructured_rows(self):
        valid = jsonl(self.rows)
        for text in ("", "\n" + valid, valid + "\n", valid.replace("\n", "\n \n", 1),
                     "progress: passed\n" + valid, valid + "not JSON\n",
                     valid.replace("\n", " trailing\n", 1), json.dumps(self.rows),
                     "\ufeff" + valid, valid.replace('"protocol": 1', '"protocol": 01', 1),
                     valid[:-20]):
            with self.subTest(text=text[:40]):
                self.raw.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.collect(self.raw)

    def test_lf_crlf_and_no_final_lf(self):
        valid = jsonl(self.rows)
        for text in (valid, valid.rstrip("\n"), valid.replace("\n", "\r\n")):
            with self.subTest(ending=text[-30:]):
                self.raw.write_bytes(text.encode("utf-8"))
                self.assertTrue(MODULE.collect(self.raw)["passed"])
        for separator in ("\r", "\v", "\f", "\x85", "\u2028", "\u2029"):
            with self.subTest(separator=separator):
                self.raw.write_bytes(valid.replace("\n", separator).encode("utf-8"))
                with self.assertRaises(ValueError):
                    MODULE.collect(self.raw)

    def test_invalid_utf8_and_deep_json_are_value_errors(self):
        for content in (b"\xff\n", b'{"deep":' + b"[" * 1500 + b"]" * 1500 + b"}\n"):
            with self.subTest(content=content[:30]):
                self.raw.write_bytes(content)
                with self.assertRaises(ValueError):
                    MODULE.collect(self.raw)

    def test_nonfinite_json_constants_and_exponent_overflow(self):
        valid = jsonl(self.rows)
        for original in ('"resident_ms": 0.03', '"max_abs": 1e-05',
                         '"max_bound_ratio": 0.05', '"rms_epsilon": 9.999999974752427e-07'):
            for token in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
                with self.subTest(original=original, token=token):
                    field = original.split(":")[0]
                    text = valid.replace(original, field + ": " + token, 1)
                    self.assertNotEqual(text, valid)
                    self.raw.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        MODULE.collect(self.raw)

    def test_duplicate_keys_in_source_gate_measurement_and_footer(self):
        valid = jsonl(self.rows)
        for original, duplicate in (
                ('"dirty": true', '"dirty": false, "dirty": true'),
                ('"abs": 0.0002', '"abs": 1, "abs": 0.0002'),
                ('"elements": 8451328', '"elements": 1, "elements": 8451328'),
                ('"max_bound_ratio": 0.05', '"max_bound_ratio": 2, "max_bound_ratio": 0.05'),
                ('"token_count": 2048', '"token_count": 2051, "token_count": 2048'),
                ('"ids_exact": true', '"ids_exact": false, "ids_exact": true'),
                ('"passed": true}', '"passed": false, "passed": true}')):
            with self.subTest(duplicate=duplicate):
                text = valid.replace(original, duplicate, 1)
                self.assertNotEqual(text, valid)
                self.raw.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.collect(self.raw)

    def test_oversized_file_and_single_row_are_rejected(self):
        for text in (" " * (MODULE.MAX_RAW_BYTES + 1),
                     " " * MODULE.MAX_ROW_BYTES + jsonl(self.rows),
                     jsonl(self.rows).replace("\n", " " * MODULE.MAX_ROW_BYTES + "\n", 1)):
            with self.subTest(size=len(text)):
                self.raw.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.collect(self.raw)

    def test_cli_appends_exactly_one_validated_record(self):
        self.raw.write_text(jsonl(self.rows), encoding="utf-8")
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
        self.assertEqual(result["kind"], "r2d_qsa")
        self.assertEqual(result["revision"], self.rows[0]["revision"])
        self.assertEqual(result["dirty"], 1)
        self.assertIs(type(result["dirty"]), int)
        self.assertEqual(result["raw_log"], str(self.raw.resolve()))
        self.assertEqual(result["source"], self.rows[0])
        self.assertEqual(result["correctness"], self.rows[1:3])
        self.assertEqual(result["measurements"], self.rows[3:61])
        self.assertEqual(result["complete"], self.rows[61])
        for field in ("speedup", "inference_tps", "improvement"):
            self.assertNotIn(field, result)

    def test_cli_creates_one_record_without_modifying_raw(self):
        raw_text = jsonl(self.rows)
        self.raw.write_text(raw_text, encoding="utf-8")
        process = self.run_cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        lines = self.results.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["kind"], "r2d_qsa")
        self.assertEqual(self.raw.read_text(encoding="utf-8"), raw_text)

    def test_cli_rejection_does_not_append_or_create_results(self):
        prior = '{"kind":"prior"}\n'
        for rows in (self.rows[:-1], self.rows[:1] + self.rows[3:] + self.rows[1:3]):
            with self.subTest(length=len(rows)):
                self.raw.write_text(jsonl(rows), encoding="utf-8")
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
