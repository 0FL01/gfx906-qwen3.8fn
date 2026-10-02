"""Bounded synthetic proofs for the current R3b result collector (no HIP/model)."""

import copy
import datetime
import importlib.util
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
TMP = Path(tempfile.gettempdir())
SCRIPT = ROOT / "tools/record_session.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load("record_session", SCRIPT)
FIXTURES = load("session_compare_fixtures", ROOT / "tests/session_compare_test.py")
OWN = "e9f1dfe57cf8fdc0abd9ba10ab91cfbe01d9e0db"
MX = "dcd685463d597d31f5ca759d32c94592a2740fa4"
HF = "a005fc82babfe8871d87746decad2dbee100a125"
VOCAB = 248320
IDS = [248044, *range(100, 131)]


def write_json(path, obj):
    path.write_text(json.dumps(obj, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def controls(meta, gdn=True, qsa=True, passes=48):
    """oracle.cpp::write_metadata schema, including independent disabled controls."""
    meta["expert_cache_warmup"] = {"passes": passes, "tokens_per_pass": 32,
                                   "state_clear_after_each_pass": True, "warmup_logits_discarded": True}
    for name, enabled, per_token, mode, filename in (
        ("hf_gdn_l2_control", gdn, 72, "experimental_hf_semantic_correction", "gdn-l2-control.jsonl"),
        ("hf_qsa_f32_control", qsa, 12, "experimental_hf_normalized_attention_precision", "qsa-f32-control.jsonl"),
    ):
        obj = {"enabled": enabled, "mode": mode if enabled else "disabled", "diagnostic_only": True,
               "baseline_performance_reference": False, "weights_unchanged": True,
               "hf_source_revision": HF, "production_source_revision": MX,
               "records": filename if enabled else None,
               "recorded_control_count": 32 * per_token if enabled else 0,
               "warm_control_count": 32 * per_token * passes if enabled else 0,
               "recorded_tokens_checked": 32 if enabled else 0,
               "warm_tokens_checked": 32 * passes if enabled else 0,
               "warm_pass_expected_count": 32 * per_token if enabled else 0,
               "warm_pass_control_counts": [32 * per_token] * passes if enabled else [],
               "last_evaluation": {"epoch": 32 * (passes + 1) if gdn or qsa else 32,
                                   "warm_pass": -1, "position": 31, "active": False,
                                   "input_token_id": 130, "control_count": per_token if enabled else 0}}
        if name == "hf_gdn_l2_control":
            obj.update(production_formula="x/sqrt(max(sum(x*x),eps*eps))",
                       control_formula="x/sqrt(sum(x*x)+1e-6f)", epsilon_type="f32",
                       epsilon_f32=9.99999997e-7,
                       epsilon_f32_bits=struct.unpack("<I", struct.pack("<f", 1e-6))[0],
                       arithmetic="ascending 128-element FP32 sum; separately rounded multiply/add, no FMA; raw source-0 recomputation",
                       handshake={"name_patterns": ["q_conv_predelta-L", "k_conv_predelta-L"],
                                  "layers": "0..47 except L%4==3", "op": "L2_NORM", "type": "f32",
                                  "ne": [128, 16, 1, 1], "source_index": 0, "source_ne": [128, 16, 1, 1],
                                  "source_type": "f32", "source_destination_disjoint_required": True,
                                  "strided_storage_checked": True, "preserve_destination_padding": True,
                                  "layer_count": 36, "corrections_per_token": 72})
        else:
            obj.update(independent_of_hf_gdn_l2_control=True, library_revision_attested=MX,
                       max_control_visible=2048, supported_positions=[0, 2047],
                       q4_cache_unchanged=True, query_and_mask_unchanged=True, raw_quant_type="q4_0",
                       gather="Q4_0->FP16 RNE->FP32 for K and V",
                       gather_helpers=["ggml_fp16_to_fp32", "ggml_fp32_to_fp16"],
                       fp32_rounding="round-to-nearest-even required; environment is not modified",
                       softmax_accumulation="FP32",
                       arithmetic="ascending separate FP32 mul/add; scores=dot*0.0625; exp(score-max); denominator=sum(weights); output=sum(weights*V)*(1/denominator), probability-one copies gathered V; no FMA or GGML max-offset",
                       original_output_used_only_for_finite_guards_and_informational_metrics=True,
                       stage_change_before_inverse_hadamard=True, no_math_ancestor_wrappers=True,
                       recorded_expected_count=32 * per_token if enabled else 0,
                       warm_expected_count=32 * per_token * passes if enabled else 0,
                       handshake={"name_pattern": "kqv_out-L", "layers": list(range(3, 48, 4)),
                                  "target_type": "f32", "target_ne": [6144, 1, 1, 1],
                                  "ancestor_op": "FLASH_ATTN_EXT", "ancestor_count": 1,
                                  "ancestor_ne": [256, 24, 1, 1],
                                  "wrapper_ops": ["VIEW", "RESHAPE", "PERMUTE", "CONT"], "max_wrappers": 8,
                                  "head_major_flattening_proved": True, "query_type": "f32",
                                  "query_ne": [256, 1, 24, 1], "kv_type": "q4_0",
                                  "kv_ne": [256, "physicalKV", 2, 1], "q4_block_bytes": 18, "q4_row_bytes": 144,
                                  "mask_type": "f16", "mask_ne": ["physicalKV", ">=1", 1, 1],
                                  "mask_row": 0, "mask_selected_value": 0, "mask_excluded_value": "-inf",
                                  "selected_ids_exact": "0..currentPosition", "query_count": 1, "seq_id": 0,
                                  "gqa_group": 12, "scale": .0625, "max_bias": 0, "softcap": 0,
                                  "precision": "F32", "sinks": False,
                                  "source_destination_disjoint_required": True, "strided_storage_checked": True,
                                  "preserve_destination_padding": True, "exact_readback_required": True,
                                  "corrections_per_token": 12})
        meta[name] = obj


def expand_report(seed):
    """Reuse actual comparator-produced row schemas without a 64 MiB fixture."""
    report = copy.deepcopy(seed)
    report.update(tokens=32, input_token_ids=IDS[:])
    report["session_source"].update(revision=OWN, dirty=1)
    report["session_complete"].update(input_tokens=32, consumed_tokens=32)
    template = seed["logits"]["rows"][0]
    report["logits"]["rows"] = [{**copy.deepcopy(template), "token_index": i,
                                  "position": i, "input_token_id": token} for i, token in enumerate(IDS)]
    report["logits"].update(elements=32 * VOCAB, expected_bytes_each=32 * VOCAB * 4, argmax_agreeing_rows=32)
    for key in ("intermediates", "optional_intermediates"):
        report[key] = [{**copy.deepcopy(row), "token_index": i, "position": i}
                       for i in range(32) for row in seed[key]]
    report["trace_counts"] = {key: value * 32 for key, value in seed["trace_counts"].items()}
    meta = report["oracle_metadata"]
    meta.update(token_ids=IDS[:], completed_tokens=32)
    meta["logits"].update(bytes_written=32 * VOCAB * 4,
                          semantics="row i is the next-token distribution after consuming token_ids[i] at position i")
    meta["callbacks"]["captures"] *= 32
    meta["callbacks"]["observed_counts"] = {key: value * 32 for key, value in meta["callbacks"]["observed_counts"].items()}
    meta["context"].update(batch_requested=1, ubatch_requested=1, threads=16, threads_batch=16,
                           flash_attention_requested="enabled", offload_kqv=True, op_offload=True, recurrent_snapshots=0)
    meta["placement"] = {"split_mode": "layer", "tensor_split": [1, 1], "n_gpu_layers": -1,
                          "cpu_expert_pattern": r"\.ffn_(up|down|gate|gate_up)_(ch|)exps", "use_extra_bufts": False,
                          "load_mode": "direct_io", "lazy_mode": "off", "moe_cache_slots_requested": 112,
                          "moe_cache_inserts_requested": 48, "cache_activation_checked_layers": [0, 1, 3, 47]}
    meta["callbacks"].update(all_layer_outputs_and_slots=False, gdn_probe_layers=[],
                             binary_layout="native little-endian storage span with original ne/nb; includes stride gaps",
                             indices="token_index,node_index,occurrence are zero-based; node_index counts scheduler ask calls")
    meta["devices"] = [{"name": f"ROCm{i}", "description": "synthetic gfx906", "backend": "ROCm",
                        "device_id": f"synthetic-{i}", "memory_total": 17163091968,
                        "memory_free_before_load": 17163091968} for i in range(2)]
    controls(meta)
    return report


def generation_fixture(source):
    source = copy.deepcopy(source)
    source["trace"] = False
    rows = [source]
    pending = 248044
    for i in range(32):
        # Include EOS before the final output to exercise --ignore-eos behavior.
        argmax = 248046 if i == 7 else 1000 + i
        rows += [{"kind": "session_token", "position": i, "token": pending, "argmax": argmax,
                  "completed_ms": 12.5, "expert_hits": 0, "expert_misses": 480 * (i + 1),
                  "expert_upload_bytes": 480 * (i + 1) * 2867200, "finite": True},
                 {"kind": "session_output", "index": i, "token": argmax}]
        pending = argmax
    rows.append({"kind": "session_complete", "input_tokens": 1, "output_tokens": 32,
                 "consumed_tokens": 32, "load_ms": 2000.0, "request_ms": 410.0,
                 "scope": "ordered_decode_no_prefill_no_sampling_speed_claim", "passed": True})
    return rows


def reset_fixture(source):
    zero = {"consumed_tokens": 0, "expert_hits": 0, "expert_misses": 0,
            "expert_upload_bytes": 0, "last_completed_ms": 0}
    baseline, replay = [], []
    for i, token in enumerate([248044, 100, 101, 102]):
        baseline.append({"token": token, "stats": {"consumed_tokens": i + 1, "expert_hits": 0,
                         "expert_misses": 480 * (i + 1), "expert_upload_bytes": 480 * (i + 1) * 2867200,
                         "last_completed_ms": 12.5}})
        # Single-slot replay may hit the first route; do not require identical
        # routing statistics/timings across replay, only bitwise logit proof.
        hits = 48 * (i + 1)
        misses = 432 * (i + 1)
        replay.append({"token": token, "stats": {"consumed_tokens": i + 1, "expert_hits": hits,
                       "expert_misses": misses, "expert_upload_bytes": misses * 2867200,
                       "last_completed_ms": 10.5}})
    return {"kind": "session_test_complete", "revision": source["revision"], "dirty": bool(source["dirty"]),
            "model": source["model"], "runtime": "own_48_layer_HIP", "capacity": 4, "expert_slots": 1,
            "trace": False, "sampling": "teacher_forced", "scope": "reset_replay_and_rejection_self_parity",
            "performance_claim": False, "initial_stats": copy.deepcopy(zero), "baseline_steps": baseline,
            "after_reset_stats": copy.deepcopy(zero), "replay_steps": replay, "final_reset_stats": copy.deepcopy(zero),
            "checks": {"finite_logit_values": 8 * VOCAB, "bitwise_compared_logit_values": 4 * VOCAB,
                       "invalid_token_rejections": 2, "capacity_rejections": 2,
                       "rejection_stats_preserved": True, "rejection_logits_preserved": True,
                       "invalid_token_continuation": True, "reset_reused": True,
                       "minimum_slot_reuse_misses_per_step": 432},
            "invalid_config_checks": {"tested": False, "capacity_cases": [0, 3], "expert_slots_cases": [0],
                                      "reason": "constructor validation follows Model and PLE initialization"}, "passed": True}


class SessionResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed_dir = tempfile.TemporaryDirectory(prefix="session-results-seed-", dir=TMP)
        try:
            fixture = FIXTURES.Fixture(Path(cls.seed_dir.name), n=1, optional=True)
            seed = fixture.compare()
            if not seed["passed"]:
                raise AssertionError(seed["failures"])
            cls.template = expand_report(seed)
        except BaseException:
            cls.seed_dir.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.seed_dir.cleanup()

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="session-results-test-", dir=TMP)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.comparison_path = self.root / "comparison.json"
        self.generation_path = self.root / "generation.jsonl"
        self.reset_path = self.root / "reset.jsonl"
        self.paths = [self.comparison_path, self.generation_path, self.reset_path]
        self.results = self.root / "results.jsonl"
        self.restore()

    def restore(self):
        self.report = copy.deepcopy(self.template)
        self.generated = generation_fixture(self.report["session_source"])
        self.reset = reset_fixture(self.report["session_source"])

    def save(self):
        write_json(self.comparison_path, self.report)
        write_rows(self.generation_path, self.generated)
        write_rows(self.reset_path, [self.reset])

    def collect(self):
        self.save()
        return MODULE.collect(*self.paths)

    def reject_mutations(self, mutations):
        for i, mutation in enumerate(mutations):
            with self.subTest(mutation=i):
                self.restore()
                mutation()
                self.save()
                with self.assertRaises(ValueError):
                    MODULE.collect(*self.paths)

    def cli(self, results=None):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--comparison", str(self.comparison_path),
                               "--generation", str(self.generation_path), "--reset", str(self.reset_path),
                               "--results", str(results or self.results)], cwd=self.root,
                              env={**os.environ, "TMPDIR": str(TMP)}, text=True, capture_output=True, check=False)

    def test_complete_proofs_counts_paths_timestamp_and_dirty_normalization(self):
        self.save()
        original = [path.read_bytes() for path in self.paths]
        record = MODULE.collect(*self.paths)
        self.assertEqual(record["kind"], "r3b_session")
        self.assertIs(record["passed"], True)
        self.assertEqual((record["revision"], record["dirty"]), (OWN, 1))
        self.assertEqual(record["comparison"], self.report)
        self.assertEqual(record["generation"]["records"], self.generated)
        self.assertEqual(record["reset"], self.reset)
        self.assertEqual(record["counts"], {"teacher_tokens": 32, "teacher_finite_logit_pairs": 7946240,
                         "output_tokens": 32, "consumed_tokens": 32, "reset_steps": 8,
                         "reset_bitwise_logit_values": 993280})
        self.assertEqual(record["generation"]["pending_token"], self.generated[-2]["token"])
        self.assertEqual(len(record["generation"]["tokens"]), 32)
        self.assertEqual(len(record["generation"]["outputs"]), 32)
        self.assertEqual(list(record["raw_logs"].values()), [str(path.resolve()) for path in self.paths])
        self.assertEqual(datetime.datetime.fromisoformat(record["timestamp"]).utcoffset(), datetime.timedelta(0))
        self.assertIs(record["provenances"]["reset"]["dirty"], True)
        self.assertIs(type(record["provenances"]["generation"]["dirty"]), int)
        self.assertIs(record["performance_claim"], False)
        self.assertIs(record["bitwise_hf_claim"], False)
        self.assertIn("invalid constructor configurations explicitly untested", record["reset_scope"])
        self.assertEqual([path.read_bytes() for path in self.paths], original)
        json.dumps(record, allow_nan=False)

    def test_parent_controller_and_oracle_provenance_are_not_relabelled(self):
        # compare_session itself emits no compare_controller: an optional parent
        # attachment must be retained as-is, not made into a compiled revision.
        attachment = {"revision": "b" * 40, "dirty": False,
                      "oracle_controller_revision": "c" * 40, "oracle_controller_dirty": 1}
        self.report["compare_controller"] = attachment
        self.report["artifacts"]["session_log"] = "/core/runs/parent-session.jsonl"
        result = self.collect()
        self.assertEqual(result["comparison"]["compare_controller"], attachment)
        self.assertEqual(result["comparison"]["artifacts"], self.report["artifacts"])
        self.assertEqual(result["revision"], OWN)
        self.assertEqual(result["provenances"]["oracle_source_revision"], MX)

    def test_another_consistent_compiled_revision_clean_or_dirty_preserves_sources(self):
        for revision, dirty in (("0123456789abcdef" * 2 + "01234567", 0), ("ABCDEF0123456789" * 2 + "ABCDEF01", 1)):
            with self.subTest(revision=revision, dirty=dirty):
                self.restore()
                self.report["session_source"].update(revision=revision, dirty=dirty)
                self.generated = generation_fixture(self.report["session_source"])
                self.reset = reset_fixture(self.report["session_source"])
                record = self.collect()
                self.assertEqual((record["revision"], record["dirty"]), (revision, dirty))
                self.assertEqual(record["comparison"], self.report)
                self.assertEqual(record["generation"]["records"], self.generated)
                self.assertEqual(record["reset"], self.reset)
                for label, original in (("session", self.report["session_source"]),
                                        ("generation", self.generated[0]), ("reset", self.reset)):
                    self.assertEqual(record["provenances"][label], {k: original[k] for k in ("revision", "dirty")})
                    self.assertIs(type(record["provenances"][label]["dirty"]), type(original["dirty"]))

    def test_compiled_revision_and_dirty_must_match_across_every_artifact(self):
        for artifact in range(3):
            for field, value in (("revision", "f" * 40), ("dirty", False if artifact == 2 else 0)):
                with self.subTest(artifact=artifact, field=field):
                    self.restore()
                    sources = [self.report["session_source"], self.generated[0], self.reset]
                    sources[artifact][field] = value
                    self.save()
                    with self.assertRaisesRegex(ValueError, "compiled provenance mismatch"):
                        MODULE.collect(*self.paths)

    def test_compiled_revision_syntax_and_schema_specific_dirty_types_are_strict(self):
        for artifact in range(3):
            bad_dirty = (0, 1, None, "false") if artifact == 2 else (False, True, -1, 2, 0.0, "0", None)
            for field, values in (("revision", ("a" * 39, "a" * 41, "g" * 40, None, 123)), ("dirty", bad_dirty)):
                for value in values:
                    with self.subTest(artifact=artifact, field=field, value=value):
                        self.restore()
                        sources = [self.report["session_source"], self.generated[0], self.reset]
                        sources[artifact][field] = value
                        self.save()
                        with self.assertRaisesRegex(ValueError, "invalid compiled revision|invalid integer|invalid boolean dirty"):
                            MODULE.collect(*self.paths)

    def test_independent_controls_and_default_production_are_explicit_no_performance_reference(self):
        for gdn, qsa, passes in ((True, True, 48), (True, False, 1), (False, True, 2), (False, False, 0),
                                  (False, False, 48)):
            with self.subTest(gdn=gdn, qsa=qsa, passes=passes):
                controls(self.report["oracle_metadata"], gdn, qsa, passes)
                result = self.collect()
                meta = result["comparison"]["oracle_metadata"]
                for key, enabled in (("hf_gdn_l2_control", gdn), ("hf_qsa_f32_control", qsa)):
                    self.assertIs(meta[key]["enabled"], enabled)
                    self.assertIs(meta[key]["baseline_performance_reference"], False)
                self.assertIs(result["performance_claim"], False)
                self.assertIs(result["bitwise_hf_claim"], False)

    def test_previous_failure_one_token_and_missing_report_footers_cannot_qualify(self):
        self.reject_mutations([
            lambda: self.report.update(passed=False),
            lambda: self.report.update(failures=[{"category": "gate", "location": "logits[1]", "message": "failed"}]),
            lambda: self.report.update(tokens=1),
            lambda: self.report["logits"]["rows"].__delitem__(slice(1, None)),
            lambda: self.report.pop("session_complete"),
            lambda: self.report["session_complete"].update(passed=False),
            lambda: self.report["oracle_metadata"].update(status="running"),
            lambda: self.report["oracle_metadata"].update(error="failed callback"),
        ])

    def test_frozen_global_gates_and_semantic_notes_never_waive_failure(self):
        self.reject_mutations([
            lambda: self.report["gates"]["logits"].update(absolute=.021),
            lambda: self.report["gates"]["logits"].update(relative=.003),
            lambda: self.report["gates"]["intermediates"].update(absolute=.02),
            lambda: self.report["gates"]["intermediates"].update(relative=.0021),
            lambda: self.report["gates"].update(hc_init="bounded"),
            lambda: self.report["logits"].update(passed=False),
            lambda: self.report["logits"]["rows"][31].update(passed=False),
            lambda: self.report["logits"]["rows"][31].update(violating_elements=1),
            lambda: self.report["logits"]["rows"][0].update(max_bound_ratio=1.001),
            lambda: self.report["intermediates"][1].update(max_bound_ratio=1.001),
            lambda: self.report["optional_intermediates"][0].update(passed=False),
            lambda: self.report["intermediates"][0].update(max_abs=1e-30, rms=1e-30),
            lambda: self.report["intermediates"][0].update(gate="bounded"),
        ])

    def test_false_pass_failed_finite_pairs_and_inconsistent_aggregates_rejected(self):
        self.reject_mutations([
            lambda: self.report["logits"].update(elements=32 * VOCAB - 1),
            lambda: self.report["logits"].update(expected_bytes_each=32 * VOCAB * 4 - 4),
            lambda: self.report["logits"].update(exact_byte_counts=False),
            lambda: self.report["logits"].update(all_finite=False),
            lambda: self.report["logits"]["rows"][31].update(elements=VOCAB - 1),
            lambda: self.report["logits"]["rows"][31].update(finite_pairs=VOCAB - 1),
            lambda: self.report["logits"]["rows"][31].update(nonfinite_session=1),
            lambda: self.report["intermediates"][-1].update(nonfinite_oracle=1),
            lambda: self.report["intermediates"][-1].update(all_finite=False),
            lambda: self.report["intermediates"][-1].update(max_abs=.001, rms=0.0),
            lambda: self.report["logits"].update(max_abs=.001),
            lambda: self.report["logits"].update(rms=.001),
            lambda: self.report["logits"].update(max_bound_ratio=.5),
            lambda: self.report["logits"].update(argmax_agreeing_rows=31),
            lambda: self.report["logits"]["rows"][0].update(oracle_argmax=VOCAB),
            lambda: self.report["logits"]["rows"][0].update(argmax_agreement=False),
        ])

    def test_bounded_nonzero_error_and_argmax_disagreement_are_diagnostic(self):
        row = self.report["logits"]["rows"][0]
        error = abs(.26 - .25)
        row.update(max_abs=error, rms=error / math.sqrt(VOCAB), max_bound_ratio=error / (.02 + .002 * .25),
                   worst_bound_element={"element": VOCAB - 1, "session": .26, "oracle": .25},
                   session_argmax=200, argmax_agreement=False)
        self.report["logits"].update(max_abs=row["max_abs"], max_bound_ratio=row["max_bound_ratio"],
                                      rms=row["rms"] / math.sqrt(32), argmax_agreeing_rows=31)
        result = self.collect()
        self.assertIs(result["passed"], True)
        self.assertIs(result["comparison"]["logits"]["rows"][0]["argmax_agreement"], False)
        for bad in ({"max_bound_ratio": .5}, {"worst_bound_element": None},
                    {"worst_bound_element": {"element": VOCAB, "session": .26, "oracle": .25}},
                    {"worst_bound_element": {"element": 0, "session": .29, "oracle": .25}}):
            with self.subTest(bad=bad):
                original = copy.deepcopy(row)
                row.update(bad)
                self.save()
                with self.assertRaises(ValueError):
                    MODULE.collect(*self.paths)
                row.clear()
                row.update(original)

    def test_teacher_ids_positions_counts_and_every_intermediate_required(self):
        self.reject_mutations([
            lambda: self.report["input_token_ids"].__setitem__(31, 129),
            lambda: self.report["oracle_metadata"]["token_ids"].__setitem__(0, 248046),
            lambda: self.report["logits"]["rows"][31].update(input_token_id=129),
            lambda: self.report["logits"]["rows"][31].update(token_index=30),
            lambda: self.report["logits"]["rows"][0].update(position=1),
            lambda: self.report["session_complete"].update(consumed_tokens=31),
            lambda: self.report["session_complete"].update(output_tokens=1),
            lambda: self.report["oracle_metadata"].update(completed_tokens=31),
            lambda: self.report["intermediates"].pop(),
            lambda: self.report["intermediates"].__setitem__(1, copy.deepcopy(self.report["intermediates"][0])),
            lambda: self.report["intermediates"][1].update(layer=1),
            lambda: self.report["intermediates"][1].update(position=1),
            lambda: self.report["intermediates"][1].update(oracle_occurrence=1),
            lambda: self.report["intermediates"][1].update(status="missing_required"),
            lambda: self.report["intermediates"][1].pop("rms"),
            lambda: self.report["trace_counts"].update(oracle=1),
            lambda: self.report["trace_counts"].update(session=192),
            lambda: self.report["oracle_metadata"]["callbacks"]["observed_counts"].update(hc_init=31),
        ])

    def test_missing_optional_schema_supported_without_waiving_compared_gates(self):
        first = self.report["optional_intermediates"][0]
        first.clear()
        first.update(token_index=0, position=0, name="hc_attn_mix", layer=0, optional=True,
                     status="missing_optional", missing=["oracle"])
        self.report["oracle_metadata"]["callbacks"]["observed_counts"]["hc_mixed-0"] -= 1
        self.report["oracle_metadata"]["callbacks"]["captures"] -= 1
        self.report["trace_counts"]["oracle"] -= 1
        self.assertIs(self.collect()["passed"], True)
        first["missing"] = []
        self.save()
        with self.assertRaises(ValueError):
            MODULE.collect(*self.paths)

    def test_all_layer_and_qsa_probe_metadata_preserved_without_inventing_extra_gates(self):
        meta = self.report["oracle_metadata"]
        meta["callbacks"]["all_layer_outputs_and_slots"] = True
        meta["placement"]["cache_activation_checked_layers"] = None
        meta["qsa_probes"] = {"format": "gfx906-mx-qsa-probe-v1", "records": "qsa-probes.jsonl",
                              "read_only_observation": True, "warmup_suppressed": True,
                              "recorded_tokens_checked": 32, "row_byte_limit": 8192,
                              "manifest_byte_limit": 2048 * 12 * 8192, "rows": 32, "expected_rows": 32,
                              "requested_layers": [3], "layer_counts": [{"layer": 3, "rows": 32}]}
        self.assertEqual(self.collect()["comparison"]["oracle_metadata"], meta)
        meta["qsa_probes"]["rows"] = 31
        self.save()
        with self.assertRaises(ValueError):
            MODULE.collect(*self.paths)

    def test_generation_counts_order_ids_argmax_pending_and_source(self):
        self.reject_mutations([
            lambda: self.generated[0].update(runtime="llama_decode"),
            lambda: self.generated[0].update(sampling="user_sampling"),
            lambda: self.generated[0].update(capacity=4095),
            lambda: self.generated[0].update(expert_slots=1),
            lambda: self.generated[0].update(trace=True),
            lambda: self.generated[0].update(model="/models/full-PLE.gguf"),
            lambda: self.generated[1].update(token=7),
            lambda: self.generated[1].update(finite=False),
            lambda: self.generated[1].update(argmax=VOCAB),
            lambda: self.generated[2].update(token=101),
            lambda: self.generated[3].update(token=101),
            lambda: self.generated[4].update(index=0),
            lambda: self.generated[3].update(position=0),
            lambda: self.generated.__setitem__(slice(2, 4), self.generated[2:4][::-1]),
            lambda: self.generated[-1].update(input_tokens=2),
            lambda: self.generated[-1].update(output_tokens=31),
            lambda: self.generated[-1].update(consumed_tokens=33),
            lambda: self.generated[-1].update(passed=False),
            lambda: self.generated[-1].update(scope="prefill_performance"),
            lambda: self.generated.insert(-1, {**self.generated[-3], "position": 32, "token": self.generated[-2]["token"]}),
            lambda: self.generated.pop(),
            lambda: self.generated.insert(-1, copy.deepcopy(self.generated[0])),
            lambda: self.generated[1].update(logits_finite=True),
        ])

    def test_generation_durations_routes_uploads_and_boolean_numbers(self):
        self.reject_mutations([
            lambda: self.generated[1].update(completed_ms=0),
            lambda: self.generated[1].update(completed_ms=-1),
            lambda: self.generated[-1].update(request_ms=0),
            lambda: self.generated[-1].update(load_ms=0),
            lambda: self.generated[3].update(expert_hits=1),
            lambda: self.generated[3].update(expert_misses=479),
            lambda: self.generated[3].update(expert_upload_bytes=self.generated[1]["expert_upload_bytes"]),
            lambda: self.generated[1].update(expert_hits=-1),
            lambda: self.generated[1].update(expert_misses=1 << 64),
            lambda: self.generated[1].update(token=True),
            lambda: self.generated[1].update(position=False),
            lambda: self.generated[1].update(argmax=True),
            lambda: self.generated[1].update(finite=1),
            lambda: self.generated[1].update(completed_ms=True),
            lambda: self.generated[1].update(expert_hits=False),
            lambda: self.generated[0].update(dirty=True),
            lambda: self.generated[-1].update(input_tokens=True),
            lambda: self.generated[-1].update(load_ms=True),
            lambda: self.report["logits"]["rows"][0].update(nonfinite_session=False),
            lambda: self.report["logits"]["rows"][0].update(max_abs=False),
            lambda: self.report["oracle_metadata"]["token_ids"].__setitem__(1, True),
            lambda: self.report["oracle_metadata"]["context"].update(batch_requested=True),
            lambda: self.report["oracle_metadata"]["context"].update(type_v="f16"),
            lambda: self.report["oracle_metadata"]["placement"].update(moe_cache_inserts_requested=True),
            lambda: self.report["oracle_metadata"]["devices"][0].update(memory_total=True),
            lambda: self.report["oracle_metadata"]["callbacks"].update(all_layer_outputs_and_slots=1),
        ])

    def test_reset_complete_self_parity_rejections_stats_slots_and_provenance(self):
        self.reject_mutations([
            lambda: self.reset.update(passed=False),
            lambda: self.reset.update(runtime="llama_decode"),
            lambda: self.reset.update(capacity=8),
            lambda: self.reset.update(expert_slots=112),
            lambda: self.reset.update(trace=True),
            lambda: self.reset.update(performance_claim=True),
            lambda: self.reset.update(sampling="greedy_diagnostic"),
            lambda: self.reset["baseline_steps"].pop(),
            lambda: self.reset["replay_steps"][0].update(token=248046),
            lambda: self.reset["checks"].update(finite_logit_values=8 * VOCAB - 1),
            lambda: self.reset["checks"].update(bitwise_compared_logit_values=VOCAB),
            lambda: self.reset["checks"].update(invalid_token_rejections=1),
            lambda: self.reset["checks"].update(capacity_rejections=1),
            lambda: self.reset["checks"].update(rejection_stats_preserved=False),
            lambda: self.reset["checks"].update(rejection_logits_preserved=False),
            lambda: self.reset["checks"].update(invalid_token_continuation=False),
            lambda: self.reset["checks"].update(reset_reused=False),
            lambda: self.reset["checks"].update(minimum_slot_reuse_misses_per_step=431),
            lambda: self.reset["initial_stats"].update(expert_hits=1),
            lambda: self.reset["after_reset_stats"].update(consumed_tokens=1),
            lambda: self.reset["final_reset_stats"].update(last_completed_ms=1),
            lambda: self.reset["replay_steps"][0]["stats"].update(expert_hits=49, expert_misses=431),
            lambda: self.reset["replay_steps"][1]["stats"].update(consumed_tokens=1),
            lambda: self.reset["replay_steps"][1]["stats"].update(expert_upload_bytes=0),
            lambda: self.reset["invalid_config_checks"].update(tested=True),
            lambda: self.reset.update(dirty=1),
            lambda: self.reset["initial_stats"].update(consumed_tokens=False),
        ])

    def test_source_matching_and_independent_control_proof_failures(self):
        self.reject_mutations([
            lambda: self.report["session_source"].update(revision="f" * 40),
            lambda: self.generated[0].update(revision="f" * 40),
            lambda: self.reset.update(revision="f" * 40),
            lambda: self.generated[0].update(dirty=0),
            lambda: self.reset.update(dirty=False),
            lambda: self.reset.update(model="/different/qwen38-keep1-Q4_0.gguf"),
            lambda: self.report["oracle_metadata"].update(source_revision="f" * 40),
            lambda: self.report["oracle_metadata"].update(library_revision_attested="f" * 40),
            lambda: self.report["oracle_metadata"].update(library_revision_runtime_verified=True),
            lambda: self.report["oracle_metadata"].pop("hf_gdn_l2_control"),
            lambda: self.report["oracle_metadata"].pop("hf_qsa_f32_control"),
            lambda: self.report["oracle_metadata"]["hf_gdn_l2_control"].update(recorded_control_count=2303),
            lambda: self.report["oracle_metadata"]["hf_qsa_f32_control"].update(warm_control_count=0),
            lambda: self.report["oracle_metadata"]["hf_gdn_l2_control"].update(baseline_performance_reference=True),
            lambda: self.report["oracle_metadata"]["hf_gdn_l2_control"].update(epsilon_f32_bits=897988540),
            lambda: self.report["oracle_metadata"]["hf_gdn_l2_control"]["warm_pass_control_counts"].pop(),
            lambda: self.report["oracle_metadata"]["hf_qsa_f32_control"].update(independent_of_hf_gdn_l2_control=False),
            lambda: self.report["oracle_metadata"]["hf_qsa_f32_control"]["handshake"].update(exact_readback_required=False),
            lambda: self.report["oracle_metadata"]["hf_qsa_f32_control"]["last_evaluation"].update(active=True),
            lambda: self.report["oracle_metadata"]["expert_cache_warmup"].update(state_clear_after_each_pass=False),
        ])

    def test_strict_json_duplicates_nonfinite_overflow_encoding_blank_and_truncation(self):
        self.save()
        originals = {path: path.read_bytes() for path in self.paths}
        for path in self.paths:
            raw = originals[path]
            bads = [raw[:-1], raw[:-5], raw + (b"{}\n" if path == self.comparison_path else b"\n"),
                    b"[]\n", b"null\n", raw + b"failure\n",
                    raw.replace(b'"dirty":', b'"dirty":1,"dirty":', 1),
                    raw.replace(b'"dirty":', b'"extra":{"x":1,"x":2},"dirty":', 1),
                    raw.replace(b'"dirty":', b'"extra":NaN,"dirty":', 1),
                    raw.replace(b'"dirty":', b'"extra":Infinity,"dirty":', 1),
                    raw.replace(b'"dirty":', b'"extra":-Infinity,"dirty":', 1),
                    raw.replace(b'"dirty":', b'"extra":1e999,"dirty":', 1), b"\xff\n",
                    b'{"x":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}\n"]
            for i, bad in enumerate(bads):
                with self.subTest(path=path.name, bad=i):
                    path.write_bytes(bad)
                    with self.assertRaises(ValueError):
                        MODULE.collect(*self.paths)
                    path.write_bytes(raw)
        for path, field in ((self.comparison_path, b'"max_abs": 0.0'),
                            (self.generation_path, b'"completed_ms": 12.5'),
                            (self.reset_path, b'"last_completed_ms": 12.5')):
            for bad in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
                with self.subTest(path=path.name, value=bad):
                    path.write_bytes(originals[path].replace(field, field.split(b":")[0] + b": " + bad, 1))
                    with self.assertRaises(ValueError):
                        MODULE.collect(*self.paths)
                    path.write_bytes(originals[path])

    def test_byte_bounds_and_large_comparison_preserve_full_provenance(self):
        self.report["compare_controller"] = {"fixture_padding": "x" * (1024 * 1024)}
        result = self.collect()
        self.assertGreater(self.comparison_path.stat().st_size, 1024 * 1024)
        self.assertEqual(result["comparison"]["compare_controller"], self.report["compare_controller"])
        for path, limit in ((self.comparison_path, MODULE.MAX_COMPARISON_BYTES),
                            (self.generation_path, MODULE.MAX_LOG_BYTES), (self.reset_path, MODULE.MAX_LOG_BYTES)):
            with self.subTest(path=path.name):
                raw = path.read_bytes()
                with path.open("r+b") as stream:
                    stream.truncate(limit + 1)
                with self.assertRaises(ValueError):
                    MODULE.collect(*self.paths)
                path.write_bytes(raw)
        raw = self.reset_path.read_bytes()
        self.reset_path.write_bytes(b" " * MODULE.MAX_LINE_BYTES + raw)
        with self.assertRaises(ValueError):
            MODULE.collect(*self.paths)
        self.reset_path.write_bytes(raw)

    def test_cli_appends_exactly_one_preserves_existing_records_and_inputs(self):
        self.save()
        original = [path.read_bytes() for path in self.paths]
        self.results.write_bytes(b'{"kind":"previous"}\n')
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = self.results.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], '{"kind":"previous"}')
        record = json.loads(lines[1])
        self.assertEqual(record["kind"], "r3b_session")
        self.assertEqual(record["comparison"], self.report)
        self.assertEqual(record["generation"]["records"], self.generated)
        self.assertEqual(record["reset"], self.reset)
        self.assertEqual([path.read_bytes() for path in self.paths], original)

    def test_cli_failed_evidence_never_appends_or_creates_results(self):
        mutations = [lambda: self.report["logits"]["rows"][1].update(violating_elements=1),
                     lambda: self.generated.pop(), lambda: self.reset.update(passed=False)]
        self.results.write_bytes(b'{"kind":"previous"}\n')
        for i, mutation in enumerate(mutations):
            with self.subTest(mutation=i):
                self.restore()
                mutation()
                self.save()
                result = self.cli()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("record_session:", result.stderr)
                self.assertEqual(self.results.read_bytes(), b'{"kind":"previous"}\n')
                absent = self.root / "absent-results.jsonl"
                self.assertEqual(self.cli(absent).returncode, 1)
                self.assertFalse(absent.exists())

    def test_cli_result_aliases_and_incomplete_destination_never_mutate_inputs(self):
        self.save()
        aliases = []
        for path in self.paths:
            symlink = self.root / (path.name + ".symlink")
            symlink.symlink_to(path)
            hardlink = self.root / (path.name + ".hardlink")
            os.link(path, hardlink)
            aliases += [(path, path), (symlink, path), (hardlink, path)]
        for destination, source in aliases:
            with self.subTest(destination=destination.name):
                raw = source.read_bytes()
                result = self.cli(destination)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(source.read_bytes(), raw)
        for key in ("session_trace", "oracle_dir"):
            destination = Path(self.report["artifacts"][key]) / "results.jsonl"
            self.assertEqual(self.cli(destination).returncode, 1)
            self.assertFalse(destination.exists())
        self.results.write_bytes(b'{"kind":"incomplete"}')
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), b'{"kind":"incomplete"}')

    def test_crlf_is_supported_and_nonregular_inputs_rejected(self):
        self.save()
        for path in self.paths:
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.assertIs(MODULE.collect(*self.paths)["passed"], True)
        with self.assertRaises(ValueError):
            MODULE.collect(self.root, self.generation_path, self.reset_path)

    def test_serialization_failure_happens_before_destination_is_opened(self):
        record = self.collect()
        with mock.patch.object(MODULE.json, "dumps", side_effect=ValueError("serialization failure")):
            with self.assertRaises(ValueError):
                MODULE.append_result(self.results, record)
        self.assertFalse(self.results.exists())


if __name__ == "__main__":
    unittest.main()
