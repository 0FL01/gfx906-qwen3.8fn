import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "record_hc_ple", Path(__file__).parents[1] / "tools" / "record_hc_ple.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    hc = {"kind": "hc_actual_model", "passed": True, "token_cases": 9,
          "geometry": {"hidden_size": 2560, "branches": 4, "low_rank": 320},
          "mtp_tap_preserved": True, "root_injection_rejected": True,
          "model_path": "/models/qwen38-keep1-Q4_0.gguf", "cases": []}
    for i, name in enumerate(["blk.0.hc_attn", "blk.0.hc_ffn", "output_hc"]):
        hc["cases"].append({"mixer": name, "tokens": [1, 2, 3], "injection": i != 2,
                            "prefix_exact": True, "residual_preserved": True,
                            "finite_outputs": True, "success_call_allocations": 0,
                            "mix_max_abs_error": 0, "injection_max_abs_error": 0,
                            "combine_max_abs_error": 0})
    end = {"passed": True, "actual_model": True, "revision": "9" * 40,
           "dirty": 1, "checks": 100}
    hc_end = dict(end, kind="hc_cpu", actual_cases=9, success_call_allocations=0)
    ple = {"kind": "ple_actual_model", "passed": True, "actual_model_cases": 5,
           "model": hc["model_path"], "hidden_size": 2560, "hc_count": 4,
           "table_rows": 40000085, "heads": 16, "head_dim": 160, "table_row_bytes": 90,
           "table_type": "Q4_0", "history_length": 9, "conv_kernel": 4, "dilation": 3,
           "ple_eos_token_id": 248044, "tokenizer_eos_token_id": 248046,
           "layer_index_zero_based": 1, "hf_layer_one_based": 2,
           "canonical_tokens": 12, "canonical_prefixes": 13,
           "hot_allocations": 0, "token_vocab_size": 248320,
           "multipliers": [23703573157769, 20109073645365, 8052911324071],
           "maximum_valid_token_id_including_eos": 248319,
           "product_fits_int64_per_multiplier": [True, True, True],
           "nonnegative_hash_proven_for_all_valid_ids": True,
           "runtime_vs_formula_mismatched_row_count": 0, "negative_hash_count": 0,
           "mismatched_row_count": 0, "verify3": []}
    for name in ("hash_history_proven_exact", "conv_history_proven_against_raw_fp32",
                 "eos_does_not_reset_conv_history", "chunk_bitwise_exact",
                 "verify3_prefixes_bitwise_exact", "continuations_bitwise_exact"):
        ple[name] = True
    zero = {"elements": 100, "bit_mismatches": 0, "max_abs": 0, "max_scaled": 0}
    for name in ("raw_fp32_output_error", "raw_fp32_history_error",
                 "chunk_output_error", "chunk_history_error"):
        ple[name] = dict(zero)
    for accepts in range(3):
        ple["verify3"].append({"accepted_drafts": accepts, "prefix_slot": 1 + accepts,
                              "base_consumed_tokens": 4, "restored_consumed_tokens": 5 + accepts,
                              "continued_consumed_tokens": 6 + accepts,
                              "prefix_hash_and_conv_bitwise_exact": True,
                              "continuation_hash_and_conv_bitwise_exact": True,
                              "output_error": dict(zero), "history_error": dict(zero)})
    return [hc, hc_end, ple, dict(end, test="ple", actual_model_cases=5)]


class ResultsTest(unittest.TestCase):
    def collect(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            return MODULE.collect(raw)

    def test_valid(self):
        result = self.collect(fixture())
        self.assertEqual(result["kind"], "r2c_hc_ple")
        self.assertTrue(result["passed"])
        self.assertEqual(result["revision"], "9" * 40)
        # Independent raw-FP32 reference differences are bounded, not forced exact.
        rows = fixture()
        rows[2]["raw_fp32_output_error"].update(bit_mismatches=4, max_abs=2e-8, max_scaled=1e-8)
        self.assertTrue(self.collect(rows)["passed"])

    def test_invalid(self):
        changes = [(0, ["passed"], False), (1, ["actual_model"], False),
                   (1, ["revision"], "unknown"), (3, ["dirty"], True),
                   (3, ["revision"], "8" * 40), (0, ["token_cases"], 8),
                   (0, ["mtp_tap_preserved"], False), (2, ["hot_allocations"], 1),
                   (2, ["table_row_bytes"], 80), (2, ["dilation"], 1),
                   (2, ["ple_eos_token_id"], 248046),
                   (2, ["nonnegative_hash_proven_for_all_valid_ids"], False),
                   (2, ["product_fits_int64_per_multiplier"], [1, 1, 1]),
                   (2, ["negative_hash_count"], 1), (2, ["mismatched_row_count"], 1),
                   (2, ["runtime_vs_formula_mismatched_row_count"], 1),
                   (2, ["raw_fp32_output_error", "max_scaled"], 4e-6),
                   (2, ["chunk_output_error", "bit_mismatches"], 1),
                   (2, ["raw_fp32_history_error", "max_abs"], float("inf")),
                   (2, ["raw_fp32_history_error", "max_abs"], float("nan")),
                   (2, ["verify3", 0, "prefix_slot"], 0),
                   (2, ["verify3", 0, "prefix_slot"], True),
                   (2, ["verify3", 1, "continued_consumed_tokens"], 1)]
        for row, keys, value in changes:
            with self.subTest(keys=keys, value=value):
                rows = copy.deepcopy(fixture())
                target = rows[row]
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaises(ValueError):
                    self.collect(rows)
        for rows in (fixture()[:-1], list(reversed(fixture())), [{"passed": True}] * 4):
            with self.assertRaises(ValueError):
                self.collect(rows)

    def test_duplicate_key_and_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            raw.write_text('{"passed":true,"passed":false}\n')
            with self.assertRaises(ValueError):
                MODULE.collect(raw)
            raw.write_text(" " * (1024 * 1024 + 1))
            with self.assertRaises(ValueError):
                MODULE.collect(raw)


if __name__ == "__main__":
    unittest.main()
