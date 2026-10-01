#!/usr/bin/env python3
"""Record completed CPU HC/PLE semantic fixtures, never inference throughput."""
import argparse
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(values):
    result = {}
    for key, value in values:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def bad_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def numeric(value, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            "invalid numeric measurement")


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif type(value) is float:
        require(math.isfinite(value), "nonfinite measurement")


def flags(row, names):
    for name in names:
        require(row[name] is True, f"failed {name}")


def error(row, exact=False):
    numeric(row["elements"], 1)
    require(type(row["elements"]) is int, "invalid element count")
    require(type(row["bit_mismatches"]) is int and row["bit_mismatches"] >= 0,
            "invalid mismatch count")
    numeric(row["max_abs"])
    numeric(row["max_scaled"])
    require(row["max_scaled"] <= 3e-6, "PLE reference gate failed")
    if exact:
        require(row["bit_mismatches"] == row["max_abs"] == row["max_scaled"] == 0,
                "bitwise gate failed")


def collect(path):
    path = Path(path)
    require(path.stat().st_size <= 1024 * 1024, "raw fixture too large")
    try:
        rows = [json.loads(line, object_pairs_hook=pairs, parse_constant=bad_constant)
                for line in path.read_text().splitlines()]
        require(len(rows) == 4, "incomplete HC/PLE protocol")
        hc, hc_end, ple, ple_end = rows
        require(hc["kind"] == "hc_actual_model" and hc_end["kind"] == "hc_cpu" and
                ple["kind"] == "ple_actual_model" and ple_end["test"] == "ple",
                "incorrect fixture order")
        for row in rows:
            flags(row, ["passed"])
            finite_tree(row)
        for row in (hc_end, ple_end):
            flags(row, ["actual_model"])
            revision = row["revision"]
            require(isinstance(revision, str) and len(revision) == 40 and
                    all(c in "0123456789abcdef" for c in revision), "invalid revision")
            require(type(row["dirty"]) is int and row["dirty"] in (0, 1), "invalid dirty flag")
            numeric(row["checks"], 1)
        require((hc_end["revision"], hc_end["dirty"]) ==
                (ple_end["revision"], ple_end["dirty"]), "mixed build provenance")
        require(hc_end["actual_cases"] == hc["token_cases"] == 9 and
                ple_end["actual_model_cases"] == ple["actual_model_cases"] == 5,
                "incorrect actual case count")
        require(hc["geometry"]["hidden_size"] == ple["hidden_size"] == 2560 and
                hc["geometry"]["branches"] == ple["hc_count"] == 4 and
                hc["geometry"]["low_rank"] == 320, "incorrect target geometry")
        flags(hc, ["mtp_tap_preserved", "root_injection_rejected"])
        require(hc_end["success_call_allocations"] == 0 and ple["hot_allocations"] == 0,
                "hot allocations")
        require(hc["model_path"] == ple["model"], "mixed models")
        require([c["mixer"] for c in hc["cases"]] ==
                ["blk.0.hc_attn", "blk.0.hc_ffn", "output_hc"], "incorrect HC cases")
        for i, case in enumerate(hc["cases"]):
            require(case["tokens"] == [1, 2, 3] and case["injection"] is (i != 2),
                    "incorrect HC token/injection cases")
            flags(case, ["prefix_exact", "residual_preserved", "finite_outputs"])
            require(case["success_call_allocations"] == 0, "HC hot allocations")
            for name in ("mix_max_abs_error", "injection_max_abs_error", "combine_max_abs_error"):
                numeric(case[name])
        require(ple["table_rows"] == 40000085 and ple["heads"] == 16 and
                ple["head_dim"] == 160 and ple["table_row_bytes"] == 90 and
                ple["table_type"] == "Q4_0" and ple["history_length"] == 9 and
                ple["conv_kernel"] == 4 and ple["dilation"] == 3 and
                ple["ple_eos_token_id"] == 248044 and ple["tokenizer_eos_token_id"] == 248046,
                "incorrect PLE geometry/EOS")
        require(ple["layer_index_zero_based"] == 1 and ple["hf_layer_one_based"] == 2 and
                ple["canonical_tokens"] == 12 and ple["canonical_prefixes"] == 13,
                "incorrect PLE layer/prefix count")
        flags(ple, ["hash_history_proven_exact", "conv_history_proven_against_raw_fp32",
                    "eos_does_not_reset_conv_history", "chunk_bitwise_exact",
                    "verify3_prefixes_bitwise_exact", "continuations_bitwise_exact"])
        require(ple["runtime_vs_formula_mismatched_row_count"] == 0, "incorrect PLE rows")
        maximum = max(ple["token_vocab_size"] - 1, ple["ple_eos_token_id"])
        multipliers = ple["multipliers"]
        require(len(multipliers) == 3 and all(type(m) is int and 0 <= m < 2**64 for m in multipliers),
                "invalid multiplier bits")
        proof = [maximum * m <= 2**63 - 1 for m in multipliers]
        require(type(ple["token_vocab_size"]) is int and ple["token_vocab_size"] > 0 and
                all(type(p) is bool for p in ple["product_fits_int64_per_multiplier"]),
                "invalid proof operand types")
        require(ple["maximum_valid_token_id_including_eos"] == maximum and
                ple["product_fits_int64_per_multiplier"] == proof and
                ple["nonnegative_hash_proven_for_all_valid_ids"] is all(proof),
                "incorrect signed/unsigned proof")
        if all(proof):
            require(ple["negative_hash_count"] == ple["mismatched_row_count"] == 0,
                    "nonnegative proof contradicts diagnostic")
        error(ple["raw_fp32_output_error"])
        error(ple["raw_fp32_history_error"])
        error(ple["chunk_output_error"], exact=True)
        error(ple["chunk_history_error"], exact=True)
        require([r["accepted_drafts"] for r in ple["verify3"]] == [0, 1, 2],
                "incorrect acceptance cases")
        for case in ple["verify3"]:
            require(all(type(case[k]) is int for k in
                        ("accepted_drafts", "prefix_slot", "restored_consumed_tokens",
                         "base_consumed_tokens", "continued_consumed_tokens")),
                    "invalid restore counts")
            require(case["prefix_slot"] == 1 + case["accepted_drafts"] and
                    case["restored_consumed_tokens"] == case["base_consumed_tokens"] + case["prefix_slot"] and
                    case["continued_consumed_tokens"] == case["restored_consumed_tokens"] + 1,
                    "incorrect consumed-input restore")
            flags(case, ["prefix_hash_and_conv_bitwise_exact", "continuation_hash_and_conv_bitwise_exact"])
            error(case["output_error"], exact=True)
            error(case["history_error"], exact=True)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("malformed fixture schema") from exc
    return {"kind": "r2c_hc_ple", "revision": hc_end["revision"], "dirty": hc_end["dirty"],
            "raw_log": str(path.resolve()), "scope": "CPU raw-FP32 semantic actual-weight fixtures; not GPU projections or inference throughput",
            "hc": hc, "hc_tests": hc_end, "ple": ple, "ple_tests": ple_end,
            "timing_scope": "correctness only; no performance claim", "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    record = collect(args.raw)
    with Path(args.results).open("a") as output:
        output.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
