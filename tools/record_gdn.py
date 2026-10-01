#!/usr/bin/env python3
"""Validate the completed R2b GDN fixture and append one scoped result."""
import argparse
import datetime
import json
import math
from pathlib import Path
import re


SOURCE_SCOPE = "loaded layer0 weights; eight synthetic projected tokens repeated for N128; not full inference"
ERROR_FIELDS = ("max_output_error", "max_state_error")
RATIO_FIELDS = ("output_gate_ratio", "state_gate_ratio")


def invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def records(path):
    result = []
    with Path(path).open(encoding="utf-8", newline="") as input_log:
        lines = input_log.read().split("\n")
    if lines[-1] == "":
        lines.pop()  # A terminating LF is not an extra JSONL row.
    for number, line in enumerate(lines, 1):
        try:
            item = json.loads(line, parse_constant=invalid_constant,
                              object_pairs_hook=unique_object)
        except ValueError as error:
            raise ValueError(f"GDN JSONL line {number}: {error}") from error
        if not isinstance(item, dict):
            raise ValueError(f"GDN JSONL line {number}: expected an object")
        result.append(item)
    return result


def expect(item, expected, variable=()):
    kind = expected["kind"]
    if set(item) != set(expected) | set(variable):
        raise ValueError(f"{kind}: unexpected or missing fields")
    for field, value in expected.items():
        if type(item[field]) is not type(value) or item[field] != value:
            raise ValueError(f"{kind}: unexpected {field}")


def finite_number(item, field, minimum=0, maximum=None, positive=False):
    value = item[field]
    if type(value) not in (int, float):
        raise ValueError(f"{item['kind']}: {field} must be a nonboolean number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite or value < minimum or (positive and value <= minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{item['kind']}: invalid {field}")


def parity(item):
    for field in ERROR_FIELDS:
        finite_number(item, field)
    for field in RATIO_FIELDS:
        finite_number(item, field, maximum=1)


def collect(path):
    """Return one r2b_gdn record; reject any incomplete or changed JSONL protocol."""
    path = Path(path)
    rows = records(path)
    if len(rows) != 32:
        raise ValueError("GDN fixture must contain exactly 32 rows")
    source = rows[0]
    expect(source, {"kind": "gdn_source", "model": "qwen38-keep1-Q4_0.gguf",
                    "layer": 0, "projection": "CPU raw FP32 oracle", "scope": SOURCE_SCOPE},
           ("revision", "dirty", "rms_epsilon"))
    if (not isinstance(source["revision"], str)
            or re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is None
            or type(source["dirty"]) is not int or source["dirty"] not in (0, 1)):
        raise ValueError("GDN source provenance")
    finite_number(source, "rms_epsilon", positive=True)

    offset = 1
    measurements, restore, rejections = [], [], []
    for device in (0, 1):
        for tokens in (1, 2, 3, 128):
            item = rows[offset]
            offset += 1
            expect(item, {"kind": "gdn_gpu", "device": device, "tokens": tokens,
                          "repeats": 20, "history_byte_parity": True},
                   ("reset_and_resident_ms", "cpu_oracle_ms") + ERROR_FIELDS + RATIO_FIELDS)
            finite_number(item, "reset_and_resident_ms", positive=True)
            finite_number(item, "cpu_oracle_ms", positive=True)
            parity(item)
            measurements.append(item)
            if tokens == 1:
                continue
            for dispatch in ("decode_steps", "resident_chunk", "decode_steps"):
                item = rows[offset]
                offset += 1
                expect(item, {"kind": "gdn_dispatch", "device": device, "tokens": tokens,
                              "path": dispatch, "repeats": 20, "history_byte_parity": True},
                       ("reset_and_resident_ms",) + ERROR_FIELDS + RATIO_FIELDS)
                finite_number(item, "reset_and_resident_ms", positive=True)
                parity(item)
                measurements.append(item)
        item = rows[offset]
        offset += 1
        expect(item, {"kind": "gdn_restore", "device": device, "cases": 8, "windows": 2,
                      "history_byte_parity": True, "split_chunk_parity": True},
               ERROR_FIELDS + RATIO_FIELDS)
        parity(item)
        restore.append(item)
        item = rows[offset]
        offset += 1
        expect(item, {"kind": "gdn_invalid", "device": device, "rejections": 11,
                      "chunk_atomic": True, "reuse": True})
        rejections.append(item)
    expect(rows[offset], {"kind": "gdn_complete", "passed": True})

    return {"kind": "r2b_gdn",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "revision": source["revision"], "dirty": source["dirty"], "source": source,
            "raw_log": str(path.resolve()),
            "scope": "actual layer0 weights + CPU raw FP32 projections + GPU recurrence through convolution/RMSNorm/sigmoid gate, before out_proj; eight synthetic projected tokens repeated for N128; not whole-model inference or end-to-end speed",
            "timing_scope": "reset_and_resident_ms: completed HIP events, mean of 20 zero-state reset + resident recurrence transactions, excluding projections/host transfers/readback; cpu_oracle_ms: one CPU FP32 recurrence, excluding projections",
            "state_abi": "FP32 [V-head][V-component][K-component] = [48][128][128], index (h*128+v)*128+k, K contiguous",
            "history_abi": "FP32 [qkv-feature][age] = [10240][3], index feature*3+age, oldest -> newest RAW projections; no padding, ring index or layout conversion",
            "head_mapping": "GGUF tiled V-head h uses Q/K head h%16; all V-side tensors share this order",
            "prefix_abi": {"slot_0": "pre-chunk state", "slot_n": "after n consumed inputs, not emitted tokens",
                           "verify_inputs": "one pending input + two drafts", "accept_slot": "1 + a",
                           "accepted_drafts_a": [0, 1, 2]},
            "normalization": {"qk_l2_epsilon": 1e-6, "rms_epsilon": source["rms_epsilon"],
                              "convolution_activation": "SiLU", "output_gate": "sigmoid"},
            "fixture_gates": {"output_abs": 2e-4, "output_rel": 2e-4,
                              "state_abs": 2e-5, "state_rel": 2e-4,
                              "rule": "error <= abs + rel * |ref|", "gate_ratio_limit": 1,
                              "frozen_before_first_gpu_run": True},
            "donors": {"mx": "dcd685463d597d31f5ca759d32c94592a2740fa4",
                       "furnace": "905021dbad71c5056ef51f9fd45d545403fc989c",
                       "reinstinct": "0b79e326351d90d4554a1c18df92da5d0ab692e8"},
            "reference": "transformers a005fc82babfe8871d87746decad2dbee100a125",
            "measurements": measurements, "restore": restore, "rejections": rejections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.raw)
    with args.results.open("a", encoding="utf-8") as output:
        output.write(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
