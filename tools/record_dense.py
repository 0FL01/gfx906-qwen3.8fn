#!/usr/bin/env python3
"""Validate completed core-dense/core-head logs and append one R3a result."""
import argparse
import datetime
import json
import math
from pathlib import Path
import re


MAX_RAW_BYTES = 4 * 1024 * 1024
DENSE_ROWS, HEAD_ROWS = 20, 10
DENSE_SCOPE = "full unchanged F32 and BF16 tensors converted once to F32; deterministic raw-FP32 inputs; ascending FP32 CPU oracle"
DENSE_TIMING = "completed HIP events around resident dense linear only; excludes allocation, conversion, upload, CPU reference, quantization, readback and warmup"
HEAD_SCOPE = "full actual LM head with deterministic common-Q8 input; sampled CPU oracle, full finite and prefix checks"
HEAD_TIMING = "completed resident linear events; excludes upload, quantization, reset, CPU oracle and readback"
DENSE_COUNTERS = {
    "matrix_cases": 28, "synthetic_cases": 20, "actual_cases": 8,
    "host_rejects": 26, "prefix_cases": 21,
    "matrix_elements": 2220648, "prefix_elements": 66288}
DENSE_TENSORS = (
    ("blk.0.ssm_alpha.weight", "F32", 2560, 48, 4),
    ("blk.3.indexer.k_proj.weight", "BF16", 2560, 128, 2))
SAMPLE_ROWS = [0, 1, 7, 31, 63, 127, 255, 511, 1023, 4095, 8191, 16383,
               32767, 65535, 131071, 200003, 248043, 248044, 248045, 248046,
               248047, 248127, 248318, 248319]
UINT64_MAX = 2**64 - 1


def invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite JSON number: {value}")
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def records(path, count):
    """Bound even a growing input; require exactly count LF/CRLF object rows."""
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_RAW_BYTES + 1)
    if len(raw) > MAX_RAW_BYTES:
        raise ValueError("dense/head raw log exceeds 4 MiB")
    if not raw.endswith(b"\n"):
        raise ValueError("dense/head raw log missing completed final line")
    lines = raw.split(b"\n")[:-1]
    if len(lines) != count:
        raise ValueError(f"dense/head raw log must contain exactly {count} rows")
    result = []
    for number, line in enumerate(lines, 1):
        try:
            item = json.loads(line.decode("utf-8"), parse_constant=invalid_constant,
                              parse_float=finite_float, object_pairs_hook=unique_object)
        except (ValueError, RecursionError) as error:
            raise ValueError(f"dense/head JSONL line {number}: {error}") from error
        if type(item) is not dict:
            raise ValueError(f"dense/head JSONL line {number}: expected an object")
        result.append(item)
    return result


def exact(value, expected, label):
    """Fixed values AND types: neither bool nor float can be an int counter."""
    if type(value) is not type(expected):
        raise ValueError(f"{label}: incorrect type")
    if type(expected) is dict:
        expect(value, expected, label=label)
    elif type(expected) is list:
        if len(value) != len(expected):
            raise ValueError(f"{label}: incorrect length")
        for index, (got, want) in enumerate(zip(value, expected)):
            exact(got, want, f"{label}[{index}]")
    elif value != expected:
        raise ValueError(f"{label}: unexpected value")


def expect(item, expected, variable=(), label=None):
    label = label or expected.get("kind", "dense/head")
    if type(item) is not dict or set(item) != set(expected) | set(variable):
        raise ValueError(f"{label}: unexpected or missing fields")
    for field, value in expected.items():
        exact(item[field], value, f"{label}.{field}")


def number(value, label, maximum=None, positive=False, integer=False):
    if type(value) not in ((int,) if integer else (int, float)):
        raise ValueError(f"{label}: expected a nonboolean {'integer' if integer else 'number'}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite or value < 0 or (positive and value <= 0)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{label}: invalid numeric value")


def provenance(source):
    if type(source["revision"]) is not str or re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is None:
        raise ValueError("dense/head invalid compiled revision")
    number(source["dirty"], "dense/head compiled dirty", maximum=1, integer=True)


def text(value, label):
    if (type(value) is not str or not value
            or any(ord(c) < 32 or 0xd800 <= ord(c) <= 0xdfff for c in value)):
        raise ValueError(f"{label}: invalid string")


def errors(item, elements, label):
    expect(item, {"elements": elements}, ("max_abs", "max_bound_ratio"), label)
    number(item["max_abs"], f"{label}.max_abs")
    number(item["max_bound_ratio"], f"{label}.max_bound_ratio", maximum=1)


def dense_views(actual):
    if type(actual) is not list or len(actual) != len(DENSE_TENSORS):
        raise ValueError("dense_source.actual: expected both full tensors")
    origins, ranges = set(), []
    for view, (name, kind, width, rows, element_bytes) in zip(actual, DENSE_TENSORS):
        expect(view, {
            "type": kind, "tensor": name, "rank": 2, "dimensions": [width, rows],
            "strides_bytes": [element_bytes, width * element_bytes],
            "elements": width * rows, "bytes": width * rows * element_bytes,
            "converted_type": "F32", "converted_bytes": width * rows * 4},
            ("file_offset", "relative_offset"), "dense_source.actual")
        for field in ("file_offset", "relative_offset"):
            number(view[field], f"dense_source.actual.{field}", maximum=UINT64_MAX, integer=True)
        origin = view["file_offset"] - view["relative_offset"]
        end = view["file_offset"] + view["bytes"]
        if origin <= 0 or end > UINT64_MAX:
            raise ValueError("dense_source.actual: invalid offset/extent")
        origins.add(origin)
        ranges.append((view["file_offset"], end))
    # Model file size/alignment/data_offset are NOT emitted. Only a common
    # positive data origin, uint64 extents and nonoverlap can be checked here.
    if len(origins) != 1:
        raise ValueError("dense_source.actual: inconsistent data origin")
    ranges.sort()
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("dense_source.actual: overlapping tensors")


def dense(rows):
    source = rows[0]
    expect(source, {
        "kind": "dense_source", "absolute_gate": 2e-4, "relative_gate": 2e-4,
        "scope": DENSE_SCOPE, "timing_scope": DENSE_TIMING,
        "tokens": [1, 2, 3, 128], "repeats": 20, "measurements": 16, "protocol_rows": 20,
        "correctness_per_device": DENSE_COUNTERS,
        "synthetic_shapes": [[1, 3], [7, 5], [3, 1], [16384, 3], [3, 16384]]},
        ("revision", "dirty", "model", "actual"))
    provenance(source)
    text(source["model"], "dense_source.model")
    dense_views(source["actual"])
    for device, item in enumerate(rows[1:3]):
        expect(item, {
            "kind": "dense_correctness", "device": device,
            **{key: value for key, value in DENSE_COUNTERS.items()
               if key not in ("matrix_elements", "prefix_elements")},
            "all_finite": True, "canaries": True, "immutable": True, "passed": True},
            ("error", "prefix_error"))
        errors(item["error"], 2220648, "dense_correctness.error")
        errors(item["prefix_error"], 66288, "dense_correctness.prefix_error")
    offset = 3
    for device in (0, 1):
        for name, kind, width, count, _ in DENSE_TENSORS:
            for n in (1, 2, 3, 128):
                item = rows[offset]
                expect(item, {
                    "kind": "dense_measurement", "device": device, "type": kind, "tensor": name,
                    "width": width, "rows": count, "tokens": n, "repeats": 20,
                    "all_finite": True, "canaries": True, "immutable": True},
                    ("resident_ms", "error"))
                number(item["resident_ms"], "dense_measurement.resident_ms", positive=True)
                errors(item["error"], count * n * 20, "dense_measurement.error")
                offset += 1
    expect(rows[offset], {"kind": "dense_complete", "measurements": 16, "rows": 20, "passed": True})
    return {"source": source, "correctness": rows[1:3],
            "measurements": rows[3:offset], "complete": rows[offset]}


def head(rows):
    source = rows[0]
    expect(source, {
        "kind": "head_source", "model_variant": "qwen38-keep1-Q4_0", "type": "Q6_K",
        "input": 2560, "output": 248320, "bytes": 521472000,
        "absolute_gate": 2e-4, "relative_gate": 2e-5, "sample_rows": SAMPLE_ROWS,
        "scope": HEAD_SCOPE, "timing_scope": HEAD_TIMING}, ("revision", "dirty"))
    provenance(source)
    offset, measurements, correctness = 1, [], []
    for device in (0, 1):
        for n in (1, 2, 3):
            item = rows[offset]
            expect(item, {
                "kind": "head_measurement", "device": device, "tokens": n,
                "rows": 248320, "repeats": 20, "all_finite": True,
                "canaries": True, "q8_bytes_exact": True}, ("resident_ms", "error"))
            number(item["resident_ms"], "head_measurement.resident_ms", positive=True)
            errors(item["error"], 24 * n * 20, "head_measurement.error")
            measurements.append(item)
            offset += 1
        item = rows[offset]
        expect(item, {"kind": "head_correctness", "device": device, "host_rejects": 11,
                      "prefix_elements": 744960, "prefix_exact": True, "passed": True})
        correctness.append(item)
        offset += 1
    expect(rows[offset], {"kind": "head_complete", "measurements": 6, "rows": 10, "passed": True})
    return {"source": source, "correctness": correctness,
            "measurements": measurements, "complete": rows[offset]}


def collect(dense_path, head_path):
    """Return ONE r3a_dense_head record after both frozen protocols validate."""
    dense_result = dense(records(dense_path, DENSE_ROWS))
    head_result = head(records(head_path, HEAD_ROWS))
    ds, hs = dense_result["source"], head_result["source"]
    if ds["revision"] != hs["revision"] or ds["dirty"] != hs["dirty"]:
        raise ValueError("dense/head compiled source revision/dirty mismatch")
    return {
        "kind": "r3a_dense_head", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": ds["revision"], "dirty": ds["dirty"],
        "raw_logs": {"dense": str(Path(dense_path).resolve()), "head": str(Path(head_path).resolve())},
        "scope": "diagnostic component qualification: full unchanged dense matmuls and actual LM head with deterministic inputs; sampled CPU common-Q8 head oracle with full-output finite/prefix checks; not full-network teacher-forced logits, A/B speedup or inference performance",
        "timing_scope": "arithmetic mean of 20 individually completed HIP event intervals; every repetition validated; resident linear only, excluding allocation, conversion, transfers, quantization, CPU oracle, output reset, readback, validation and warmup",
        "dense_contract": {
            "conversion": "exactly once before GPU work: F32 bits preserved; little-endian BF16 bits shifted left16 into F32; no changed weight values or activation quantization",
            "storage": "unchanged complete GGUF tensors [K,M], contiguous K per output row; prepared F32 weights, input [N,K] and output [N,M]",
            "layout": "rocBLAS column-major A[K,M], transpose-A / no-transpose-B; lda=ldb=K, ldc=M",
            "alpha": 1, "beta": 0,
            "input": "float((k*17+column*23+3)%67-33)/64; same deterministic raw-FP32 columns for N1/2/3/128",
            "reference": "ascending unfused FP32 multiply/add over the once-converted F32 view; every output compared at every completed repetition",
            "prefix": "bounded comparisons N2->N1, N3->N1, N3->N2; not byte equality between different GEMM reduction orders",
            "checks": "full finite outputs, poisoned old beta output, output canaries and byte-immutable weights/input; 26 host rejects per device with whole-allocation readbacks"},
        "head_contract": {
            "tensor": "output.weight", "dimensions": [2560, 248320], "bytes": 521472000,
            "storage": "unchanged canonical Q6_K: block256/210 bytes, ql[128], qh[64], signed scales[16], FP16 d; 2100 bytes per row",
            "input": "0.5f*sin(float(i)*0.137f)+0.125f*cos(float(i)*0.071f); contiguous column-major inputs for N1/2/3",
            "activation_abi": "Q8_1: 36 bytes per block32; FP16 d, FP16 raw input sum s, 32 signed int8 codes",
            "codes": "FP32 d=amax/127 selects roundf(x/d) codes, not rounded stored half d",
            "raw_sum": "ascending XOR32 FP32 input sum then FP16 RNE; not sum of quantized codes",
            "arithmetic": "canonical signed 6-bit codes and int8 subscales; int32 scaled dot4, paired FP32 activation scales then weight scale",
            "reference": "independent scalar canonical common-Q8 decoder on identical Q8 bytes, 24 exact sampled row IDs; not a full-output CPU oracle or dequantized-F32 parity",
            "prefix": "full output byte equality N1->N2 and N2->N3; 744960 compared elements per device",
            "checks": "full-output finite and canaries every repetition; producer and post-run Q8 byte parity; 11 host shape/type rejects per device",
            "provenance_limits": "head_source emits model_variant, not model path/offsets; dense_source emits offsets, not model file size/alignment; cross-log identity is compiled revision/dirty"},
        "fixture_gates": {
            "dense": {"absolute": 2e-4, "relative_cpu_magnitude": 2e-4,
                      "frozen_before_gpu": True, "max_bound_ratio_limit": 1},
            "head": {"absolute": 2e-4, "relative_cpu_magnitude": 2e-5,
                     "frozen_before_gpu": True, "max_bound_ratio_limit": 1},
            "bound": "absolute+relative_cpu_magnitude*abs(CPU_reference)"},
        "donors": {"mx": "dcd685463d597d31f5ca759d32c94592a2740fa4",
                   "furnace": "905021dbad71c5056ef51f9fd45d545403fc989c",
                   "reinstinct": "0b79e326351d90d4554a1c18df92da5d0ab692e8"},
        "dense": dense_result, "head": head_result, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    # Exit status is checked by the caller, not invented as stdout metadata.
    # Validate AND serialize both complete logs before opening the destination.
    result = json.dumps(collect(args.dense, args.head), allow_nan=False, separators=(",", ":"))
    with args.results.open("a", encoding="utf-8") as output:
        output.write(result + "\n")


if __name__ == "__main__":
    main()
