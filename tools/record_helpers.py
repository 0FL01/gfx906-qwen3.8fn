#!/usr/bin/env python3
"""Validate completed core-linear/core-blocks logs and append one R2e result."""
import argparse
import datetime
import json
import math
from pathlib import Path
import re


MAX_RAW_BYTES = 4 * 1024 * 1024
LINEAR_ROWS, BLOCKS_ROWS = 214, 20
TYPES = ("Q4_0", "Q4_1", "Q5_0", "Q8_0", "Q6_K")
LINEAR_SCOPE = "resident common-Q8 linear; synthetic matrices and first up-to-nine unchanged actual weight rows; not full projection/inference"
BLOCKS_SCOPE = "isolated_resident_fp32_primitives_same_synthetic_projections_actual_gamma_conv"
BLOCKS_GATES = {
    "frozen_before_gpu": True, "absolute": 2e-4, "relative_cpu_magnitude": 2e-4,
    "groups": ["rms", "hc", "ple", "history"], "max_bound_ratio_limit": 1,
    "finite_required": True, "exact_history": "identical_norm_gated_inputs_only"}
BLOCKS_COUNTERS = {
    "norm_cases": 86, "activation_cases": 5, "mix_cases": 5, "injection_cases": 5,
    "combine_cases": 5, "gate_cases": 8, "conv_cases": 19, "head_cases": 5,
    "broadcast_cases": 56, "signed_zero_cases": 2, "large_finite_cases": 1,
    "prefix_slots": 21, "restore_cases": 10, "repeated_accept0_cases": 4,
    "mask_cases": 2, "readonly_active_checks": 65, "publication_checks": 73,
    "rejected_publications": 46, "numeric_rejects": 46, "host_rejects": 224,
    "readonly_alias_cases": 2, "valid_reuses": 46, "numeric_invalid_cases": 39,
    "numeric_arithmetic_cases": 6, "numeric_combined_cases": 1}
POSITIVE_COUNTERS = ("immutable_checks", "canary_checks")
COMMON_PARITY = {name: True for name in (
    "cpu_equations", "same_projections", "finite", "byte_canaries", "immutable_inputs")}
CORRECTNESS_PARITY = {**COMMON_PARITY, **{name: True for name in (
    "root_head_no_injection", "original_widened_tap_unchanged", "history_exact_common_inputs",
    "history_bounded_own_parallel_rms", "all_four_prefix_tails", "restore_1_plus_accepted",
    "repeated_accept0", "keep_mask_0_1_no_EOS_reset", "active_readonly_until_publish",
    "rejected_entire_active_and_marker_unchanged", "valid_reuse")}}
TIMING_EXCLUDES = ["model_load", "fixture_allocation", "CPU_reference", "H2D", "D2H",
                   "history_restore", "output_reset", "error_clear", "validation", "warmup"]
TENSOR_SPECS = (
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
    ("blk.3.attn_k_norm.weight", "F32", [256]))
# Gate::elements counts isolated checks plus all 46 valid reuses. Ordinary
# isolated N sums to 134; the additional large case is N=3. Transactions add
# 28 convolution-output tokens and 22 history tails; isolated adds 18 tails.
SUMMARY_ELEMENTS = {
    "rms": 134 * (7 * 10240 + 28544) + 3 * 7 * 10240 + 46 * 3 * 4 * 10240,
    "hc": 137 * (320 + 2560 + 4 + 10240 + 2560) + 3 * 2560
          + 46 * 3 * (320 + 2560 + 4 + 10240),
    "ple": (2 * 137 + 3 + 6 + 3 + 28 + 46 * 6) * 10240,
    "history": (18 + 22 + 46 * 2) * 92160}


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
    """Bound reads even if a log grows; require LF/CRLF-terminated object rows."""
    with Path(path).open("rb") as stream:
        if Path(path).stat().st_size > MAX_RAW_BYTES:
            raise ValueError("helpers raw log exceeds 4 MiB")
        raw = stream.read(MAX_RAW_BYTES + 1)
    if len(raw) > MAX_RAW_BYTES:
        raise ValueError("helpers raw log exceeds 4 MiB")
    if not raw.endswith(b"\n"):
        raise ValueError("helpers raw log missing completed final line")
    lines = raw.split(b"\n")[:-1]
    if len(lines) != count:
        raise ValueError(f"helpers raw log must contain exactly {count} rows")
    result = []
    for number, line in enumerate(lines, 1):
        try:
            item = json.loads(line.decode("utf-8"), parse_constant=invalid_constant,
                              parse_float=finite_float, object_pairs_hook=unique_object)
        except (ValueError, RecursionError) as error:
            raise ValueError(f"helpers JSONL line {number}: {error}") from error
        if type(item) is not dict:
            raise ValueError(f"helpers JSONL line {number}: expected an object")
        result.append(item)
    return result


def exact(value, expected, label):
    """Fixed schema comparison, including types: True is never a counter of 1."""
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
    label = label or expected.get("kind", "helpers")
    if type(item) is not dict or set(item) != set(expected) | set(variable):
        raise ValueError(f"{label}: unexpected or missing fields")
    for field, value in expected.items():
        exact(item[field], value, f"{label}.{field}")


def number(value, label, minimum=0, maximum=None, positive=False, integer=False):
    if type(value) not in ((int,) if integer else (int, float)):
        raise ValueError(f"{label}: expected a nonboolean {'integer' if integer else 'number'}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite or value < minimum or (positive and value <= minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{label}: invalid numeric value")


def text(value, label):
    if (type(value) is not str or not value
            or any(ord(c) < 32 or 0xd800 <= ord(c) <= 0xdfff for c in value)):
        raise ValueError(f"{label}: invalid string")


def revision(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("helpers invalid compiled revision")


def error_metrics(item, label):
    number(item["max_abs"], f"{label}.max_abs")
    number(item["max_bound_ratio"], f"{label}.max_bound_ratio", maximum=1)


def errors(item, elements):
    expect(item, {}, elements, "blocks.error")
    for group, count in elements.items():
        gate = item[group]
        expect(gate, {"elements": count}, ("max_abs", "max_bound_ratio"), f"blocks.error.{group}")
        error_metrics(gate, f"blocks.error.{group}")
        if count == 0 and (gate["max_abs"] != 0 or gate["max_bound_ratio"] != 0):
            raise ValueError(f"blocks.error.{group}: unused group must be zero")


def counters(item, expected):
    expect(item, expected, POSITIVE_COUNTERS, "blocks.counters")
    for field in POSITIVE_COUNTERS:
        number(item[field], f"blocks.counters.{field}", integer=True, positive=True)


def linear(rows):
    source = rows[0]
    expect(source, {"kind": "linear_source", "absolute_gate": 2e-4,
                    "relative_gate": 2e-5, "scope": LINEAR_SCOPE},
           ("revision", "dirty", "model", "actual"))
    revision(source["revision"])
    number(source["dirty"], "linear_source.dirty", integer=True, maximum=1)
    text(source["model"], "linear_source.model")
    actual = source["actual"]
    if type(actual) is not list or len(actual) != 5:
        raise ValueError("linear_source.actual: expected all five types")
    names = set()
    for kind, matrix in zip(TYPES, actual):
        expect(matrix, {"type": kind}, ("tensor", "width", "rows"), "linear_source.actual")
        text(matrix["tensor"], "linear_source.actual.tensor")
        if matrix["tensor"] == "synthetic" or matrix["tensor"] in names:
            raise ValueError("linear_source.actual: invalid actual tensor name")
        names.add(matrix["tensor"])
        number(matrix["width"], "linear_source.actual.width", integer=True, positive=True, maximum=16384)
        number(matrix["rows"], "linear_source.actual.rows", integer=True, positive=True, maximum=9)
        if matrix["width"] % (256 if kind == "Q6_K" else 32):
            raise ValueError("linear_source.actual: invalid block width")

    offset, correctness, measurements = 1, [], []
    for device in (0, 1):
        matrices = []
        for kind in TYPES:
            widths = (256, 512, 1024, 1280, 2560, 16384) if kind == "Q6_K" else (32, 64, 2048, 2080, 2560, 16384)
            matrices.extend({"type": kind, "tensor": "synthetic", "width": w, "rows": r}
                            for w, r in zip(widths, (1, 3, 9, 7, 9, 9)))
        matrices.extend(actual)
        for matrix in matrices:
            for n in (1, 2, 3):
                item = rows[offset]
                expect(item, {"kind": "linear_measurement", "device": device, **matrix,
                              "columns": n, "repeats": 20, "q8_byte_parity": True,
                              "canary": True, "immutable": True},
                       ("resident_ms", "max_abs", "max_bound_ratio"))
                number(item["resident_ms"], "linear_measurement.resident_ms", positive=True)
                error_metrics(item, "linear_measurement")
                measurements.append(item)
                offset += 1
        item = rows[offset]
        expect(item, {"kind": "linear_correctness", "device": device, "matrix_cases": 110,
                      "quantization_cases": 123, "numeric_rejects": 5, "host_rejects": 32,
                      "corner_cases": 5, "passed": True})
        correctness.append(item)
        offset += 1
    expect(rows[offset], {"kind": "linear_complete", "measurements": 210,
                          "rows": LINEAR_ROWS, "passed": True})
    return {"source": source, "correctness": correctness,
            "measurements": measurements, "complete": rows[offset]}


def tensor_views(source):
    views = source["actual_tensors"]
    if type(views) is not list or len(views) != len(TENSOR_SPECS):
        raise ValueError("blocks_source.actual_tensors: incomplete actual tensor proof")
    origins, ranges = set(), []
    for view, (name, kind, dims) in zip(views, TENSOR_SPECS):
        stride, strides = (4 if kind == "F32" else 2), []
        for dim in dims:
            strides.append(stride)
            stride *= dim
        expect(view, {"name": name, "type": kind, "rank": len(dims), "dimensions": dims,
                      "strides_bytes": strides, "elements": math.prod(dims), "bytes": stride},
               ("relative_offset", "file_offset"), "blocks_source.actual_tensors")
        for field in ("relative_offset", "file_offset"):
            number(view[field], f"blocks_source.actual_tensors.{field}", integer=True, maximum=2**64 - 1)
        origin = view["file_offset"] - view["relative_offset"]
        end = view["file_offset"] + view["bytes"]
        if origin <= 0 or end > source["model_file_bytes"]:
            raise ValueError("blocks_source.actual_tensors: invalid offset/file extent")
        origins.add(origin)
        ranges.append((view["file_offset"], end))
    # data_offset and general.alignment are not emitted. A common positive data
    # origin, canonical strides, disjoint extents and file bounds are provable;
    # do not invent a fixed GGUF alignment or absolute tensor offsets.
    if len(origins) != 1:
        raise ValueError("blocks_source.actual_tensors: inconsistent data origin")
    ranges.sort()
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("blocks_source.actual_tensors: overlapping tensor extents")


def blocks(rows):
    source = rows[0]
    expect(source, {
        "kind": "blocks_source", "protocol": 1, "scope": BLOCKS_SCOPE,
        "build_provenance_complete": True, "gates": BLOCKS_GATES,
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
        "correctness_exact_counters": BLOCKS_COUNTERS,
        "correctness_positive_counters": list(POSITIVE_COUNTERS),
        "host_validation": {"expected_status": "hipErrorInvalidValue",
                            "checks": ["geometry", "epsilon", "null_required", "all_writable_overlap", "misalignment", "wrapping"],
                            "invalid_addresses_enqueued": False},
        "timing_scope": "individual_completed_HIP_event_intervals_resident_primitives_only",
        "launch_counts": {"HC": 5, "PLE": 7}, "full_pipeline_qualification": False,
        "request_performance_claim": False},
        ("revision", "dirty", "model_path", "model_file_bytes", "actual_metadata", "actual_tensors", "devices"))
    revision(source["revision"])
    if type(source["dirty"]) is not bool:
        raise ValueError("blocks_source.dirty: expected compiled boolean")
    text(source["model_path"], "blocks_source.model_path")
    number(source["model_file_bytes"], "blocks_source.model_file_bytes", integer=True, positive=True, maximum=2**64 - 1)
    expect(source["actual_metadata"], {
        "architecture": "qwen4exp", "hidden": 2560, "branches": 4, "rank": 320,
        "rms_epsilon_type": "FLOAT32", "ple_layer": 1, "ple_embedding_dim": 2560,
        "ple_conv_kernel": 4, "ple_ngram_size": 3, "ple_history_length": 9},
        ("rms_epsilon",), "blocks_source.actual_metadata")
    number(source["actual_metadata"]["rms_epsilon"], "blocks_source.rms_epsilon", positive=True)
    tensor_views(source)
    devices = source["devices"]
    if type(devices) is not list or len(devices) != 2:
        raise ValueError("blocks_source.devices: expected devices 0/1")
    for device, proof in enumerate(devices):
        expect(proof, {"device": device, "wave_size": 64}, ("arch",), "blocks_source.devices")
        text(proof["arch"], "blocks_source.devices.arch")
        if proof["arch"] != "gfx906" and not proof["arch"].startswith("gfx906:"):
            raise ValueError("blocks_source.devices: expected gfx906")

    for device, item in enumerate(rows[1:3]):
        expect(item, {"kind": "blocks_correctness", "protocol": 1, "device": device,
                      "parity": CORRECTNESS_PARITY, "passed": True}, ("counters", "error"))
        counters(item["counters"], BLOCKS_COUNTERS)
        errors(item["error"], SUMMARY_ELEMENTS)
    offset = 3
    for device in (0, 1):
        for n in (1, 2, 3, 128):
            for pipeline in ("HC", "PLE"):
                ple = pipeline == "PLE"
                item = rows[offset]
                expect(item, {"kind": "blocks_measurement", "protocol": 1, "device": device,
                              "tokens": n, "pipeline": pipeline, "repeats": 20, "validated_repetitions": 20,
                              "launchcount": 7 if ple else 5, "timing_excludes": TIMING_EXCLUDES,
                              "history_comparison": "bounded_own_parallel_RMS" if ple else "not_applicable",
                              "parity": {**COMMON_PARITY, "history_bounded_own_parallel_rms": True if ple else None,
                                         "passed": True}}, ("resident_ms", "error", "counters"))
                number(item["resident_ms"], "blocks_measurement.resident_ms", positive=True)
                errors(item["error"], {
                    "rms": 20 * n * 10240 * (3 if ple else 1),
                    "hc": 0 if ple else 20 * n * (320 + 2560 + 4 + 10240),
                    "ple": 20 * n * 10240 * 2 if ple else 0,
                    "history": 20 * 92160 * 2 if ple else 0})
                counters(item["counters"], {**dict.fromkeys(BLOCKS_COUNTERS, 0),
                                            "publication_checks": 20 if ple else 0})
                offset += 1
    expect(rows[offset], {"kind": "blocks_complete", "protocol": 1, "correctness_rows": 2,
                          "measurement_rows": 16, "measurement_repetitions": 320,
                          "jsonl_rows": BLOCKS_ROWS, "passed": True})
    return {"source": source, "correctness": rows[1:3],
            "measurements": rows[3:offset], "complete": rows[offset]}


def collect(linear_path, blocks_path):
    """Return one r2e_helpers record only after both exact protocols validate."""
    linear_result = linear(records(linear_path, LINEAR_ROWS))
    blocks_result = blocks(records(blocks_path, BLOCKS_ROWS))
    ls, bs = linear_result["source"], blocks_result["source"]
    if ls["revision"] != bs["revision"] or ls["dirty"] != int(bs["dirty"]):
        raise ValueError("helpers compiled source revision/dirty mismatch")
    if ls["model"] != bs["model_path"]:
        raise ValueError("helpers model path mismatch")
    return {
        "kind": "r2e_helpers", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": ls["revision"], "dirty": ls["dirty"],
        "raw_logs": {"linear": str(Path(linear_path).resolve()), "blocks": str(Path(blocks_path).resolve())},
        "scope": "diagnostic component qualification: common-Q8 synthetic matrices and unchanged actual row slices; resident FP32 HC/PLE primitives with synthetic projections and actual gamma/conv; not whole projections, full pipeline, A/B speedup or inference performance",
        "timing_scope": "arithmetic mean of 20 individually completed HIP event intervals; every repetition validated; resident launches only, excluding quantization, transfers, CPU oracle, reset/restore and validation",
        "linear_contract": {
            "scope": LINEAR_SCOPE, "activation_abi": "Q8_1: 36 bytes per block32; FP16 d, FP16 raw input sum s, 32 signed int8 codes",
            "codes": "FP32 d=amax/127 selects roundf(x/d) codes, not rounded stored half d",
            "raw_sum": "ascending XOR32 FP32 input sum then FP16 RNE; not sum of quantized codes",
            "zero": "all +0/-0 inputs produce positive-zero d/s and zero codes",
            "tiny": "representable tiny half scales retain FP32-scale code selection; nonzero scale rounding to half zero is rejected, with zeroed failed blocks",
            "arithmetic": {"Q4_0": "d4*(integer_dot*d8-8*s8)",
                           "Q4_1": "integer_dot*half_RNE(d4*d8)+half_RNE(m4*s8)",
                           "Q5_0": "d5*(integer_dot*d8-16*s8)",
                           "Q8_0": "(d8_weight*d8_activation)*integer_dot in FP32",
                           "Q6_K": "canonical signed 6-bit codes and int8 subscales; int32 scaled dot4, paired FP32 activation scales then weight scale"},
            "reference": "scalar canonical decoder on identical Q8 bytes; not dequantized-F32 parity",
            "actual_rows": "first up-to-nine unchanged stored rows of first eligible tensor per type; not whole projection",
            "numeric_status": "sticky bit1 invalid input, bit2 unrepresentable arithmetic; failed Q8 blocks zero; valid reuse checked"},
        "blocks_contract": {
            "scope": BLOCKS_SCOPE, "rms": "ascending unfused FP32 CPU sum; direct stored gamma; parallel GPU RMS bounded",
            "hc": "SiLU(raw_down/4); mean4(sigmoid(raw_up)*normalized); injection=2*sigmoid(raw_inject/4); original_wide+injection*synthetic_block",
            "readonly_tap": "root head normalizes/mixes without injection; original widened 4x2560 MTP tap remains byte-identical",
            "ple": "signed-root sigmoid of normalized key/query score times shared value; actual decoded F16 dilation3 conv with four taps",
            "history_abi": "FP32 [10240][9], chronological oldest-to-newest; staged before owner publication",
            "cpu_history_byte_parity": "ONLY identical gated and normalized-gated inputs; own parallel-RMS pipeline state is bounded, not CPU byte parity",
            "publication_byte_parity": "active is exact copy of identical GPU staged input; rejected entire active history and owner marker unchanged",
            "prefix_abi": {"slot_0": "pre-chunk history", "slot_n": "after n consumed inputs",
                           "verify_inputs": "one pending input plus two drafts", "accepted_drafts": [0, 1, 2],
                           "restore_slot": "1+accepted", "tails": 4},
            "mask": "keep 0/1 masks current input; does not reset history or perform EOS/hash handling",
            "numeric_status": "sticky bit1 invalid input, bit2 arithmetic failure, status3 combined; whole publication rejection and valid reuse checked"},
        "fixture_gates": {
            "linear": {"absolute": 2e-4, "relative_cpu_magnitude": 2e-5, "frozen_before_gpu": True,
                       "max_bound_ratio_limit": 1}, "blocks": bs["gates"],
            "bound": "absolute+relative_cpu_magnitude*abs(CPU_reference)"},
        "donors": {"mx": "dcd685463d597d31f5ca759d32c94592a2740fa4",
                   "furnace": "905021dbad71c5056ef51f9fd45d545403fc989c",
                   "reinstinct": "0b79e326351d90d4554a1c18df92da5d0ab692e8"},
        "linear": linear_result, "blocks": blocks_result, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear", type=Path, required=True)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    # Driver exit status is checked by the caller; it is not stdout metadata.
    # Validate and serialize both completed logs before touching the destination.
    result = json.dumps(collect(args.linear, args.blocks), allow_nan=False, separators=(",", ":"))
    with args.results.open("a", encoding="utf-8") as output:
        output.write(result + "\n")


if __name__ == "__main__":
    main()
