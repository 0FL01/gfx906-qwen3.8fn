#!/usr/bin/env python3
"""Validate core-mmq JSONL v1 and append one component A/B/A evidence record.

The frozen driver is src/mmq_main.hip. Diagnostic sliced-linear is a fixture
baseline, not production PP. These measurements imply no universal speedup,
large-prompt, end-to-end, independent HF, or MTP qualification.
"""

import argparse
import datetime
import json
import math
from pathlib import Path
import stat
import sys

if __package__:
    from .record_memory import (UINT64_MAX, append_result, expect, finite_float,
                                integer, invalid_constant, number, require,
                                text, unique_object)
else:
    from record_memory import (UINT64_MAX, append_result, expect, finite_float,
                               integer, invalid_constant, number, require,
                               text, unique_object)


MAX_RAW_BYTES = 2 * 1024 * 1024
REPEATS, RECORDS = 20, 94
COLUMNS, PHASES = (1, 3, 8, 32, 128), ("A1", "B", "A2")
# Source v1 emits type NAMES, not GGUF numeric type IDs or donor/HF pins.
TENSORS = (
    ("blk.0.ffn_gate_exps.weight", "Q4_0", 2560, 640, 512, 18),
    ("blk.0.ffn_down_exps.weight", "Q4_1", 640, 2560, 512, 20),
    ("blk.0.hc_attn_down.weight", "Q4_0", 10240, 320, 1, 18),
)
COUNTERS = {
    "grid_cases": 936, "actual_cases": 39, "corner_cases": 3, "reuse_cases": 1,
    "quantizer_edges": 8, "matrix_cases": 979, "quantization_cases": 984,
    "packing_cases": 987, "poison_cases": 316, "poisoned_subblocks": 27540,
    "compared_elements": 2568652, "reference_slices": 9904,
    "quantizer_rejects": 30, "packer_rejects": 23, "mmq_rejects": 36,
    "host_rejects": 89, "manual_cases": 3, "int8_min_cases": 3,
    "raw_sum_difference_cases": 3, "signed_q8_scale_cases": 3, "half_product_pairs": 21,
}
PROTOCOL = {
    "devices": 2, "source_records": 1, "correctness_records": 2,
    "measurements_per_device": 45, "measurement_records": 90,
    "validated_intervals": 1800, "complete_records": 1, "jsonl_records": 94,
}
SCHEMAS = {
    "mmq_source": [
        "kind", "protocol", "revision", "dirty", "model", "model_variant", "architecture",
        "absolute_gate", "relative_gate", "scope", "numeric_contract", "oracle", "baseline",
        "timing_scope", "timing_excludes", "ordering", "stream", "q8_1", "ds4", "tiles",
        "buffers", "coverage", "benchmark_columns", "benchmark_phases", "repeats",
        "expected_per_device", "expected_protocol", "actual", "devices", "schemas",
    ],
    "mmq_correctness": [
        "kind", "protocol", "device", "counters", "max_abs", "max_bound_ratio", "q8_bytes_exact",
        "packed_bytes_exact", "missing_subblocks_zero", "poison_masked", "active_subblocks_unchanged",
        "half_RNE_corners", "rejects_unchanged", "valid_reuse", "finite", "canary", "readonly",
        "cleanup", "passed",
    ],
    "mmq_measurement": [
        "kind", "protocol", "sequence", "device", "tensor_index", "column_index", "phase_index",
        "phase", "path", "tensor", "type", "width", "rows", "columns", "tile_j", "column_tiles",
        "launches_per_repeat", "repeats", "validated_intervals", "elements_per_interval",
        "completed_ms", "completed_ms_total", "resident_ms", "max_abs", "max_bound_ratio",
        "warmup_max_abs", "warmup_max_bound_ratio", "finite", "canary", "readonly",
        "q8_bytes_exact", "packed_bytes_exact", "passed",
    ],
    "mmq_complete": [
        "kind", "protocol", "devices", "correctness_records", "measurement_records",
        "validated_intervals", "jsonl_records", "cleanup", "passed",
    ],
}
SOURCE_FIXED = {
    "kind": "mmq_source", "protocol": 1, "model_variant": "qwen38-keep1-Q4_0",
    "architecture": "qwen4exp", "absolute_gate": .0002, "relative_gate": .00002,
    "scope": "component common-Q8 quantization/DS4 packing/canonical Q4 MMQ; no PP or inference speed claim",
    "numeric_contract": "finite static weight halves and valid finite Q8 halves/codes; numeric validity is caller-owned",
    "oracle": "unchanged matmul_q8_reference, ordered 1..3-column CPU slices, full-N finite comparison",
    "baseline": "diagnostic sliced current qualified launch_quantized_linear Columns<=3; benchmark only",
    "timing_scope": "20 individually completed resident HIP-event intervals per phase; all output/read-only/redzones validated each interval",
    "timing_excludes": ["allocation", "model_reads", "H2D", "quantization", "packing", "CPU_reference", "output_reset", "readback"],
    "ordering": "source; correctness device0,device1 BEFORE performance; measurements device,tensor,column,phase A1/B/A2; complete after all cleanup",
    "stream": "explicit hipStreamNonBlocking",
    "q8_1": {"bytes": 36, "qs_offset": 4, "sum": "original half raw-input sum, not d*sum(codes)"},
    "ds4": {
        "bytes": 144, "alignment": 16, "ds_offset": 0, "qs_offset": 16,
        "layout": "[K128][column]", "ds_order": "d0,s0,d1,s1,d2,s2,d3,s3",
        "missing_subblocks": "zero halves/codes; unused final subblocks alone poisoned before qualification MMQ",
    },
    "tiles": {"rows": 64, "k": 256, "block": [64, 4], "j": [8, 16, 32, 64],
              "selection": "smallest J>=N capped64; N65..128 two J64 tiles"},
    "buffers": {
        "exact_payloads": True, "weight_pointer_offset": 2, "weight_alignment": 2,
        "q8_alignment": 4, "packed_alignment": 16, "output_alignment": 4,
        "redzones": "prefix/suffix bytes weights2/2, raw4/4, Q8_1 4/4, DS4 16/16, output4/4, flag4/4; half-qNaN poison",
    },
    "coverage": {
        "types": ["Q4_0", "Q4_1"], "cartesian": True,
        "widths": [32, 160, 640, 2560, 10240, 16384], "rows": [1, 7, 63, 64, 65, 67],
        "columns": [1, 2, 3, 4, 8, 9, 16, 17, 32, 33, 64, 65, 128],
        "quantizer_edges": {"width": 256, "columns": 1, "cases": [
            "zero", "negative_zero", "half_rounding", "minimum_half_scale", "signed_alternating",
            "small_sine", "unrounded_scale_codes", "large_cancellation"]},
        "manual_corners": [
            {"type": "Q4_0", "width": 160, "rows": 7, "columns": 3},
            {"type": "Q4_1", "width": 160, "rows": 7, "columns": 3},
            {"type": "Q4_1", "width": 32, "rows": 7, "columns": 3}],
        "reuse": {"type": "Q4_1", "width": 160, "rows": 7, "columns": 3},
        "reject_categories": ["null", "shape", "type", "alignment", "range_wrap", "write_alias"],
    },
    "benchmark_columns": list(COLUMNS), "benchmark_phases": list(PHASES), "repeats": REPEATS,
    "expected_per_device": COUNTERS, "expected_protocol": PROTOCOL, "schemas": SCHEMAS,
}
CORRECTNESS_PROOFS = (
    "q8_bytes_exact", "packed_bytes_exact", "missing_subblocks_zero", "poison_masked",
    "active_subblocks_unchanged", "half_RNE_corners", "rejects_unchanged", "valid_reuse",
    "finite", "canary", "readonly", "cleanup", "passed",
)
MEASUREMENT_PROOFS = ("finite", "canary", "readonly", "q8_bytes_exact", "packed_bytes_exact", "passed")


def json_integer(value):
    # Unsigned driver coordinates/counters never spell zero as "-0". Preserve
    # that floating spelling rather than silently accepting it as an integer.
    return -0.0 if value == "-0" else int(value)


def records(path):
    """Bound even a growing log, and require all 94 newline-completed objects."""
    require(stat.S_ISREG(path.stat().st_mode), str(path) + ": not a regular file")
    require(path.stat().st_size <= MAX_RAW_BYTES, str(path) + ": oversized input")
    with path.open("rb") as stream:
        raw = stream.read(MAX_RAW_BYTES + 1)
    require(len(raw) <= MAX_RAW_BYTES, str(path) + ": oversized input")
    require(raw.endswith(b"\n"), str(path) + ": missing completed final line")
    lines = raw.split(b"\n")[:-1]
    require(len(lines) == RECORDS, str(path) + ": expected exactly 94 JSONL rows")
    rows = []
    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line.decode("utf-8"), object_pairs_hook=unique_object,
                             parse_int=json_integer, parse_float=finite_float, parse_constant=invalid_constant)
            require(type(obj) is dict, "expected an object")
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{path}:row {i}: {error}") from error
        rows.append(obj)
    return rows


def source(obj):
    expect(obj, SOURCE_FIXED, ("revision", "dirty", "model", "actual", "devices"), "source")
    revision = obj["revision"]
    require(type(revision) is str and len(revision) == 40 and
            all(c in "0123456789abcdef" for c in revision), "source: invalid compiled revision")
    integer(obj["dirty"], "source.dirty", 0, 1)
    # The driver preserves argv[1]; neither an absolute model path nor a
    # particular basename is required by its architecture/tensor binding gate.
    text(obj["model"], "source.model")
    devices = obj["devices"]
    require(type(devices) is list and len(devices) == 2, "source: expected both devices")
    for i, device in enumerate(devices):
        label = f"source.devices[{i}]"
        expect(device, {"device": i, "wave": 64}, ("arch",), label)
        arch = text(device["arch"], label + ".arch")
        require(arch == "gfx906" or arch.startswith("gfx906:"), label + ": requires gfx906")
    actual = obj["actual"]
    require(type(actual) is list and len(actual) == 3, "source: expected three unchanged tensors")
    ranges = []
    for i, (tensor, type_name, width, rows, experts, block_bytes) in enumerate(TENSORS):
        label = f"source.actual[{i}]"
        row_bytes = width // 32 * block_bytes
        payload = row_bytes * rows
        expert = i < 2
        tensor_bytes = payload * experts
        expect(actual[i], {
            "tensor_index": i, "tensor": tensor, "type": type_name, "rank": 3 if expert else 2,
            "dimensions": [width, rows, experts] if expert else [width, rows],
            "strides_bytes": [block_bytes, row_bytes, payload] if expert else [block_bytes, row_bytes],
            "elements": width * rows * experts, "tensor_bytes": tensor_bytes,
            "read_api": "Model.read_expert" if expert else "Model.read_slice",
            "expert": 0 if expert else None, "slice_offset": 0, "selected_bytes": payload,
            "width": width, "rows": rows, "unchanged": True,
        }, ("file_offset",), label)
        offset = integer(actual[i]["file_offset"], label + ".file_offset", 1, UINT64_MAX - tensor_bytes)
        ranges.append((offset, offset + tensor_bytes))
    ranges.sort()
    require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), "source: overlapping tensor ranges")
    # File size, GGUF alignment, data_offset and relative_offset are NOT emitted.
    # The loader/driver checks them; do not invent journal fields or fixed offsets.


def errors(obj, label, prefix=""):
    number(obj[prefix + "max_abs"], label + "." + prefix + "max_abs")
    ratio = number(obj[prefix + "max_bound_ratio"], label + "." + prefix + "max_bound_ratio")
    require(ratio <= 1, label + ": frozen common-Q8 bound ratio exceeded")


def correctness(obj, device):
    label = f"correctness[{device}]"
    expect(obj, {"kind": "mmq_correctness", "protocol": 1, "device": device, "counters": COUNTERS,
                 **dict.fromkeys(CORRECTNESS_PROOFS, True)}, ("max_abs", "max_bound_ratio"), label)
    errors(obj, label)


def rounded_timing(actual, expected, label):
    # C++ prints max_digits10 doubles, but its sequential sum and Python's fsum
    # need not use the same addition order. 32 double ulps (no fixed ms floor)
    # cover 20 rounded additions/division without hiding meaningful discrepancies.
    require(math.isfinite(expected) and expected > 0, label + ": invalid interval aggregate")
    require(math.isclose(actual, expected, rel_tol=32 * sys.float_info.epsilon,
                         abs_tol=32 * math.ulp(expected)), label + ": inconsistent completed intervals")


def measurement(obj, device, tensor_index, column_index, phase_index, sequence):
    label = f"measurement[{sequence}]"
    tensor, type_name, width, rows, _, _ = TENSORS[tensor_index]
    n = COLUMNS[column_index]
    tile_j = next((j for j in (8, 16, 32, 64) if n <= j), 64)
    expect(obj, {
        "kind": "mmq_measurement", "protocol": 1, "sequence": sequence, "device": device,
        "tensor_index": tensor_index, "column_index": column_index, "phase_index": phase_index,
        "phase": PHASES[phase_index], "path": "mmq" if phase_index == 1 else "diagnostic_sliced_linear",
        "tensor": tensor, "type": type_name, "width": width, "rows": rows, "columns": n,
        "tile_j": tile_j, "column_tiles": (n + tile_j - 1) // tile_j,
        "launches_per_repeat": 1 if phase_index == 1 else (n + 2) // 3,
        "repeats": REPEATS, "validated_intervals": REPEATS, "elements_per_interval": rows * n,
        **dict.fromkeys(MEASUREMENT_PROOFS, True),
    }, ("completed_ms", "completed_ms_total", "resident_ms", "max_abs", "max_bound_ratio",
        "warmup_max_abs", "warmup_max_bound_ratio"), label)
    errors(obj, label)
    errors(obj, label, "warmup_")
    intervals = obj["completed_ms"]
    require(type(intervals) is list and len(intervals) == REPEATS, label + ": expected 20 completed intervals")
    for i, value in enumerate(intervals):
        require(number(value, f"{label}.completed_ms[{i}]") > 0, label + ": interval must be positive")
    total = number(obj["completed_ms_total"], label + ".completed_ms_total")
    mean = number(obj["resident_ms"], label + ".resident_ms")
    require(total > 0 and mean > 0, label + ": aggregate times must be positive")
    try:
        expected = math.fsum(intervals)
    except OverflowError as error:
        raise ValueError(label + ": interval aggregate overflow") from error
    rounded_timing(total, expected, label + ".completed_ms_total")
    rounded_timing(mean, expected / REPEATS, label + ".resident_ms")


def complete(obj):
    expect(obj, {"kind": "mmq_complete", "protocol": 1, "devices": 2,
                 "correctness_records": 2, "measurement_records": 90, "validated_intervals": 1800,
                 "jsonl_records": RECORDS, "cleanup": True, "passed": True}, label="complete")


def collect(raw_path):
    """Return ONE r4b_mmq_primitives record after full validation; never write inputs."""
    raw_path = Path(raw_path).resolve()
    rows = records(raw_path)
    origin, proofs, metrics, footer = rows[0], rows[1:3], rows[3:-1], rows[-1]
    source(origin)
    for device, proof in enumerate(proofs):
        correctness(proof, device)
    sequence = 0
    for device in (0, 1):
        for tensor in range(3):
            for column in range(len(COLUMNS)):
                for phase in range(len(PHASES)):
                    measurement(metrics[sequence], device, tensor, column, phase, sequence)
                    sequence += 1
    complete(footer)
    return {
        "kind": "r4b_mmq_primitives", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": origin["revision"], "dirty": origin["dirty"], "model": origin["model"],
        "model_variant": origin["model_variant"], "raw_logs": {"mmq": str(raw_path)},
        "scope": "component A/B/A on two gfx906 GPUs: common-Q8 quantization, bit-preserving DS4 packing and canonical unchanged Q4_0/Q4_1 MMQ; diagnostic_sliced_linear is ONLY a fixture baseline, not production PP; no large-prompt, end-to-end, independent HF, MTP or performance-win claim; no universal speedup asserted",
        "performance_claim": False, "universal_speedup_claim": False, "pp_qualified": False,
        "large_prompt_qualified": False, "end_to_end_qualified": False, "independent_hf_claim": False, "mtp": False,
        "timing_scope": origin["timing_scope"], "timing_excludes": origin["timing_excludes"],
        "fixture_gates": {"absolute": .0002, "relative_cpu_magnitude": .00002,
                          "bound": "absolute+relative_cpu_magnitude*abs(CPU_reference)",
                          "max_bound_ratio_limit": 1, "frozen_before_execution": True},
        "provenance_limits": "compiled revision/dirty and original model path preserved; no donor/HF pins, file size or GGUF alignment emitted by v1; file-range/alignment gates are driver/loader checks",
        "source": origin, "correctness": proofs, "measurements": metrics, "complete": footer, "passed": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        append_result(args.results, collect(args.raw))
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print("record_mmq: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
