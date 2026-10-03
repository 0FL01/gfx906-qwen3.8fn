#!/usr/bin/env python3
"""Validate frozen qwen.mmq-wide.v1 JSONL and append one component record.

The driver is src/mmq_wide_main.hip. All GPU outputs are finite-checked, but
the LM-head CPU oracle covers ONLY 24 original rows. Diagnostic sliced linear
is a component fixture baseline, not production PP or an inference speed claim.
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
PROTOCOL = "qwen.mmq-wide.v1"
REPEATS, RECORDS = 20, 94
COLUMNS, PHASES = (1, 3, 8, 32, 128), ("A1", "B", "A2")
HEAD_ROWS = [0, 1, 7, 31, 63, 127, 255, 511, 1023, 4095, 8191, 16383,
             32767, 65535, 131071, 200003, 248043, 248044, 248045, 248046,
             248047, 248127, 248318, 248319]
# name, type NAME, K, M, block elements, canonical block bytes, CPU rows
TENSORS = (
    ("blk.6.ffn_down_shexp.weight", "Q5_0", 640, 2560, 32, 22, 2560),
    ("blk.0.ffn_down_shexp.weight", "Q8_0", 640, 2560, 32, 34, 2560),
    ("output.weight", "Q6_K", 2560, 248320, 256, 210, 24),
)
# Frozen expected_counters(), not assertions supplied by a log's source record.
COUNTERS = {
    "grid_cases": 588, "q4_forward_cases": 48, "limit_cases": 5,
    "manual_cases": 3, "reuse_cases": 5, "actual_cases": 15,
    "matrices": 664, "quantization": 669, "packing": 672,
    "poison_cases": 202, "poison_subblocks": 21018,
    "finite_elements": 45769172, "reference_elements": 3062260,
    "reference_slices": 9497, "readonly_validations": 664,
    "quantizer_rejects": 30, "packer_rejects": 23, "mmq_rejects": 153,
    "int8_min_cases": 3, "int8_max_cases": 3, "raw_sum_difference_cases": 3,
    "signed_scale_cases": 3, "subnormal_scale_cases": 3, "extreme_scale_cases": 3,
    "q6_subscale_cases": 1, "q6_activation_scale_cases": 1, "q6_integer_grouping_cases": 1,
}
EXPECTED_PROTOCOL = {
    "source": 1, "correctness": 2, "metrics": 90,
    "intervals": 1800, "complete": 1, "rows": RECORDS,
}
SCHEMAS = {
    "mmq_wide_source": [
        "kind", "protocol", "revision", "dirty", "model", "model_variant", "architecture",
        "donor_revision", "donor_license", "absolute_gate", "relative_gate", "scope", "oracle",
        "baseline", "timing", "ordering", "abi", "resources", "coverage", "expected_per_device",
        "expected_protocol", "actual", "devices", "schemas",
    ],
    "mmq_wide_correctness": [
        "kind", "protocol", "device", "counters", "error", "proofs", "cleanup", "passed",
    ],
    "mmq_wide_metric": [
        "kind", "protocol", "sequence", "device", "tensor_index", "column_index", "phase_index",
        "phase", "path", "tensor", "type", "width", "rows", "columns", "weight_bytes",
        "reference_coverage", "reference_rows", "sample_rows", "launches_per_repeat", "repeats",
        "validated_intervals", "finite_elements", "reference_elements", "completed_ms",
        "completed_ms_total", "resident_ms", "warmup_error", "error", "proofs", "cleanup", "passed",
    ],
    "mmq_wide_complete": [
        "kind", "protocol", "devices", "source_records", "correctness_records", "metric_records",
        "validated_intervals", "finite_elements", "reference_elements", "jsonl_records",
        "live_owners", "cleanup", "passed",
    ],
    "counters": list(COUNTERS),
    "error": ["max_abs", "max_bound_ratio"],
    "proofs": ["q8_bytes_exact", "ds4_bytes_exact", "finite", "redzones", "readonly"],
    "actual": [
        "tensor_index", "tensor", "type", "rank", "dimensions", "strides_bytes", "elements",
        "bytes", "file_offset", "read_api", "payload_reads", "unchanged", "reference_coverage",
        "reference_rows", "sample_rows", "finite_rows", "finite_halfscales", "all_halfscales_finite",
    ],
    "devices": ["device", "arch", "wave"],
    "timing": ["clock", "repeats", "phases", "columns", "excludes", "validation", "element_counters"],
    "abi": ["q8_bytes", "q8_qs_offset", "ds4_bytes", "ds4_alignment", "ds4_qs_offset", "layout",
            "ds_order", "sum", "padding"],
    "resources": ["stream", "exact_payloads", "weight_alignment", "weight_pointer_mod4",
                  "q8_alignment", "output_alignment", "redzones_bytes", "redzone_bits",
                  "weight_readonly", "head_host_reads"],
    "redzones_bytes": ["weights", "raw", "q8", "ds4", "flag", "output"],
    "coverage": ["types", "new_format_cartesian", "q5_q8_widths", "q6_widths", "rows", "columns",
                 "q4_forward", "limits", "manual", "quantizer_edges", "rejects", "reuse"],
    "q4_forward": ["widths", "rows", "columns"],
    "manual": ["q5_q8", "q6", "checks"],
    "quantizer_edges": ["shape", "cases"],
    "expected_protocol": ["source", "correctness", "metrics", "intervals", "complete", "rows"],
    "live_owners": ["buffers", "streams", "events"],
}
PROOFS = dict.fromkeys(SCHEMAS["proofs"], True)
SOURCE_FIXED = {
    "kind": "mmq_wide_source", "protocol": PROTOCOL,
    "model_variant": "qwen38-keep1-Q4_0", "architecture": "qwen4exp",
    "donor_revision": "dcd685463d597d31f5ca759d32c94592a2740fa4", "donor_license": "MIT",
    "absolute_gate": .0002, "relative_gate": .00002,
    "scope": "component qualification; no PP-production/full-model/inference speed claim",
    "oracle": "unchanged matmul_q8_reference in ordered <=3-column slices; full synthetic/shared-expert outputs, 24 original head rows ONLY",
    "baseline": "diagnostic_sliced_linear",
    "timing": {
        "clock": "individually completed HIP events", "repeats": REPEATS,
        "phases": list(PHASES), "columns": list(COLUMNS),
        "excludes": ["allocation", "Model", "H2D", "quantization", "DS4_pack", "CPU_reference",
                     "output_reset", "D2H"],
        "validation": "all GPU outputs finite/redzones; all applicable CPU references and complete readonly payloads EVERY warmup/repeat",
        "element_counters": "sum over 20 validated intervals; excludes warmup",
    },
    "ordering": "source; correctness0/1 after device cleanup BEFORE any timing; device,tensor,column,A1/B/A2 metrics after device cleanup; complete after Model/host/device cleanup",
    "abi": {
        "q8_bytes": 36, "q8_qs_offset": 4, "ds4_bytes": 144, "ds4_alignment": 16,
        "ds4_qs_offset": 16, "layout": "[K128][column]", "ds_order": "d0,s0,d1,s1,d2,s2,d3,s3",
        "sum": "original raw-input half sum, not d8*sum(codes)",
        "padding": "missing halves/codes zero; unused K128 subblocks alone poisoned",
    },
    "resources": {
        "stream": "borrowed explicit nonblocking", "exact_payloads": True,
        "weight_alignment": 2, "weight_pointer_mod4": 2, "q8_alignment": 4, "output_alignment": 4,
        "redzones_bytes": {"weights": [2, 2], "raw": [4, 4], "q8": [4, 4],
                           "ds4": [16, 16], "flag": [4, 4], "output": [4, 4]},
        "redzone_bits": "half qNaN 0x7e00",
        "weight_readonly": "full original bytes in <=4MiB D2H chunks, no weight backup",
        "head_host_reads": 1,
    },
    "coverage": {
        "types": ["Q4_0", "Q4_1", "Q5_0", "Q8_0", "Q6_K"], "new_format_cartesian": True,
        "q5_q8_widths": [32, 160, 640, 2560, 16384], "q6_widths": [256, 768, 2560, 16384],
        "rows": [1, 63, 64, 65, 127, 128, 129], "columns": [1, 3, 4, 17, 65, 128],
        "q4_forward": {"widths": [160, 2560], "rows": [65, 129], "columns": [1, 3, 4, 17, 65, 128]},
        "limits": "all five types M16384 N3; new formats K16384 grid; sole large exception Q6_K[2560,248320]",
        "manual": {
            "q5_q8": [160, 65, 3], "q6": [768, 65, 3],
            "checks": ["int8 -128/127", "rawsum != d8*sumcodes", "signed/subnormal/extreme halves",
                       "16 distinct Q6 subscales", "all 8 distinct Q6 activation scales",
                       "integer dot*subscale before FP32"],
        },
        "quantizer_edges": {
            "shape": [256, 1], "cases": ["zero", "negative_zero", "half_rounding", "minimum_half_scale",
                                        "signed_alternating", "small_sine", "unrounded_scale_codes",
                                        "large_cancellation"],
        },
        "rejects": ["null", "shape", "type", "alignment", "range_wrap", "write_alias", "exact_head_exception"],
        "reuse": "all five formats; same producer/pack/linear allocations after rejection",
    },
    "expected_per_device": COUNTERS, "expected_protocol": EXPECTED_PROTOCOL, "schemas": SCHEMAS,
}


def json_integer(value):
    # Unsigned driver counters/coordinates never emit the spelling -0.
    return -0.0 if value == "-0" else int(value)


def records(path):
    """Bound even a growing log and require exactly 94 newline-completed objects."""
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
    # argv[1] is preserved; architecture and tensor bindings, not basename, are gated.
    text(obj["model"], "source.model")
    require(type(obj["devices"]) is list and len(obj["devices"]) == 2, "source: expected both devices")
    for i, device in enumerate(obj["devices"]):
        label = f"source.devices[{i}]"
        expect(device, {"device": i, "wave": 64}, ("arch",), label)
        arch = text(device["arch"], label + ".arch")
        require(arch == "gfx906" or arch.startswith("gfx906:"), label + ": requires gfx906")
    actual = obj["actual"]
    require(type(actual) is list and len(actual) == 3, "source: expected three unchanged tensors")
    ranges = []
    for i, (tensor, type_name, k, m, block_elements, block_bytes, cpu_rows) in enumerate(TENSORS):
        label = f"source.actual[{i}]"
        row_bytes = k // block_elements * block_bytes
        payload = row_bytes * m
        expect(actual[i], {
            "tensor_index": i, "tensor": tensor, "type": type_name, "rank": 2,
            "dimensions": [k, m], "strides_bytes": [block_bytes, row_bytes],
            "elements": k * m, "bytes": payload, "read_api": "Model.read_tensor", "payload_reads": 1,
            "unchanged": True, "reference_coverage": "sampled_rows" if i == 2 else "all_rows",
            "reference_rows": cpu_rows, "sample_rows": HEAD_ROWS if i == 2 else [], "finite_rows": m,
            "finite_halfscales": payload // block_bytes, "all_halfscales_finite": True,
        }, ("file_offset",), label)
        offset = integer(actual[i]["file_offset"], label + ".file_offset", 1, UINT64_MAX - payload)
        ranges.append((offset, offset + payload))
    ranges.sort()
    require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), "source: overlapping tensor ranges")
    # File size/alignment/data_offset/relative_offset are absent, checked by the
    # loader/driver. Do not fabricate values or assume a particular GGUF alignment.


def errors(obj, label):
    expect(obj, {}, SCHEMAS["error"], label)
    number(obj["max_abs"], label + ".max_abs")
    require(number(obj["max_bound_ratio"], label + ".max_bound_ratio") <= 1,
            label + ": frozen common-Q8 bound ratio exceeded")


def correctness(obj, device):
    label = f"correctness[{device}]"
    expect(obj, {"kind": "mmq_wide_correctness", "protocol": PROTOCOL, "device": device,
                 "counters": COUNTERS, "proofs": PROOFS, "cleanup": True, "passed": True}, ("error",), label)
    errors(obj["error"], label + ".error")


def rounded_timing(actual, expected, label):
    # max_digits10 output, sequential C++ double sum vs fsum: 32 double ULPs
    # cover 20 additions/division, with no fixed millisecond tolerance floor.
    require(math.isfinite(expected) and expected > 0, label + ": invalid interval aggregate")
    require(math.isclose(actual, expected, rel_tol=32 * sys.float_info.epsilon,
                         abs_tol=32 * math.ulp(expected)), label + ": inconsistent completed intervals")


def metric(obj, device, tensor_index, column_index, phase_index, sequence):
    label = f"metric[{sequence}]"
    tensor, type_name, k, m, block_elements, block_bytes, cpu_rows = TENSORS[tensor_index]
    n = COLUMNS[column_index]
    expect(obj, {
        "kind": "mmq_wide_metric", "protocol": PROTOCOL, "sequence": sequence, "device": device,
        "tensor_index": tensor_index, "column_index": column_index, "phase_index": phase_index,
        "phase": PHASES[phase_index], "path": "mmq_wide" if phase_index == 1 else "diagnostic_sliced_linear",
        "tensor": tensor, "type": type_name, "width": k, "rows": m, "columns": n,
        "weight_bytes": k // block_elements * block_bytes * m,
        "reference_coverage": "sampled_rows" if tensor_index == 2 else "all_rows",
        "reference_rows": cpu_rows, "sample_rows": HEAD_ROWS if tensor_index == 2 else [],
        "launches_per_repeat": 1 if phase_index == 1 else (n + 2) // 3,
        "repeats": REPEATS, "validated_intervals": REPEATS,
        "finite_elements": m * n * REPEATS, "reference_elements": cpu_rows * n * REPEATS,
        "proofs": PROOFS, "cleanup": True, "passed": True,
    }, ("completed_ms", "completed_ms_total", "resident_ms", "warmup_error", "error"), label)
    errors(obj["error"], label + ".error")
    errors(obj["warmup_error"], label + ".warmup_error")
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


def complete(obj, metrics):
    # Reconcile the live footer with both the validated rows and frozen geometry.
    expect(obj, {
        "kind": "mmq_wide_complete", "protocol": PROTOCOL, "devices": 2, "source_records": 1,
        "correctness_records": 2, "metric_records": 90, "validated_intervals": 1800,
        "finite_elements": 2 * 3 * REPEATS * sum(COLUMNS) * sum(t[3] for t in TENSORS),
        "reference_elements": 2 * 3 * REPEATS * sum(COLUMNS) * sum(t[6] for t in TENSORS),
        "jsonl_records": RECORDS, "live_owners": {"buffers": 0, "streams": 0, "events": 0},
        "cleanup": True, "passed": True,
    }, label="complete")
    for footer_key, metric_key in (("metric_records", None), ("validated_intervals", "validated_intervals"),
                                   ("finite_elements", "finite_elements"), ("reference_elements", "reference_elements")):
        observed = len(metrics) if metric_key is None else sum(row[metric_key] for row in metrics)
        require(obj[footer_key] == observed, "complete: inconsistent live " + footer_key)


def collect(raw_path):
    """Return ONE r4b_mmq_wide record after full validation; never write inputs."""
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
                    metric(metrics[sequence], device, tensor, column, phase, sequence)
                    sequence += 1
    complete(footer, metrics)
    return {
        "kind": "r4b_mmq_wide", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": origin["revision"], "dirty": origin["dirty"], "model": origin["model"],
        "model_variant": origin["model_variant"], "raw_logs": {"mmq_wide": str(raw_path)},
        "scope": "component A/B/A on two gfx906 GPUs: unchanged common-Q8/DS4 Q5_0/Q8_0/Q6_K MMQ with Q4 forwarding; all synthetic/shared-expert outputs numerically compared; full LM-head GPU outputs finite/redzones/readonly, numerical CPU coverage ONLY the exact 24 original sampled rows including 248319; diagnostic_sliced_linear is ONLY a fixture baseline, not production PP; no large-prompt, end-to-end, independent HF, MTP or performance-win claim; no universal speedup asserted",
        "performance_claim": False, "universal_speedup_claim": False, "pp_qualified": False,
        "large_prompt_qualified": False, "end_to_end_qualified": False, "independent_hf_claim": False, "mtp": False,
        "timing_scope": origin["timing"]["clock"], "timing_excludes": origin["timing"]["excludes"],
        "fixture_gates": {"absolute": .0002, "relative_cpu_magnitude": .00002,
                          "bound": "absolute+relative_cpu_magnitude*abs(CPU_reference)",
                          "max_bound_ratio_limit": 1, "frozen_before_execution": True},
        "provenance_limits": "compiled revision/dirty, emitted mx donor revision/MIT, original model path and source records preserved; no HF pin, file size, GGUF alignment, data_offset or relative_offset emitted; file-range/alignment gates are driver/loader checks; LM-head numerical CPU oracle covers only 24 sampled original rows, not all 248320 rows",
        "source": origin, "correctness": proofs, "measurements": metrics, "complete": footer, "passed": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    # Caller supplies the canonical ROOT/results.jsonl explicitly; no cwd default.
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        append_result(args.results, collect(args.raw))
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print("record_mmq_wide: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
