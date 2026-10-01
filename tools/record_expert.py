#!/usr/bin/env python3
"""Validate the completed R1 fixture before appending its structured record."""
import argparse
import json
import math
from pathlib import Path


def invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def collect(path):
    records = [json.loads(line, parse_constant=invalid_constant)
               for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 38 or records[-1] != {"kind": "expert_complete", "passed": True}:
        raise ValueError("incomplete R1 fixture")
    source = records[0]
    if (source.get("kind") != "expert_source" or source.get("layer") != 0
            or source.get("expert") != 0 or source.get("weight_bytes") != 2867200
            or source.get("model") != "qwen38-keep1-Q4_0.gguf"
            or not isinstance(source.get("revision"), str) or not source["revision"]
            or source.get("dirty") not in (0, 1)
            or not isinstance(source.get("cpu_affinity"), int) or source["cpu_affinity"] < 0):
        raise ValueError("unexpected expert source/geometry")
    offset = 1
    measurements = []
    for columns in (1, 2, 3, 128):
        repeats = 5 if columns == 128 else 50
        for kernel in ("block_calls", "inlined_f16c", "block_calls"):
            item = records[offset]
            offset += 1
            if (item.get("kind") != "expert_cpu" or item.get("columns") != columns
                    or item.get("kernel") != kernel or item.get("workers") != 1
                    or item.get("repeats") != repeats):
                raise ValueError("CPU A/B/A sequence")
            if item["ms"] <= 0 or item["max_abs_error"] != 0 or item["rms_error"] != 0:
                raise ValueError("CPU fixture timing/parity")
            measurements.append(item)
        for device in (0, 1):
            for layout in ("canonical", "planar", "canonical"):
                item = records[offset]
                offset += 1
                if (item.get("kind") != "expert_gpu" or item.get("columns") != columns
                        or item.get("device") != device or item.get("layout") != layout
                        or item.get("repeats") != repeats):
                    raise ValueError("GPU A/B/A sequence")
                for field in ("upload_pack_complete_ms", "resident_ms", "completed_wall_ms"):
                    if item[field] <= 0:
                        raise ValueError("GPU duration")
                if item["completed_wall_ms"] < item["resident_ms"]:
                    raise ValueError("completed wall excludes GPU runtime")
                for field in ("gate_error", "up_error", "down_error", "max_abs_error", "rms_error"):
                    if item[field] < 0:
                        raise ValueError("negative error")
                measurements.append(item)
    for item in measurements:
        for value in item.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("nonfinite measurement")
    return {"kind": "r1_expert", "source": source,
            "scope": "real layer0 expert0; fixed synthetic float inputs; hot repeated weights; not full-model inference or DDR miss throughput",
            "activation_abi": "mx Q8_1 36 bytes, fp16 d/raw sum, roundf, ascending-XOR raw sum",
            "weight_precision": "canonical Q4_0 gate/up and Q4_1 down; byte-preserving planar Q4_0 candidate",
            "q4_1_products": "fp16-rounded d4*d8 and m4*s8 as mx FAST_FP16_AVAILABLE",
            "gpu_input_output_scope": "input uploaded once before warmup, timed resident pipeline repeated, final output read back; no whole-request latency claim",
            "upload_scope": "completed weights allocations/pack/copies, includes initial warmup effects; not an isolated steady-state DMA miss benchmark",
            "fixture_gates": {"linear_abs": 0.0002, "linear_rel": 0.00002,
                              "float_pipeline_abs": 0.002, "float_pipeline_rel": 0.0002},
            "gpu_abi_fixtures": "zero/negative-zero, signed rounding ties, half-subnormal scale, invalid NaN/Inf/scale underflow/overflow and subsequent valid reuse, int8 extrema",
            "cpu_default": "inlined_f16c", "gpu_default": "canonical",
            "measurements": measurements, "raw_log": str(path.resolve())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    record = collect(args.log)
    with args.results.open("a") as output:
        output.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
