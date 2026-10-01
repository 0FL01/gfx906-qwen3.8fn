#!/usr/bin/env python3
"""Validate a completed core-probe JSONL log and append one hardware record."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def collect(path):
    def reject(value):
        raise ValueError(f"invalid JSON constant: {value}")
    records = [json.loads(line, parse_constant=reject) for line in path.read_text().splitlines()]
    if not records or records[-1] != {"kind": "probe_complete", "passed": True}:
        raise ValueError("probe did not complete successfully")
    def selected(kind, count):
        values = [record for record in records if record.get("kind") == kind]
        if len(values) != count:
            raise ValueError(f"expected {count} {kind} records, got {len(values)}")
        return values
    source, = selected("source", 1)
    cpus, = selected("cpu", 1)
    hip, = selected("hip", 1)
    gpus = selected("gpu", 2)
    if (hip["device_count"] != 2 or {gpu["device"] for gpu in gpus} != {0, 1}
            or any(gpu["wave_size"] != 64 or not gpu["arch"].startswith("gfx906") for gpu in gpus)):
        raise ValueError("unexpected assigned GPU geometry")
    transfers = selected("h2d", 16)
    gemms = selected("rocblas_sgemm", 4)
    peers = selected("peer", 2)
    dual = selected("dual_h2d", 2)
    launches = selected("empty_launch", 2)
    copies = selected("device_copy", 2)
    ram = selected("ram_read_fma", 3)
    for record in transfers + gemms + dual + launches + copies + ram:
        for key, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite measurement: {key}")
    for record in gemms:
        if record["max_abs_error"] != 0:
            raise ValueError("binary-exact GEMM fixture failed")
    for record in peers:
        if record["available"] and record["epochs"] != 20:
            raise ValueError("peer ordering fixture incomplete")
    return {"kind": "r0_hardware", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "revision": source["revision"], "dirty": bool(source["dirty"]),
            "raw_log": str(path.resolve()), "cpu": cpus, "hip": hip, "gpus": gpus,
            "ram_read_fma": ram, "h2d": transfers, "rocblas_sgemm": gemms,
            "peer": peers, "dual_h2d": dual, "empty_launch": launches, "device_copy": copies,
            "scope": "R0 hardware smoke; CPU read/FMA is not an expert benchmark; no speedup claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    record = collect(args.log)
    with args.results.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, allow_nan=False) + "\n")
    print("Validated R0 probe and recorded hardware measurements")


if __name__ == "__main__":
    main()
