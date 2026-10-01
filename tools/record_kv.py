#!/usr/bin/env python3
"""Validate completed R2a KV/QSA fixtures and append one scoped result."""
import argparse
import datetime
import json
import math
from pathlib import Path


def invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def records(path):
    return [json.loads(line, parse_constant=invalid_constant)
            for line in Path(path).read_text().splitlines() if line.strip()]


def collect(kv_path, qsa_path):
    kv, qsa = records(kv_path), records(qsa_path)
    if (len(kv) != 50 or kv[0].get("kind") != "kv_source"
            or kv[-1] != {"kind": "kv_complete", "passed": True}):
        raise ValueError("KV fixture incomplete")
    source = kv[0]
    if (not isinstance(source.get("revision"), str) or len(source["revision"]) != 40
            or any(c not in "0123456789abcdef" for c in source["revision"])
            or source.get("dirty") not in (0, 1) or source.get("kv") != "Q4_0"):
        raise ValueError("KV source provenance")
    expected = [(device, columns, role)
                for device in (0, 1) for columns in (1, 2, 3, 128)
                for role in ("q", "k", "v")]
    pipelines, packs = [], {}
    for record in kv[1:-1]:
        if (record.get("byte_parity") is not True or record.get("repeats") != 100
                or not isinstance(record.get("resident_ms"), (int, float))
                or not math.isfinite(record["resident_ms"]) or record["resident_ms"] <= 0):
            raise ValueError("KV timing/parity")
        key = (record.get("device"), record.get("columns"))
        if key[0] not in (0, 1) or key[1] not in (1, 2, 3, 128):
            raise ValueError("KV device/columns")
        if record.get("kind") == "kv_gpu":
            role = record.get("role")
            if (role not in ("q", "k", "v") or record.get("max_abs_error") != 0
                    or record.get("group") != (64 if role == "v" else 256)
                    or record.get("elements") != (24 if role == "q" else 2) * 256 * key[1]):
                raise ValueError("KV transform geometry/error")
            pipelines.append((*key, role))
        elif record.get("kind") == "kv_pack":
            packs.setdefault(key, []).append(record.get("layout"))
        else:
            raise ValueError("unknown KV measurement")
    if pipelines != expected or packs != {
            (d, n): ["mx_serial", "cooperative", "mx_serial"]
            for d in (0, 1) for n in (1, 2, 3, 128)}:
        raise ValueError("KV measurement sequence/A-B-A")
    if (len(qsa) != 11 or qsa[-1].get("test") != "qsa"
            or qsa[-1].get("passed") is not True or qsa[-1].get("checks", 0) <= 0
            or qsa[-1].get("rejections", 0) <= 0 or qsa[-1].get("chunk_prefix_queries", 0) <= 0):
        raise ValueError("QSA fixture incomplete")
    counts = [2047, 2048, 2049, 2050, 2051, 2048, 2049, 2050, 2051, 2048]
    extras = [[], [], [], [], [], [0, 1, 2], [0, 2], [0], [], [4, 5, 6]]
    for i, record in enumerate(qsa[:-1]):
        visible = 2047 + i
        if record != {"diagnostic": "mx-expanded-position", "visible": visible,
                      "n_kv": (visible + 255) // 256 * 256,
                      "official_count": counts[i], "fork_width": min((visible + 255) // 256 * 256, 2051),
                      "fork_valid_count": min(visible, 2051),
                      "fork_extra_valid_ids": len(extras[i]), "extra_token_ids": extras[i]}:
            raise ValueError("QSA boundary diagnostic changed")
    return {"kind": "r2a_kv_qsa",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "revision": source["revision"], "dirty": bool(source["dirty"]),
            "kv_log": str(Path(kv_path).resolve()), "qsa_log": str(Path(qsa_path).resolve()),
            "scope": "synthetic resident transforms + CPU QSA source emulation; not attention, baseline GPU IDs or inference speed",
            "kv": "Q4_0, signed first-max scale, original FP32 reciprocal codes, FP16 RNE",
            "hadamard": "Q/K256 after RoPE, V64 before storage, inverse V before gate",
            "reference": "transformers a005fc82babfe8871d87746decad2dbee100a125",
            "baseline": "mx dcd685463d597d31f5ca759d32c94592a2740fa4",
            "tie_policy": "equal completed-block scores: lower block ID first; HF ties unspecified",
            "measurements": kv[1:-1], "qsa": qsa}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-log", required=True)
    parser.add_argument("--qsa-log", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    result = collect(args.kv_log, args.qsa_log)
    with Path(args.results).open("a") as output:
        output.write(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
