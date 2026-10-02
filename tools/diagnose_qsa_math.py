#!/usr/bin/env python3
"""Read-only captured QSA12 diagnosis: exp rounding and whole-block order."""
import argparse
import ctypes
import itertools
import json
import math
import struct
from pathlib import Path

import compare_session as check


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def run(own_directory, oracle_directory, layer, position):
    own, oracle = Path(own_directory), Path(oracle_directory)
    probes = check._jsonl(oracle / "qsa-probes.jsonl", check.INDEX_LIMIT,
                         check.MAX_TOKENS * 12)
    matching = [r for r in probes if r["layer"] == layer and r["position"] == position]
    if len(matching) != 1 or matching[0]["selected_count"] != 12:
        raise ValueError("this narrow witness requires exactly twelve visible keys")
    probe = matching[0]
    artifacts = probe["artifacts"]

    def blob(role, count):
        return check._file(check._inside(oracle, artifacts[role]["path"]), count, count).read_bytes()

    query = list(struct.unpack("<6144f", blob("query", 24576)))
    reference = list(struct.unpack("<6144f", blob("corrected", 24576)))
    if not all(math.isfinite(x) for x in query + reference):
        raise ValueError("nonfinite original query/reference")
    keys, values = blob("key", 3456), blob("value", 3456)
    rows = check._jsonl(own / "tensors.jsonl", check.INDEX_LIMIT, check.MAX_TOKENS * 256)

    def captured(name):
        found = [r for r in rows if r["name"] in (name, "debug_" + name)
                 and r["layer"] == layer and r["position"] == position]
        if len(found) != 1:
            raise ValueError("missing or ambiguous capture: " + name)
        return check._tensor((found[0], check._inside(own, found[0]["file"])))

    actual = captured("qsa_attention_rotated")
    query_error = check._metrics(captured("qsa_query_rotated"), query, 0, 0, exact=True)
    if not query_error["passed"]:
        raise ValueError("original and own queries are not numerically identical")

    def gather(data, token, head):
        row = []
        for i in range(256):
            offset = token * 288 + head * 144 + (i // 32) * 18
            scale = struct.unpack_from("<e", data, offset)[0]
            packed = data[offset + 2 + i % 16]
            code = (packed & 15) if i % 32 < 16 else packed >> 4
            value = f32(scale * (code - 8))
            row.append(struct.unpack("<e", struct.pack("<e", value))[0])
        return row

    k = [[gather(keys, t, h) for t in range(12)] for h in range(2)]
    v = [[gather(values, t, h) for t in range(12)] for h in range(2)]
    libm = ctypes.CDLL("libm.so.6")
    expf = libm.expf
    expf.argtypes, expf.restype = [ctypes.c_float], ctypes.c_float
    weights, mismatches = [], []
    for h in range(24):
        scores = []
        for t in range(12):
            score = 0.0
            for i in range(256):
                score = f32(score + f32(query[h * 256 + i] * k[h // 12][t][i]))
            scores.append(f32(score * 0.0625))
        shifted = [f32(x - max(scores)) for x in scores]
        a = [expf(x) for x in shifted]
        b = [f32(math.exp(x)) for x in shifted]
        for t, (x, y) in enumerate(zip(a, b)):
            if bits(x) != bits(y):
                mismatches.append({"head": h, "key": t, "shift": shifted[t],
                                   "expf_bits": bits(x), "double_exp_bits": bits(y)})
        weights.append((a, b))

    permutations = []
    for order in itertools.permutations(range(3)):
        ids = [4 * block + member for block in order for member in range(4)]
        for method in range(2):
            output = []
            for h in range(24):
                w = weights[h][method]
                denominator = 0.0
                for t in ids:
                    denominator = f32(denominator + w[t])
                inverse = f32(1.0 / denominator)
                for i in range(256):
                    total = 0.0
                    for t in ids:
                        total = f32(total + f32(w[t] * v[h // 12][t][i]))
                    output.append(f32(total * inverse))
            permutations.append({"blocks": order, "exp": ("libm_expf", "rounded_double_exp")[method],
                                 "own_bit_mismatches": sum(bits(x) != bits(y) for x, y in zip(output, actual)),
                                 "reference_bit_mismatches": sum(bits(x) != bits(y) for x, y in zip(output, reference)),
                                 "own_max_abs": max(abs(x - y) for x, y in zip(output, actual))})
    return {"scope": "read_only_same_Q_K_V_exp_and_whole_block_order_witness_not_a_global_gate",
            "layer": layer, "position": position, "query_exact": True,
            "exp_mismatches": mismatches, "permutations": permutations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-trace", required=True)
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--position", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.session_trace, args.oracle_dir, args.layer, args.position)
    text = json.dumps(report, allow_nan=False)
    with open(args.output, "x", encoding="utf-8") as output:
        output.write(text + "\n")
    print(text)
