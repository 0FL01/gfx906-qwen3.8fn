#!/usr/bin/env python3
"""Read-only matched-node diagnostic; never changes/waives comparison gates."""
import argparse
import json
import struct
from pathlib import Path

import compare_session as compare


PAIRS = (
    ("hc_attn_norm", "hc_norm", 0),
    ("hc_attn_mix", "hc_mixed", 0),
    ("hc_attn_inject_raw", "hc_inject", 0),
    ("qkv", "linear_attn_qkv_mixed", 0),
    ("z", "z", 0),
    ("alpha", "alpha", 0),
    ("beta", "beta", 0),
    ("final_output", "final_output", 0),
    ("block_attention_out", "linear_attn_out", 0),
    ("hc_ffn_norm", "hc_norm", 1),
    ("hc_ffn_mix", "hc_mixed", 1),
    ("hc_ffn_inject_raw", "hc_inject", 1),
    ("ffn_out", "ffn_out", 0),
)


def diagnose(session_root, oracle_root, position=0):
    a, b = Path(session_root).resolve(), Path(oracle_root).resolve()
    own = compare._jsonl(a / "tensors.jsonl", compare.INDEX_LIMIT, compare.MAX_TOKENS * 256)
    ref = compare._jsonl(b / "tensors.jsonl", compare.INDEX_LIMIT, compare.MAX_TOKENS * 256)
    results, missing, slots, outputs = [], [], [], []
    layers = sorted({r["layer"] for r in own if r["position"] == position and r["layer"] >= 0})
    for layer in layers:
        label = f"l_last-{layer}"
        left = [r for r in own if r["position"] == position and r["layer"] == layer
                and r["name"] in (label, "debug_" + label)]
        right = [r for r in ref if r["position"] == position and r["name"] == label]
        if left and right:
            if len(left) != 1 or len(right) != 1:
                raise ValueError("ambiguous layer output")
            x, y = left[0], right[0]
            xp, yp = compare._inside(a, x["file"]), compare._inside(b, y["file"])
            compare._file(xp, compare.TENSOR_LIMIT, x["elements"] * 4)
            compare._file(yp, compare.TENSOR_LIMIT, compare._shape(y))
            outputs.append({"layer": layer, "metrics": compare._metrics(
                compare._tensor((x, xp)), compare._tensor((y, yp), True),
                compare.TENSOR_ABS, compare.TENSOR_REL)})
        for name, oracle_name, occurrence in PAIRS:
            left = [r for r in own if r["position"] == position and r["layer"] == layer
                    and r["name"] in (name, "debug_" + name)]
            right = [r for r in ref if r["position"] == position and r["name"] == f"{oracle_name}-{layer}"
                     and r["occurrence"] == occurrence]
            if not left or not right:
                missing.append({"layer": layer, "name": name, "session_missing": not left, "oracle_missing": not right})
                continue
            if len(left) != 1 or len(right) != 1:
                raise ValueError("ambiguous matched tensor")
            x, y = left[0], right[0]
            x_path, y_path = compare._inside(a, x["file"]), compare._inside(b, y["file"])
            compare._file(x_path, compare.TENSOR_LIMIT, x["elements"] * 4)
            compare._file(y_path, compare.TENSOR_LIMIT, compare._shape(y))
            metric = compare._metrics(
                compare._tensor((x, x_path)),
                compare._tensor((y, y_path), True),
                compare.TENSOR_ABS, compare.TENSOR_REL)
            results.append({"layer": layer, "name": name, "oracle_name": y["name"], "metrics": metric})
        cache = [r for r in ref if r["position"] == position and r["name"] == f"ffn_moe_cache_slots-{layer}"]
        if cache:
            if len(cache) != 1 or cache[0]["type"] != "i32" or cache[0]["ne"] != [10, 1, 1, 1]:
                raise ValueError("unexpected cache-slot tensor")
            row = cache[0]
            path = compare._inside(b, row["file"])
            compare._file(path, compare.TENSOR_LIMIT, compare._shape(row))
            data = path.read_bytes()
            ids = [struct.unpack_from("<i", data, i * row["nb"][0])[0] for i in range(10)]
            if any(v < 0 or v > 112 for v in ids):
                raise ValueError("cache-slot ID outside captured 112-slot contract")
            slots.append({"layer": layer, "slots": ids, "misses_at_capacity112": sum(v == 112 for v in ids)})
    return {"scope": "read_only_matched_nodes_not_whole_session_gate", "position": position,
            "gate": {"abs": compare.TENSOR_ABS, "rel": compare.TENSOR_REL},
            "nodes": results, "missing": missing, "cache_slots": slots,
            "layer_outputs": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-trace", required=True)
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--position", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.session_trace, args.oracle_dir, args.position), allow_nan=False))


if __name__ == "__main__":
    main()
