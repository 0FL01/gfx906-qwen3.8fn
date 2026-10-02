#!/usr/bin/env python3
"""Read-only matched GDN state/controls diagnosis; never waives Session gates."""

import argparse
import json
import math
import struct
from pathlib import Path

import compare_session as checked


def binary(root, descriptor, elements):
    if descriptor.get("type") != "f32" or descriptor.get("elements") != elements:
        raise ValueError("wrong GDN probe artifact type/element count")
    if descriptor.get("bytes") != elements * 4 or descriptor.get("endianness") != "little":
        raise ValueError("wrong GDN probe artifact bytes/endianness")
    path = checked._inside(root, descriptor["path"])
    checked._file(path, elements * 4, elements * 4)
    raw = path.read_bytes()
    values = struct.unpack(f"<{elements}f", raw)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("nonfinite GDN probe artifact")
    return raw, values


def own_tensor(root, entries, name, layer, position, elements):
    candidates = [row for row in entries if row.get("name") in (name, "debug_" + name)
                  and row.get("layer") == layer and row.get("position") == position]
    if len(candidates) != 1:
        raise ValueError(f"missing/ambiguous own {name} layer={layer} position={position}")
    row = candidates[0]
    if row.get("elements") != elements or row.get("type") != "F32":
        raise ValueError("wrong own GDN artifact shape/type")
    path = checked._inside(root, row["file"])
    checked._file(path, elements * 4, elements * 4)
    values = checked._tensor((row, path))
    return path.read_bytes(), values


def metrics(a, b, absolute, relative):
    if len(a[0]) != len(b[0]) or len(a[1]) != len(b[1]):
        raise ValueError("GDN matched artifact length mismatch")
    result = checked._metrics(a[1], b[1], absolute, relative, exact=absolute == relative == 0)
    result["bit_mismatches"] = sum(a[0][i:i+4] != b[0][i:i+4]
                                    for i in range(0, len(a[0]), 4))
    result["byte_exact"] = a[0] == b[0]
    return result


def diagnose(session_trace, oracle_directory, layer=22):
    own_root, ref_root = Path(session_trace), Path(oracle_directory)
    own = checked._jsonl(own_root / "tensors.jsonl", checked.INDEX_LIMIT, checked.MAX_TOKENS * 256)
    probes = checked._jsonl(ref_root / "gdn-state-probes.jsonl", 64 * 1024 * 1024,
                           checked.MAX_TOKENS * 36)
    probes = [row for row in probes if row.get("layer") == layer]
    if not probes or [row.get("position") for row in probes] != list(range(len(probes))):
        raise ValueError("missing/nonchronological original GDN state probes")
    previous_own = previous_ref = None
    rows = []
    for probe in probes:
        if (probe.get("format") != "gfx906-mx-gdn-state-probe-v1"
                or probe.get("phase") != "recorded" or probe.get("warm_pass") != -1
                or probe.get("post_from_persistent_cache_destination") is not True
                or probe.get("primitive_state_tail_read") is not False
                or probe.get("all_finite") is not True):
            raise ValueError("unvalidated original GDN state probe")
        position = probe["position"]
        reference = {role: binary(ref_root, probe["artifacts"][role], count)
                     for role, count in (("pre_state", 786432), ("post_state", 786432),
                                         ("log_decay", 48), ("beta", 48))}
        current = {role: own_tensor(own_root, own, name, layer, position, count)
                   for role, name, count in (
                       ("pre_state", "gdn_state_pre", 786432),
                       ("post_state", "gdn_state_post", 786432),
                       ("log_decay", "gdn_control_log_decay_recomputed", 48),
                       ("beta", "gdn_control_beta_recomputed", 48))}
        comparisons = {role: metrics(current[role], reference[role],
                                      2e-5 if role.endswith("state") else 2e-4, 2e-4)
                       for role in current}
        continuity = None
        if previous_own is not None:
            continuity = {"own_post_to_next_pre": metrics(current["pre_state"], previous_own, 0, 0),
                          "reference_post_to_next_pre": metrics(reference["pre_state"], previous_ref, 0, 0)}
        rows.append({"position": position, "token_id": probe["input_token_id"],
                     "comparisons": comparisons, "continuity": continuity})
        previous_own, previous_ref = current["post_state"], reference["post_state"]
    return {"kind": "gdn_state_diagnosis", "layer": layer, "rows": rows,
            "scope": "original primitive inputs/persistent states versus own captured states; "
                     "own controls recomputed by the same GPU helper, not captured registers",
            "whole_session_gate_waived": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-trace", required=True)
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--layer", type=int, default=22)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.session_trace, args.oracle_dir, args.layer), allow_nan=False))
