#!/usr/bin/env python3
"""Read-only layer0 FFN experiment; frozen tensor gates, never acceptance."""
import argparse
import array
import collections
import json
import math
from pathlib import Path

import diagnose_prefill as legacy

BASE_GEOMETRY = legacy._geometry
FFN_WIDTHS = {
    "ffn_router_logits_raw": 512, "ffn_router_probabilities": 512,
    "ffn_selected_expert_ids_exact_f32": 10, "ffn_selected_weights": 10,
    **{"ffn_expert_down_unweighted_rank_" + str(rank): 2560 for rank in range(10)},
    "ffn_routed_weighted_aggregate": 2560, "ffn_shared_gate_raw": 640,
    "ffn_shared_up_raw": 640, "ffn_shared_silu_middle": 640,
    "ffn_shared_down": 2560, "ffn_shared_scalar_gate_raw": 1,
}


def geometry(tokens, n1=False):
    expected = BASE_GEOMETRY(tokens, n1)
    expected.update({(name, 0, token): width for name, width in FFN_WIDTHS.items()
                     for token in range(tokens)})
    return expected


def index(root, capture_log, expanded):
    capture, tokens, boundary, end = legacy._capture(capture_log)
    if tokens != 32:
        raise ValueError("this experiment requires explicit N32 protocol2")
    root = Path(root).resolve()
    manifest = legacy.compare._file(root / "tensors.jsonl", legacy.compare.INDEX_LIMIT, end).read_bytes()
    if manifest[boundary - 1] != 10 or manifest[-1] != 10:
        raise ValueError("incomplete phase newline boundaries")
    seen = set()
    phases = []
    for phase, blob in enumerate((manifest[:boundary], manifest[boundary:])):
        expected = (geometry if expanded else BASE_GEOMETRY)(tokens, n1=phase == 0)
        rows = {}
        for line in blob.splitlines():
            row = legacy.compare._json(line.decode("utf-8"), "FFN manifest")
            if set(row) != {"name", "layer", "position", "type", "elements", "file"}:
                raise ValueError("unexpected tensor schema")
            if type(row["layer"]) is not int or type(row["position"]) is not int:
                raise ValueError("invalid layer/position types")
            key = row["name"], row["layer"], row["position"]
            if key in rows or key not in expected or row["type"] != "F32" or type(row["elements"]) is not int or row["elements"] != expected[key]:
                raise ValueError("duplicate, extra or incorrect-geometry tensor")
            path = legacy.compare._inside(root, row["file"])
            if path in seen or not row["file"].endswith(".f32.bin"):
                raise ValueError("unsafe/reused artifact")
            legacy.compare._file(path, legacy.compare.TENSOR_LIMIT, row["elements"] * 4)
            seen.add(path)
            rows[key] = row, path
        if set(rows) != set(expected):
            raise ValueError("exact phase keyset mismatch")
        phases.append(rows)
    return capture, phases, len(seen)


def floats(blob):
    values = array.array("f")
    values.frombytes(blob)
    return values


def independent_metrics(actual, reference):
    """Full rows, FP64 differences and the frozen .002+.002*abs(N1)."""
    if len(actual) != len(reference) or not actual or len(actual) % 4:
        raise ValueError("invalid compared row extent")
    a, r = floats(actual), floats(reference)
    if not all(math.isfinite(x) for x in a) or not all(math.isfinite(x) for x in r):
        raise ValueError("nonfinite compared row")
    errors = [abs(float(x) - float(y)) for x, y in zip(a, r)]
    ratios = [error / (.002 + .002 * abs(float(y))) for error, y in zip(errors, r)]
    return {"elements": len(a), "all_finite": True, "bytes_exact": actual == reference,
            "bit_mismatches": sum(actual[i:i + 4] != reference[i:i + 4] for i in range(0, len(actual), 4)),
            "max_abs": max(errors), "rms": math.sqrt(sum(x * x for x in errors) / len(a)),
            "max_bound_ratio": max(ratios), "violations": sum(x > 1 for x in ratios),
            "passed": all(x <= 1 for x in ratios)}


def diagnose(root, capture_log, prior_root, prior_log):
    capture, phases, files = index(root, capture_log, expanded=True)
    prior_capture, old, old_files = index(prior_root, prior_log, expanded=False)
    # Extend only the declared probe geometry in-memory. The existing helper's
    # schemas, phase boundaries, five N1-only exclusions and gates are unchanged.
    legacy._geometry = geometry
    try:
        existing = legacy.diagnose(root, capture_log)
    finally:
        legacy._geometry = BASE_GEOMETRY
    n1, wide = phases
    invariant = []
    for phase in range(2):
        mismatches = []
        for key, entry in old[phase].items():
            if phases[phase][key][1].read_bytes() != entry[1].read_bytes():
                mismatches.append(list(key))
        invariant.append({"phase": "N1" if phase == 0 else "N32", "old_rows": len(old[phase]),
                          "bytes_exact_rows": len(old[phase]) - len(mismatches), "mismatches": mismatches})
    rows_unchanged = capture[1:-1] == prior_capture[1:-1]
    if any(row["mismatches"] for row in invariant) or not rows_unchanged:
        raise ValueError("observers changed old tensors or full-logit metrics")
    nodes = []
    for name in (*FFN_WIDTHS, "ffn_out"):
        for token in range(32):
            key = name, 0, token
            metric = independent_metrics(wide[key][1].read_bytes(), n1[key][1].read_bytes())
            nodes.append({"name": name, "position": token, "metrics": metric})
    selected = []
    counts = collections.Counter()
    for token in range(32):
        key = "ffn_selected_expert_ids_exact_f32", 0, token
        ids = list(floats(wide[key][1].read_bytes()))
        if len(set(ids)) != 10 or any(x != int(x) or not 0 <= x <= 511 for x in ids):
            raise ValueError("semantic selected ID contract failed")
        ids = [int(x) for x in ids]
        counts.update(ids)
        weight_key = "ffn_selected_weights", 0, token
        selected.append({"position": token, "ids": ids,
                         "weights": list(floats(wide[weight_key][1].read_bytes())),
                         "ids_bytes_exact": wide[key][1].read_bytes() == n1[key][1].read_bytes(),
                         "weights_bytes_exact": wide[weight_key][1].read_bytes() == n1[weight_key][1].read_bytes()})
    down_differences = []
    for node in nodes:
        if not node["name"].startswith("ffn_expert_down_unweighted_rank_") or node["metrics"]["bytes_exact"]:
            continue
        rank = int(node["name"].rsplit("_", 1)[1])
        expert = selected[node["position"]]["ids"][rank]
        down_differences.append({"position": node["position"], "rank": rank, "expert": expert,
                                 "group_columns": counts[expert],
                                 "dispatch": "MMQ" if counts[expert] > 8 else "canonical_short",
                                 "metrics": node["metrics"]})
    input_exact = [token for token in range(32) if wide["hc_ffn_mix", 0, token][1].read_bytes() == n1["hc_ffn_mix", 0, token][1].read_bytes()]
    summaries = []
    for name in (*FFN_WIDTHS, "ffn_out"):
        group = [node["metrics"] for node in nodes if node["name"] == name]
        summaries.append({"name": name, "rows": len(group), "exact_rows": sum(m["bytes_exact"] for m in group),
                          "max_abs": max(m["max_abs"] for m in group),
                          "max_bound_ratio": max(m["max_bound_ratio"] for m in group),
                          "bit_mismatches": sum(m["bit_mismatches"] for m in group),
                          "violations": sum(m["violations"] for m in group)})
    scalar = []
    for token in range(32):
        key = "ffn_shared_scalar_gate_raw", 0, token
        scalar.append({"position": token, "N1": floats(n1[key][1].read_bytes())[0],
                       "N32": floats(wide[key][1].read_bytes())[0]})
    return {"scope": "read_only_layer0_FFN_same_input_diagnostic_not_acceptance",
            "source": capture[0], "footer": capture[-1],
            "frozen_tensor_bound": {"absolute": .002, "relative": .002, "reference": "N1"},
            "strict_manifest": {"exact_keysets": True, "exact_geometry": True, "unique_F32_files": files,
                                "phase_rows": [len(p) for p in phases], "new_rows_per_phase": 32 * len(FFN_WIDTHS)},
            "existing_helper": {"phase_rows": existing["phase_rows"], "manifest": existing["manifest"],
                                "n1_only_state_control_probes": existing["n1_only_state_control_probes"],
                                "first_nonzero": existing["first_nonzero"],
                                "first_failed_tensor_bound": existing["first_failed_tensor_bound"]},
            "observer_invariance": {"prior_trace": str(prior_root), "prior_files": old_files,
                                    "phases": invariant, "full_logit_metric_rows_exact": rows_unchanged,
                                    "full_logit_rows": capture[1:-1]},
            "input_hc_ffn_mix_exact_positions": input_exact,
            "dependency_order": [*FFN_WIDTHS, "ffn_out"],
            "first_nonexact_observed_FFN": next((node for node in nodes if not node["metrics"]["bytes_exact"]), None),
            "first_failed_new_tensor_bound": next((node for node in nodes if node["name"] in FFN_WIDTHS and not node["metrics"]["passed"]), None),
            "summaries": summaries, "full_row_metrics": nodes, "selected_routes": selected,
            "expert_group_columns": [{"expert": expert, "columns": count} for expert, count in sorted(counts.items())],
            "unweighted_down_differences": down_differences, "shared_scalar_values": scalar,
            "projection_input_capture_limit": "Expert gate/up/SiLU and Q8 down inputs were not captured; unweighted down is the earliest observed routed projection subnode."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--capture-log", required=True)
    parser.add_argument("--prior-trace", required=True)
    parser.add_argument("--prior-log", required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.trace, args.capture_log, args.prior_trace, args.prior_log), allow_nan=False))


if __name__ == "__main__":
    main()
