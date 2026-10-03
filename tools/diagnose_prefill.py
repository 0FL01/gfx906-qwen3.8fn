#!/usr/bin/env python3
"""Read-only N1/wide trace localization; never replaces the full-logits gate."""
import argparse
import json
import math
from pathlib import Path

import compare_session as compare

N1_ONLY = frozenset(("debug_gdn_state_pre", "debug_gdn_state_post",
                     "debug_gdn_control_log_decay_recomputed",
                     "debug_gdn_control_beta_recomputed", "debug_gdn_control_decay_recomputed"))
SOURCE_KEYS = frozenset(("kind", "revision", "dirty", "trace", "diagnostic_only",
                         "absolute_gate", "relative_gate"))
ROW_KEYS = frozenset(("kind", "position", "token", "elements", "max_abs", "rms",
                      "max_bound_ratio", "violations", "bit_mismatches"))
FOOTER_KEYS = frozenset(("kind", "diagnostic_only", "capture_completed", "raii_cleanup_completed",
                         "numeric_gate_passed", "n1_manifest_end_bytes", "batch_manifest_end_bytes"))
PROBE_LAYERS = frozenset((0, 1, 2, 3, 6, 8, 19, 22, 47))
DEBUG_LAYERS = frozenset((2, 6, 8, 19, 22))
FFN_WIDTHS = {
    "ffn_router_logits_raw": 512, "ffn_router_probabilities": 512,
    "ffn_selected_expert_ids_exact_f32": 10, "ffn_selected_weights": 10,
    **{"ffn_expert_down_unweighted_rank_" + str(rank): 2560 for rank in range(10)},
    "ffn_routed_weighted_aggregate": 2560, "ffn_shared_gate_raw": 640,
    "ffn_shared_up_raw": 640, "ffn_shared_silu_middle": 640,
    "ffn_shared_down": 2560, "ffn_shared_scalar_gate_raw": 1,
}


def _geometry(tokens, n1=False):
    """Exact existing Session capture allowlist; not a runtime/math specification."""
    probes = {("hc_init", -1): 10240, ("result_norm", -1): 2560}
    for layer in range(48):
        prefix = "debug_" if layer in DEBUG_LAYERS else ""
        probes[prefix + "l_last-" + str(layer), layer] = 10240
        if layer not in PROBE_LAYERS:
            continue
        widths = {"block_attention_out": 2560, "ffn_out": 2560}
        for phase in ("attn", "ffn"):
            widths.update({"hc_" + phase + "_mix": 2560, "hc_" + phase + "_norm": 10240,
                           "hc_" + phase + "_gate_raw": 10240, "hc_" + phase + "_inject_raw": 4})
        if (layer + 1) % 4:
            widths.update(qkv=10240, z=6144, alpha=48, beta=48,
                          gdn_convolved_normalized=10240, final_output=6144)
        else:
            widths.update(qsa_q_gate_raw=12288, qsa_key_raw=512, qsa_value_raw=512,
                          qsa_gate_raw=6144, qsa_query_rotated=6144, qsa_key_rotated=512,
                          qsa_value_rotated=512, qsa_attention_rotated=6144,
                          qsa_attention_inverse_h64=6144, qsa_attention_gated=6144)
        probes.update({(prefix + name, layer): width for name, width in widths.items()})
    probes.update({(name, 1): width for name, width in
                   {"ple_embedding": 2560, "ple_key_raw": 10240, "ple_value_raw": 2560,
                    "ple_gated_value": 10240, "ple_residual": 10240}.items()})
    if n1:
        probes.update({(name, 22): 786432 if name.endswith(("_pre", "_post")) else 48
                       for name in N1_ONLY})
    return {(name, layer, position): width for (name, layer), width in probes.items()
            for position in range(tokens)}


def _capture(path):
    capture = compare._jsonl(Path(path), 1024 * 1024, 34)
    if not capture or capture[0].get("kind") != "prefill_trace_source":
        raise ValueError("unexpected trace protocol")
    source = capture[0]
    if set(source) == SOURCE_KEYS:
        tokens = 4
    elif set(source) == SOURCE_KEYS | {"protocol", "token_count"}:
        compare._equal(source, "protocol", 2, "trace source")
        compare._equal(source, "token_count", 32, "trace source")
        tokens = 32
    else:
        raise ValueError("only legacy N4 and explicit protocol2 N32 are supported")
    if len(capture) != tokens + 2:
        raise ValueError("unexpected capture record count")
    compare._equal(source, "diagnostic_only", True, "trace source")
    revision = compare._text(source.get("revision"), "trace revision")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("invalid trace revision")
    compare._int(source.get("dirty"), "trace dirty", 0, 1)
    compare._text(source.get("trace"), "trace directory")
    for key, value in (("absolute_gate", .02), ("relative_gate", .002)):
        compare._equal(source, key, value, "trace source")
    footer = capture[-1]
    if set(footer) != FOOTER_KEYS or footer.get("kind") != "prefill_trace_complete" or any(
        footer.get(k) is not True for k in
        ("diagnostic_only", "capture_completed", "raii_cleanup_completed")
    ):
        raise ValueError("incomplete diagnostic capture")
    for position, row in enumerate(capture[1:-1]):
        if set(row) != ROW_KEYS:
            raise ValueError("unexpected capture row schema")
        for key, value in (("kind", "prefill_trace_row"), ("position", position),
                           ("token", 248044 if position == 0 else 99 + position),
                           ("elements", compare.VOCAB)):
            compare._equal(row, key, value, "capture row")
        compare._numbers(row, ("max_abs", "rms", "max_bound_ratio"), "capture row")
        compare._int(row["violations"], "capture violations", 0, compare.VOCAB)
        compare._int(row["bit_mismatches"], "capture bit mismatches", row["violations"], compare.VOCAB)
        if (row["max_bound_ratio"] > 1) != (row["violations"] > 0):
            raise ValueError("inconsistent capture bound metrics")
    compare._equal(footer, "numeric_gate_passed", all(row["violations"] == 0 for row in capture[1:-1]),
                   "trace footer")
    boundary, end = footer["n1_manifest_end_bytes"], footer["batch_manifest_end_bytes"]
    if type(boundary) is not int or type(end) is not int or not 0 < boundary < end:
        raise ValueError("invalid manifest byte boundaries")
    return capture, tokens, boundary, end


def diagnose(trace_root, capture_log):
    root = Path(trace_root).resolve()
    capture, tokens, boundary, end = _capture(capture_log)
    manifest = compare._file(root / "tensors.jsonl", compare.INDEX_LIMIT, end).read_bytes()
    if manifest[boundary - 1] != 10 or manifest[-1] != 10:
        raise ValueError("manifest phase boundary is not newline-complete")

    files = set()

    def index(blob, expected):
        rows = [compare._json(line.decode("utf-8"), "trace row")
                for line in blob.splitlines()]
        if len(rows) != len(expected):
            raise ValueError("unexpected manifest phase extent")
        result = {}
        for row in rows:
            if set(row) != {"name", "layer", "position", "type", "elements", "file"}:
                raise ValueError("unexpected trace schema")
            if row["type"] != "F32" or type(row["elements"]) is not int or row["elements"] < 1:
                raise ValueError("invalid trace type or element count")
            if type(row["position"]) is not int or not 0 <= row["position"] < tokens:
                raise ValueError("unexpected trace position")
            if type(row["layer"]) is not int or not -1 <= row["layer"] <= 48:
                raise ValueError("unexpected trace layer")
            compare._text(row["name"], "trace name")
            key = row["name"], row["layer"], row["position"]
            if key in result:
                raise ValueError("ambiguous trace inside a phase")
            if key not in expected or row["elements"] != expected[key]:
                raise ValueError("unexpected probe or trace geometry")
            path = compare._inside(root, row["file"])
            if not row["file"].endswith(".f32.bin") or path in files:
                raise ValueError("unexpected or reused F32 artifact path")
            compare._file(path, compare.TENSOR_LIMIT, row["elements"] * 4)
            files.add(path)
            result[key] = row, path
        if set(result) != set(expected):
            raise ValueError("missing or extra phase probes")
        return result

    # The layer0 FFN observer adds exactly these twenty probes per token in
    # BOTH phases. Recognize that complete geometry without allowing omissions
    # or arbitrary extra captures in either the old or the expanded protocol.
    expanded_ffn = any(compare._json(line.decode("utf-8"), "trace row").get("name") in FFN_WIDTHS
                       for line in manifest.splitlines())
    def expected(n1=False):
        probes = _geometry(tokens, n1=n1)
        if expanded_ffn:
            probes.update({(name, 0, position): width for name, width in FFN_WIDTHS.items()
                           for position in range(tokens)})
        return probes

    n1, wide = index(manifest[:boundary], expected(n1=True)), index(manifest[boundary:], expected())
    n1_only = [key for key in n1 if key[0] in N1_ONLY]
    if set(n1_only) != {(name, 22, position) for name in N1_ONLY for position in range(tokens)}:
        raise ValueError("unexpected N1-only state/control probe coverage")
    for key in n1_only:
        if not all(math.isfinite(value) for value in compare._tensor(n1[key])):
            raise ValueError("nonfinite N1-only state/control probe")
        del n1[key]
    if any(key[0] in N1_ONLY for key in wide):
        raise ValueError("wide state/control probes have no per-token snapshot contract")
    if set(n1) != set(wide):
        raise ValueError("phase capture coverage differs")
    nodes = []
    # Legacy reports retain token-major N1 order. N32 localization must follow
    # the actual batch capture order, including QSA's per-query subloop.
    for key in (wide if tokens == 32 else n1):
        left = n1[key]
        right = wide[key]
        if left[0]["elements"] != right[0]["elements"]:
            raise ValueError("matched node geometry differs")
        nodes.append({"name": key[0], "layer": key[1], "position": key[2],
                      "metrics": compare._metrics(compare._tensor(right), compare._tensor(left),
                                                  compare.TENSOR_ABS, compare.TENSOR_REL)})
    report = {"scope": "read_only_same_session_N1_wide_matched_nodes_not_acceptance",
            "capture": capture, "gate": {"abs": compare.TENSOR_ABS, "rel": compare.TENSOR_REL},
            "nodes": nodes, "phase_rows": [len(n1), len(wide)],
             "n1_only_state_control_probes": len(n1_only)}
    if tokens == 32:
        report.update(protocol=2, token_count=32, node_order="N32_manifest_execution_order",
                      manifest={"n1_end_bytes": boundary, "batch_end_bytes": end,
                                "exact_extent": True, "newline_complete_boundaries": True,
                                "exact_phase_keysets_and_geometry": True, "unique_safe_F32_paths": len(files)},
                      first_nonzero=next((node for node in nodes if node["metrics"]["max_abs"] != 0), None),
                      first_failed_tensor_bound=next((node for node in nodes if not node["metrics"]["passed"]), None))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--capture-log", required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.trace, args.capture_log), allow_nan=False))


if __name__ == "__main__":
    main()
