#!/usr/bin/env python3
"""Collect the current R3b 32-teacher/32-output/reset qualification evidence.

Only completed, finite, frozen-gate proofs qualify. This consumes the report
from compare_session.py, not the model/binary captures again. The complete
report (including controller provenance, oracle controls and operational
limitations) is retained verbatim as JSON data. No inference speed is claimed.
"""

import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import sys


BASELINE_REVISION = "dcd685463d597d31f5ca759d32c94592a2740fa4"
HF_REVISION = "a005fc82babfe8871d87746decad2dbee100a125"
VOCAB = 248320
TEACHER_IDS = [248044, *range(100, 131)]
RESET_IDS = [248044, 100, 101, 102]
MODEL = "qwen38-keep1-Q4_0.gguf"
RUNTIME = "own_48_layer_HIP"
DECODE_SCOPE = "ordered_decode_no_prefill_no_sampling_speed_claim"
RESET_SCOPE = "reset_replay_and_rejection_self_parity"
MAX_COMPARISON_BYTES = 64 * 1024 * 1024
MAX_LOG_BYTES = 1024 * 1024
MAX_LINE_BYTES = 64 * 1024
UINT64_MAX = (1 << 64) - 1
F32_MAX = 3.4028234663852886e38
REQUIRED = {"hc_init": (-1, 10240), "l_last-0": (0, 10240),
            "l_last-1": (1, 10240), "l_last-3": (3, 10240),
            "l_last-47": (47, 10240), "result_norm": (-1, 2560)}
LAYERS = (0, 1, 3, 47)
METRIC_KEYS = {"elements", "finite_pairs", "all_finite", "nonfinite_session",
               "nonfinite_oracle", "max_abs", "rms", "max_bound_ratio",
               "violating_elements", "worst_bound_element", "passed",
               "session_argmax", "oracle_argmax", "oracle_max", "argmax_agreement"}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def exact(actual, expected, label):
    require(type(actual) is type(expected), label + ": incorrect type")
    if type(expected) is dict:
        expect(actual, expected, label=label)
    elif type(expected) is list:
        require(len(actual) == len(expected), label + ": incorrect length")
        for i, (got, want) in enumerate(zip(actual, expected)):
            exact(got, want, f"{label}[{i}]")
    else:
        require(actual == expected, label + ": unexpected value")


def expect(obj, fixed, variable=(), label="record", extensions=False):
    require(type(obj) is dict, label + ": expected an object")
    fields = set(fixed) | set(variable)
    require(fields <= set(obj) if extensions else fields == set(obj),
            label + ": unexpected or missing fields")
    for key, value in fixed.items():
        exact(obj[key], value, label + "." + key)


def integer(value, label, low=0, high=UINT64_MAX):
    require(type(value) is int and low <= value <= high, label + ": invalid integer")
    return value


def number(value, label, positive=False, maximum=None):
    require(type(value) in (int, float), label + ": expected a nonboolean number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and value >= 0 and (not positive or value > 0) and
            (maximum is None or value <= maximum), label + ": invalid finite number")
    return value


def text(value, label):
    require(type(value) is str and 0 < len(value) <= 4096 and
            not any(ord(c) < 32 or 0xd800 <= ord(c) <= 0xdfff for c in value),
            label + ": invalid string")
    return value


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "duplicate JSON key: " + key)
        obj[key] = value
    return obj


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite JSON number")
    return result


def invalid_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def json_object(raw, label):
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                         parse_float=finite_float, parse_constant=invalid_constant)
        require(type(obj) is dict, "expected an object")
    except (ValueError, RecursionError) as error:
        raise ValueError(label + ": " + str(error)) from error
    return obj


def raw_bytes(path, limit):
    path = Path(path)
    require(stat.S_ISREG(path.stat().st_mode), str(path) + ": not a regular file")
    require(path.stat().st_size <= limit, str(path) + ": oversized input")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, str(path) + ": oversized input")
    require(raw.endswith(b"\n"), str(path) + ": missing completed final line")
    return raw


def records(path, count):
    lines = raw_bytes(path, MAX_LOG_BYTES).split(b"\n")[:-1]
    require(len(lines) == count, str(path) + ": incorrect JSONL row count")
    rows = []
    for i, line in enumerate(lines):
        require(0 < len(line) <= MAX_LINE_BYTES, str(path) + ": invalid JSONL line size")
        rows.append(json_object(line, f"{path}:row {i}"))
    return rows


def compiled_provenance(obj, label, boolean_dirty=False):
    revision = obj["revision"]
    require(type(revision) is str and len(revision) == 40 and
            all(c in "0123456789abcdefABCDEF" for c in revision), label + ": invalid compiled revision")
    if boolean_dirty:
        require(type(obj["dirty"]) is bool, label + ": invalid boolean dirty")
    else:
        integer(obj["dirty"], label + ".dirty", high=1)


def source(obj, trace):
    expect(obj, {"kind": "session_source",
                 "capacity": 4096, "expert_slots": 112, "trace": trace,
                 "sampling": "greedy_diagnostic", "runtime": RUNTIME}, ("model", "revision", "dirty"),
           "Session source")
    compiled_provenance(obj, "Session source")
    require(Path(text(obj["model"], "Session model")).name == MODEL, "unexpected model variant")


def completion(obj, inputs, outputs, consumed):
    expect(obj, {"kind": "session_complete", "input_tokens": inputs,
                 "output_tokens": outputs, "consumed_tokens": consumed,
                 "scope": DECODE_SCOPE, "passed": True}, ("load_ms", "request_ms"),
           "Session complete")
    for key in ("load_ms", "request_ms"):
        number(obj[key], "Session complete." + key, positive=True)


def route_stats(obj, previous, position, completed_key, single_slot=False):
    number(obj[completed_key], "step." + completed_key, positive=not single_slot)
    for key in ("expert_hits", "expert_misses", "expert_upload_bytes"):
        integer(obj[key], "step." + key)
        require(obj[key] >= previous[key], "expert counters decreased")
    hits = obj["expert_hits"] - previous["expert_hits"]
    misses = obj["expert_misses"] - previous["expert_misses"]
    require(hits + misses == 480 and obj["expert_hits"] + obj["expert_misses"] == 480 * (position + 1),
            "incomplete expert route accounting")
    uploaded = obj["expert_upload_bytes"] - previous["expert_upload_bytes"]
    require((uploaded > 0) == (misses > 0), "expert upload/miss accounting mismatch")
    if single_slot:
        require(misses >= 432, "single-slot minimum misses not exercised")


def generation(rows):
    source(rows[0], False)
    completion(rows[-1], 1, 32, 32)
    tokens, outputs = [], []
    previous = {"expert_hits": 0, "expert_misses": 0, "expert_upload_bytes": 0}
    pending = 248044
    for i in range(32):
        token, output = rows[1 + 2 * i:3 + 2 * i]
        expect(token, {"kind": "session_token", "position": i, "token": pending,
                       "finite": True}, ("argmax", "completed_ms", "expert_hits",
                                        "expert_misses", "expert_upload_bytes"), "generation token")
        integer(token["argmax"], "generation argmax", high=VOCAB - 1)
        route_stats(token, previous, i, "completed_ms")
        expect(output, {"kind": "session_output", "index": i, "token": token["argmax"]},
               label="generation output")
        previous, pending = token, output["token"]
        tokens.append(token)
        outputs.append(output)
    # There is no consumed row for output[31], even if its ID appeared earlier.
    return {"source": rows[0], "tokens": tokens, "outputs": outputs,
            "complete": rows[-1], "records": rows, "pending_token": pending}


def metric(row, count, absolute, relative, exact_gate=False):
    label = "comparison metric"
    for key, value in {"elements": count, "finite_pairs": count, "all_finite": True,
                       "nonfinite_session": 0, "nonfinite_oracle": 0,
                       "violating_elements": 0, "passed": True}.items():
        exact(row[key], value, label + "." + key)
    maximum = number(row["max_abs"], label + ".max_abs", maximum=2 * F32_MAX)
    rms = number(row["rms"], label + ".rms", maximum=maximum)
    # A full finite vector cannot have zero RMS with nonzero maximum error.
    require(rms >= maximum / math.sqrt(count) * (1 - 1e-12), "inconsistent maximum/RMS")
    for key in ("session_argmax", "oracle_argmax"):
        integer(row[key], label + "." + key, high=count - 1)
    exact(row["argmax_agreement"], row["session_argmax"] == row["oracle_argmax"],
          label + ".argmax_agreement")
    require(type(row["oracle_max"]) in (int, float) and -F32_MAX <= row["oracle_max"] <= F32_MAX,
            label + ": invalid oracle maximum")
    if exact_gate:
        exact(row["max_abs"], 0.0, label + ".max_abs")
        exact(row["rms"], 0.0, label + ".rms")
        exact(row["max_bound_ratio"], None, label + ".max_bound_ratio")
        exact(row["worst_bound_element"], None, label + ".worst_bound_element")
        return
    ratio = number(row["max_bound_ratio"], label + ".max_bound_ratio", maximum=1)
    worst = row["worst_bound_element"]
    if ratio == 0:
        require(maximum == rms == 0, "zero bound ratio with nonzero errors")
        exact(worst, None, label + ".worst_bound_element")
    else:
        require(maximum > 0, "nonzero bound ratio with zero maximum error")
        expect(worst, {}, ("element", "session", "oracle"), "worst bound element")
        integer(worst["element"], "worst element index", high=count - 1)
        for key in ("session", "oracle"):
            require(type(worst[key]) in (int, float) and -F32_MAX <= worst[key] <= F32_MAX,
                    "nonfinite/non-numeric worst element")
        error = abs(worst["session"] - worst["oracle"])
        computed = error / (absolute + relative * abs(worst["oracle"]))
        require(error <= maximum and computed <= 1 and
                math.isclose(computed, ratio, rel_tol=1e-12, abs_tol=0),
                "inconsistent/failed worst-element frozen gate")


def control_metadata(meta):
    """Check the independent controls as emitted by oracle.cpp; never infer HF parity."""
    warm = meta["expert_cache_warmup"]
    expect(warm, {"tokens_per_pass": 32, "state_clear_after_each_pass": True,
                  "warmup_logits_discarded": True}, ("passes",), "oracle warmup")
    passes = integer(warm["passes"], "oracle warm passes", high=64)
    enabled = []
    for name in ("hf_gdn_l2_control", "hf_qsa_f32_control"):
        obj = meta[name]
        require(type(obj) is dict and type(obj.get("enabled")) is bool, name + ": missing enabled boolean")
        enabled.append(obj["enabled"])
    for name, on, per_token, mode, filename in (
        ("hf_gdn_l2_control", enabled[0], 72, "experimental_hf_semantic_correction", "gdn-l2-control.jsonl"),
        ("hf_qsa_f32_control", enabled[1], 12, "experimental_hf_normalized_attention_precision", "qsa-f32-control.jsonl"),
    ):
        obj = meta[name]
        expected = {"enabled": on, "mode": mode if on else "disabled", "diagnostic_only": True,
                    "baseline_performance_reference": False, "weights_unchanged": True,
                    "hf_source_revision": HF_REVISION, "production_source_revision": BASELINE_REVISION,
                    "records": filename if on else None,
                    "recorded_control_count": per_token * 32 if on else 0,
                    "warm_control_count": per_token * 32 * passes if on else 0,
                    "recorded_tokens_checked": 32 if on else 0,
                    "warm_tokens_checked": 32 * passes if on else 0,
                    "warm_pass_expected_count": per_token * 32 if on else 0,
                    "warm_pass_control_counts": [per_token * 32] * passes if on else []}
        if name == "hf_gdn_l2_control":
            expected.update(production_formula="x/sqrt(max(sum(x*x),eps*eps))",
                            control_formula="x/sqrt(sum(x*x)+1e-6f)", epsilon_type="f32",
                            epsilon_f32=9.99999997e-7, epsilon_f32_bits=897988541,
                            arithmetic="ascending 128-element FP32 sum; separately rounded multiply/add, no FMA; raw source-0 recomputation")
            handshake = {"name_patterns": ["q_conv_predelta-L", "k_conv_predelta-L"],
                         "layers": "0..47 except L%4==3", "op": "L2_NORM", "type": "f32",
                         "ne": [128, 16, 1, 1], "source_index": 0, "source_ne": [128, 16, 1, 1],
                         "source_type": "f32", "source_destination_disjoint_required": True,
                         "strided_storage_checked": True, "preserve_destination_padding": True,
                         "layer_count": 36, "corrections_per_token": 72}
        else:
            expected.update(independent_of_hf_gdn_l2_control=True,
                            library_revision_attested=BASELINE_REVISION,
                            recorded_expected_count=per_token * 32 if on else 0,
                            warm_expected_count=per_token * 32 * passes if on else 0,
                            max_control_visible=2048, supported_positions=[0, 2047],
                            q4_cache_unchanged=True, query_and_mask_unchanged=True,
                            raw_quant_type="q4_0", gather="Q4_0->FP16 RNE->FP32 for K and V",
                            gather_helpers=["ggml_fp16_to_fp32", "ggml_fp32_to_fp16"],
                            fp32_rounding="round-to-nearest-even required; environment is not modified",
                            softmax_accumulation="FP32",
                            arithmetic="ascending separate FP32 mul/add; scores=dot*0.0625; exp(score-max); denominator=sum(weights); output=sum(weights*V)*(1/denominator), probability-one copies gathered V; no FMA or GGML max-offset",
                            original_output_used_only_for_finite_guards_and_informational_metrics=True,
                            stage_change_before_inverse_hadamard=True, no_math_ancestor_wrappers=True)
            handshake = {"name_pattern": "kqv_out-L", "layers": list(range(3, 48, 4)),
                         "target_type": "f32", "target_ne": [6144, 1, 1, 1],
                         "ancestor_op": "FLASH_ATTN_EXT", "ancestor_count": 1,
                         "ancestor_ne": [256, 24, 1, 1],
                         "wrapper_ops": ["VIEW", "RESHAPE", "PERMUTE", "CONT"], "max_wrappers": 8,
                         "head_major_flattening_proved": True, "query_type": "f32", "query_ne": [256, 1, 24, 1],
                         "kv_type": "q4_0", "kv_ne": [256, "physicalKV", 2, 1],
                         "q4_block_bytes": 18, "q4_row_bytes": 144,
                         "mask_type": "f16", "mask_ne": ["physicalKV", ">=1", 1, 1],
                         "mask_row": 0, "mask_selected_value": 0, "mask_excluded_value": "-inf",
                         "selected_ids_exact": "0..currentPosition", "query_count": 1, "seq_id": 0,
                         "gqa_group": 12, "scale": 0.0625, "max_bias": 0, "softcap": 0,
                         "precision": "F32", "sinks": False, "source_destination_disjoint_required": True,
                         "strided_storage_checked": True, "preserve_destination_padding": True,
                         "exact_readback_required": True, "corrections_per_token": 12}
        expected["handshake"] = handshake
        expected["last_evaluation"] = {"epoch": 32 * (passes + 1) if any(enabled) else 32,
                                       "warm_pass": -1, "position": 31, "active": False,
                                       "input_token_id": 130, "control_count": per_token if on else 0}
        expect(obj, expected, label=name, extensions=True)


def oracle_metadata(meta):
    expect(meta, {"format": "gfx906-mx-oracle-v1", "status": "complete", "error": "",
                  "source_revision": BASELINE_REVISION, "library_revision_attested": BASELINE_REVISION,
                  "library_revision_runtime_verified": False, "model": MODEL,
                  "model_layers_actual": 48, "vocab_actual": VOCAB, "sampler": None,
                  "mtp": False, "token_ids": TEACHER_IDS, "completed_tokens": 32},
           ("library_version", "context", "placement", "devices", "logits", "callbacks", "expert_cache_warmup",
            "hf_gdn_l2_control", "hf_qsa_f32_control"), "oracle metadata", extensions=True)
    text(meta["library_version"], "oracle library version")
    expect(meta["context"], {"capacity_requested": 4096, "capacity_actual": 4096,
                            "batch_requested": 1, "ubatch_requested": 1, "batch_actual": 1,
                            "ubatch_actual": 1, "seq_id": 0, "seq_max": 1, "threads": 16,
                            "threads_batch": 16, "type_k": "q4_0", "type_v": "q4_0",
                            "flash_attention_requested": "enabled", "offload_kqv": True,
                            "op_offload": True, "recurrent_snapshots": 0},
           label="oracle context", extensions=True)
    expect(meta["placement"], {"split_mode": "layer", "tensor_split": [1, 1], "n_gpu_layers": -1,
                              "cpu_expert_pattern": r"\.ffn_(up|down|gate|gate_up)_(ch|)exps",
                              "use_extra_bufts": False, "load_mode": "direct_io", "lazy_mode": "off",
                              "moe_cache_slots_requested": 112},
           ("moe_cache_inserts_requested", "cache_activation_checked_layers"), "oracle placement", extensions=True)
    integer(meta["placement"]["moe_cache_inserts_requested"], "oracle cache inserts", 1, 112)
    devices = meta["devices"]
    require(type(devices) is list and len(devices) == 2, "oracle requires two recorded devices")
    for device in devices:
        expect(device, {}, ("name", "description", "backend", "device_id", "memory_total", "memory_free_before_load"),
               "oracle device")
        for key in ("name", "description", "backend"):
            text(device[key], "oracle device." + key)
        require(device["backend"] in ("ROCm", "HIP"), "unexpected oracle GPU backend")
        require(type(device["device_id"]) is str, "invalid oracle device ID")
        if device["device_id"]:
            text(device["device_id"], "oracle device ID")
        total = integer(device["memory_total"], "oracle device memory", 1)
        integer(device["memory_free_before_load"], "oracle device free memory", high=total)
    expect(meta["logits"], {"dtype": "float32", "endianness": "little",
                           "layout": "token-major,vocabulary-minor", "columns": VOCAB,
                           "header_bytes": 0, "row_bytes": VOCAB * 4, "bytes_written": 32 * VOCAB * 4},
           ("file",), "oracle logits", extensions=True)
    exact(meta["logits"]["file"], "logits.f32.bin", "oracle logits file")
    callbacks = meta["callbacks"]
    expect(callbacks, {"records": "tensors.jsonl", "tensor_byte_limit": 16777216,
                       "token_byte_limit": 67108864,
                       "binary_layout": "native little-endian storage span with original ne/nb; includes stride gaps",
                       "indices": "token_index,node_index,occurrence are zero-based; node_index counts scheduler ask calls"},
           ("captures", "allowlist", "observed_counts", "all_layer_outputs_and_slots", "gdn_probe_layers"),
           "oracle callbacks", extensions=True)
    require(type(callbacks["all_layer_outputs_and_slots"]) is bool, "invalid oracle all-layers flag")
    # --all-layers has a different capture contract; compare_session still gates
    # only the six required sites per token and whatever optional sites exist.
    checked_layers = meta["placement"]["cache_activation_checked_layers"]
    if checked_layers is not None:
        exact(checked_layers, [0, 1, 3, 47], "oracle checked cache layers")
    probes = callbacks["gdn_probe_layers"]
    require(type(probes) is list, "invalid oracle GDN probe layers")
    for layer in probes:
        integer(layer, "oracle GDN probe layer", high=47)
        require(layer % 4 != 3, "invalid QSA layer in GDN probes")
    require(len(probes) == len(set(probes)), "duplicate GDN probe layers")
    integer(callbacks["captures"], "oracle captures", 32 * 6, 32 * 256)
    names = callbacks["allowlist"]
    require(type(names) is list and all(type(n) is str for n in names) and
            len(names) == len(set(names)) and set(REQUIRED) <= set(names), "invalid oracle allowlist")
    observed = callbacks["observed_counts"]
    require(type(observed) is dict and set(observed) <= set(names), "invalid observed callback counts")
    for name, count in observed.items():
        text(name, "callback name")
        integer(count, "callback count", 1, 32 * 256)
    require(sum(observed.values()) == callbacks["captures"], "callback captures/counts mismatch")
    for name in REQUIRED:
        exact(observed.get(name), 32, "required callback count." + name)
    control_metadata(meta)
    if "qsa_probes" in meta:
        probe = meta["qsa_probes"]
        expect(probe, {"format": "gfx906-mx-qsa-probe-v1", "records": "qsa-probes.jsonl",
                       "read_only_observation": True, "warmup_suppressed": True,
                       "recorded_tokens_checked": 32, "row_byte_limit": 8192,
                       "manifest_byte_limit": 2048 * 12 * 8192},
               ("rows", "expected_rows", "requested_layers", "layer_counts"), "oracle QSA probes")
        layers = probe["requested_layers"]
        require(type(layers) is list and 1 <= len(layers) <= 12, "invalid QSA probe layers")
        for layer in layers:
            integer(layer, "QSA probe layer", 3, 47)
            require(layer % 4 == 3, "non-QSA probe layer")
        require(layers == sorted(set(layers)), "duplicate/unordered QSA probe layers")
        exact(probe["rows"], 32 * len(layers), "QSA probe rows")
        exact(probe["expected_rows"], 32 * len(layers), "QSA probe expected rows")
        exact(probe["layer_counts"], [{"layer": layer, "rows": 32} for layer in layers], "QSA layer counts")
        exact(meta["hf_qsa_f32_control"]["enabled"], True, "QSA probes require independent control")


def comparison(report):
    expect(report, {"kind": "session_comparison", "format": "gfx906-session-compare-v1",
                    "scope": "teacher_forced_operational_baseline", "baseline_revision": BASELINE_REVISION,
                    "gates": {"logits": {"absolute": .02, "relative": .002},
                              "intermediates": {"absolute": .002, "relative": .002},
                              "hc_init": "exact_numerical_float"},
                    "tokens": 32, "input_token_ids": TEACHER_IDS, "failures": [], "passed": True},
           ("artifacts", "notes", "logits", "intermediates", "optional_intermediates",
            "session_source", "session_complete", "oracle_metadata", "trace_counts"),
           "comparison report", extensions=True)
    # compare_controller, when supplied by the parent, describes the comparison
    # tool, not CORE_REVISION. Keep its actual fields without inventing equality
    # with either the compiled Session or the production library attestation.
    if "compare_controller" in report:
        require(type(report["compare_controller"]) is dict, "invalid compare_controller provenance")
    source(report["session_source"], True)
    completion(report["session_complete"], 32, 0, 32)
    oracle_metadata(report["oracle_metadata"])
    expect(report["artifacts"], {}, ("session_log", "session_logits", "session_trace", "oracle_dir"),
           "comparison artifacts")
    for path in report["artifacts"].values():
        text(path, "comparison artifact path")
    require(type(report["notes"]) is list and len(report["notes"]) >= 6, "missing operational baseline notes")
    for note in report["notes"]:
        text(note, "comparison note")
    logits = report["logits"]
    expect(logits, {"exact_byte_counts": True, "expected_bytes_each": 32 * VOCAB * 4,
                    "elements": 32 * VOCAB, "all_finite": True, "passed": True},
           ("rows", "max_abs", "rms", "max_bound_ratio", "argmax_agreeing_rows"), "comparison logits")
    rows = logits["rows"]
    require(type(rows) is list and len(rows) == 32, "comparison must have 32 full-vocabulary rows")
    for i, row in enumerate(rows):
        expect(row, {"token_index": i, "position": i, "input_token_id": TEACHER_IDS[i]},
               METRIC_KEYS, "logits row")
        metric(row, VOCAB, .02, .002)
    for field in ("max_abs", "max_bound_ratio"):
        number(logits[field], "aggregate logits." + field, maximum=1 if field == "max_bound_ratio" else None)
        require(logits[field] == max(row[field] for row in rows), "logits aggregate " + field + " mismatch")
    number(logits["rms"], "aggregate logits.rms")
    expected_rms = math.sqrt(math.fsum(row["rms"] ** 2 * row["finite_pairs"] for row in rows) / (32 * VOCAB))
    require(math.isclose(logits["rms"], expected_rms, rel_tol=1e-12, abs_tol=0), "logits aggregate RMS mismatch")
    exact(logits["argmax_agreeing_rows"], sum(row["argmax_agreement"] for row in rows), "argmax agreeing rows")
    for optional, key, expected_count in ((False, "intermediates", 192), (True, "optional_intermediates", 128)):
        entries = report[key]
        require(type(entries) is list and len(entries) == expected_count, key + ": wrong row count")
        seen = set()
        for row in entries:
            require(type(row) is dict, key + ": invalid row")
            i = integer(row.get("token_index"), key + " token index", high=31)
            name, layer = row.get("name"), row.get("layer")
            integer(layer, key + " layer", -1, 47)
            require(type(name) is str and (name == "hc_attn_mix" and layer in LAYERS if optional else name in REQUIRED),
                    key + ": unexpected name/layer")
            if not optional:
                exact(layer, REQUIRED[name][0], key + " layer")
            identity = (i, name, layer)
            require(identity not in seen, key + ": duplicate tensor")
            seen.add(identity)
            fixed = {"token_index": i, "position": i, "name": name, "layer": layer, "optional": optional}
            if optional and row.get("status") == "missing_optional":
                expect(row, {**fixed, "status": "missing_optional"}, ("missing",), key)
                require(type(row["missing"]) is list and row["missing"] in
                        (["session"], ["oracle"], ["session", "oracle"]), "invalid optional missing sides")
                continue
            exact_gate = name == "hc_init"
            expect(row, {**fixed, "status": "compared", "oracle_occurrence": 0,
                         "oracle_name": "hc_mixed-" + str(layer) if optional else name,
                         "gate": "exact_numerical_float" if exact_gate else "bounded"}, METRIC_KEYS, key)
            metric(row, 2560 if optional else REQUIRED[name][1], .002, .002, exact_gate)
    counts = report["trace_counts"]
    expect(counts, {}, ("session", "oracle"), "trace counts")
    for side in ("session", "oracle"):
        integer(counts[side], "trace count." + side, 192, 32 * 256)
    exact(counts["oracle"], report["oracle_metadata"]["callbacks"]["captures"], "oracle trace count")
    minimum_session = 192 + sum(row["status"] == "compared" or "session" not in row.get("missing", [])
                                for row in report["optional_intermediates"])
    require(counts["session"] >= minimum_session, "session trace count missing compared/captured tensors")
    observed = report["oracle_metadata"]["callbacks"]["observed_counts"]
    for layer in LAYERS:
        minimum = sum(row["layer"] == layer and (row["status"] == "compared" or
                      "oracle" not in row.get("missing", [])) for row in report["optional_intermediates"])
        require(observed.get("hc_mixed-" + str(layer), 0) >= minimum,
                "oracle observed count missing optional captured tensors")
    return report


def reset(row):
    expect(row, {"kind": "session_test_complete",
                 "runtime": RUNTIME, "capacity": 4, "expert_slots": 1, "trace": False,
                 "sampling": "teacher_forced", "scope": RESET_SCOPE, "performance_claim": False,
                 "checks": {"finite_logit_values": 8 * VOCAB, "bitwise_compared_logit_values": 4 * VOCAB,
                            "invalid_token_rejections": 2, "capacity_rejections": 2,
                            "rejection_stats_preserved": True, "rejection_logits_preserved": True,
                            "invalid_token_continuation": True, "reset_reused": True,
                            "minimum_slot_reuse_misses_per_step": 432},
                 "invalid_config_checks": {"tested": False, "capacity_cases": [0, 3], "expert_slots_cases": [0],
                                           "reason": "constructor validation follows Model and PLE initialization"},
                 "passed": True}, ("model", "revision", "dirty", "initial_stats", "baseline_steps", "after_reset_stats",
                                    "replay_steps", "final_reset_stats"), "reset proof")
    compiled_provenance(row, "reset proof", boolean_dirty=True)
    require(Path(text(row["model"], "reset model")).name == MODEL, "reset model variant mismatch")
    zero = {"consumed_tokens": 0, "expert_hits": 0, "expert_misses": 0,
            "expert_upload_bytes": 0, "last_completed_ms": 0}
    for key in ("initial_stats", "after_reset_stats", "final_reset_stats"):
        obj = row[key]
        expect(obj, {k: v for k, v in zero.items() if k != "last_completed_ms"}, ("last_completed_ms",), key)
        number(obj["last_completed_ms"], key + ".last_completed_ms", maximum=0)
    for key in ("baseline_steps", "replay_steps"):
        steps = row[key]
        require(type(steps) is list and len(steps) == 4, key + ": expected four steps")
        previous = zero
        for i, step in enumerate(steps):
            expect(step, {"token": RESET_IDS[i]}, ("stats",), key)
            obj = step["stats"]
            expect(obj, {"consumed_tokens": i + 1}, ("expert_hits", "expert_misses",
                                                   "expert_upload_bytes", "last_completed_ms"), key + ".stats")
            route_stats(obj, previous, i, "last_completed_ms", single_slot=True)
            previous = obj
    return row


def collect(comparison_path, generation_path, reset_path):
    """Return ONE r3b_session record, or raise ValueError/OSError; never write inputs."""
    report = comparison(json_object(raw_bytes(comparison_path, MAX_COMPARISON_BYTES), "comparison"))
    generated = generation(records(generation_path, 66))
    replay = reset(records(reset_path, 1)[0])
    sources = (report["session_source"], generated["source"], replay)
    compiled = [(item["revision"], int(item["dirty"])) for item in sources]
    require(len(set(compiled)) == 1, "Session/generation/reset compiled provenance mismatch")
    require(len({item["model"] for item in sources}) == 1, "Session/generation/reset model path mismatch")
    return {"kind": "r3b_session", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "revision": compiled[0][0], "dirty": compiled[0][1], "runtime": RUNTIME,
            "model_variant": "qwen38-keep1-Q4_0", "model": sources[0]["model"],
            "raw_logs": {"comparison": str(Path(comparison_path).resolve()),
                         "generation": str(Path(generation_path).resolve()), "reset": str(Path(reset_path).resolve())},
            "scope": "48-layer short-session diagnostic qualification with frozen full-vocabulary teacher gates, ordered greedy output and reset/rejection self-parity",
            "scopes": {"comparison": report["scope"], "generation": generated["complete"]["scope"],
                       "reset": replay["scope"]},
            "performance_claim": False, "bitwise_hf_claim": False,
            "oracle_scope": "pinned production operational forward with independently declared optional GDN additive-L2 and QSA FP32 diagnostic corrections; controls are not bitwise HF or a baseline-performance reference",
            "generation_scope": "one prompt plus 31 emitted-token consumes; output 31 remains pending; --ignore-eos is not attested in the emitted schema; greedy diagnostic, no user-sampling or prefill speed claim",
            "reset_scope": "four teacher IDs replayed after reset, full-vocabulary bitwise self-parity; two invalid IDs and two capacity rejections preserve statistics/logits; invalid constructor configurations explicitly untested",
            "counts": {"teacher_tokens": 32, "teacher_finite_logit_pairs": 32 * VOCAB,
                       "output_tokens": 32, "consumed_tokens": 32, "reset_steps": 8,
                       "reset_bitwise_logit_values": 4 * VOCAB},
            "provenances": {"session": {k: sources[0][k] for k in ("revision", "dirty")},
                            "generation": {k: sources[1][k] for k in ("revision", "dirty")},
                            "reset": {k: sources[2][k] for k in ("revision", "dirty")},
                            "oracle_source_revision": report["oracle_metadata"]["source_revision"],
                            "oracle_library_revision_attested": report["oracle_metadata"]["library_revision_attested"],
                            "oracle_library_revision_runtime_verified": False},
            "comparison": report, "generation": generated, "reset": replay, "passed": True}


def append_result(destination, record):
    """Serialize before opening results and protect both raw and reported artifacts."""
    payload = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    destination = Path(destination).resolve()
    files = [Path(p).resolve() for p in record["raw_logs"].values()]
    artifacts = record["comparison"]["artifacts"]
    files.extend(Path(artifacts[k]).resolve() for k in ("session_log", "session_logits"))
    files.append(Path(record["model"]).resolve())
    for path in files:
        require(destination != path and not (destination.exists() and path.exists() and os.path.samefile(destination, path)),
                "results would overwrite an input artifact")
    for key in ("session_trace", "oracle_dir"):
        require(not destination.is_relative_to(Path(artifacts[key]).resolve()),
                "results would write inside input captures")
    if destination.exists():
        require(stat.S_ISREG(destination.stat().st_mode), "results is not a regular file")
    with destination.open("ab+") as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        output.seek(0, os.SEEK_END)
        if output.tell():
            output.seek(-1, os.SEEK_END)
            require(output.read(1) == b"\n", "results has an incomplete final line")
        output.write(payload)
        output.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "generation", "reset", "results"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = collect(args.comparison, args.generation, args.reset)
        append_result(args.results, record)
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print("record_session: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
