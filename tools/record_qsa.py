#!/usr/bin/env python3
"""Validate protocol-v1 QSA GPU fixtures and append one scoped R2d result."""
import argparse
import datetime
import json
import math
from pathlib import Path
import re


MAX_RAW_BYTES = 1024 * 1024
MAX_ROW_BYTES = 64 * 1024
PROTOCOL_ROWS = 62
SOURCE_SCOPE = "fixture_only_loaded_indexer_weights_CPU_raw_FP32_projection_of_eight_synthetic_columns_repeated_in_cache_timeline; synthetic_attention_KV; not_prompt_or_full_QSA_block_or_inference"
EVENT_PROTOCOL = "20_individual_completed_GPU_event_intervals; arithmetic_mean_ms; reset_restore_and_flag_clear_outside_events"
ROW_COUNTS = {"qsa_source": 1, "qsa_correctness": 2, "qsa_append": 32,
              "qsa_score": 6, "qsa_select": 8, "qsa_attention": 12,
              "qsa_complete": 1, "total": PROTOCOL_ROWS}
GATES = {"key": {"abs": 2e-4, "rel": 2e-4},
         "score": {"abs": 2e-4, "rel": 2e-4},
         "attention": {"abs": 2e-4, "rel": 2e-4},
         "bound": "abs+rel*abs(CPU_reference)", "max_bound_ratio_limit": 1}
CONFIG = {"capacity": 131072, "index_D": 128, "index_Q": 4, "index_KV": 1,
          "compress": 4, "budget": 2048, "rotary_dim": 64,
          "rope_scale": 1, "sections": [11, 11, 10, 0],
          "text_position_axes": "identical_split_half",
          "frequencies": "FP32_1/pow(base,2*i/64); bit_identical_CPU_and_uploaded_GPU",
          "pooled_cache_bytes_per_device": 16777216,
          "Q4_cache_bytes_each_per_device": 37748736}
COUNTERS = {"cache_cases": 32, "prefix_checks": 648, "restore_cases": 24,
            "reject_windows": 8, "q4_blocks": 8, "append_rejects": 6,
            "selection_checks": 200, "count_checks": 200, "selection_rejects": 3,
            "score_checks": 60, "score_rejects": 2,
            "attention_checks": 26, "attention_rejects": 11}
SUMMARY_FLAGS = {name: True for name in (
    "q4_byte_parity", "q4_dequant_bit_parity", "tail_byte_parity",
    "unused_id_canaries", "cache_readonly_byte_parity", "chunk_atomic",
    "sticky_reuse", "future_poison_unread", "passed")}
# Gate::elements counts every comparison, including post-rejection valid reuse.
# These sizes follow q4/cache/score/attention_cases in frozen qsa_main.hip.
SUMMARY_ELEMENTS = {"key_error": 8451328, "common_score_error": 47157,
                    "actual_score_error": 96362, "attention_error": 233472}


def invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite JSON number: {value}")
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def records(path):
    """Read bounded LF/CRLF JSONL, never skip blank or unstructured rows."""
    path = Path(path)
    if path.stat().st_size > MAX_RAW_BYTES:
        raise ValueError("QSA raw fixture too large")
    result, total = [], 0
    with path.open("rb") as input_log:
        while True:
            line = input_log.readline(MAX_ROW_BYTES + 1)
            if not line:
                break
            total += len(line)
            if len(line) > MAX_ROW_BYTES or total > MAX_RAW_BYTES:
                raise ValueError("QSA JSONL row or fixture too large")
            if len(result) >= PROTOCOL_ROWS:
                raise ValueError("QSA fixture must contain exactly 62 rows")
            try:
                item = json.loads(line.decode("utf-8"), parse_constant=invalid_constant,
                                  parse_float=finite_float, object_pairs_hook=unique_object)
            except (ValueError, RecursionError) as error:
                raise ValueError(f"QSA JSONL line {len(result) + 1}: {error}") from error
            if type(item) is not dict:
                raise ValueError(f"QSA JSONL line {len(result) + 1}: expected an object")
            result.append(item)
    return result


def exact(value, expected, label):
    """Recursive fixed schema comparison; bool never equals an integer counter."""
    if type(value) is not type(expected):
        raise ValueError(f"{label}: incorrect type")
    if type(expected) is dict:
        expect(value, expected, label=label)
    elif type(expected) is list:
        if len(value) != len(expected):
            raise ValueError(f"{label}: incorrect length")
        for index, (got, want) in enumerate(zip(value, expected)):
            exact(got, want, f"{label}[{index}]")
    elif value != expected:
        raise ValueError(f"{label}: unexpected value")


def expect(item, expected, variable=(), label=None):
    label = label or expected.get("kind", "QSA")
    if type(item) is not dict or set(item) != set(expected) | set(variable):
        raise ValueError(f"{label}: unexpected or missing fields")
    for field, value in expected.items():
        exact(item[field], value, f"{label}.{field}")


def finite_number(value, label, minimum=0, maximum=None, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f"{label}: expected a nonboolean number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite or value < minimum or (positive and value <= minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{label}: invalid numeric value")


def gate(value, elements, label):
    expect(value, {"elements": elements}, ("max_abs", "max_bound_ratio"), label)
    finite_number(value["max_abs"], f"{label}.max_abs")
    finite_number(value["max_bound_ratio"], f"{label}.max_bound_ratio", maximum=1)


def measurement(item, expected, errors=()):
    expect(item, expected, ("resident_ms",) + tuple(name for name, _ in errors))
    finite_number(item["resident_ms"], f"{item['kind']}.resident_ms", positive=True)
    for name, elements in errors:
        gate(item[name], elements, f"{item['kind']}.{name}")


def collect(path):
    """Return one r2d_qsa record from the exact completed 62-row GPU protocol."""
    path = Path(path)
    rows = records(path)
    if len(rows) != PROTOCOL_ROWS:
        raise ValueError("QSA fixture must contain exactly 62 rows")
    source = rows[0]
    expect(source, {
        "kind": "qsa_source", "protocol": 1, "model_variant": "qwen38-keep1-Q4_0",
        "layer": 3, "scope": SOURCE_SCOPE,
        "synthetic_x": "FP32_0.3*sin(i*0.137)+0.1*cos(i*0.071); eight_2560_element_columns",
        "projection_types": {"q_proj": "BF16[2560,512]", "k_proj": "BF16[2560,128]",
                             "q_norm": "F32[128]", "k_norm": "F32[128]"},
        "metadata_types": {"rope.dimension_count": "UINT32", "rope.freq_base": "FLOAT32",
                           "attention.layer_norm_rms_epsilon": "FLOAT32",
                            "rope.dimension_sections": "ARRAY_INT32"},
        "gates": GATES, "correctness_before_any_metrics": True,
        "validate_after_each_timed_repeat": True, "event_protocol": EVENT_PROTOCOL,
        "rows": ROW_COUNTS}, ("revision", "dirty", "model_path", "config"))
    if (type(source["revision"]) is not str
            or re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is None
            or type(source["dirty"]) is not bool):
        raise ValueError("QSA invalid compiled source provenance")
    if (type(source["model_path"]) is not str or not source["model_path"]
            or any(ord(c) < 32 for c in source["model_path"])):
        raise ValueError("QSA invalid model path")
    config = source["config"]
    expect(config, CONFIG, ("rope_base", "rms_epsilon"), "qsa_source.config")
    for field, expected in (("rope_base", 10000000), ("rms_epsilon", 9.999999974752427e-7)):
        finite_number(config[field], f"qsa_source.config.{field}", positive=True)
        if config[field] != expected:
            raise ValueError(f"qsa_source.config: unexpected {field}")

    correctness = rows[1:3]
    for device, item in enumerate(correctness):
        expect(item, {"kind": "qsa_correctness", "device": device,
                      **COUNTERS, **SUMMARY_FLAGS}, SUMMARY_ELEMENTS)
        for name, elements in SUMMARY_ELEMENTS.items():
            gate(item[name], elements, f"qsa_correctness.{name}")

    offset = 3
    for device in (0, 1):
        for phase in range(4):
            for n in (1, 2, 3, 128):
                base = 4096 + phase
                visible = base + n
                measurement(rows[offset], {
                    "kind": "qsa_append", "device": device, "N": n,
                    "base": base, "phase": phase, "visible": visible, "repeats": 20,
                    "launches": 4 if phase + n >= 4 else 3,
                    "scope": "resident_raw_K_q4_roundtrip_new_pool_norm_rope_publish; no_prefix_snapshots",
                    "timing_excludes": "projection,H2D,D2H,reset,tail_restore,error_clear,validation",
                    "completed_blocks": visible // 4, "new_blocks": (phase + n) // 4,
                    "tail_byte_parity": True, "prefix_byte_parity": True},
                    (("key_error", 20 * (visible // 4) * 128),))
                offset += 1
        for visible in (4096, 32768, 131072):
            measurement(rows[offset], {
                "kind": "qsa_score", "device": device, "visible": visible,
                "prepared": 131072, "repeats": 20, "launches": 2,
                "scope": "resident_raw_Q_norm_rope_and_causal_score; two_launches; actual_GPU_keys",
                "timing_excludes": "projection,H2D,D2H,append,error_clear,validation",
                "completed_blocks": visible // 4, "future_output_canary": True},
                (("score_error", 20 * (visible // 4)),))
            offset += 1
        for visible in (2052, 4096, 32768, 131072):
            measurement(rows[offset], {
                "kind": "qsa_select", "device": device, "visible": visible,
                "repeats": 20, "launches": 19,
                "scope": "resident_selection_only; nineteen_launches; random_fixed_same_score_floats",
                "timing_excludes": "projection,H2D,D2H,index_score,error_clear,validation",
                "token_count": 2048, "block_count": 512, "ids_exact": True,
                "counts_exact": True, "unused_capacity_byte_parity": True})
            offset += 1
        for selected in (1, 63, 64, 65, 2048, 2051):
            measurement(rows[offset], {
                "kind": "qsa_attention", "device": device, "visible": 4096,
                "capacity": 131072, "selected": selected, "repeats": 20, "launches": 6,
                "scope": "resident_selected_Q4_gather_FP16_and_attention_publish; Q24_KV2_D256",
                "cache_source": "synthetic_canonical_Q4; all_unselected_scales_Inf",
                "reference": "CPU_gathered_fp16_same_ID_order_and_repeats",
                "timing_excludes": "projection,Hadamard,RoPE,gate,index_select,H2D,D2H,error_clear,validation",
                "token_count": selected, "block_count": min(512, selected // 4),
                "cache_readonly_byte_parity": True, "output_canaries": True},
                (("attention_error", 20 * 6144),))
            offset += 1
    footer = rows[offset]
    expect(footer, {"kind": "qsa_complete", "protocol": 1, "rows": PROTOCOL_ROWS,
                    "devices": 2, "passed": True})

    return {
        "kind": "r2d_qsa", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": source["revision"], "dirty": int(source["dirty"]),
        "raw_log": str(path.resolve()), "source": source,
        "scope": "actual layer3 F32 indexer norms and BF16 Q/K weights; CPU raw FP32 projections of eight synthetic columns repeated in a numeric cache timeline; GPU append/score/selection and synthetic read-only Q4 gather/FP16 attention; not GPU projections, full QSA block, full-model inference or inference speed",
        "timing_scope": "completed GPU events, arithmetic mean of 20 individually completed intervals, validated after every repeat; projections, transfers, reset/restore, error clear and validation excluded as specified by each raw measurement",
        "state_abi": {"pooled_keys": "FP32 [32768][128], completed blocks only",
                      "raw_tail": "FP32 [3][128], chronological Q4 roundtrip keys, unused floats zero",
                      "logical_length": "consumed inputs; completed_blocks=length//4; actual_tail=length%4",
                      "publication": "caller accepts logical length only after completed stream and sticky error==0; failed append preserves cache/tail bytes",
                      "restore": "384 tail floats plus logical length; no pooled-history copy or full raw history"},
        "prefix_abi": {"slot_0": "pre-chunk tail", "slot_n": "after n consumed inputs",
                       "verify_inputs": "one pending input + two drafts", "accept_slot": "1 + a",
                       "accepted_drafts_a": [0, 1, 2]},
        "indexer_contract": {"key": "raw K -> canonical Q4_0 quantize/dequantize -> mean4 -> direct stored gamma RMS -> split-half RoPE at block start 4*b",
                             "query": "raw Q[4][128] -> direct stored gamma RMS -> split-half RoPE at visible-1",
                             "score": "sum_h ReLU(dot(Q_h,K_b))/sqrt(128); only visible//4 completed blocks"},
        "selection_contract": {"blocks": "up to 512 whole blocks, descending score; ties use smaller block ID",
                               "tail": "actual visible%4 positions, never padding",
                               "capacity": 2051, "actual_count": "4*min(512,visible//4)+visible%4",
                               "ids_reference": "exact CPU selection on identical score floats; differing score arithmetic need not select identical cutoff IDs"},
        "attention_contract": {"geometry": "Q24/KV2/D256", "cache": "synthetic canonical Q4_0 K/V; read-only; unselected scales Inf",
                               "reference": "CPU gathered FP16 with identical selected ID order and repetitions",
                               "workspace": "bounded selected-row FP16 gather, wave64 64-key chunks and FP32 softmax merge"},
        "fixture_gates": {**source["gates"], "frozen_before_first_gpu_run": True},
        "donors": {"mx": "dcd685463d597d31f5ca759d32c94592a2740fa4",
                   "furnace": "905021dbad71c5056ef51f9fd45d545403fc989c",
                   "reinstinct": "0b79e326351d90d4554a1c18df92da5d0ab692e8"},
        "reference": "transformers a005fc82babfe8871d87746decad2dbee100a125",
        "correctness": correctness, "measurements": rows[3:offset], "complete": footer,
        "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    # Serialize before opening the destination: failed validation adds no record.
    result = json.dumps(collect(args.raw), allow_nan=False, separators=(",", ":"))
    with args.results.open("a", encoding="utf-8") as output:
        output.write(result + "\n")


if __name__ == "__main__":
    main()
