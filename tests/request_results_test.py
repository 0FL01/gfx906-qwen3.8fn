"""Synthetic protocol unit tests only: no model, GPU evidence or ROOT journal."""

import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import record_memory
from tools import record_request as MODULE

SCRIPT = ROOT / "tools/record_request.py"
REVISION = "0123456789abcdef0123456789abcdef01234567"
VOCAB, EOS, Q40, Q41 = 248320, 248046, 2764800, 2867200
COUNTERS = ("expert_hits", "expert_misses", "expert_upload_bytes")


def fixture(mode="stochastic", outputs=None, requested=None, chunk=2, trace="", logits="", ignore=False):
    """Independent transcription of the current driver, not production evidence."""
    outputs = [101, 102, 103] if outputs is None else outputs
    requested = len(outputs) if requested is None else requested
    diagnostic = mode == "greedy_diagnostic" or bool(trace) or bool(logits)
    prompt = [248044, 100, 101, 102, 103]
    sampling = {"mode": mode}
    if mode == "stochastic":
        sampling.update(seed=42, temperature=1., top_p=.95, top_k=20,
                        rng="mt19937_64_high53_ascending_id_cdf", baseline_rng_equivalent=False)
    source = {
        "kind": "session_request_source", "protocol": 1, "revision": REVISION, "dirty": True,
        "model": '/models/данные "quoted"/qwen38-keep1-Q4_0.gguf',
        "runtime": "own_48_layer_HIP", "kv": "Q4_0", "candidate": True,
        "mtp": False, "mtp_acceptance": None, "series": "primary_sampling" if mode == "stochastic" else mode,
        "session_start": "fresh", "prefix_reuse": False, "expert_cache_warmness": "unknown",
        "config": {"capacity": 32, "expert_slots": 112, "max_batch_tokens": chunk,
                   "requested_output_tokens": requested, "ignore_eos": ignore, "trace": bool(trace),
                   "trace_directory": trace, "logits_path": logits, "diagnostic_rows": diagnostic},
        "sampling": sampling,
        "timing_scope": "completed_wall_sampling_and_cli_io_excludes_load_cleanup",
        "pp_scope": "completed_prompt_calls_and_row_io",
        "tg_scope": "remaining_output_forwards_sampling_and_io", "prompt_ids": prompt,
    }
    records, counters, position = [source], dict.fromkeys(COUNTERS, 0), 0

    def consume(tokens, phase, next_argmax):
        nonlocal position
        if diagnostic:
            for i, token in enumerate(tokens):
                records.append({"kind": "session_request_row", "protocol": 1, "phase": phase,
                                "position": position + i, "token": token,
                                "argmax": next_argmax if i == len(tokens) - 1 else 99, "finite": True})
        # A realizable repeated group: ten experts/layer, one upload/group/call.
        counters["expert_misses"] += 480
        counters["expert_hits"] += 480 * (len(tokens) - 1)
        counters["expert_upload_bytes"] += 420 * Q40 + 60 * Q41
        if diagnostic:
            records.append({"kind": "session_request_window", "protocol": 1, "phase": phase,
                            "first_position": position, "rows": len(tokens), "completed_ms": 2., **counters})
        position += len(tokens)

    for first in range(0, len(prompt), chunk):
        consume(prompt[first:first + chunk], "pp", outputs[0] if outputs else 99)
    for i, token in enumerate(outputs):
        if i:
            consume([outputs[i - 1]], "tg", token)
        records.append({"kind": "session_request_output", "protocol": 1, "index": i, "token": token})
    pp_calls, forwards = (len(prompt) + chunk - 1) // chunk, max(len(outputs) - 1, 0)
    pp_ms, first_after = pp_calls * 2. + .5, .25 if outputs else 0.
    first_ms = pp_ms + first_after + .25 if outputs else None
    tg_ms = forwards * 2. + .5 if forwards else 0.
    records.append({
        "kind": "session_request_complete", "protocol": 1, "candidate": True,
        "mtp": False, "mtp_acceptance": None, "input_tokens": len(prompt), "output_tokens": len(outputs),
        "consumed_tokens": position, "pp_calls": pp_calls, "tg_forwards": forwards,
        "random_draws": len(outputs) if mode == "stochastic" else 0, "load_ms": 12345.,
        "pp_ms": pp_ms, "tg_ms": tg_ms, "total_ms": (first_ms if outputs else pp_ms) + tg_ms + .25,
        "first_output_ms": first_ms, "first_output_after_pp_ms": first_after,
        "stop_reason": "eos" if outputs and outputs[-1] == EOS and not ignore else
                       "output_limit" if outputs else "prompt_only",
        "pending_token": outputs[-1] if outputs else None, **counters, "generated_ids": outputs,
        "scope": "candidate_non_mtp_full_request_no_speed_claim", "passed": True,
    })
    return records


def encoded(rows):
    return ("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n").encode()


def vram_fixture(request):
    """Synthetic observer transcription; BDF strings are not HIP attestation."""
    origin, config, sampling = request[0], request[0]["config"], request[0]["sampling"]
    command = ["docker", "run", "--rm", "--name", "synthetic-request",
               "--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "video",
               "--ipc", "host", "--security-opt", "seccomp=unconfined",
               "--entrypoint", "/core/build/core-session",
               "-v", "/home/radneon/gfx906-core:/core",
               "-v", "/home/radneon/models-nvme:/models:ro", "llama.cpp-gfx906:cmake-4.4.3",
               "--capacity", str(config["capacity"]), "--slots", str(config["expert_slots"]),
               "--prefill-chunk", str(config["max_batch_tokens"]),
               "--generate", str(config["requested_output_tokens"])]
    if config["ignore_eos"]:
        command.append("--ignore-eos")
    for flag, key in (("--trace", "trace_directory"), ("--logits", "logits_path")):
        if config[key]:
            command.extend([flag, config[key]])
    if sampling["mode"] == "stochastic":
        command.extend(["--sample", "--seed", str(sampling["seed"]),
                        "--temperature", str(sampling["temperature"]),
                        "--top-p", str(sampling["top_p"]), "--top-k", str(sampling["top_k"])])
    command.extend([origin["model"], *map(str, origin["prompt_ids"])])
    scope = "sampled_global_driver_VRAM_not_exact_instantaneous_peak"
    total, rounds = 17163091968, 130
    paths = ["/sys/devices/pci0000:00/0000:00:03.0/0000:05:00.0",
             "/sys/devices/pci0000:00/0000:00:03.0/0000:08:00.0"]
    return [{
        "kind": "vram_source", "protocol": 1, "scope": scope,
        "mapping_attestation": "caller_supplied_labels_no_HIP_index_attestation",
        "value_source": "AMD_sysfs_mem_info_vram_used_and_mem_info_vram_total",
        "sampling_window": "pre_spawn_through_child_lifetime_and_post_wait",
        "elapsed_scope": "monotonic_before_Popen_through_completed_child_wait_includes_spawn_and_observer_overhead",
        "command_argv": command, "interval_seconds": .1,
        "devices": [{"label": f"device_{i}", "supplied_path": f"/sys/class/drm/card{i}/device",
                     "resolved_path": path, "used_file": path + "/mem_info_vram_used",
                     "total_file": path + "/mem_info_vram_total", "total_bytes": total}
                    for i, path in enumerate(paths)],
    }, {
        "kind": "vram_complete", "protocol": 1, "scope": scope,
        "observation_complete": True, "child_started": True, "child_returncode": 0,
        "observer_returncode": 0, "completed_child_elapsed_seconds": 15.,
        "sample_rounds": rounds, "post_wait_sampled": True, "observer_error": None,
        "devices": [{"label": f"device_{i}", "total_bytes": total, "sample_count": rounds,
                     "observed_max_used_bytes": 12000000000 + i,
                     "observed_min_free_bytes": total - 12000000000 - i} for i in range(2)],
    }]


class RequestResultsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="request-results-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw, self.results = self.root / "request.jsonl", self.root / "results.jsonl"
        self.rows = fixture()
        self.original = encoded(self.rows)
        self.raw.write_bytes(self.original)

    def collect(self, rows=None):
        if rows is not None:
            self.raw.write_bytes(encoded(rows))
        return MODULE.collect(self.raw)

    def reject(self, rows):
        with self.assertRaises(ValueError):
            self.collect(rows)

    def change(self, index, path, bad, rows=None):
        rows = copy.deepcopy(self.rows if rows is None else rows)
        target = rows[index]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad
        with self.subTest(index=index, path=path, bad=bad):
            self.reject(rows)

    def cli(self, destination=None, raw=None, extra=()):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", str(raw or self.raw),
                               "--results", str(destination if destination is not None else self.results), *extra],
                              cwd=self.root, capture_output=True, text=True, timeout=30)

    def test_primary_exact_source_outputs_footer_counts_and_rates(self):
        result = self.collect()
        self.assertEqual(result["kind"], "r4_request")
        self.assertEqual(result["source"], self.rows[0])
        self.assertEqual(result["output_ids"], [101, 102, 103])
        self.assertEqual(result["complete"], self.rows[-1])
        self.assertEqual(result["revision"], REVISION)
        self.assertIs(result["dirty"], True)
        self.assertEqual(result["counts"], {"input_tokens": 5, "output_tokens": 3, "consumed_tokens": 7,
                                          "pp_calls": 3, "tg_forwards": 2, "random_draws": 3,
                                          "requested_output_tokens": 3, "records": 5})
        self.assertEqual(result["diagnostics"], {"row_count": 0, "window_count": 0,
                                               "pp_completed_call_ms_sum": None, "tg_completed_call_ms_sum": None})
        self.assertEqual(result["throughput"]["pp_tokens_per_second"], 5000 / 6.5)
        self.assertEqual(result["throughput"]["tg_output_tokens_per_second"], 2000 / 4.5)
        self.assertEqual(result["throughput"]["request_output_tokens_per_second"], 3000 / 11.75)
        self.assertEqual(result["raw_logs"], {"request": str(self.raw.resolve())})
        for field in ("mtp", "performance_claim", "speedup_claim", "paired_ab_claim",
                      "long_context_qualification_claim", "peak_vram_qualification_claim", "independent_reference_claim"):
            self.assertIs(result[field], False)
        self.assertIsNone(result["mtp_acceptance"])
        self.assertEqual(result["scope"], "candidate_non_mtp_full_request_no_speed_claim")
        self.assertEqual(self.raw.read_bytes(), self.original)
        self.assertNotIn("model_bytes", result)
        self.assertNotIn("argv", result["source"])

    def test_custom_filters_top_k_zero_and_unsigned_seed_endpoints(self):
        for temperature, top_p, top_k in ((.7, .8, 0), (2., 1., VOCAB), (1., .95, 1)):
            rows = fixture()
            rows[0]["series"] = "custom_sampling"
            rows[0]["sampling"].update(temperature=temperature, top_p=top_p, top_k=top_k)
            self.assertIs(self.collect(rows)["passed"], True)
        for seed in (0, (1 << 64) - 1):
            rows = fixture()
            rows[0]["sampling"]["seed"] = seed
            rows[0]["revision"], rows[0]["dirty"] = REVISION.upper(), False
            self.assertEqual(self.collect(rows)["source"]["sampling"]["seed"], seed)

    def test_greedy_grouped_diagnostic_chronology(self):
        rows = fixture("greedy_diagnostic")
        result = self.collect(rows)
        self.assertEqual(result["counts"]["records"], 17)
        self.assertEqual(result["diagnostics"], {"row_count": 7, "window_count": 5,
                                               "pp_completed_call_ms_sum": 6., "tg_completed_call_ms_sum": 4.})
        self.assertEqual(result["counts"]["random_draws"], 0)
        self.assertEqual(result["source"]["sampling"], {"mode": "greedy_diagnostic"})

    def test_prompt_only_zero_outputs_no_rng_first_output_or_pending(self):
        rows = fixture("greedy_diagnostic", outputs=[])
        result = self.collect(rows)
        self.assertEqual(result["counts"], {"input_tokens": 5, "output_tokens": 0, "consumed_tokens": 5,
                                          "pp_calls": 3, "tg_forwards": 0, "random_draws": 0,
                                          "requested_output_tokens": 0, "records": 10})
        self.assertIsNone(result["complete"]["pending_token"])
        self.assertIsNone(result["timings_ms"]["first_output_ms"])
        self.assertEqual(result["timings_ms"]["first_output_after_pp_ms"], 0)
        self.assertIsNone(result["throughput"]["tg_output_tokens_per_second"])
        self.assertIsNone(result["throughput"]["request_output_tokens_per_second"])

    def test_sampled_diagnostics_logit_artifact_or_trace(self):
        for rows in (fixture(logits="captured.bin"), fixture(chunk=1, trace="trace-dir")):
            result = self.collect(rows)
            self.assertEqual(result["diagnostics"]["row_count"], 7)
            self.assertEqual(result["counts"]["random_draws"], 3)
            self.assertEqual(result["source"]["series"], "primary_sampling")
        self.assertIn("logits_artifact", self.collect(fixture(logits="captured.bin"))["raw_logs"])

    def test_eos_early_at_limit_and_first_output_remains_pending(self):
        for outputs, requested in (([EOS], 3), ([101, EOS], 3), ([101, 102, EOS], 3)):
            rows = fixture(outputs=outputs, requested=requested)
            result = self.collect(rows)
            self.assertEqual(result["complete"]["stop_reason"], "eos")
            self.assertEqual(result["complete"]["pending_token"], EOS)
            self.assertEqual(result["counts"]["consumed_tokens"], 5 + len(outputs) - 1)
            self.assertEqual(result["counts"]["random_draws"], len(outputs))
        result = self.collect(fixture(outputs=[101]))
        self.assertEqual(result["counts"]["tg_forwards"], 0)
        self.assertEqual(result["complete"]["tg_ms"], 0)
        self.assertIsNone(result["throughput"]["tg_output_tokens_per_second"])

    def test_ignored_eos_can_be_forwarded_and_last_token_is_pending(self):
        rows = fixture("greedy_diagnostic", outputs=[EOS, 101, EOS], ignore=True)
        result = self.collect(rows)
        self.assertEqual(result["complete"]["stop_reason"], "output_limit")
        tg = [r for r in rows if r["kind"] == "session_request_row" and r["phase"] == "tg"]
        self.assertEqual([r["token"] for r in tg], [EOS, 101])
        self.assertEqual(result["complete"]["pending_token"], EOS)

    def test_512_outputs_mean_511_tg_forwards_not_requested_or_draft_rate(self):
        rows = fixture(outputs=[100] * 512)
        rows[0]["config"]["capacity"] = 520
        result = self.collect(rows)
        self.assertEqual(result["counts"]["tg_forwards"], 511)
        self.assertEqual(result["counts"]["consumed_tokens"], 516)
        self.assertEqual(result["throughput"]["tg_output_tokens_per_second"], 511000 / 1022.5)
        self.assertEqual(result["counts"]["records"], 514)

    def test_actual_cli_strings_relative_paths_escaping_no_inventory_inference(self):
        rows = fixture(logits='relative "artifact".bin')
        rows[0]["model"] = 'relative/renamed\\model\n.gguf'
        self.assertEqual(self.collect(rows)["model"], rows[0]["model"])
        for path in (("model",), ("config", "logits_path"), ("config", "trace_directory")):
            self.change(0, path, "bad\0argv", rows)
            self.change(0, path, "bad\ud800argv", rows)
        self.change(0, ("model",), "")

    def test_exact_fields_missing_extra_unknown_and_legacy_failure(self):
        for baseline in (fixture(), fixture("greedy_diagnostic")):
            for i, row in enumerate(baseline):
                for field in row:
                    rows = copy.deepcopy(baseline)
                    del rows[i][field]
                    with self.subTest(kind=row["kind"], missing=field):
                        self.reject(rows)
                self.change(i, ("unknown",), True, baseline)
                self.change(i, ("protocol",), True, baseline)
            for key in ("config", "sampling"):
                for field in baseline[0][key]:
                    rows = copy.deepcopy(baseline)
                    del rows[0][key][field]
                    self.reject(rows)
                self.change(0, (key, "unknown"), 0, baseline)
        for key in ("model_bytes", "argv", "build_provenance_complete", "peak_vram", "checksum"):
            self.change(0, (key,), 0)
        for kind in ("session_source", "session_complete", "session_failure", "session_request_failure"):
            self.change(0, ("kind",), kind)
            self.change(-1, ("kind",), kind)
        rows = fixture()
        rows.insert(-1, {"kind": "session_request_failure", "passed": False})
        self.reject(rows)

    def test_source_provenance_nonmtp_fresh_scopes_and_sampling_series(self):
        for path, bad in (
            (("revision",), "2e9848d"), (("revision",), "g" * 40), (("revision",), None),
            (("dirty",), 1), (("dirty",), "true"), (("runtime",), "llama_decode"), (("kv",), "F16"),
            (("candidate",), False), (("mtp",), True), (("mtp_acceptance",), 0),
            (("session_start",), "reset"), (("prefix_reuse",), True), (("expert_cache_warmness",), "warm"),
            (("series",), "custom_sampling"), (("timing_scope",), "GPU_events"),
            (("pp_scope",), "host_enqueue"), (("tg_scope",), "all_requested_outputs"),
            (("sampling", "rng"), "baseline"), (("sampling", "baseline_rng_equivalent"), True),
        ):
            self.change(0, path, bad)
        for path in (("candidate",), ("mtp",), ("mtp_acceptance",), ("passed",), ("scope",)):
            self.change(-1, path, 1)
        rows = fixture()
        rows[0]["sampling"]["top_k"] = 0
        self.reject(rows)  # Changed filters must be labelled custom.
        rows = fixture("greedy_diagnostic")
        self.change(0, ("sampling", "seed"), 0, rows)
        self.change(0, ("series",), "primary_sampling", rows)

    def test_cli_config_shapes_types_trace_consistency_and_capacity(self):
        for field, values in {
            "capacity": (0, 3, 6, 131076, 32., True), "expert_slots": (0, 513, 112., True),
            "max_batch_tokens": (0, 1025, 2., True), "requested_output_tokens": (-1, 131073, 3., True),
            "ignore_eos": (0, "false"), "trace": (0, True), "diagnostic_rows": (0, True),
            "trace_directory": (False, []), "logits_path": (False, []),
        }.items():
            for bad in values:
                self.change(0, ("config", field), bad)
        self.change(0, ("config", "capacity"), 4)
        self.change(0, ("config", "requested_output_tokens"), 29)  # 5+28 exceeds32.
        rows = fixture(chunk=1, trace="trace-dir")
        self.change(0, ("config", "trace"), False, rows)
        self.change(0, ("config", "max_batch_tokens"), 2, rows)
        self.change(0, ("config", "diagnostic_rows"), False, rows)
        self.change(0, ("config", "diagnostic_rows"), False, fixture("greedy_diagnostic"))

    def test_capacity_slots_chunks_and_vocabulary_valid_boundaries(self):
        for slots, chunk, capacity in ((1, 1, 8), (512, 1024, 131072)):
            rows = fixture(chunk=chunk)
            rows[0]["config"].update(expert_slots=slots, capacity=capacity)
            rows[0]["prompt_ids"][1:3] = [0, VOCAB - 1]
            self.assertIs(self.collect(rows)["passed"], True)
        rows = fixture(outputs=[100], chunk=1024)
        rows[0]["config"]["capacity"] = 131072
        rows[0]["prompt_ids"] = [0] * 131072
        rows[-1].update(input_tokens=131072, consumed_tokens=131072, pp_calls=128,
                        expert_hits=131072 * 480, expert_misses=0, expert_upload_bytes=0)
        self.assertIs(self.collect(rows)["passed"], True)

    def test_sampling_seed_numeric_filters_bounds_and_labels(self):
        for field, bads in {
            "seed": (-1, 1 << 64, 42., True, "42"),
            "temperature": (0, -1, float("nan"), float("inf"), True, "1"),
            "top_p": (0, -1, math.nextafter(1., math.inf), float("nan"), True, ".95"),
            "top_k": (-1, VOCAB + 1, 20., True), "mode": ("greedy", True),
        }.items():
            for bad in bads:
                self.change(0, ("sampling", field), bad)
        rows = fixture(outputs=[], requested=0)
        self.reject(rows)

    def test_id_shapes_counts_ranges_and_nonboolean_integers(self):
        for bad in ([], "248044", None, [False], [-1], [VOCAB], [100.], [0] * 33):
            self.change(0, ("prompt_ids",), bad)
        for bad in ([101, 102], [101, 102, 103, 104], [101, 102, True], [101, 102, VOCAB], "101,102,103"):
            self.change(-1, ("generated_ids",), bad)
        self.change(1, ("index",), 0.)
        self.change(1, ("token",), True)
        for key in ("input_tokens", "output_tokens", "consumed_tokens", "pp_calls", "tg_forwards", "random_draws"):
            self.change(-1, (key,), float(self.rows[-1][key]))
            self.change(-1, (key,), True)
            self.change(-1, (key,), self.rows[-1][key] + 1)
        self.change(-1, ("output_tokens",), -1)

    def test_stop_eos_ignore_limit_generated_ids_and_pending_consistency(self):
        for path, bad in ((("pending_token",), None), (("pending_token",), 102),
                          (("pending_token",), 103.), (("stop_reason",), "eos"),
                          (("stop_reason",), "prompt_only")):
            self.change(-1, path, bad)
        self.reject(fixture(outputs=[101], requested=3))
        self.reject(fixture(outputs=[EOS, 101, 102]))
        self.reject(fixture(outputs=[EOS], requested=3, ignore=True))
        self.reject(fixture(outputs=[101, 102, 103], requested=2))
        self.change(-1, ("stop_reason",), "output_limit", fixture(outputs=[EOS], requested=3))
        prompt_only = fixture("greedy_diagnostic", outputs=[])
        for path, bad in ((("pending_token",), 100), (("first_output_ms",), 0),
                          (("first_output_after_pp_ms",), .1), (("random_draws",), 1),
                          (("stop_reason",), "output_limit")):
            self.change(-1, path, bad, prompt_only)

    def test_record_order_no_missing_extra_duplicate_or_post_complete(self):
        for baseline in (fixture(), fixture("greedy_diagnostic")):
            for i in range(len(baseline)):
                self.reject(baseline[:i] + baseline[i + 1:])
                self.reject(baseline[:i] + [baseline[i]] + baseline[i:])
            for i in range(len(baseline) - 1):
                rows = copy.deepcopy(baseline)
                rows[i], rows[i + 1] = rows[i + 1], rows[i]
                self.reject(rows)
        for index in (0, 1, 2):
            self.change(index + 1, ("index",), index + 1)
            self.change(index + 1, ("token",), 99)

    def test_pp_exact_chunk_segmentation_tg_forward_pending_and_greedy_argmax(self):
        rows = fixture("greedy_diagnostic")
        for i, row in enumerate(rows):
            if row["kind"] == "session_request_row":
                for field, bad in (("position", row["position"] + 1), ("token", 99),
                                   ("finite", False), ("argmax", VOCAB), ("argmax", True),
                                   ("phase", "tg" if row["phase"] == "pp" else "pp")):
                    self.change(i, (field,), bad, rows)
            if row["kind"] == "session_request_window":
                for field, bad in (("first_position", row["first_position"] + 1), ("rows", row["rows"] + 1),
                                   ("rows", float(row["rows"])), ("phase", "invalid")):
                    self.change(i, (field,), bad, rows)
        final_prompt = next(i for i, r in enumerate(rows) if r["kind"] == "session_request_row" and r["position"] == 4)
        self.change(final_prompt, ("argmax",), 99, rows)
        first_tg = next(i for i, r in enumerate(rows) if r["kind"] == "session_request_row" and r["phase"] == "tg")
        self.change(first_tg, ("token",), 102, rows)  # Must forward output0, not newly drawn output1.
        self.change(0, ("config", "max_batch_tokens"), 3, rows)

    def test_diagnostics_required_iff_driver_enabled_and_no_extra_timing_bits(self):
        diagnostic = fixture("greedy_diagnostic")
        self.reject([r for r in diagnostic if r["kind"] not in ("session_request_row", "session_request_window")])
        rows = fixture()
        rows.insert(1, diagnostic[1])
        self.reject(rows)
        window = next(i for i, r in enumerate(diagnostic) if r["kind"] == "session_request_window")
        self.change(window, ("completed_ms_bits",), 0, diagnostic)
        self.change(-1, ("pp_ms_bits",), 0)

    def test_expert_totals_deltas_upload_bounds_counter_types_and_final_window(self):
        for key in COUNTERS:
            for bad in (-1, True, float(self.rows[-1][key]), 1 << 64):
                self.change(-1, (key,), bad)
        self.change(-1, ("expert_hits",), self.rows[-1]["expert_hits"] - 1)
        self.change(-1, ("expert_upload_bytes",), self.rows[-1]["expert_misses"] * Q40 - 1)
        self.change(-1, ("expert_upload_bytes",), self.rows[-1]["expert_misses"] * Q41 + 1)
        for q in (Q40, Q41):
            rows = fixture()
            rows[-1]["expert_upload_bytes"] = rows[-1]["expert_misses"] * q
            self.assertIs(self.collect(rows)["passed"], True)
        rows = fixture()
        rows[-1].update(expert_hits=480 * 7, expert_misses=0, expert_upload_bytes=0)
        self.assertIs(self.collect(rows)["passed"], True)
        self.change(-1, ("expert_upload_bytes",), 1, rows)
        diagnostic = fixture("greedy_diagnostic")
        windows = [i for i, r in enumerate(diagnostic) if r["kind"] == "session_request_window"]
        for i in windows:
            self.change(i, ("expert_hits",), diagnostic[i]["expert_hits"] + 1, diagnostic)
            self.change(i, ("expert_upload_bytes",), diagnostic[i]["expert_upload_bytes"] + Q41 * 481, diagnostic)
            self.change(i, ("expert_misses",), 0, diagnostic)
        rows = copy.deepcopy(diagnostic)
        second = rows[windows[1]]
        second.update(expert_hits=480 * 4 - 1, expert_misses=1, expert_upload_bytes=Q40)
        self.reject(rows)  # Aggregate is valid, but misses/uploads decreased.
        self.change(-1, ("expert_upload_bytes",), diagnostic[-1]["expert_upload_bytes"] + 1, diagnostic)

    def test_timings_finite_nonboolean_positive_used_intervals_and_chronology(self):
        for key in ("load_ms", "pp_ms", "tg_ms", "total_ms", "first_output_ms", "first_output_after_pp_ms"):
            for bad in (-1, float("inf"), float("nan"), True, None, "1"):
                self.change(-1, (key,), bad)
        for key in ("pp_ms", "tg_ms", "total_ms"):
            self.change(-1, (key,), 0)
        self.change(-1, ("first_output_ms",), 6.6)  # pp6.5 + first-sampling.25 cannot fit.
        self.change(-1, ("total_ms",), 11.)  # first7 + completed TG4.5 cannot fit.
        self.change(-1, ("first_output_after_pp_ms",), 1.)
        for baseline in (fixture(outputs=[101]), fixture("greedy_diagnostic", outputs=[])):
            self.change(-1, ("tg_ms",), .1, baseline)
        rows = fixture("greedy_diagnostic")
        for i, row in enumerate(rows):
            if row["kind"] == "session_request_window":
                self.change(i, ("completed_ms",), -1, rows)
                self.change(i, ("completed_ms",), True, rows)
                self.change(i, ("completed_ms",), 20., rows)
        self.change(-1, ("pp_ms",), 5., rows)
        self.change(-1, ("tg_ms",), 3., rows)

    def test_no_arbitrary_speed_gate_load_separate_and_flush_overhead_not_event_sum(self):
        for factor in (1e-9, 1e15):
            rows = fixture("greedy_diagnostic")
            for row in rows:
                for key in ("completed_ms", "pp_ms", "tg_ms", "total_ms", "first_output_ms", "first_output_after_pp_ms"):
                    if key in row:
                        row[key] *= factor
            rows[-1]["load_ms"] = 1e20
            result = self.collect(rows)
            self.assertIs(result["passed"], True)
        rows = fixture()
        rows[-1]["total_ms"] = 99.
        self.assertEqual(self.collect(rows)["throughput"]["request_output_tokens_per_second"], 3000 / 99.)
        rows = fixture()
        rows[-1]["load_ms"] = 0.
        self.assertIs(self.collect(rows)["passed"], True)
        rows = fixture()
        rows[-1].update(pp_ms=1e-320, first_output_after_pp_ms=0, first_output_ms=1e-320)
        self.reject(rows)  # Nonfinite derived rate is not serializable evidence.

    def test_json_duplicates_nonfinite_utf8_final_newline_blank_and_nonobjects(self):
        diagnostic = encoded(fixture("greedy_diagnostic"))
        for field in ("revision", "capacity", "mode", "phase", "position", "expert_hits", "passed"):
            key = ('"' + field + '":').encode()
            self.raw.write_bytes(diagnostic.replace(key, key + b"0," + key, 1))
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                self.collect()
        for bad in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            self.raw.write_bytes(self.original.replace(b'"pp_ms":6.5', b'"pp_ms":' + bad))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.collect()
        for bad in (b"", self.original[:-1], self.original[:-20], self.original + b"\n",
                    b"\xef\xbb\xbf" + self.original, self.original.replace(b"session_request_source", b"\xff", 1),
                    b"[]\n" + self.original, b"null\n" + self.original, b"\n" + self.original,
                    self.original + b"core-session: failure\n"):
            self.raw.write_bytes(bad)
            with self.assertRaises(ValueError):
                self.collect()
        self.raw.write_bytes(self.original.replace(b"\n", b"\r\n"))
        self.assertIs(self.collect()["passed"], True)

    def test_raw_read_bound_before_open_and_during_read_regular_file(self):
        self.assertEqual(MODULE.MAX_RAW_BYTES, 16 * 1024 * 1024)
        self.raw.write_bytes(b" " * (MODULE.MAX_RAW_BYTES + 1))
        with mock.patch.object(Path, "open") as opened:
            with self.assertRaisesRegex(ValueError, "oversized"):
                self.collect()
            opened.assert_not_called()
        self.raw.write_bytes(self.original)
        with mock.patch.object(Path, "open", return_value=mock.MagicMock()) as opened:
            stream = opened.return_value.__enter__.return_value
            stream.read.return_value = b"x" * (MODULE.MAX_RAW_BYTES + 1)
            with mock.patch.object(MODULE.os, "fstat", return_value=self.raw.stat()):
                with self.assertRaisesRegex(ValueError, "oversized"):
                    self.collect()
            stream.read.assert_called_once_with(MODULE.MAX_RAW_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "regular file"):
            MODULE.collect(self.root)
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "regular file"):
            MODULE.collect(fifo)

    def test_cli_append_one_compact_record_entire_history_unchanged(self):
        history = b'{ "kind": "old_protocol", "values":[1,2,3] }\n{"kind":"second","dirty":1}\n'
        self.results.write_bytes(history)
        process = self.cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "")
        result = self.results.read_bytes()
        self.assertTrue(result.startswith(history))
        self.assertEqual(len(result.splitlines()), 3)
        self.assertEqual([json.loads(line) for line in result.splitlines()[:2]],
                         [json.loads(line) for line in history.splitlines()])
        record = json.loads(result.splitlines()[-1])
        self.assertEqual(record["source"], self.rows[0])
        self.assertEqual(record["complete"], self.rows[-1])
        self.assertEqual(self.raw.read_bytes(), self.original)
        self.assertIs(MODULE.append_record, record_memory.append_result)
        with mock.patch.object(MODULE, "append_record") as append:
            self.assertEqual(MODULE.main(["--raw", str(self.raw), "--results", str(self.results)]), 0)
            append.assert_called_once()
        self.assertEqual(self.cli(Path("explicit-relative.jsonl")).returncode, 0)
        self.results.unlink()
        self.assertEqual(self.cli().returncode, 0)
        self.assertEqual(len(self.results.read_bytes().splitlines()), 1)

    def test_cli_validation_and_missing_results_never_touch_journal(self):
        history = b'{"kind":"previous"}\n'
        self.results.write_bytes(history)
        self.raw.write_bytes(self.original[:-1])
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)
        self.results.unlink()
        self.assertEqual(self.cli().returncode, 1)
        self.assertFalse(self.results.exists())
        self.raw.write_bytes(self.original)
        process = subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", str(self.raw)],
                                 cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 2)
        self.assertFalse(self.results.exists())
        with mock.patch.object(MODULE, "collect", side_effect=OSError("injected input failure")):
            with mock.patch("sys.stderr", new_callable=io.StringIO) as errors:
                self.assertEqual(MODULE.main(["--raw", str(self.raw), "--results", str(self.results)]), 1)
                self.assertIn("record_request: injected input failure", errors.getvalue())

    def test_raw_model_logits_alias_symlink_hardlink_and_trace_directory_protected(self):
        model, logits = self.root / "model.gguf", self.root / "logits.bin"
        model.write_bytes(b"synthetic only, never inspected")
        logits.write_bytes(b"synthetic binary capture")
        rows = fixture(logits=str(logits))
        rows[0]["model"] = str(model)
        self.collect(rows)
        for artifact in (self.raw, model, logits):
            before = artifact.read_bytes()
            symlink, hardlink = self.root / (artifact.name + ".sym"), self.root / (artifact.name + ".hard")
            symlink.symlink_to(artifact)
            os.link(artifact, hardlink)
            for destination in (artifact, symlink, hardlink):
                self.assertEqual(self.cli(destination).returncode, 1)
                self.assertEqual(artifact.read_bytes(), before)
        trace = self.root / "trace"
        trace.mkdir()
        self.collect(fixture(chunk=1, trace=str(trace)))
        destination = trace / "results.jsonl"
        self.assertEqual(self.cli(destination).returncode, 1)
        self.assertFalse(destination.exists())
        alias = self.root / "trace-alias"
        alias.symlink_to(trace, target_is_directory=True)
        self.assertEqual(self.cli(alias / "results.jsonl").returncode, 1)

    def test_incomplete_results_nonregular_missing_parent_and_fullhistory_preservation(self):
        history = b'{"kind":"first"}\n{"kind":"unfinished"}'
        self.results.write_bytes(history)
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)
        self.assertEqual(self.cli(self.root).returncode, 1)
        self.assertEqual(self.cli(self.root / "missing" / "results.jsonl").returncode, 1)
        fifo = self.root / "journal-fifo"
        os.mkfifo(fifo)
        self.assertEqual(self.cli(fifo).returncode, 1)

    def test_shared_lock_serialization_failure_and_short_write_io_rollback(self):
        record = self.collect()
        history = b'{"kind":"first"}\n{"kind":"second","proof":[1,2]}\n'
        self.results.write_bytes(history)
        with mock.patch.object(record_memory.fcntl, "flock", wraps=record_memory.fcntl.flock) as lock:
            MODULE.append_result(self.results, record)
            self.assertEqual(lock.call_args.args[1], record_memory.fcntl.LOCK_EX)
        with mock.patch.object(record_memory.json, "dumps", side_effect=ValueError("serialization failure")):
            with mock.patch.object(Path, "open") as opened:
                with self.assertRaisesRegex(ValueError, "serialization failure"):
                    MODULE.append_result(self.results, record)
                opened.assert_not_called()
        real_open = Path.open
        for failure in ("short", "write", "flush"):
            self.results.write_bytes(history)

            def faulty_open(path, *args, **kwargs):
                stream = real_open(path, *args, **kwargs)
                proxy = mock.MagicMock(wraps=stream)
                proxy.__enter__.return_value = proxy
                proxy.__exit__.side_effect = lambda *_: stream.close()

                def write(value):
                    if failure == "flush":
                        return stream.write(value)
                    stream.write(value[:17])
                    if failure == "write":
                        raise OSError("injected write failure")
                    return 17

                proxy.write.side_effect = write
                if failure == "flush":
                    proxy.flush.side_effect = OSError("injected flush failure")
                return proxy

            with mock.patch.object(Path, "open", autospec=True, side_effect=faulty_open):
                with self.assertRaises((ValueError, OSError)):
                    MODULE.append_result(self.results, record)
            self.assertEqual(self.results.read_bytes(), history)

    def test_alias_check_repeated_after_locked_open(self):
        record = self.collect()
        self.results.write_bytes(b'{"kind":"old"}\n')
        before = self.raw.read_bytes()
        real_open = Path.open

        def replace_before_open(path, *args, **kwargs):
            self.results.unlink()
            os.link(self.raw, self.results)
            return real_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", autospec=True, side_effect=replace_before_open):
            with self.assertRaisesRegex(ValueError, "input artifact"):
                MODULE.append_result(self.results, record)
        self.assertEqual(self.raw.read_bytes(), before)

    def test_concurrent_cli_append_records_under_shared_lock(self):
        history = b'{"kind":"first"}\n{"kind":"second"}\n'
        self.results.write_bytes(history)
        command = [sys.executable, "-B", str(SCRIPT), "--raw", str(self.raw), "--results", str(self.results)]
        processes = [subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True) for _ in range(2)]
        try:
            for process in processes:
                _, errors = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, errors)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        result = self.results.read_bytes()
        self.assertTrue(result.startswith(history))
        self.assertEqual(len(result.splitlines()), 4)
        for line in result.splitlines()[2:]:
            self.assertEqual(json.loads(line)["complete"], self.rows[-1])


class VramRequestResultsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="request-vram-results-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw, self.vram, self.results = (self.root / name for name in
                                            ("request.jsonl", "vram.jsonl", "results.jsonl"))
        self.rows = fixture()
        self.observation = vram_fixture(self.rows)
        self.raw.write_bytes(encoded(self.rows))
        self.vram.write_bytes(encoded(self.observation))

    def collect(self, observation=None, request=None):
        if request is not None:
            self.raw.write_bytes(encoded(request))
        if observation is not None:
            self.vram.write_bytes(encoded(observation))
        return MODULE.collect(self.raw, self.vram)

    def reject(self, observation):
        with self.assertRaises(ValueError):
            self.collect(observation)

    def change(self, index, path, bad):
        observation = copy.deepcopy(self.observation)
        target = observation[index]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad
        with self.subTest(index=index, path=path, bad=bad):
            self.reject(observation)

    def reject_command(self, command):
        self.change(0, ("command_argv",), command)

    def cli(self, destination=None, vram=None):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", str(self.raw),
                               "--vram-log", str(self.vram if vram is None else vram), "--results",
                               str(self.results if destination is None else destination)],
                              cwd=self.root, capture_output=True, text=True, timeout=30)

    def test_optional_object_preserves_all_observer_metadata_and_native_record(self):
        before = self.raw.read_bytes(), self.vram.read_bytes()
        plain = MODULE.collect(self.raw)
        result = self.collect()
        observation = result.pop("vram_observation")
        self.assertEqual(observation, {
            "scope": "sampled_global_driver_VRAM_not_exact_instantaneous_peak",
            "mapping_attestation": "caller_supplied_labels_no_HIP_index_attestation",
            "command_join_scope": "direct_Docker_core_session_structured_argv_matches_request_source_only_no_binary_or_mount_identity_attestation",
            "source": self.observation[0], "complete": self.observation[1], "passed": True,
        })
        self.assertEqual(result["raw_logs"].pop("vram"), str(self.vram.resolve()))
        result.pop("timestamp")
        plain.pop("timestamp")
        self.assertEqual(result, plain)
        self.assertEqual((self.raw.read_bytes(), self.vram.read_bytes()), before)
        for key in ("peak_vram_qualification_claim", "performance_claim", "speedup_claim", "mtp"):
            self.assertIs(result[key], False)
        self.assertNotIn("device", observation["source"]["devices"][0])
        self.assertEqual(observation["source"]["command_argv"], self.observation[0]["command_argv"])

    def test_without_vram_exact_compatibility_and_explicit_none(self):
        self.vram.write_bytes(b"bad observer log, deliberately ignored")
        with mock.patch.object(MODULE, "vram_records") as read:
            plain, explicit = MODULE.collect(self.raw), MODULE.collect(self.raw, None)
            read.assert_not_called()
        plain.pop("timestamp")
        explicit.pop("timestamp")
        self.assertEqual(plain, explicit)
        self.assertNotIn("vram_observation", plain)
        self.assertEqual(plain["raw_logs"], {"request": str(self.raw.resolve())})
        process = subprocess.run([sys.executable, "-B", str(SCRIPT), "--raw", str(self.raw),
                                  "--results", str(self.results)], cwd=self.root,
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        saved = json.loads(self.results.read_bytes())
        saved.pop("timestamp")
        self.assertEqual(saved, plain)

    def test_join_accepts_diagnostics_prompt_only_eos_and_exact_custom_sampling(self):
        cases = (fixture("greedy_diagnostic"), fixture("greedy_diagnostic", outputs=[]),
                 fixture(outputs=[EOS], requested=3), fixture(ignore=True),
                 fixture(chunk=1, trace='trace "dir"'), fixture(logits="logits\\capture\n.bin"))
        for request in cases:
            with self.subTest(config=request[0]["config"]):
                self.assertIs(self.collect(vram_fixture(request), request)["vram_observation"]["passed"], True)
        for seed in (0, (1 << 64) - 1):
            request = fixture()
            request[0]["series"] = "custom_sampling"
            request[0]["sampling"].update(seed=seed, temperature=.7, top_p=1, top_k=0)
            self.assertIs(self.collect(vram_fixture(request), request)["passed"], True)

    def test_native_defaults_typed_numbers_and_options_among_positionals(self):
        request = fixture(chunk=1)
        request[0]["config"]["capacity"] = 4096
        request[0]["sampling"]["temperature"] = 1  # Native prints binary64 1 as JSON integer.
        observation = vram_fixture(request)
        argv = observation[0]["command_argv"]
        start = argv.index("--capacity")
        argv[start:] = [request[0]["model"], "0248044", "--generate", "03", "100", "--sample",
                        "--seed", "00042", "101", "--temperature", "1.e+0", "--top-p", ".95e0",
                        "102", "103"]
        self.assertIs(self.collect(observation, request)["passed"], True)
        # Omitted default filters, slots and capacity still join by their typed values.
        argv[start:] = ["--sample", "--seed", "42", "--generate", "3", request[0]["model"],
                        *map(str, request[0]["prompt_ids"])]
        self.assertIs(self.collect(observation, request)["passed"], True)
        request = fixture("greedy_diagnostic", outputs=[], chunk=1)
        request[0]["config"]["capacity"] = 4096
        observation = vram_fixture(request)
        observation[0]["command_argv"][start:] = ["--prefill-chunk", "1", request[0]["model"],
                                                 *map(str, request[0]["prompt_ids"])]
        self.assertIs(self.collect(observation, request)["passed"], True)
        del observation[0]["command_argv"][start:start + 2]
        self.reject(observation)  # Same defaults, but the legacy CLI path is not protocol 1.

    def test_exact_closed_source_footer_and_device_schemas(self):
        for index, record in enumerate(self.observation):
            for key in record:
                observation = copy.deepcopy(self.observation)
                del observation[index][key]
                with self.subTest(index=index, missing=key):
                    self.reject(observation)
            for key in ("unknown", "passed", "peak_vram", "HIP_index_attestation"):
                self.change(index, (key,), True)
            for i, device in enumerate(record["devices"]):
                for key in device:
                    observation = copy.deepcopy(self.observation)
                    del observation[index]["devices"][i][key]
                    self.reject(observation)
                self.change(index, ("devices", i, "unknown"), 0)
        for index, key, bad in (
            (0, "protocol", True), (1, "protocol", True), (0, "scope", "exact_peak"),
            (1, "scope", "per_phase_peak"), (0, "mapping_attestation", "HIP_index_verified"),
            (0, "value_source", "HIP_owned"), (0, "sampling_window", "TG_only"),
            (0, "elapsed_scope", "GPU_events"),
        ):
            self.change(index, (key,), bad)

    def test_rejects_failure_nonzero_children_observer_errors_and_incomplete_sampling(self):
        for field, bads in {
            "kind": ("vram_failure", "session_request_complete"), "observation_complete": (False, 1),
            "child_started": (False, 1), "child_returncode": (1, -9, None, False, 0.),
            "observer_returncode": (1, -1, None, False, 0.), "post_wait_sampled": (False, 1),
            "observer_error": ({"stage": "sample", "type": "OSError", "message": "synthetic"}, "", False),
            "sample_rounds": (0, 1, 2, True, 130., -1, 1 << 64),
        }.items():
            for bad in bads:
                self.change(1, (field,), bad)
        for index in (0, 1):
            for bad in ([], self.observation[index]["devices"][:1],
                        self.observation[index]["devices"] * 2, {}, None):
                self.change(index, ("devices",), bad)
            self.change(index, ("devices",), list(reversed(self.observation[index]["devices"])))
        for i in range(2):
            for bad in (129, 131, True, 130., 1 << 64):
                self.change(1, ("devices", i, "sample_count"), bad)

    def test_device_totals_extrema_uint64_bounds_and_paths_not_mapping_attestation(self):
        for i in range(2):
            for index in (0, 1):
                for bad in (0, -1, True, 17163091968., 1 << 64):
                    self.change(index, ("devices", i, "total_bytes"), bad)
                self.change(index, ("devices", i, "label"), f"device_{1 - i}")
            for bad in (-1, True, 12000000000., 17163091969, 1 << 64):
                self.change(1, ("devices", i, "observed_max_used_bytes"), bad)
            for bad in (-1, True, 5163091968., 0, 17163091969):
                self.change(1, ("devices", i, "observed_min_free_bytes"), bad)
            for bad in ("", "bad\0path", "bad\ud800path", False):
                self.change(0, ("devices", i, "supplied_path"), bad)
            for bad in ("relative/device", "/sys/../device", "/sys/./device", "/sys//device",
                        "//sys/device", "/sys/device/", "bad\0path", False):
                self.change(0, ("devices", i, "resolved_path"), bad)
            for key in ("used_file", "total_file"):
                self.change(0, ("devices", i, key), self.observation[0]["devices"][1 - i][key])
        observation = copy.deepcopy(self.observation)
        for key in ("resolved_path", "used_file", "total_file"):
            observation[0]["devices"][1][key] = observation[0]["devices"][0][key]
        self.reject(observation)
        for total in (1, (1 << 64) - 1):
            for used in (0, total):
                observation = copy.deepcopy(self.observation)
                for index in (0, 1):
                    for device in observation[index]["devices"]:
                        device["total_bytes"] = total
                for device in observation[1]["devices"]:
                    device.update(observed_max_used_bytes=used, observed_min_free_bytes=total - used,
                                  sample_count=3)
                observation[1]["sample_rounds"] = 3
                # No collector-side sysfs lookup, hardcoded BDF or real-capacity assertion.
                for i, device in enumerate(observation[0]["devices"]):
                    path = f"/synthetic/remote/device_{i}"
                    device.update(resolved_path=path, used_file=path + "/mem_info_vram_used",
                                  total_file=path + "/mem_info_vram_total")
                self.assertIs(self.collect(observation)["passed"], True)

    def test_elapsed_conservative_load_plus_request_lower_bound_no_speed_or_phase_gate(self):
        lower = (self.rows[-1]["load_ms"] + self.rows[-1]["total_ms"]) / 1000
        for elapsed in (lower, math.nextafter(lower, -math.inf), 1e100):
            observation = copy.deepcopy(self.observation)
            observation[1]["completed_child_elapsed_seconds"] = elapsed
            result = self.collect(observation)
            self.assertEqual(result["timings_ms"]["total_ms"], self.rows[-1]["total_ms"])
        for bad in (0, self.rows[-1]["total_ms"] / 1000, lower - .001, -1, True, None, "15",
                    float("nan"), float("inf")):
            self.change(1, ("completed_child_elapsed_seconds",), bad)
        for interval in (.001, 10):
            observation = copy.deepcopy(self.observation)
            observation[0]["interval_seconds"] = interval
            self.assertIs(self.collect(observation)["passed"], True)
        for bad in (0, .0009, 10.001, -1, True, None, ".1", float("nan"), float("inf")):
            self.change(0, ("interval_seconds",), bad)

    def test_join_rejects_changed_model_prompt_config_sampling_and_output_budget(self):
        baseline = self.observation[0]["command_argv"]
        for flag, bad in (("--capacity", "64"), ("--slots", "111"), ("--prefill-chunk", "1"),
                          ("--generate", "4"), ("--seed", "43"), ("--temperature", ".7"),
                          ("--top-p", ".8"), ("--top-k", "0")):
            argv = list(baseline)
            argv[argv.index(flag) + 1] = bad
            self.reject_command(argv)
        argv = list(baseline)
        argv[argv.index(self.rows[0]["model"])] = "/models/different.gguf"
        self.reject_command(argv)
        for tail in (["248044", "100", "101", "102", "104"], ["248044", "100"],
                     ["248044", "100", "101", "102", "103", "104"], ["248044", "101", "100", "102", "103"]):
            self.reject_command(baseline[:-5] + tail)
        for extra in (["--ignore-eos"], ["--trace", "trace-dir"], ["--logits", "capture.bin"]):
            self.reject_command(baseline + extra)
        argv = list(baseline)
        argv.remove("--sample")
        self.reject_command(argv)
        argv = list(baseline)
        first = argv.index("--seed")
        del argv[first:first + 2]
        self.reject_command(argv)

    def test_native_closed_grammar_duplicates_unknown_ambiguous_and_numeric_values(self):
        baseline = self.observation[0]["command_argv"]
        for flag in ("--capacity", "--slots", "--prefill-chunk", "--generate", "--seed",
                     "--temperature", "--top-p", "--top-k"):
            value = baseline[baseline.index(flag) + 1]
            self.reject_command(baseline + [flag, value])
            self.reject_command(baseline + [flag])
        self.reject_command(baseline + ["--sample"])
        self.reject_command(baseline + ["--ignore-eos", "--ignore-eos"])
        for flag in ("--help", "--unknown", "--", "--seed=42", "--capacity=32"):
            self.reject_command(baseline + [flag])
        for flag, bads in {
            "--capacity": ("+32", " 32", "32 ", "32.0", "0x20", "2147483648", "-2147483649"),
            "--seed": ("-0", "-42", "+42", "42.0", "42x", str(1 << 64), "9" * 5000),
            "--temperature": ("+1", " 1", "1_0", "NaN", "Infinity", "1e999", "0x1p0", "1x"),
            "--top-p": (".95x", "0", "-0.95"), "--top-k": ("20.0", "+20", "true"),
        }.items():
            for bad in bads:
                argv = list(baseline)
                argv[argv.index(flag) + 1] = bad
                self.reject_command(argv)
        for bad in ("100.0", "+100", " 100", "-1", str(VOCAB), "$(seq 100 103)"):
            argv = list(baseline)
            argv[-4] = bad
            self.reject_command(argv)
        for flag in ("--trace", "--logits"):
            self.reject_command(baseline + [flag, ""])
            self.reject_command(baseline + [flag, "x", flag, "x"])

    def test_docker_prefix_closed_attached_shape_no_shell_detachment_or_overrides(self):
        baseline = self.observation[0]["command_argv"]
        for bad in (None, {}, "docker run ...", [], [True], ["docker", "run", None],
                    ["docker", "run", "bad\0arg"], ["docker", "run", "bad\ud800arg"]):
            self.reject_command(bad)
        for argv in (["bash", "-c", " ".join(baseline)], ["sudo", *baseline], ["env", *baseline],
                     ["sh", "-c", "exec docker run"], [*baseline[:2], "-d", *baseline[2:]],
                     [*baseline[:2], "--detach", *baseline[2:]],
                     [*baseline[:2], "-e", "HIP_VISIBLE_DEVICES=1,0", *baseline[2:]],
                     [*baseline[:2], "--rm", *baseline[2:]]):
            self.reject_command(argv)
        for flag, bad in (("--entrypoint", "/bin/bash"), ("--device", "/dev/other"),
                          ("--group-add", "root"), ("--ipc", "private"),
                          ("--security-opt", "different"), ("--name", "--detach"),
                          ("-v", "/wrong:/different")):
            argv = list(baseline)
            argv[argv.index(flag) + 1] = bad
            self.reject_command(argv)
        for value in ("relative:/core", "/host:/core:ro", "/host:/core:rw,ro", "/host:/models",
                      "/host:/models:rw", "/host:/core:rw:extra"):
            argv = list(baseline)
            argv[argv.index("-v") + 1] = value
            self.reject_command(argv)
        for flag in ("--rm", "--name", "--group-add", "--ipc", "--security-opt", "--entrypoint", "--device", "-v"):
            argv = list(baseline)
            first = argv.index(flag)
            del argv[first:first + (1 if flag == "--rm" else 2)]
            if flag == "--name":
                self.assertIs(self.collect([{**self.observation[0], "command_argv": argv}, self.observation[1]])["passed"], True)
            else:
                self.reject_command(argv)
        image = baseline.index("llama.cpp-gfx906:cmake-4.4.3")
        self.reject_command(baseline[:image] + ["different:image"] + baseline[image + 1:])
        self.reject_command(baseline[:image])
        self.reject_command(baseline[:image] + ["--entrypoint", "/core/build/core-session"] + baseline[image:])
        self.reject_command(baseline[:image] + ["--device", "/dev/dri"] + baseline[image:])
        self.reject_command(baseline[:image] + ["-v", "/other:/core"] + baseline[image:])

    def test_vram_json_duplicates_nonfinite_utf8_and_finished_exact_two_record_order(self):
        original = encoded(self.observation)
        for field in ("kind", "protocol", "command_argv", "interval_seconds", "label", "total_bytes",
                      "sample_count", "sample_rounds", "observer_error", "child_returncode"):
            key = ('"' + field + '":').encode()
            self.vram.write_bytes(original.replace(key, key + b"0," + key, 1))
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                self.collect()
        for bad in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            self.vram.write_bytes(original.replace(b'"interval_seconds":0.1', b'"interval_seconds":' + bad))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.collect()
        for raw in (b"", original[:-1], original[:-20], original + b"\n", b"\n" + original,
                    b"[]\n" + original, b"null\n" + original, b"\xef\xbb\xbf" + original,
                    original.replace(b"vram_source", b"\xff", 1), original + b"observe_vram: failure\n"):
            self.vram.write_bytes(raw)
            with self.assertRaises(ValueError):
                self.collect()
        for rows in (self.observation[:1], self.observation[1:], list(reversed(self.observation)),
                     self.observation * 2, [self.observation[0], self.observation[0]],
                     [self.observation[1], self.observation[1]]):
            self.reject(rows)

    def test_vram_read_total_per_record_bounds_before_after_open_and_during_read(self):
        self.assertEqual(MODULE.MAX_VRAM_BYTES, 2 * 1024 * 1024)
        self.assertEqual(MODULE.MAX_VRAM_RECORD_BYTES, 1024 * 1024)
        self.vram.write_bytes(b" " * (MODULE.MAX_VRAM_BYTES + 1))
        with mock.patch.object(MODULE.os, "open") as opened:
            with self.assertRaisesRegex(ValueError, "oversized"):
                MODULE.vram_records(self.vram)
            opened.assert_not_called()
        self.vram.write_bytes(encoded(self.observation))
        with mock.patch.object(MODULE.os, "fdopen", return_value=mock.MagicMock()) as opened:
            stream = opened.return_value.__enter__.return_value
            stream.read.return_value = b"x" * (MODULE.MAX_VRAM_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "oversized"):
                MODULE.vram_records(self.vram)
            stream.read.assert_called_once_with(MODULE.MAX_VRAM_BYTES + 1)
        oversized = self.root / "oversized"
        oversized.write_bytes(b" " * (MODULE.MAX_VRAM_BYTES + 1))
        with mock.patch.object(MODULE.os, "fstat", return_value=oversized.stat()):
            with mock.patch.object(MODULE.os, "fdopen") as opened:
                with self.assertRaisesRegex(ValueError, "oversized"):
                    MODULE.vram_records(self.vram)
                opened.assert_not_called()
        lines = encoded(self.observation).splitlines()
        self.vram.write_bytes(lines[0].ljust(MODULE.MAX_VRAM_RECORD_BYTES, b" ") + b"\n" + lines[1] + b"\n")
        with self.assertRaisesRegex(ValueError, "oversized VRAM record"):
            self.collect()
        self.vram.write_bytes(b"".join(line.ljust(MODULE.MAX_VRAM_RECORD_BYTES - 1, b" ") + b"\n" for line in lines))
        self.assertEqual(self.vram.stat().st_size, MODULE.MAX_VRAM_BYTES)
        self.assertIs(self.collect()["passed"], True)

    def test_vram_regular_path_missing_symlink_and_nonblocking_substitution_guard(self):
        for path in (self.root, self.root / "fifo"):
            if path.name == "fifo":
                os.mkfifo(path)
            with self.assertRaisesRegex(ValueError, "regular file"):
                MODULE.collect(self.raw, path)
        with self.assertRaises(OSError):
            MODULE.collect(self.raw, self.root / "missing")
        alias = self.root / "vram-symlink"
        alias.symlink_to(self.vram)
        self.assertEqual(MODULE.collect(self.raw, alias)["raw_logs"]["vram"], str(self.vram.resolve()))
        fifo, real_open = self.root / "fifo", os.open
        with mock.patch.object(MODULE.os, "open", side_effect=lambda _, flags: real_open(fifo, flags)) as opened:
            with self.assertRaisesRegex(ValueError, "regular file"):
                MODULE.vram_records(self.vram)
            self.assertEqual(opened.call_args.args[1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)

    def test_cli_appends_after_both_validate_preserves_history_inputs_and_optional_object(self):
        history = b'{ "kind":"first" }\n{"kind":"second","vram":null}\n'
        self.results.write_bytes(history)
        before = self.raw.read_bytes(), self.vram.read_bytes()
        process = self.cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "")
        saved = self.results.read_bytes()
        self.assertTrue(saved.startswith(history))
        self.assertEqual(len(saved.splitlines()), 3)
        record = json.loads(saved.splitlines()[-1])
        self.assertEqual(record["vram_observation"]["source"], self.observation[0])
        self.assertEqual(record["vram_observation"]["complete"], self.observation[1])
        self.assertEqual(record["raw_logs"], {"request": str(self.raw.resolve()), "vram": str(self.vram.resolve())})
        self.assertEqual((self.raw.read_bytes(), self.vram.read_bytes()), before)
        with mock.patch.object(MODULE, "append_record") as append:
            self.assertEqual(MODULE.main(["--raw", str(self.raw), "--vram-log", str(self.vram),
                                          "--results", str(self.results)]), 0)
            append.assert_called_once()
            self.assertEqual(append.call_args.args[1]["vram_observation"]["complete"], self.observation[1])

    def test_bad_vram_or_native_input_never_appends_or_creates_results(self):
        original = encoded(self.observation)
        failure = copy.deepcopy(self.observation)
        failure[1].update(kind="vram_failure", child_returncode=1, observer_returncode=1)
        command = copy.deepcopy(self.observation)
        argv = command[0]["command_argv"]
        argv[argv.index("--seed") + 1] = "43"
        bads = (original[:-1], encoded(failure), encoded(command), original + b"\n",
                original.replace(b'"observer_error":null', b'"observer_error":{"unknown":true}'))
        history = b'{ "kind":"history" }\n'
        for bad in bads:
            for existing in (False, True):
                self.vram.write_bytes(bad)
                if self.results.exists():
                    self.results.unlink()
                if existing:
                    self.results.write_bytes(history)
                process = self.cli()
                self.assertEqual(process.returncode, 1, process.stderr)
                self.assertEqual(self.raw.read_bytes(), encoded(self.rows))
                self.assertEqual(self.vram.read_bytes(), bad)
                if existing:
                    self.assertEqual(self.results.read_bytes(), history)
                else:
                    self.assertFalse(self.results.exists())
        self.vram.write_bytes(original)
        self.raw.write_bytes(encoded(self.rows)[:-1])
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)
        self.raw.write_bytes(encoded(self.rows))
        self.assertEqual(self.cli(vram=self.root / "missing").returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)

    def test_raw_vram_results_alias_preflight_lexical_symlink_and_hardlink(self):
        history = b'{"kind":"history"}\n'
        self.results.write_bytes(history)
        for artifact in (self.raw, self.vram):
            before = artifact.read_bytes()
            symlink, hardlink = self.root / (artifact.name + ".sym"), self.root / (artifact.name + ".hard")
            symlink.symlink_to(artifact)
            os.link(artifact, hardlink)
            for destination in (artifact, symlink, hardlink):
                process = self.cli(destination=destination)
                self.assertEqual(process.returncode, 1)
                self.assertIn("paths must be distinct", process.stderr)
                self.assertEqual(artifact.read_bytes(), before)
                with mock.patch.object(MODULE, "collect") as collect, mock.patch.object(MODULE, "append_record") as append:
                    with mock.patch("sys.stderr", new_callable=io.StringIO):
                        self.assertEqual(MODULE.main(["--raw", str(self.raw), "--vram-log", str(self.vram),
                                                      "--results", str(destination)]), 1)
                    collect.assert_not_called()
                    append.assert_not_called()
        for vram in (self.raw, self.root / (self.raw.name + ".sym"), self.root / (self.raw.name + ".hard")):
            with mock.patch.object(MODULE, "records") as read, mock.patch.object(MODULE, "vram_records") as observe:
                with self.assertRaisesRegex(ValueError, "paths must be distinct"):
                    MODULE.collect(self.raw, vram)
                read.assert_not_called()
                observe.assert_not_called()
            self.assertEqual(self.cli(vram=vram).returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)

    def test_locked_append_rechecks_vram_alias_and_retains_shared_guards(self):
        record = self.collect()
        self.results.write_bytes(b'{"kind":"old"}\n')
        before = self.vram.read_bytes()
        real_open = Path.open

        def replace_before_open(path, *args, **kwargs):
            self.results.unlink()
            os.link(self.vram, self.results)
            return real_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", autospec=True, side_effect=replace_before_open):
            with self.assertRaisesRegex(ValueError, "input artifact"):
                MODULE.append_result(self.results, record)
        self.assertEqual(self.vram.read_bytes(), before)
        self.assertIs(MODULE.append_record, record_memory.append_result)


if __name__ == "__main__":
    unittest.main()
