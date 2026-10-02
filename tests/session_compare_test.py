"""Independent binary fixtures for the Session/production-oracle parity gate."""

from array import array
from collections import Counter
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "tools/compare_session.py"
SPEC = importlib.util.spec_from_file_location("compare_session", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TMP = Path(tempfile.gettempdir())
REVISION = "dcd685463d597d31f5ca759d32c94592a2740fa4"
VOCAB = 248320
ROW_BYTES = VOCAB * 4
REQUIRED = [("hc_init", -1, 10240), ("l_last-0", 0, 10240), ("l_last-1", 1, 10240),
            ("l_last-3", 3, 10240), ("l_last-47", 47, 10240), ("result_norm", -1, 2560)]


def write_json(path, obj):
    path.write_text(json.dumps(obj, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def f32_bytes(values):
    result = array("f", values)
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def capture(values, ne, nb):
    """Storage-span capture with deliberate NaN stride gaps (not live values)."""
    size = 4 + sum((dim - 1) * stride for dim, stride in zip(ne, nb))
    data = bytearray(struct.pack("<f", math.nan) * (size // 4))
    cursor = iter(values)
    for w in range(ne[3]):
        for z in range(ne[2]):
            for y in range(ne[1]):
                for x in range(ne[0]):
                    offset = w * nb[3] + z * nb[2] + y * nb[1] + x * nb[0]
                    struct.pack_into("<f", data, offset, next(cursor))
    return bytes(data)


class Fixture:
    def __init__(self, root, n=3, optional=True):
        self.root = root
        self.session_trace = root / "session-trace"
        self.oracle_dir = root / "oracle"
        self.session_trace.mkdir()
        self.oracle_dir.mkdir()
        self.session_log = root / "session.jsonl"
        self.session_logits = root / "session.f32.bin"
        self.oracle_logits = self.oracle_dir / "logits.f32.bin"
        first_ids = [7, 11, 248044]
        self.ids = [first_ids[i] if i < 3 else (17 * i) % VOCAB for i in range(n)]
        self.log = [{"kind": "session_source", "revision": "a" * 40, "dirty": 1,
                     "model": "/models/qwen38-keep1-Q4_0.gguf", "capacity": 4096, "expert_slots": 112,
                     "trace": True, "sampling": "greedy_diagnostic", "runtime": "own_48_layer_HIP"}]
        self.tokens = []
        blobs = []
        self.session_tensors = []
        self.oracle_tensors = []
        for i, token in enumerate(self.ids):
            values = array("f", [0.25]) * VOCAB
            values[0], values[1], values[2], values[3] = -3, 0, 5, 10
            values[100 + i], values[200 + i] = 30 + i, 30 + i - 0.005
            blobs.append(f32_bytes(values))
            self.log.append({"kind": "session_token", "position": i, "token": token,
                             "argmax": 100 + i, "completed_ms": 12.5, "expert_hits": i,
                             "expert_misses": 10, "expert_upload_bytes": 1024, "finite": True})
            self.tokens.append({"token_index": i, "position": i, "seq_id": 0, "input_token_id": token,
                                "logits_byte_offset": i * ROW_BYTES, "logits_bytes": ROW_BYTES,
                                "all_finite": True, "argmax_token_id": 100 + i, "max_logit": 30 + i,
                                "callback_captures": 0, "diagnostic_decode_with_callbacks_ms": 20.5})
            for name, layer, count in REQUIRED:
                self.add_tensor(i, name, layer, count)
            if optional:
                for layer in (0, 1, 3, 47):
                    self.add_tensor(i, "hc_attn_mix", layer, 2560, "hc_mixed-" + str(layer))
        self.log.append({"kind": "session_complete", "input_tokens": n, "output_tokens": 0,
                         "consumed_tokens": n, "load_ms": 2.0, "request_ms": 50.0,
                         "scope": "ordered_decode_no_prefill_no_sampling_speed_claim", "passed": True})
        blob = b"".join(blobs)
        self.session_logits.write_bytes(blob)
        self.oracle_logits.write_bytes(blob)
        self.meta = {
            "format": "gfx906-mx-oracle-v1", "status": "complete", "error": "",
            "source_revision": REVISION, "library_revision_attested": REVISION,
            "library_revision_runtime_verified": False, "library_version": "diagnostic",
            "model": "qwen38-keep1-Q4_0.gguf", "model_layers_actual": 48, "vocab_actual": 248320,
            "context": {"capacity_requested": 4096, "capacity_actual": 4096, "batch_actual": 1,
                        "ubatch_actual": 1, "seq_id": 0, "seq_max": 1, "type_k": "q4_0", "type_v": "q4_0"},
            "sampler": None, "mtp": False, "token_ids": self.ids[:], "completed_tokens": n,
            "logits": {"file": "logits.f32.bin", "dtype": "float32", "endianness": "little",
                       "layout": "token-major,vocabulary-minor", "columns": 248320, "header_bytes": 0,
                       "row_bytes": 993280, "bytes_written": n * 993280},
            "callbacks": {"records": "tensors.jsonl", "captures": 0, "tensor_byte_limit": 16777216,
                          "token_byte_limit": 67108864, "allowlist": [], "observed_counts": {}}}
        self.sync_counts()
        self.save()

    def add_tensor(self, i, name, layer, count, oracle_name=None):
        values = array("f", ((j % 23 - 11) / 16 + i / 8 for j in range(count)))
        values[0], values[1], values[2] = 0, 1, 10
        filename = "t" + str(i) + "-" + name + "-" + str(layer) + ".bin"
        self.session_tensors.append({"name": name, "layer": layer, "position": i,
                                     "type": "F32", "elements": count, "file": filename})
        ne = [2560, count // 2560, 1, 1]
        nb = [4, 10240, count * 4, count * 4]
        self.oracle_tensors.append({"name": oracle_name or name, "token_index": i, "position": i,
                                    "seq_id": 0, "input_token_id": self.ids[i], "occurrence": 0,
                                    "node_index": len(self.oracle_tensors), "type": "f32", "type_id": 0,
                                    "op": "ADD", "op_id": 2, "ne": ne, "nb": nb, "bytes": count * 4,
                                    "contiguous": True, "view_offset": 0, "file": filename})
        self.session_trace.joinpath(filename).write_bytes(f32_bytes(values))
        self.oracle_dir.joinpath(filename).write_bytes(f32_bytes(values))

    def sync_counts(self):
        counts = Counter(row["name"] for row in self.oracle_tensors)
        self.meta["callbacks"].update(captures=len(self.oracle_tensors), allowlist=sorted(counts), observed_counts=dict(counts))
        per_token = Counter(row["token_index"] for row in self.oracle_tensors)
        for row in self.tokens:
            row["callback_captures"] = per_token[row["token_index"]]

    def save(self):
        write_jsonl(self.session_log, self.log)
        write_jsonl(self.session_trace / "tensors.jsonl", self.session_tensors)
        write_jsonl(self.oracle_dir / "tensors.jsonl", self.oracle_tensors)
        write_jsonl(self.oracle_dir / "tokens.jsonl", self.tokens)
        write_json(self.oracle_dir / "metadata.json", self.meta)

    def compare(self):
        return MODULE.compare_session(self.session_log, self.session_logits, self.session_trace, self.oracle_dir)

    def tensor(self, side, name, token=0, layer=None):
        rows = self.oracle_tensors if side == "oracle" else self.session_tensors
        for row in rows:
            if row["name"] == name and row["position"] == token and (layer is None or row.get("layer") == layer):
                return row
        raise AssertionError("fixture tensor missing")

    def perturb_logit(self, side, token, column, value):
        path = self.oracle_logits if side == "oracle" else self.session_logits
        with path.open("r+b") as stream:
            stream.seek(token * ROW_BYTES + column * 4)
            stream.write(struct.pack("<f", value))

    def perturb_tensor(self, side, name, value, token=0, element=0):
        row = self.tensor(side, name, token)
        root = self.oracle_dir if side == "oracle" else self.session_trace
        with (root / row["file"]).open("r+b") as stream:
            stream.seek(element * 4)
            stream.write(struct.pack("<f", value))


class SessionCompareTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="session-compare-test-", dir=os.environ.get("TMPDIR", TMP))
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.f = Fixture(self.root)

    def assert_rejected(self, report):
        self.assertIs(report["passed"], False)
        self.assertTrue(report["failures"])
        json.dumps(report, allow_nan=False)

    def test_correct_full_vocab_and_every_required_token(self):
        report = self.f.compare()
        self.assertIs(report["passed"], True, report["failures"])
        self.assertEqual(report["input_token_ids"], [7, 11, 248044])
        self.assertEqual(report["logits"]["elements"], 3 * 248320)
        self.assertEqual(report["logits"]["expected_bytes_each"], 3 * 993280)
        self.assertIs(report["logits"]["exact_byte_counts"], True)
        self.assertIs(report["logits"]["all_finite"], True)
        self.assertEqual(report["logits"]["max_abs"], 0)
        self.assertEqual(report["logits"]["rms"], 0)
        self.assertEqual(report["logits"]["max_bound_ratio"], 0)
        self.assertEqual(report["logits"]["argmax_agreeing_rows"], 3)
        self.assertEqual(len(report["intermediates"]), 18)
        self.assertEqual(len(report["optional_intermediates"]), 12)
        self.assertTrue(all(row["passed"] for row in report["intermediates"] + report["optional_intermediates"]))
        self.assertEqual(report["gates"], {"logits": {"absolute": .02, "relative": .002},
                                           "intermediates": {"absolute": .002, "relative": .002},
                                           "hc_init": "exact_numerical_float"})
        self.assertEqual(report["scope"], "teacher_forced_operational_baseline")
        self.assertTrue(any("sum+1e-6" in note and "eps^2" in note for note in report["notes"]))
        self.assertTrue(any("whole blocks" in note and "expanded positions" in note for note in report["notes"]))

    def test_one_row_diagnostic_is_supported(self):
        other = self.root / "one"
        other.mkdir()
        report = Fixture(other, n=1, optional=False).compare()
        self.assertIs(report["passed"], True, report["failures"])
        self.assertEqual(len(report["logits"]["rows"]), 1)
        self.assertEqual(len(report["intermediates"]), 6)
        self.assertTrue(all(row["status"] == "missing_optional" for row in report["optional_intermediates"]))

    def test_thirty_two_teacher_forced_rows_are_supported(self):
        other = self.root / "thirty-two"
        other.mkdir()
        report = Fixture(other, n=32, optional=False).compare()
        self.assertIs(report["passed"], True, report["failures"])
        self.assertEqual(report["tokens"], 32)
        self.assertEqual(report["logits"]["elements"], 32 * 248320)
        self.assertEqual(len(report["intermediates"]), 32 * 6)

    def test_row_count_bounds_include_4096_and_reject_zero_and_4097(self):
        # Exercise the upper protocol bound without allocating 4 GiB of logits.
        source, _, complete = copy.deepcopy((self.f.log[0], self.f.log[1:-1], self.f.log[-1]))
        tokens = [{"kind": "session_token", "position": i, "token": 7, "argmax": 100,
                   "finite": True} for i in range(4096)]
        complete.update(input_tokens=4096, consumed_tokens=4096)
        write_jsonl(self.f.session_log, [source, *tokens, complete])
        self.assertEqual(len(MODULE._session(self.f.session_log)[1]), 4096)
        complete.update(input_tokens=4097, consumed_tokens=4097)
        tokens.append({"kind": "session_token", "position": 4096, "token": 7, "argmax": 100, "finite": True})
        write_jsonl(self.f.session_log, [source, *tokens, complete])
        self.assert_rejected(self.f.compare())
        complete.update(input_tokens=0, consumed_tokens=0)
        write_jsonl(self.f.session_log, [source, complete])
        self.assert_rejected(self.f.compare())
        self.f.save()
        self.f.meta["token_ids"] = [7] * 4097
        self.f.save()
        self.assert_rejected(self.f.compare())

    def test_every_logit_is_checked_including_last_column_last_row(self):
        self.f.perturb_logit("session", 2, VOCAB - 1, .5)
        report = self.f.compare()
        self.assert_rejected(report)
        failed = report["logits"]["rows"][2]
        self.assertEqual(failed["violating_elements"], 1)
        self.assertEqual(failed["worst_bound_element"]["element"], 248319)
        self.assertAlmostEqual(failed["rms"], .25 / math.sqrt(248320))

    def test_logits_absolute_and_relative_frozen_bounds(self):
        for column, inside, outside in ((1, .019, .021), (3, 10.039, 10.041)):
            with self.subTest(column=column):
                self.f.perturb_logit("session", 0, column, inside)
                report = self.f.compare()
                self.assertIs(report["passed"], True, report["failures"])
                self.assertLessEqual(report["logits"]["max_bound_ratio"], 1)
                self.f.perturb_logit("session", 0, column, outside)
                report = self.f.compare()
                self.assert_rejected(report)
                self.assertGreater(report["logits"]["max_bound_ratio"], 1)
                self.f.perturb_logit("session", 0, column, 0 if column == 1 else 10)

    def test_intermediate_bounds_are_independent_and_not_logits_bounds(self):
        for name in ("l_last-0", "l_last-1", "l_last-3", "l_last-47", "result_norm"):
            with self.subTest(name=name):
                self.f.perturb_tensor("session", name, .0019)
                self.assertIs(self.f.compare()["passed"], True)
                self.f.perturb_tensor("session", name, .0021)
                report = self.f.compare()
                self.assert_rejected(report)
                self.assertIs(report["logits"]["passed"], True)
                metric = next(row for row in report["intermediates"] if row["name"] == name and row["token_index"] == 0)
                self.assertEqual(metric["violating_elements"], 1)
                self.assertGreater(metric["max_bound_ratio"], 1)
                self.f.perturb_tensor("session", name, 0)
        self.f.perturb_tensor("session", "result_norm", 10.021, element=2)
        self.assertIs(self.f.compare()["passed"], True)
        self.f.perturb_tensor("session", "result_norm", 10.023, element=2)
        self.assert_rejected(self.f.compare())

    def test_hc_init_requires_exact_numerical_float_but_accepts_signed_zero(self):
        self.f.perturb_tensor("session", "hc_init", -0.0)
        self.assertIs(self.f.compare()["passed"], True)
        self.f.perturb_tensor("session", "hc_init", 1e-30)
        report = self.f.compare()
        self.assert_rejected(report)
        metric = report["intermediates"][0]
        self.assertEqual(metric["gate"], "exact_numerical_float")
        self.assertEqual(metric["violating_elements"], 1)
        self.assertIsNone(metric["max_bound_ratio"])

    def test_argmax_difference_within_bounded_gate_is_diagnostic(self):
        self.f.perturb_logit("session", 0, 200, 30.005)
        self.f.log[1]["argmax"] = 200
        self.f.save()
        report = self.f.compare()
        self.assertIs(report["passed"], True, report["failures"])
        self.assertIs(report["logits"]["rows"][0]["argmax_agreement"], False)
        self.assertEqual(report["logits"]["argmax_agreeing_rows"], 2)

    def test_stride_gaps_and_permuted_axes_are_not_contiguous_values(self):
        row = self.f.tensor("oracle", "l_last-3")
        values = struct.unpack("<10240f", (self.f.session_trace / row["file"]).read_bytes())
        row.update(nb=[4, 10256, 41024, 41024], contiguous=False, view_offset=8192)
        data = capture(values, row["ne"], row["nb"])
        row["bytes"] = len(data)
        (self.f.oracle_dir / row["file"]).write_bytes(data)
        permuted = self.f.tensor("oracle", "l_last-47")
        values = struct.unpack("<10240f", (self.f.session_trace / permuted["file"]).read_bytes())
        permuted.update(ne=[4, 2560, 1, 1], nb=[10240, 4, 40960, 40960], contiguous=False)
        (self.f.oracle_dir / permuted["file"]).write_bytes(capture(values, permuted["ne"], permuted["nb"]))
        self.f.save()
        report = self.f.compare()
        self.assertIs(report["passed"], True, report["failures"])
        # Change a live strided coordinate, not a padding byte.
        with (self.f.oracle_dir / row["file"]).open("r+b") as stream:
            stream.seek(10256)
            stream.write(struct.pack("<f", 100.0))
        self.assert_rejected(self.f.compare())

    def test_records_and_tensor_rows_are_matched_by_token_index(self):
        self.f.tokens.reverse()
        self.f.oracle_tensors.reverse()
        self.f.session_tensors.reverse()
        self.f.save()
        self.assertIs(self.f.compare()["passed"], True)
        # Equal shapes cannot hide a tensor assigned to the wrong input token.
        self.f.oracle_tensors[0]["input_token_id"] = 12
        self.f.save()
        self.assert_rejected(self.f.compare())

    def test_optional_missing_is_reported_but_does_not_fail(self):
        self.f.oracle_tensors = [row for row in self.f.oracle_tensors if row["name"] != "hc_mixed-0"]
        self.f.sync_counts()
        self.f.save()
        report = self.f.compare()
        self.assertIs(report["passed"], True, report["failures"])
        self.assertEqual(sum(row["status"] == "missing_optional" for row in report["optional_intermediates"]), 3)

    def test_optional_uses_attention_occurrence_zero_not_ffn_occurrence_one(self):
        row = copy.deepcopy(self.f.tensor("oracle", "hc_mixed-0"))
        row.update(occurrence=1, file="ffn-occurrence.bin")
        (self.f.oracle_dir / row["file"]).write_bytes(f32_bytes([100.0] * 2560))
        self.f.oracle_tensors.append(row)
        self.f.sync_counts()
        self.f.save()
        self.assertIs(self.f.compare()["passed"], True)
        self.f.perturb_tensor("session", "hc_attn_mix", .01)
        report = self.f.compare()
        self.assert_rejected(report)
        self.assertTrue(any(row.get("oracle_occurrence") == 0 and row.get("passed") is False for row in report["optional_intermediates"]))

    def test_all_math_failures_are_retained_without_semantic_waivers(self):
        for i in range(3):
            self.f.perturb_logit("session", i, 1, .1)
            for name in ("l_last-0", "result_norm"):
                self.f.perturb_tensor("session", name, .01, token=i)
        self.f.perturb_tensor("session", "hc_init", 1e-7)
        report = self.f.compare()
        self.assert_rejected(report)
        self.assertEqual(len(report["logits"]["rows"]), 3)
        self.assertEqual(len(report["intermediates"]), 18)
        failures = [row for row in report["failures"] if row["category"] == "gate"]
        self.assertEqual(len(failures), 10)
        self.assertTrue(all("max_abs" in row["metrics"] and "rms" in row["metrics"] for row in failures))

    def test_teacher_forced_ids_positions_and_counts_cannot_false_pass(self):
        originals = copy.deepcopy((self.f.log, self.f.tokens, self.f.meta))
        mutations = [lambda: self.f.log[2].update(token=99), lambda: self.f.log[2].update(position=0),
                     lambda: self.f.log[-1].update(input_tokens=2), lambda: self.f.log[-1].update(consumed_tokens=4),
                     lambda: self.f.meta.update(token_ids=[7, 99, 248044]),
                     lambda: self.f.meta.update(token_ids=[7, 11]),
                     lambda: self.f.tokens[1].update(position=0), lambda: self.f.tokens[1].update(input_token_id=99),
                     lambda: self.f.tokens[1].update(token_index=0), lambda: self.f.tokens[1].update(seq_id=1),
                     lambda: self.f.tokens[1].update(logits_byte_offset=0)]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                self.f.log, self.f.tokens, self.f.meta = copy.deepcopy(originals)
                mutation()
                self.f.save()
                self.assert_rejected(self.f.compare())

    def test_missing_or_duplicate_required_tensors_fail_on_either_side(self):
        originals = copy.deepcopy((self.f.session_tensors, self.f.oracle_tensors))
        for side in ("session", "oracle"):
            for kind in ("missing", "duplicate"):
                with self.subTest(side=side, kind=kind):
                    self.f.session_tensors, self.f.oracle_tensors = copy.deepcopy(originals)
                    rows = self.f.session_tensors if side == "session" else self.f.oracle_tensors
                    if kind == "missing":
                        del rows[0]
                    else:
                        duplicate = copy.deepcopy(rows[0])
                        duplicate["file"] = "duplicate-" + side + ".bin"
                        root = self.f.session_trace if side == "session" else self.f.oracle_dir
                        (root / duplicate["file"]).write_bytes((root / rows[0]["file"]).read_bytes())
                        if side == "oracle":
                            duplicate["occurrence"] = 1
                        rows.append(duplicate)
                    self.f.sync_counts()
                    # Missing a name entirely must still retain the declared required allowlist.
                    self.f.meta["callbacks"]["allowlist"] = sorted(set(self.f.meta["callbacks"]["allowlist"]) | {row[0] for row in REQUIRED})
                    self.f.save()
                    self.assert_rejected(self.f.compare())

    def test_duplicate_optional_occurrence_and_shared_files_are_invalid(self):
        row = copy.deepcopy(self.f.tensor("oracle", "hc_mixed-0"))
        self.f.oracle_tensors.append(row)
        self.f.sync_counts()
        self.f.save()
        self.assert_rejected(self.f.compare())

    def test_incomplete_metadata_wrong_baseline_and_generation_fail(self):
        originals = copy.deepcopy((self.f.log, self.f.meta))
        mutations = [lambda: self.f.log[-1].update(passed=False), lambda: self.f.log[-1].update(output_tokens=1),
                     lambda: self.f.log.insert(-1, {"kind": "session_output", "index": 0, "token": 7}),
                     lambda: self.f.meta.update(status="running"), lambda: self.f.meta.update(status="failed"),
                     lambda: self.f.meta.update(error="callback failed"), lambda: self.f.meta.update(completed_tokens=2),
                     lambda: self.f.meta.update(source_revision="b" * 40),
                     lambda: self.f.meta.update(library_revision_attested="b" * 40),
                     lambda: self.f.meta.update(vocab_actual=248319), lambda: self.f.meta.update(model_layers_actual=47),
                     lambda: self.f.meta.update(mtp=True), lambda: self.f.meta.update(sampler={"kind": "greedy"}),
                     lambda: self.f.meta["logits"].update(bytes_written=0), lambda: self.f.meta["logits"].update(header_bytes=4),
                     lambda: self.f.meta["context"].update(type_k="f16"),
                     lambda: self.f.meta["callbacks"].update(captures=1),
                     lambda: self.f.meta["callbacks"]["observed_counts"].update(hc_init=1)]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                self.f.log, self.f.meta = copy.deepcopy(originals)
                mutation()
                self.f.save()
                self.assert_rejected(self.f.compare())

    def test_required_metadata_cannot_be_omitted_or_replaced_by_null(self):
        original = copy.deepcopy(self.f.meta)
        for key in ("sampler", "status", "completed_tokens", "token_ids", "context", "logits", "callbacks",
                    "library_revision_attested", "library_revision_runtime_verified"):
            with self.subTest(key=key):
                self.f.meta = copy.deepcopy(original)
                del self.f.meta[key]
                self.f.save()
                self.assert_rejected(self.f.compare())

    def test_corrupt_manifest_does_not_hide_failed_logits_metric(self):
        self.f.perturb_logit("session", 1, VOCAB - 1, .5)
        (self.f.session_trace / "tensors.jsonl").write_text('{"name": "hc_init", bad}\n')
        report = self.f.compare()
        self.assert_rejected(report)
        self.assertEqual(len(report["logits"]["rows"]), 3)
        self.assertEqual(report["logits"]["rows"][1]["violating_elements"], 1)
        self.assertTrue(any(row["category"] == "gate" for row in report["failures"]))
        self.assertTrue(any(row["category"] == "protocol" for row in report["failures"]))

    def test_binary_nonfinite_is_not_excused_by_true_finite_flags(self):
        for side in ("session", "oracle"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(side=side, value=value):
                    self.f.perturb_logit(side, 2, VOCAB - 1, value)
                    report = self.f.compare()
                    self.assert_rejected(report)
                    self.assertIs(report["logits"]["all_finite"], False)
                    self.assertEqual(report["logits"]["rows"][2]["nonfinite_" + side], 1)
                    self.f.perturb_logit(side, 2, VOCAB - 1, .25)
        self.f.perturb_tensor("session", "l_last-1", math.nan)
        self.assert_rejected(self.f.compare())

    def test_unpaired_optional_nonfinite_capture_is_invalid(self):
        self.f.oracle_tensors = [row for row in self.f.oracle_tensors if row["name"] != "hc_mixed-0"]
        self.f.sync_counts()
        self.f.save()
        self.f.perturb_tensor("session", "hc_attn_mix", math.nan)
        self.assert_rejected(self.f.compare())

    def test_reported_argmax_and_oracle_max_are_verified_from_binary(self):
        self.f.log[1]["argmax"] = 1
        self.f.tokens[1]["max_logit"] = 0
        self.f.save()
        report = self.f.compare()
        self.assert_rejected(report)
        self.assertIs(report["logits"]["passed"], True)
        self.assertEqual(sum(row["category"] == "protocol" for row in report["failures"]), 2)

    def test_truncated_extra_and_missing_binary_files_are_rejected(self):
        original = self.f.session_logits.read_bytes()
        for suffix in ("truncated", "extra"):
            with self.subTest(case=suffix):
                self.f.session_logits.write_bytes(original[:-1] if suffix == "truncated" else original + b"\0")
                self.assert_rejected(self.f.compare())
        self.f.session_logits.write_bytes(original)
        row = self.f.tensor("oracle", "result_norm")
        path = self.f.oracle_dir / row["file"]
        path.write_bytes(path.read_bytes()[:-4])
        self.assert_rejected(self.f.compare())
        path.unlink()
        self.assert_rejected(self.f.compare())

    def test_json_is_strict_including_nested_duplicate_keys_and_overflow(self):
        original = (self.f.oracle_dir / "metadata.json").read_text()
        for bad in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(value=bad):
                (self.f.oracle_dir / "metadata.json").write_text(original[:-2] + ',"extra":' + bad + "}\n")
                self.assert_rejected(self.f.compare())
        (self.f.oracle_dir / "metadata.json").write_text(original[:-2] + ',"extra":{"x":1,"x":2}}\n')
        self.assert_rejected(self.f.compare())
        (self.f.oracle_dir / "metadata.json").write_text(original)
        self.f.session_log.write_text(self.f.session_log.read_text() + "session_complete passed true\n")
        self.assert_rejected(self.f.compare())

    def test_boolean_numeric_fields_are_rejected(self):
        originals = copy.deepcopy((self.f.log, self.f.tokens, self.f.meta, self.f.oracle_tensors, self.f.session_tensors))
        mutations = [lambda: self.f.log[1].update(position=False), lambda: self.f.log[1].update(token=True),
                     lambda: self.f.log[1].update(argmax=True), lambda: self.f.log[1].update(completed_ms=True),
                     lambda: self.f.log[-1].update(output_tokens=False), lambda: self.f.log[0].update(dirty=True),
                     lambda: self.f.meta["token_ids"].__setitem__(0, True),
                     lambda: self.f.meta["logits"].update(header_bytes=False),
                     lambda: self.f.tokens[0].update(token_index=False),
                     lambda: self.f.tokens[0].update(max_logit=True),
                     lambda: self.f.oracle_tensors[0]["ne"].__setitem__(1, True),
                     lambda: self.f.oracle_tensors[0]["nb"].__setitem__(0, True),
                     lambda: self.f.oracle_tensors[0].update(bytes=True),
                     lambda: self.f.session_tensors[0].update(layer=False),
                     lambda: self.f.session_tensors[0].update(elements=True)]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                self.f.log, self.f.tokens, self.f.meta, self.f.oracle_tensors, self.f.session_tensors = copy.deepcopy(originals)
                mutation()
                self.f.save()
                self.assert_rejected(self.f.compare())

    def test_path_escape_including_symlinks_is_rejected(self):
        row = self.f.session_tensors[0]
        original = row["file"]
        outside = self.root / "outside.bin"
        outside.write_bytes((self.f.session_trace / original).read_bytes())
        (self.f.session_trace / "escape.bin").symlink_to(outside)
        for filename in ("../outside.bin", str(outside), "escape.bin"):
            with self.subTest(file=filename):
                row["file"] = filename
                self.f.save()
                self.assert_rejected(self.f.compare())
        row["file"] = original
        self.f.meta["logits"]["file"] = "../session.f32.bin"
        self.f.save()
        self.assert_rejected(self.f.compare())

    def test_oversize_artifacts_and_invalid_stride_metadata_are_rejected(self):
        original = copy.deepcopy(self.f.oracle_tensors[0])
        for change in ({"bytes": 16777217}, {"ne": [2560, 4, 1, 1], "nb": [4, 4, 40960, 40960]},
                       {"ne": [2560, 4, 1, 1], "nb": [4, 10239, 40960, 40960]},
                       {"ne": [2560, 4, 1], "nb": [4, 10240, 40960]}, {"nb": [4, 10240, 16777217, 40960]},
                       {"type": "f16"}, {"occurrence": 1}):
            with self.subTest(change=change):
                self.f.oracle_tensors[0] = {**original, **change}
                self.f.save()
                self.assert_rejected(self.f.compare())
        self.f.oracle_tensors[0] = original
        self.f.save()
        # Sparse oversized files exercise size checks without allocating large payloads.
        with self.f.session_logits.open("r+b") as stream:
            stream.truncate(4096 * 993280 + 1)
        self.assert_rejected(self.f.compare())
        with self.f.session_logits.open("r+b") as stream:
            stream.truncate(3 * 993280)
        with self.f.session_log.open("r+b") as stream:
            stream.truncate(8 * 1024 * 1024 + 1)
        self.assert_rejected(self.f.compare())

    def cli(self, output=None):
        command = [sys.executable, "-B", str(SCRIPT), "--session-log", str(self.f.session_log),
                   "--session-logits", str(self.f.session_logits), "--session-trace", str(self.f.session_trace),
                   "--oracle-dir", str(self.f.oracle_dir)]
        if output is not None:
            command += ["--output", str(output)]
        return subprocess.run(command, cwd=self.root, env={**os.environ, "TMPDIR": str(TMP)},
                              text=True, capture_output=True, check=False)

    def test_cli_success_failure_and_atomic_json_output(self):
        output = self.root / "report.json"
        output.write_text("previous report\n")
        result = self.cli(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report, json.loads(output.read_text()))
        self.assertIs(report["passed"], True)
        self.f.perturb_logit("session", 0, 1, .1)
        self.f.perturb_tensor("session", "l_last-47", .01)
        result = self.cli(output)
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report, json.loads(output.read_text()))
        self.assert_rejected(report)
        self.assertEqual(len([f for f in report["failures"] if f["category"] == "gate"]), 2)
        self.assertFalse(list(self.root.glob(".report.json.*.tmp")))

    def test_atomic_output_failure_preserves_previous_destination_and_cleans_temp(self):
        output = self.root / "report.json"
        output.write_text("previous report\n")
        with mock.patch.object(MODULE.os, "replace", side_effect=OSError("replacement failure")):
            with self.assertRaises(OSError):
                MODULE._write_output(output, '{"passed":false}\n')
        self.assertEqual(output.read_text(), "previous report\n")
        self.assertFalse(list(self.root.glob(".report.json.*.tmp")))

    def test_cli_protocol_failure_is_structured_and_does_not_overwrite_inputs(self):
        original = self.f.session_log.read_bytes()
        result = self.cli(self.f.session_log)
        self.assertEqual(result.returncode, 1)
        self.assert_rejected(json.loads(result.stdout))
        self.assertEqual(self.f.session_log.read_bytes(), original)
        self.f.meta["status"] = "failed"
        self.f.save()
        output = self.root / "failure.json"
        result = self.cli(output)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout), json.loads(output.read_text()))
        self.assert_rejected(json.loads(result.stdout))


if __name__ == "__main__":
    unittest.main()
