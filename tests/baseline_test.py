"""Run with: python3 -B -m unittest discover -s tests -p baseline_test.py -v"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest


SPEC = importlib.util.spec_from_file_location("baseline", Path(__file__).resolve().parents[1]
                                           / "tools" / "baseline.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


@contextmanager
def mock_server(transform=None, pieces=False, health_codes=(200,), completion_code=200,
                body_delay=0, models_responses=None, models_codes=(200,), models_calls=None):
    requests = []
    responses = []
    health_calls = []
    if models_calls is None:
        models_calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_json(self, status, value, delay=0):
            raw = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if status == 302:
                self.send_header("Location", "/completion")
            self.end_headers()
            self.wfile.flush()
            if delay:
                time.sleep(delay)
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/models" and models_responses is not None:
                index = min(len(models_calls), len(models_responses) - 1)
                code = models_codes[min(len(models_calls), len(models_codes) - 1)]
                value = models_responses[index]
                models_calls.append(value)
                self.send_json(code, value)
            elif self.path == "/health":
                index = min(len(health_calls), len(health_codes) - 1)
                code = health_codes[index]
                health_calls.append(code)
                self.send_json(code, {"status": "ok"} if code == 200 else
                               {"error": {"code": code, "message": "Loading model"}})
            else:
                self.send_json(404, {"error": "unknown endpoint"})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            if models_responses is not None and models_calls and "data" in models_calls[-1]:
                entries = models_calls[-1]["data"]
                if any("status" in entry for entry in entries):
                    target = next((entry for entry in entries if entry["id"] == payload.get("model")), {})
                    if target.get("status", {}).get("value") != "loaded":
                        self.send_json(503, {"error": "POST before selected model loaded"})
                        return
            if self.path == "/tokenize":
                # A mock tokenizer supplies an explicit BOS followed by deterministic
                # byte tokens. Real tokenization and normalization remain server-owned.
                ids = [248044] + list(payload["content"].encode("utf-8"))
                tokens = [{"id": token, "piece": "x"} for token in ids] if pieces else ids
                self.send_json(200, {"tokens": tokens})
                return
            if self.path != "/completion":
                self.send_json(404, {"error": "unknown endpoint"})
                return
            n_in, n_out = len(payload["prompt"]), payload["n_predict"]
            response = {
                "content": "accepted0=999 is generated text, not a metric",
                "tokens": list(range(n_out)), "tokens_evaluated": n_in,
                "tokens_predicted": n_out, "truncated": False,
                "timings": {"prompt_n": n_in, "predicted_n": n_out,
                            "prompt_ms": 200.0, "predicted_ms": 500.0, "cache_n": 0},
                "generation_settings": {key: value for key, value in payload.items()
                                        if key != "prompt"},
                "extra": {"kept_verbatim": [1, 2, 3]},
            }
            if transform:
                transform(response)
            if completion_code != 200:
                response = {"error": {"code": completion_code, "message": "failed"}}
            responses.append(response)
            self.send_json(completion_code, response, body_delay)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, responses, health_calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class BaselineTest(unittest.TestCase):
    def invoke(self, root, url, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        args = ["--url", url, "--runs-dir", str(root / "runs"),
                "--results", str(root / "results.jsonl"), "--revision", "production-revision",
                "--wait-seconds", "4", "--request-timeout", "4", *extra]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = baseline.main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def records(self, root):
        path = root / "results.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_protocol_sampling_counts_and_complete_artifacts(self):
        expected_sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                             "min_p": 0.0, "repeat_penalty": 1.0, "seed": 12345}
        for pieces in (False, True):
            with self.subTest(pieces=pieces), tempfile.TemporaryDirectory() as tmp, mock_server(pieces=pieces) as mock:
                root = Path(tmp)
                url, requests, responses, health_calls = mock
                status, _, error = self.invoke(root, url, "--dirty")
                self.assertEqual(status, 0, error)
                self.assertEqual(health_calls, [200])
                tokenizations = [payload for path, payload in requests if path == "/tokenize"]
                completions = [payload for path, payload in requests if path == "/completion"]
                self.assertEqual(len(tokenizations), 2)
                self.assertEqual([len(p["prompt"]) for p in completions], [32, 4096])
                self.assertEqual([p["n_predict"] for p in completions], [64, 512])
                records = self.records(root)
                self.assertEqual(len(records), 2)
                for index, (record, payload) in enumerate(zip(records, completions)):
                    self.assertEqual(payload["prompt"][0], 248044)
                    self.assertEqual(payload["model"], "current")
                    for key, expected in expected_sampling.items():
                        self.assertEqual(payload[key], expected)
                    for key, expected in {"cache_prompt": False, "stream": False,
                                          "ignore_eos": True, "return_tokens": True, "stop": []}.items():
                        self.assertEqual(payload[key], expected)
                    self.assertEqual(record["series"], "new-fixtures")
                    self.assertEqual(record["revision"], "production-revision")
                    self.assertTrue(record["dirty"])
                    self.assertEqual(record["model"], "qwen38-keep1-Q4_0.gguf")
                    self.assertEqual(record["model_id"], "current")
                    self.assertIsNone(record["models_response_artifact"])
                    self.assertEqual(record["kv"], {role: {"k": "q4_0", "v": "q4_0"}
                                                     for role in ("target", "mtp")})
                    self.assertEqual(record["input_tokens"], len(payload["prompt"]))
                    self.assertEqual(record["output_tokens"], payload["n_predict"])
                    self.assertEqual(record["sampling"], expected_sampling)
                    self.assertFalse(record["prefix_reuse"])
                    self.assertEqual(record["expert_cache"]["protocol_state"], "cold" if index == 0 else "warm")
                    self.assertEqual(record["expert_cache"]["observed_state"], "unknown")
                    self.assertEqual(record["pp_ms"], 200)
                    self.assertEqual(record["tg_ms"], 500)
                    self.assertGreater(record["wall_total_ms"], 0)
                    self.assertEqual(record["pp_tokens_per_second"], len(payload["prompt"]) * 5)
                    self.assertEqual(record["tg_tokens_per_second"], payload["n_predict"] * 2)
                    self.assertTrue(all(record[key] is None for key in ("accepted0", "accepted1", "accepted2")))
                    self.assertEqual(json.loads(Path(record["request_artifact"]).read_text()), payload)
                    self.assertEqual(json.loads(Path(record["response_artifact"]).read_text()), responses[index])
                    self.assertEqual(json.loads(Path(record["prompt_ids_artifact"]).read_text()), payload["prompt"])
                    directory = Path(record["request_artifact"]).parent
                    fixture = record["fixture"]
                    self.assertEqual((directory / f"{fixture}.txt").read_text(), tokenizations[index]["content"])
                    self.assertTrue(tokenizations[index]["add_special"])
                    self.assertEqual(tokenizations[index]["model"], "current")
                    self.assertFalse(tokenizations[index]["with_pieces"])
                    full = json.loads((directory / f"{fixture}.tokenize.response.json").read_text())
                    self.assertEqual(baseline.token_ids(full["tokens"], "fixture")[:record["input_tokens"]], payload["prompt"])

    def test_invalid_completions_are_rejected_and_preserved(self):
        def change_prompt(response):
            response["timings"]["prompt_n"] += 1  # BOS/normalization off by one

        def change_predicted(response):
            response["timings"]["predicted_n"] -= 1

        def short_output(response):
            response["tokens"].pop()

        def empty_output(response):
            response["tokens"] = []

        def nan_timing(response):
            response["timings"]["prompt_ms"] = float("nan")

        def overflow_timing(response):
            response["timings"]["predicted_ms"] = float("inf")

        changes = [change_prompt, change_predicted, short_output, empty_output, nan_timing,
                   overflow_timing, lambda r: r["timings"].update(prompt_n=0),
                   lambda r: r["timings"].update(predicted_ms=0),
                   lambda r: r["timings"].update(cache_n=1),
                   lambda r: r.update(tokens_evaluated=31),
                   lambda r: r.update(tokens_predicted=63),
                   lambda r: r.update(truncated=True),
                   lambda r: r.pop("timings")]
        for transform in changes:
            with self.subTest(transform=transform), tempfile.TemporaryDirectory() as tmp, mock_server(transform=transform) as mock:
                root = Path(tmp)
                url, requests, _, _ = mock
                status, _, error = self.invoke(root, url)
                self.assertEqual(status, 1)
                self.assertIn("baseline failed", error)
                self.assertEqual(self.records(root), [])
                self.assertEqual(len([p for path, p in requests if path == "/completion"]), 1)
                saved = list((root / "runs").glob("*/short-01.response.json"))
                self.assertEqual(len(saved), 1)
                self.assertGreater(saved[0].stat().st_size, 0)

    def test_4k_normalization_mismatch_retains_only_valid_short_result(self):
        def extra_bos(response):
            if response["timings"]["prompt_n"] == 4096:
                response["timings"]["prompt_n"] += 1

        with tempfile.TemporaryDirectory() as tmp, mock_server(transform=extra_bos) as mock:
            root = Path(tmp)
            status, _, error = self.invoke(root, mock[0])
            self.assertEqual(status, 1)
            self.assertIn("expected 4096, got 4097", error)
            self.assertEqual([r["fixture"] for r in self.records(root)], ["short"])
            self.assertEqual(len(mock[2]), 2)
            saved = next((root / "runs").glob("*/4k-01.response.json"))
            self.assertEqual(json.loads(saved.read_text()), mock[2][1])

    def test_structured_acceptance_and_repeats(self):
        def histogram(response):
            response["speculative"] = {"accepted0": 5, "accepted1": 3, "accepted2": 7}

        with tempfile.TemporaryDirectory() as tmp, mock_server(transform=histogram) as mock:
            root = Path(tmp)
            status, _, error = self.invoke(root, mock[0], "--repeats", "2")
            self.assertEqual(status, 0, error)
            records = self.records(root)
            self.assertEqual([r["repeat"] for r in records], [1, 1, 2, 2])
            self.assertEqual([r["expert_cache"]["protocol_state"] for r in records], ["cold", "warm", "warm", "warm"])
            for record in records:
                self.assertFalse(record["dirty"])
                self.assertEqual([record[k] for k in ("accepted0", "accepted1", "accepted2")], [5, 3, 7])

    def test_health_loading_retried_but_completion_failure_is_not(self):
        with tempfile.TemporaryDirectory() as tmp, mock_server(health_codes=(503, 200), completion_code=503) as mock:
            root = Path(tmp)
            url, requests, responses, health_calls = mock
            status, _, error = self.invoke(root, url)
            self.assertEqual(status, 1)
            self.assertIn("not retried", error)
            self.assertEqual(health_calls, [503, 200])
            self.assertEqual(len([p for path, p in requests if path == "/completion"]), 1)
            saved = next((root / "runs").glob("*/short-01.response.json"))
            self.assertEqual(json.loads(saved.read_text()), responses[0])
            self.assertEqual(self.records(root), [])

    def test_completion_redirect_is_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp, mock_server(completion_code=302) as mock:
            root = Path(tmp)
            status, _, error = self.invoke(root, mock[0])
            self.assertEqual(status, 1)
            self.assertIn("completion HTTP 302", error)
            self.assertEqual(len(mock[2]), 1)
            self.assertEqual(self.records(root), [])

    def test_health_wait_is_bounded_and_other_http_errors_fail(self):
        for codes, wait in (((503,), "0.12"), ((404,), "2")):
            with self.subTest(codes=codes), tempfile.TemporaryDirectory() as tmp, mock_server(health_codes=codes) as mock:
                root = Path(tmp)
                started = time.monotonic()
                status, _, error = self.invoke(root, mock[0], "--wait-seconds", wait)
                self.assertEqual(status, 1)
                self.assertIn("/health", error)
                self.assertLess(time.monotonic() - started, 1)
                self.assertEqual(mock[1], [])
                self.assertEqual(self.records(root), [])
                self.assertEqual(list((root / "runs").glob("*/*.json")), [])

    def test_router_waits_for_selected_model_and_saves_launch_configuration(self):
        for model in ("current", "custom-model"):
            loading = {"data": [
                {"id": "another-model", "status": {"value": "loaded"}},
                {"id": model, "status": {"value": "loading", "args": ["llama-server", "-c", "8192"],
                                         "preset": {"cache-type-k": "q4_0", "cache-type-v": "q4_0"}}},
            ]}
            loaded = copy.deepcopy(loading)
            loaded["data"][1]["status"]["value"] = "loaded"
            calls = []
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp, mock_server(
                    models_responses=(loading, loaded), models_calls=calls) as mock:
                root = Path(tmp)
                extra = [] if model == "current" else ["--model", model]
                status, _, error = self.invoke(root, mock[0], *extra)
                self.assertEqual(status, 0, error)
                self.assertEqual(calls, [loading, loaded])
                self.assertEqual(mock[3], [200])
                self.assertEqual([path for path, _ in mock[1]],
                                 ["/tokenize", "/tokenize", "/completion", "/completion"])
                self.assertTrue(all(payload["model"] == model for _, payload in mock[1]))
                records = self.records(root)
                self.assertEqual(len(records), 2)
                path = Path(records[0]["models_response_artifact"])
                self.assertEqual(path.read_bytes(), json.dumps(loaded).encode("utf-8"))
                self.assertLessEqual(path.stat().st_size, baseline.MODELS_MAX_BYTES)
                self.assertEqual(list(path.parent.glob("models*.json")), [path])
                for record in records:
                    self.assertEqual(record["model_id"], model)
                    self.assertEqual(record["models_response_artifact"], str(path))
                    request = json.loads(Path(record["request_artifact"]).read_text())
                    self.assertEqual(request["model"], model)
                for request_path in path.parent.glob("*.tokenize.request.json"):
                    self.assertEqual(json.loads(request_path.read_text())["model"], model)

    def test_router_missing_failed_unknown_and_malformed_models_are_rejected(self):
        cases = [
            {"data": [{"id": "other", "status": {"value": "loaded"}}]},
            {"data": []},
            {"error": {"code": 500, "message": "could not list models"}},
            {"data": [{"id": "current", "status": {"value": "unloaded", "failed": True, "exit_code": 1}}]},
            {"data": [{"id": "current", "status": {"value": "ready"}}]},
            {"data": [{"id": "current", "status": {"value": "error"}}]},
            {"data": [{"id": "current", "status": {"value": "loaded", "error": {"code": 1}}}]},
            {"data": [{"id": "current", "status": None}]},
            {"data": [{"id": "current", "status": {"value": "loaded", "failed": "false"}}]},
            {"data": [{"id": "current"}, {"id": "other", "status": {"value": "loaded"}}]},
            {"data": [{"id": "current", "status": {"value": "loaded"}}] * 2},
            {"data": "not an array"},
        ]
        for response in cases:
            calls = []
            with self.subTest(response=response), tempfile.TemporaryDirectory() as tmp, mock_server(
                    models_responses=(response,), models_calls=calls) as mock:
                root = Path(tmp)
                status, _, error = self.invoke(root, mock[0])
                self.assertEqual(status, 1)
                self.assertIn("/models", error)
                self.assertEqual(calls, [response])
                self.assertEqual(mock[1], [])
                self.assertEqual(self.records(root), [])
                path = next((root / "runs").glob("*/models.response.json"))
                self.assertEqual(json.loads(path.read_text()), response)

    def test_models_http_loading_retries_but_other_errors_fail(self):
        loaded = {"data": [{"id": "current", "status": {"value": "loaded"}}]}
        for codes, expected_status in (((503, 200), 0), ((400,), 1), ((500,), 1)):
            calls = []
            with self.subTest(codes=codes), tempfile.TemporaryDirectory() as tmp, mock_server(
                    models_responses=({"error": {"code": codes[0]}}, loaded),
                    models_codes=codes, models_calls=calls) as mock:
                root = Path(tmp)
                status, _, error = self.invoke(root, mock[0])
                self.assertEqual(status, expected_status, error)
                self.assertEqual(len(calls), len(codes))
                self.assertEqual(len(mock[2]), 2 if expected_status == 0 else 0)
                if expected_status == 1:
                    self.assertIn(f"/models returned HTTP {codes[0]}", error)
                    self.assertEqual(mock[1], [])

    def test_models_wait_and_shared_health_models_deadline_are_bounded(self):
        loading = {"data": [{"id": "current", "status": {"value": "loading"}}]}
        for health_codes, wait, upper_bound in (((200,), "0.12", 1), ((503, 200), "1.12", 1.8)):
            calls = []
            with self.subTest(health_codes=health_codes), tempfile.TemporaryDirectory() as tmp, mock_server(
                    health_codes=health_codes, models_responses=(loading,), models_calls=calls) as mock:
                root = Path(tmp)
                started = time.monotonic()
                status, _, error = self.invoke(root, mock[0], "--wait-seconds", wait)
                self.assertEqual(status, 1)
                self.assertIn("/models did not become ready", error)
                self.assertLess(time.monotonic() - started, upper_bound)
                self.assertEqual(mock[1], [])
                self.assertEqual(mock[3], list(health_codes))
                self.assertEqual(calls, [loading])
                self.assertEqual(self.records(root), [])
                path = next((root / "runs").glob("*/models.response.json"))
                self.assertEqual(json.loads(path.read_text()), loading)

    def test_status_free_non_router_models_use_health_readiness(self):
        response = {"data": [{"id": "current", "object": "model"}]}
        calls = []
        with tempfile.TemporaryDirectory() as tmp, mock_server(
                models_responses=(response,), models_calls=calls) as mock:
            root = Path(tmp)
            status, _, error = self.invoke(root, mock[0])
            self.assertEqual(status, 0, error)
            self.assertEqual(mock[3], [200])
            self.assertEqual(calls, [response])
            self.assertEqual(len(mock[2]), 2)

    def test_models_artifact_size_is_bounded(self):
        response = {"data": [{"id": "current", "status": {"value": "loaded"}}],
                    "large_metadata": "x" * (baseline.MODELS_MAX_BYTES + 1)}
        with tempfile.TemporaryDirectory() as tmp, mock_server(models_responses=(response,)) as mock:
            root = Path(tmp)
            status, _, error = self.invoke(root, mock[0])
            self.assertEqual(status, 1)
            self.assertIn("/models response exceeds", error)
            self.assertEqual(mock[1], [])
            self.assertEqual(list((root / "runs").glob("*/models.response.json")), [])

    def test_wall_time_includes_response_body_wait(self):
        with tempfile.TemporaryDirectory() as tmp, mock_server(body_delay=0.06) as mock:
            root = Path(tmp)
            status, _, error = self.invoke(root, mock[0])
            self.assertEqual(status, 0, error)
            for record in self.records(root):
                self.assertGreaterEqual(record["wall_total_ms"], 50)

    def test_json_overflow_and_invalid_ids(self):
        timings = {"prompt_n": 32, "predicted_n": 64, "prompt_ms": 10, "predicted_ms": 20}
        response = {"content": "ok", "timings": timings, "tokens": list(range(64))}
        overflow = baseline.decode_json(b'{"prompt_ms": 1e999}')
        bad = copy.deepcopy(response)
        bad["timings"].update(overflow)
        with self.assertRaises(baseline.BaselineError):
            baseline.validate_response(bad, 32, 64)
        for ids in ([True], [-1], [1.5], [{"piece": "no ID"}], []):
            with self.subTest(ids=ids), self.assertRaises(baseline.BaselineError):
                baseline.token_ids(ids, "test")


if __name__ == "__main__":
    unittest.main()
