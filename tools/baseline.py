#!/usr/bin/env python3
"""R0 client for an already running, privately bound production llama-server.

Example (revision describes the measured server):
  python3 tools/baseline.py --runs-dir runs --results results.jsonl \
      --revision dcd685463d597d31f5ca759d32c94592a2740fa4

These are new fixtures, not the unavailable original user's prompts. The exact
input is the saved token-ID prefix of /tokenize(add_special=true), including any
BOS. /completion receives IDs, not text; a normalization/count mismatch fails.
Only readiness GETs (/health and /models) are retried. A failed completion may
still be running on the server.
"""

import argparse
from datetime import datetime, timezone
from http.client import HTTPException
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


SAMPLING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20,
    "min_p": 0.0, "repeat_penalty": 1.0, "seed": 12345,
}
MODELS_MAX_BYTES = 1024 * 1024
FIXTURE_BLOCK = """A small archive keeps records of bridges, gardens, and libraries.
Each record has a name, a year, and a short description. The archivist checks
that names are unique and that dates remain in ascending order. A reader asks
for a summary of the changes and an explanation of how to find missing entries.
Keep the original records intact while comparing the old and new catalogues.
Explain the assumptions, give an example, and describe one useful boundary case.

```python
def merge_records(old, new):
    records = {item['name']: item for item in old}
    for item in new:
        records[item['name']] = item
    return sorted(records.values(), key=lambda item: (item['year'], item['name']))

old = [{'name': 'stone bridge', 'year': 1920}]
new = [{'name': 'public garden', 'year': 1950}]
assert len(merge_records(old, new)) == 2
```
Discuss how this function handles an empty list and repeated names.

"""
FIXTURES = (
    ("short", 32, 64, "Explain why a sorted catalogue helps a reader find records. "
     "Give a clear example and describe how repeated names should be handled.\n" * 4),
    ("4k", 4096, 512, "Read the following archive notes and code. Explain their design.\n\n"
     + FIXTURE_BLOCK * 64),
)


class BaselineError(ValueError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def decode_json(raw):
    def reject_constant(value):
        raise BaselineError(f"non-finite JSON value: {value}")
    try:
        return json.loads(raw, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineError(f"invalid JSON response: {exc}") from exc


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")


def token_ids(value, label):
    if not isinstance(value, list) or not value:
        raise BaselineError(f"{label} must be a nonempty tokens array")
    ids = [token.get("id") if isinstance(token, dict) else token for token in value]
    if any(type(token) is not int or token < 0 for token in ids):
        raise BaselineError(f"{label} contains an invalid token ID")
    return ids


def count(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise BaselineError(f"{label} must be an integer >= {minimum}")
    return value


def positive_number(value, label):
    if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
        raise BaselineError(f"{label} must be finite and positive")
    return value


def check_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, list):
        for item in value:
            check_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise BaselineError("non-finite timing value")


class Client:
    def __init__(self, url, model="current"):
        parts = urlsplit(url)
        if (parts.scheme not in ("http", "https") or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment):
            raise BaselineError("--url must be an HTTP(S) base URL without credentials or query")
        self.url = url.rstrip("/")
        if not isinstance(model, str) or not model.strip():
            raise BaselineError("--model must be nonempty")
        self.model = model
        # This local/private endpoint needs no environment proxy or credentials.
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, path, payload=None, timeout=1800, max_bytes=None):
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(self.url + path, data=body, headers=headers)
        with self.opener.open(request, timeout=timeout) as response:
            raw = response.read() if max_bytes is None else response.read(max_bytes + 1)
            if max_bytes is not None and len(raw) > max_bytes:
                raise BaselineError(f"{path} response exceeds {max_bytes} bytes")
            return raw

    def models_ready(self, response):
        if not isinstance(response, dict) or "error" in response:
            raise BaselineError("/models response must be an object without error")
        entries = response.get("data")
        if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
            raise BaselineError("/models response lacks a model data array")
        targets = [item for item in entries if item.get("id") == self.model]
        if len(targets) != 1:
            raise BaselineError(f"/models must list model {self.model!r} exactly once")
        target = targets[0]
        if "error" in target:
            raise BaselineError(f"/models declares an error for model {self.model!r}")
        # Single-model servers can expose a status-free model list: /health is
        # authoritative there. Do not confuse a malformed router entry with it.
        if all("status" not in item for item in entries):
            return True
        status = target.get("status")
        if not isinstance(status, dict) or type(status.get("failed", False)) is not bool:
            raise BaselineError("/models target status must be a structured status object")
        if status.get("failed") or "error" in status:
            raise BaselineError(f"/models reports model {self.model!r} failed to load")
        value = status.get("value")
        # mx dcd6854 tools/server/server-models.h: is_ready() means LOADED.
        # 'ready' is an internal child notification, not a /models status enum.
        if value == "loaded":
            return True
        if value in ("unloaded", "loading", "sleeping", "downloading", "downloaded"):
            return False
        raise BaselineError(f"/models returned an unknown target status: {value!r}")

    def wait_ready(self, wait_seconds, models_path=None):
        deadline = time.monotonic() + wait_seconds
        path = "/health"
        models_seen = False
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                raw = self.request(path, timeout=min(5, remaining),
                                   max_bytes=MODELS_MAX_BYTES if path == "/models" else 4096)
                if time.monotonic() >= deadline:
                    break
                if path == "/health":
                    health = decode_json(raw)
                    if not isinstance(health, dict) or health.get("status") != "ok":
                        raise BaselineError("unexpected /health response (expected status=ok)")
                    path = "/models"
                    continue
                models_seen = True
                if models_path is not None:
                    models_path.write_bytes(raw)  # one bounded, latest snapshot, not a poll log
                if self.models_ready(decode_json(raw)):
                    return
            except HTTPError as exc:
                exc.close()
                if path == "/models" and exc.code == 404 and not models_seen:
                    if time.monotonic() < deadline:
                        return  # non-router server without /models; /health was ready
                    break
                if exc.code != 503:  # documented model-loading health status
                    raise BaselineError(f"{path} returned HTTP {exc.code}") from exc
            except (URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        raise BaselineError(f"server {path} did not become ready for model {self.model!r} "
                            "within --wait-seconds")


def prepare_fixture(client, directory, name, text, input_count, timeout):
    directory.joinpath(f"{name}.txt").write_text(text, encoding="utf-8")
    payload = {"model": client.model, "content": text, "add_special": True, "with_pieces": False}
    save_json(directory / f"{name}.tokenize.request.json", payload)
    raw = client.request("/tokenize", payload, timeout)
    (directory / f"{name}.tokenize.response.json").write_bytes(raw)
    response = decode_json(raw)
    if not isinstance(response, dict):
        raise BaselineError("/tokenize response must be an object")
    ids = token_ids(response.get("tokens"), "/tokenize")
    if len(ids) < input_count:
        raise BaselineError(f"{name}: fixture has {len(ids)} tokens, needs {input_count}")
    ids = ids[:input_count]
    save_json(directory / f"{name}.prompt-ids.json", ids)
    return ids


def validate_response(response, input_count, output_count):
    if not isinstance(response, dict) or "error" in response:
        raise BaselineError("/completion response must be a completion object without error")
    timings = response.get("timings")
    if not isinstance(timings, dict):
        raise BaselineError("/completion response lacks structured timings")
    check_finite(timings)
    prompt_n = count(timings.get("prompt_n"), "timings.prompt_n")
    predicted_n = count(timings.get("predicted_n"), "timings.predicted_n")
    output_ids = token_ids(response.get("tokens"), "completion output")
    if prompt_n != input_count:
        raise BaselineError(f"prompt count mismatch: expected {input_count}, got {prompt_n}")
    if predicted_n != output_count or len(output_ids) != output_count:
        raise BaselineError(f"output count mismatch: expected {output_count}, "
                            f"timings={predicted_n}, token IDs={len(output_ids)}")
    for key, expected in (("tokens_evaluated", input_count), ("tokens_predicted", output_count)):
        if key in response and count(response[key], key) != expected:
            raise BaselineError(f"{key} disagrees with the actual token count")
    if "cache_n" in timings and count(timings["cache_n"], "timings.cache_n", 0) != 0:
        raise BaselineError("response reports prefix reuse despite cache_prompt=false")
    if response.get("truncated"):
        raise BaselineError("response reports a truncated context")
    if not isinstance(response.get("content"), str):
        raise BaselineError("response lacks completion content")
    positive_number(timings.get("prompt_ms"), "timings.prompt_ms")
    positive_number(timings.get("predicted_ms"), "timings.predicted_ms")
    return timings, output_ids


def acceptance(response, timings):
    # Only copy explicitly structured histogram fields. Draft token totals cannot
    # recover a 0/1/2 histogram, and server logs are not a metrics API.
    sources = [response, timings]
    if isinstance(response.get("speculative"), dict):
        sources.append(response["speculative"])
    values = {}
    for key in ("accepted0", "accepted1", "accepted2"):
        supplied = [source[key] for source in sources if source.get(key) is not None]
        values[key] = count(supplied[0], key, 0) if supplied else None
        if any(count(value, key, 0) != values[key] for value in supplied):
            raise BaselineError(f"conflicting structured {key} values")
    return values


def run(args):
    client = Client(args.url, args.model)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("r0-%Y%m%dT%H%M%SZ-")
    directory = Path(tempfile.mkdtemp(prefix=stamp, dir=args.runs_dir)).resolve()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts: {directory}", flush=True)
    models_path = directory / "models.response.json"
    client.wait_ready(args.wait_seconds, models_path)
    prepared = [(name, n_in, n_out, prepare_fixture(client, directory, name, text,
                 n_in, args.request_timeout)) for name, n_in, n_out, text in FIXTURES]
    ordinal = 0
    for repeat in range(1, args.repeats + 1):
        for name, n_in, n_out, ids in prepared:
            ordinal += 1
            label = f"{name}-{repeat:02d}"
            payload = {"model": client.model, "prompt": ids, "n_predict": n_out, **SAMPLING,
                       "cache_prompt": False, "stream": False,
                       "ignore_eos": True, "return_tokens": True, "stop": []}
            request_path = directory / f"{label}.request.json"
            response_path = directory / f"{label}.response.json"
            save_json(request_path, payload)
            started = time.perf_counter()
            try:
                raw = client.request("/completion", payload, args.request_timeout)
            except HTTPError as exc:
                response_path.write_bytes(exc.read())
                exc.close()
                raise BaselineError(f"{label}: completion HTTP {exc.code}; not retried") from exc
            wall_ms = positive_number((time.perf_counter() - started) * 1000, "wall_total_ms")
            response_path.write_bytes(raw)  # retain full response even if validation fails
            response = decode_json(raw)
            timings, output_ids = validate_response(response, n_in, n_out)
            result = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "revision": args.revision, "dirty": args.dirty,
                "series": "new-fixtures", "fixture": name, "repeat": repeat,
                "model": "qwen38-keep1-Q4_0.gguf", "mtp_draft_max": 2,
                "model_id": client.model,
                "kv": {"target": {"k": "q4_0", "v": "q4_0"},
                       "mtp": {"k": "q4_0", "v": "q4_0"}},
                "configuration_source": "parent-launch-protocol",
                "input_tokens": timings["prompt_n"], "output_tokens": len(output_ids),
                "sampling": SAMPLING, "sampling_source": "request",
                "ignore_eos": True, "prefix_reuse": False,
                "expert_cache": {"protocol_state": "cold" if ordinal == 1 else "warm",
                                 "observed_state": "unknown",
                                 "protocol": "first completion after parent server start; "
                                             "subsequent completions without cache reset"},
                "pp_ms": timings["prompt_ms"], "tg_ms": timings["predicted_ms"],
                "wall_total_ms": wall_ms, "wall_scope": "full HTTP completion including body",
                "pp_tokens_per_second": positive_number(n_in * 1000 / timings["prompt_ms"], "PP rate"),
                "tg_tokens_per_second": positive_number(n_out * 1000 / timings["predicted_ms"], "TG rate"),
                **acceptance(response, timings), "ram_bytes": None, "vram_bytes": None,
                "request_artifact": str(request_path), "response_artifact": str(response_path),
                "prompt_ids_artifact": str(directory / f"{name}.prompt-ids.json"),
                "models_response_artifact": str(models_path) if models_path.exists() else None,
            }
            line = json.dumps(result, ensure_ascii=False, allow_nan=False)
            with args.results.open("a", encoding="utf-8") as output:
                output.write(line + "\n")
            print(f"{label}: {result['input_tokens']}+{result['output_tokens']} tokens, "
                  f"PP {result['pp_tokens_per_second']:.2f}, "
                  f"TG {result['tg_tokens_per_second']:.2f} tok/s, HTTP {wall_ms / 1000:.3f}s",
                  flush=True)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="current", help="model ID for router POST requests")
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--revision", required=True, help="revision of the measured server")
    parser.add_argument("--dirty", action="store_true", help="measured server source is dirty")
    parser.add_argument("--repeats", type=int, default=1, help="R0 pairs; first request only is cold by protocol")
    parser.add_argument("--wait-seconds", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    if args.repeats < 1 or not args.revision.strip():
        parser.error("--repeats and --revision must be positive/nonempty")
    for key in ("wait_seconds", "request_timeout"):
        try:
            positive_number(getattr(args, key), key)
        except BaselineError as exc:
            parser.error(str(exc))
    try:
        run(args)
    except (BaselineError, OSError, URLError, HTTPException) as exc:
        print(f"baseline failed: {exc}. Completion requests are never retried.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
