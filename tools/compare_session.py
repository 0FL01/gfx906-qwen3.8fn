#!/usr/bin/env python3
"""Strict, stdlib-only comparison with the pinned *operational* mx baseline.

compare_session(session_log, session_logits, session_trace, oracle_dir) returns a
JSON-serializable report, including failures (it does not raise on bad artifacts).
Logits are streamed one full vocabulary row at a time. Tensor coordinates follow
ggml order: ne[0] varies fastest, and nb contains byte strides. Capture files
already start at the view's data pointer; view_offset must NOT be applied again.
"""

import argparse
from array import array
from collections import Counter
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile


REVISION = "dcd685463d597d31f5ca759d32c94592a2740fa4"
VOCAB = 248320
MAX_TOKENS = 4096
ROW_BYTES = VOCAB * 4
LOGITS_ABS, LOGITS_REL = 2e-2, 2e-3
TENSOR_ABS, TENSOR_REL = 2e-3, 2e-3
TENSOR_LIMIT = 16 * 1024 * 1024
TOKEN_LIMIT = 64 * 1024 * 1024
JSON_LIMIT = 1024 * 1024
LOG_LIMIT = 8 * 1024 * 1024
INDEX_LIMIT = 256 * 1024 * 1024
LINE_LIMIT = 64 * 1024
LAYERS = (0, 1, 3, 47)
REQUIRED = {"hc_init": -1, **{"l_last-" + str(i): i for i in LAYERS}, "result_norm": -1}
ELEMENTS = {name: (2560 if name == "result_norm" else 10240) for name in REQUIRED}
NOTES = [
    "This is parity with the pinned production mx operational baseline, not architectural authority.",
    "Source-established GDN L2 difference: HF uses sum+1e-6; mx uses max(sum,eps^2).",
    "Source-established QSA difference: HF selects whole blocks and the actual tail; mx selects expanded positions.",
    "Cold expert CPU/GPU FP summation order may differ. Only measured bounded gates determine parity.",
    "These notes never waive a failed gate. Argmax agreement is diagnostic, not a universal bit-generation requirement.",
    "Oracle library revision is build-attested; metadata does not claim runtime revision verification.",
]


class ProtocolError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ProtocolError(message)


def _int(value, label, low=0, high=(1 << 63) - 1):
    _require(type(value) is int and low <= value <= high, label + ": invalid integer")
    return value


def _text(value, label):
    _require(type(value) is str and 0 < len(value) <= 4096 and "\0" not in value,
             label + ": invalid string")
    return value


def _equal(obj, key, value, label):
    actual = obj.get(key)
    _require(key in obj and type(actual) is type(value) and actual == value, label + "." + key + ": unexpected value")


def _numbers(obj, keys, label):
    for key in keys:
        if key in obj:
            value = obj[key]
            _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                     label + "." + key + ": invalid nonnegative number")


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        _require(key not in obj, "duplicate JSON key: " + key)
        obj[key] = value
    return obj


def _float(text):
    value = float(text)
    _require(math.isfinite(value), "non-finite JSON number")
    return value


def _constant(text):
    raise ProtocolError("non-finite JSON constant: " + text)


def _json(text, label):
    try:
        obj = json.loads(text, object_pairs_hook=_pairs, parse_float=_float, parse_constant=_constant)
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise ProtocolError(label + ": " + str(exc)) from exc
    _require(type(obj) is dict, label + ": expected a JSON object")
    return obj


def _file(path, limit, expected=None):
    path = Path(path)
    info = path.stat()
    _require(stat.S_ISREG(info.st_mode), str(path) + ": not a regular file")
    _require(info.st_size <= limit, str(path) + ": oversized artifact")
    if expected is not None:
        _require(info.st_size == expected, str(path) + ": byte count mismatch (expected " + str(expected) + ")")
    return path


def _inside(root, name):
    name = _text(name, "artifact file")
    relative = Path(name)
    _require(not relative.is_absolute() and ".." not in relative.parts, "artifact path escape: " + name)
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), "artifact path escape: " + name)
    return path


def _json_file(path):
    path = _file(path, JSON_LIMIT)
    with path.open("rb") as stream:
        data = stream.read(JSON_LIMIT + 1)
    _require(len(data) <= JSON_LIMIT, str(path) + ": oversized JSON artifact")
    return _json(data, str(path))


def _jsonl(path, limit, row_limit):
    path = _file(path, limit)
    rows = []
    total = 0
    with path.open("rb") as stream:
        while True:
            line = stream.readline(LINE_LIMIT + 1)
            if not line:
                break
            total += len(line)
            _require(len(line) <= LINE_LIMIT and total <= limit, str(path) + ": oversized JSONL artifact")
            _require(len(rows) < row_limit, str(path) + ": too many JSONL rows")
            # Blank lines/free text are invalid too; no keyword parsing or log scraping.
            rows.append(_json(line, str(path) + ":row " + str(len(rows))))
    return rows


def _session(path):
    rows = _jsonl(path, LOG_LIMIT, MAX_TOKENS + 2)
    _require(len(rows) >= 3, "Session source/tokens/complete are required")
    source, complete = rows[0], rows[-1]
    _equal(source, "kind", "session_source", "Session source")
    _equal(source, "runtime", "own_48_layer_HIP", "Session source")
    _equal(source, "trace", True, "Session source")
    _text(source.get("revision"), "Session revision")
    _text(source.get("model"), "Session model")
    _int(source.get("dirty"), "Session dirty", 0, 1)
    _int(source.get("capacity"), "Session capacity", 4, (1 << 31) - 1)
    _int(source.get("expert_slots"), "Session expert_slots", 0, 512)
    _equal(complete, "kind", "session_complete", "Session complete")
    _equal(complete, "passed", True, "Session complete")
    _equal(complete, "output_tokens", 0, "Session complete")
    tokens = rows[1:-1]
    n = len(tokens)
    _require(1 <= n <= MAX_TOKENS and n <= source["capacity"], "invalid Session token count")
    for key in ("input_tokens", "consumed_tokens"):
        _int(complete.get(key), "Session " + key, 1, MAX_TOKENS)
        _equal(complete, key, n, "Session complete")
    _numbers(complete, ("load_ms", "request_ms"), "Session complete")
    for i, row in enumerate(tokens):
        label = "Session token " + str(i)
        _equal(row, "kind", "session_token", label)
        _int(row.get("position"), label + " position", 0, n - 1)
        _equal(row, "position", i, label)
        _int(row.get("token"), label + " input ID", 0, VOCAB - 1)
        _int(row.get("argmax"), label + " argmax", 0, VOCAB - 1)
        _equal(row, "finite", True, label)
        _numbers(row, ("completed_ms",), label)
        for key in ("expert_hits", "expert_misses", "expert_upload_bytes"):
            if key in row:
                _int(row[key], label + " " + key)
    return source, tokens, complete


def _oracle(root, session_source, session_tokens):
    meta = _json_file(_inside(root, "metadata.json"))
    for key, value in (("format", "gfx906-mx-oracle-v1"), ("status", "complete"), ("error", ""),
                       ("source_revision", REVISION), ("library_revision_attested", REVISION),
                       ("library_revision_runtime_verified", False),
                       ("model_layers_actual", 48), ("vocab_actual", VOCAB), ("mtp", False), ("sampler", None)):
        _equal(meta, key, value, "oracle metadata")
    model = _text(meta.get("model"), "oracle model")
    _require(Path(session_source["model"]).name == model, "Session/oracle model mismatch")
    ids = meta.get("token_ids")
    _require(type(ids) is list and 1 <= len(ids) <= MAX_TOKENS, "invalid oracle input ID count")
    for value in ids:
        _int(value, "oracle input ID", 0, VOCAB - 1)
    n = len(ids)
    _int(meta.get("completed_tokens"), "oracle completed_tokens", 1, MAX_TOKENS)
    _equal(meta, "completed_tokens", n, "oracle metadata")
    _require(ids == [row["token"] for row in session_tokens], "Session/oracle teacher-forced input ID/count mismatch")
    context = meta.get("context")
    _require(type(context) is dict, "missing oracle context metadata")
    for key, value in (("capacity_requested", 4096), ("capacity_actual", 4096), ("batch_actual", 1),
                       ("ubatch_actual", 1), ("seq_id", 0), ("seq_max", 1), ("type_k", "q4_0"), ("type_v", "q4_0")):
        _equal(context, key, value, "oracle context")
    logits = meta.get("logits")
    _require(type(logits) is dict, "missing oracle logits metadata")
    for key, value in (("dtype", "float32"), ("endianness", "little"),
                       ("layout", "token-major,vocabulary-minor"), ("columns", VOCAB),
                       ("header_bytes", 0), ("row_bytes", ROW_BYTES), ("bytes_written", n * ROW_BYTES)):
        _equal(logits, key, value, "oracle logits")
    logits_path = _inside(root, logits.get("file"))
    callbacks = meta.get("callbacks")
    _require(type(callbacks) is dict, "missing oracle callbacks metadata")
    _equal(callbacks, "tensor_byte_limit", TENSOR_LIMIT, "oracle callbacks")
    _equal(callbacks, "token_byte_limit", TOKEN_LIMIT, "oracle callbacks")
    _int(callbacks.get("captures"), "oracle captures", n * len(REQUIRED), n * 256)
    allowlist = callbacks.get("allowlist")
    _require(type(allowlist) is list, "invalid oracle allowlist")
    for name in allowlist:
        _text(name, "oracle allowlist name")
    _require(len(allowlist) == len(set(allowlist)) and set(REQUIRED) <= set(allowlist), "invalid oracle allowlist")
    records = _jsonl(_inside(root, "tokens.jsonl"), LOG_LIMIT, MAX_TOKENS)
    _require(len(records) == n, "oracle token_records count mismatch")
    indexed = {}
    for row in records:
        i = _int(row.get("token_index"), "oracle token_index", 0, n - 1)
        _require(i not in indexed, "duplicate oracle token_record at " + str(i))
        for key, value in (("position", i), ("seq_id", 0), ("input_token_id", ids[i]),
                           ("logits_byte_offset", i * ROW_BYTES), ("logits_bytes", ROW_BYTES), ("all_finite", True)):
            _equal(row, key, value, "oracle token " + str(i))
        _int(row.get("argmax_token_id"), "oracle argmax", 0, VOCAB - 1)
        value = row.get("max_logit")
        _require(type(value) in (int, float) and math.isfinite(value), "invalid oracle max_logit")
        _int(row.get("callback_captures"), "oracle callback_captures", len(REQUIRED), 256)
        _numbers(row, ("diagnostic_decode_with_callbacks_ms",), "oracle token")
        indexed[i] = row
    return meta, [indexed[i] for i in range(n)], logits_path


def _f32(data):
    values = array("f")
    _require(values.itemsize == 4, "host float array is not F32")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _metrics(actual, reference, absolute, relative, exact=False):
    _require(len(actual) == len(reference) and len(actual) > 0, "tensor element count mismatch")
    count = len(actual)
    bad_actual = bad_reference = violations = pairs = 0
    max_abs = squared = ratio = 0.0
    worst = None
    actual_max = reference_max = -math.inf
    actual_best = reference_best = None
    for i, (value, ref) in enumerate(zip(actual, reference)):
        a_finite, r_finite = math.isfinite(value), math.isfinite(ref)
        bad_actual += not a_finite
        bad_reference += not r_finite
        if a_finite and value > actual_max:
            actual_max, actual_best = value, i
        if r_finite and ref > reference_max:
            reference_max, reference_best = ref, i
        if not (a_finite and r_finite):
            continue
        pairs += 1
        error = abs(value - ref)
        squared += error * error
        max_abs = max(max_abs, error)
        if exact:
            violations += value != ref  # numerical equality: +0 and -0 are equal
        else:
            bound_ratio = error / (absolute + relative * abs(ref))
            violations += bound_ratio > 1
            if bound_ratio > ratio:
                ratio = bound_ratio
                worst = {"element": i, "session": value, "oracle": ref}
    finite = bad_actual == bad_reference == 0
    return {"elements": count, "finite_pairs": pairs, "all_finite": finite,
            "nonfinite_session": bad_actual, "nonfinite_oracle": bad_reference,
            "max_abs": max_abs if pairs else None, "rms": math.sqrt(squared / pairs) if pairs else None,
            "max_bound_ratio": None if exact or not pairs else ratio, "violating_elements": violations,
            "worst_bound_element": worst, "passed": finite and violations == 0,
            "session_argmax": actual_best if bad_actual == 0 else None,
            "oracle_argmax": reference_best if bad_reference == 0 else None,
            "oracle_max": reference_max if bad_reference == 0 else None,
            "argmax_agreement": finite and actual_best == reference_best}


def _failure(report, category, location, message, metrics=None):
    row = {"category": category, "location": location, "message": message}
    if metrics is not None:
        row["metrics"] = metrics
    report["failures"].append(row)


def _error(report, location, exc):
    _failure(report, "protocol", location, str(exc))


def _compare_logits(report, session_path, oracle_path, tokens, records):
    n = len(tokens)
    _file(session_path, MAX_TOKENS * ROW_BYTES, n * ROW_BYTES)
    _file(oracle_path, MAX_TOKENS * ROW_BYTES, n * ROW_BYTES)
    rows = report["logits"]["rows"]
    with Path(session_path).open("rb") as actual, oracle_path.open("rb") as reference:
        for i, (token, record) in enumerate(zip(tokens, records)):
            a, r = actual.read(ROW_BYTES), reference.read(ROW_BYTES)
            _require(len(a) == len(r) == ROW_BYTES, "truncated logits row " + str(i))
            metric = _metrics(_f32(a), _f32(r), LOGITS_ABS, LOGITS_REL)
            rows.append({"token_index": i, "position": token["position"], "input_token_id": token["token"], **metric})
            if not metric["passed"]:
                _failure(report, "gate", "logits[" + str(i) + "]", "frozen logits gate failed", metric)
            if metric["session_argmax"] is not None and token["argmax"] != metric["session_argmax"]:
                _error(report, "Session token " + str(i), "reported argmax does not match binary logits")
            if metric["oracle_argmax"] is not None:
                if record["argmax_token_id"] != metric["oracle_argmax"] or record["max_logit"] != metric["oracle_max"]:
                    _error(report, "oracle token " + str(i), "reported argmax/max_logit does not match binary logits")
        _require(not actual.read(1) and not reference.read(1), "trailing logits bytes")
    pairs = sum(row["finite_pairs"] for row in rows)
    report["logits"].update({"exact_byte_counts": True, "expected_bytes_each": n * ROW_BYTES,
        "elements": n * VOCAB, "all_finite": all(row["all_finite"] for row in rows),
        "max_abs": max((row["max_abs"] for row in rows if row["max_abs"] is not None), default=None),
        "rms": math.sqrt(math.fsum(row["rms"] ** 2 * row["finite_pairs"] for row in rows if row["rms"] is not None) / pairs) if pairs else None,
        "max_bound_ratio": max((row["max_bound_ratio"] for row in rows if row["max_bound_ratio"] is not None), default=None),
        "argmax_agreeing_rows": sum(row["argmax_agreement"] for row in rows),
        "passed": all(row["passed"] for row in rows)})


def _shape(row):
    ne, nb = row.get("ne"), row.get("nb")
    _require(type(ne) is list and type(nb) is list and len(ne) == len(nb) == 4, "oracle ne/nb must have four dimensions")
    for dim, stride in zip(ne, nb):
        _int(dim, "oracle dimension", 1, TENSOR_LIMIT)
        _int(stride, "oracle stride", 1, TENSOR_LIMIT)
    size = _int(row.get("bytes"), "oracle tensor bytes", 1, TENSOR_LIMIT)
    if row["type"] in ("f32", "F32"):
        _require(math.prod(ne) <= TENSOR_LIMIT // 4, "oversized F32 element count")
        # Reject overlapping/unaligned layouts, permitting permutations and gaps.
        span = 4
        for stride, dim in sorted((stride, dim) for dim, stride in zip(ne, nb) if dim > 1):
            _require(stride % 4 == 0 and stride >= span, "invalid/overlapping F32 strides")
            span += (dim - 1) * stride
        _require(all(stride % 4 == 0 for stride in nb), "unaligned F32 stride")
        _require(size == span, "oracle F32 storage span/bytes mismatch")
    return size


def _manifest(root, path, n, tokens, oracle=False):
    rows = _jsonl(path, INDEX_LIMIT if oracle else LOG_LIMIT, n * 256)
    indexed = {}
    files = set()
    token_bytes = Counter()
    for row in rows:
        name = _text(row.get("name"), "tensor name")
        dtype = _text(row.get("type"), "tensor type")
        position = _int(row.get("position"), "tensor position", 0, n - 1)
        if oracle:
            i = _int(row.get("token_index"), "tensor token_index", 0, n - 1)
            _equal(row, "position", i, "oracle tensor")
            _equal(row, "input_token_id", tokens[i]["token"], "oracle tensor")
            _equal(row, "seq_id", 0, "oracle tensor")
            occurrence = _int(row.get("occurrence"), "tensor occurrence", 0, 255)
            for key in ("node_index", "type_id", "op_id", "view_offset"):
                if key in row:
                    _int(row[key], "oracle tensor " + key)
            size = _shape(row)
            key = (i, name, occurrence)
        else:
            _equal(row, "type", "F32", "Session tensor")
            layer = _int(row.get("layer"), "Session tensor layer", -1, 47)
            count = _int(row.get("elements"), "Session tensor elements", 1, TENSOR_LIMIT // 4)
            if name in REQUIRED:
                _equal(row, "layer", REQUIRED[name], "Session " + name)
                _equal(row, "elements", ELEMENTS[name], "Session " + name)
            if name == "hc_attn_mix":
                _require(layer in LAYERS and count == 2560, "invalid Session hc_attn_mix shape/layer")
            i, size, key = position, count * 4, (position, name, layer)
        _require(key not in indexed, "duplicate tensor row: " + str(key))
        file = _inside(root, row.get("file"))
        _require(file not in files, "duplicate tensor artifact file: " + str(file))
        _file(file, TENSOR_LIMIT, size)
        files.add(file)
        token_bytes[i] += size
        _require(token_bytes[i] <= TOKEN_LIMIT, "oversized per-token tensor captures")
        indexed[key] = (row, file)
    return rows, indexed


def _tensor(entry, oracle=False):
    row, path = entry
    size = row["bytes"] if oracle else row["elements"] * 4
    with path.open("rb") as stream:
        data = stream.read(size + 1)
    _require(len(data) == size, "tensor byte count changed: " + str(path))
    if not oracle:
        return _f32(data)
    _require(row["type"] in ("f32", "F32"), "required comparison tensor is not F32")
    ne, nb = row["ne"], row["nb"]
    result = array("f")
    for w in range(ne[3]):
        for z in range(ne[2]):
            for y in range(ne[1]):
                base = w * nb[3] + z * nb[2] + y * nb[1]
                for x in range(ne[0]):
                    result.append(struct.unpack_from("<f", data, base + x * nb[0])[0])
    return result


def _compare_traces(report, session_root, oracle_root, meta, tokens, records):
    n = len(tokens)
    session_rows, session = _manifest(session_root, _inside(session_root, "tensors.jsonl"), n, tokens)
    oracle_rows, oracle = _manifest(oracle_root, _inside(oracle_root, meta["callbacks"].get("records")), n, tokens, True)
    callbacks = meta["callbacks"]
    _equal(callbacks, "captures", len(oracle_rows), "oracle callbacks")
    counts = dict(Counter(row["name"] for row in oracle_rows))
    observed = callbacks.get("observed_counts")
    _require(type(observed) is dict, "missing oracle observed_counts")
    for value in observed.values():
        _int(value, "oracle observed count", 1, n * 256)
    _require(observed == counts, "oracle observed_counts mismatch")
    _require(set(counts) <= set(callbacks["allowlist"]), "oracle tensor outside exact allowlist")
    per_token = Counter(row["token_index"] for row in oracle_rows)
    for i, record in enumerate(records):
        _equal(record, "callback_captures", per_token[i], "oracle token " + str(i))
    # Different occurrence numbers do not make duplicate required captures valid.
    required_counts = Counter((row["token_index"], row["name"]) for row in oracle_rows if row["name"] in REQUIRED)
    for i in range(n):
        for name, layer in REQUIRED.items():
            location = "tensor[" + str(i) + "]." + name
            if required_counts[i, name] != 1:
                _error(report, location, "missing/duplicate required oracle tensor")
            _pair(report, location, i, name, layer, session.get((i, name, layer)), oracle.get((i, name, 0)), False)
        for layer in LAYERS:
            name = "hc_mixed-" + str(layer)
            location = "tensor[" + str(i) + "].hc_attn_mix-" + str(layer)
            _pair(report, location, i, "hc_attn_mix", layer,
                  session.get((i, "hc_attn_mix", layer)), oracle.get((i, name, 0)), True)
    compared = report["intermediates"] + report["optional_intermediates"]
    used_session = {(row["token_index"], row["name"], row["layer"]) for row in compared if row["status"] == "compared"}
    used_oracle = {(row["token_index"], row["oracle_name"], row["oracle_occurrence"]) for row in compared if row["status"] == "compared"}
    # Missing optional counterparts do not excuse corrupt captures. Check other
    # F32 diagnostics too, but never interpret padding bytes as tensor elements.
    for side, entries, used, is_oracle in (("session", session, used_session, False), ("oracle", oracle, used_oracle, True)):
        for key, entry in entries.items():
            if key in used or entry[0]["type"] not in ("f32", "F32"):
                continue
            try:
                _require(all(math.isfinite(value) for value in _tensor(entry, is_oracle)), "non-finite F32 diagnostic tensor")
            except (OSError, ValueError, OverflowError) as exc:
                _error(report, side + " tensor " + str(key), exc)
    report["trace_counts"] = {"session": len(session_rows), "oracle": len(oracle_rows)}


def _pair(report, location, i, name, layer, session, oracle, optional):
    results = report["optional_intermediates"] if optional else report["intermediates"]
    row = {"token_index": i, "position": i, "name": name, "layer": layer, "optional": optional}
    if session is None or oracle is None:
        row.update(status="missing_optional" if optional else "missing_required", missing=[
            side for side, entry in (("session", session), ("oracle", oracle)) if entry is None])
        results.append(row)
        if not optional:
            _error(report, location, "missing required tensor: " + ",".join(row["missing"]))
        return
    try:
        actual, reference = _tensor(session), _tensor(oracle, True)
        expected = 2560 if optional else ELEMENTS[name]
        _require(len(actual) == len(reference) == expected, "required tensor geometry mismatch")
        exact = name == "hc_init"
        metric = _metrics(actual, reference, TENSOR_ABS, TENSOR_REL, exact)
        row.update(status="compared", oracle_name=oracle[0]["name"], oracle_occurrence=oracle[0]["occurrence"],
                   gate="exact_numerical_float" if exact else "bounded", **metric)
        if not metric["passed"]:
            _failure(report, "gate", location, "frozen intermediate gate failed", metric)
    except (OSError, ValueError, OverflowError) as exc:
        row.update(status="invalid", passed=False)
        _error(report, location, exc)
    results.append(row)


def compare_session(session_log, session_logits, session_trace, oracle_dir):
    """Return all measured gates and structured protocol/gate failures; no sampling."""
    report = {"kind": "session_comparison", "format": "gfx906-session-compare-v1",
              "scope": "teacher_forced_operational_baseline", "baseline_revision": REVISION,
              "gates": {"logits": {"absolute": LOGITS_ABS, "relative": LOGITS_REL},
                        "intermediates": {"absolute": TENSOR_ABS, "relative": TENSOR_REL},
                        "hc_init": "exact_numerical_float"},
              "artifacts": {"session_log": str(session_log), "session_logits": str(session_logits),
                            "session_trace": str(session_trace), "oracle_dir": str(oracle_dir)},
              "notes": list(NOTES), "logits": {"rows": [], "passed": False},
              "intermediates": [], "optional_intermediates": [], "failures": [], "passed": False}
    try:
        source, tokens, complete = _session(session_log)
        meta, records, oracle_logits = _oracle(Path(oracle_dir), source, tokens)
        report.update(tokens=len(tokens), session_source=source, session_complete=complete,
                      oracle_metadata=meta, input_token_ids=[row["token"] for row in tokens])
    except (OSError, ValueError, OverflowError) as exc:
        _error(report, "input_protocol", exc)
        return report
    for location, function, args in (
        ("logits", _compare_logits, (session_logits, oracle_logits, tokens, records)),
        ("traces", _compare_traces, (Path(session_trace), Path(oracle_dir), meta, tokens, records)),
    ):
        try:
            function(report, *args)
        except (OSError, ValueError, OverflowError) as exc:
            _error(report, location, exc)
    report["passed"] = not report["failures"]
    return report


def _write_output(path, text):
    # One atomic replacement, with no partial report left at the destination.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-log", type=Path, required=True)
    parser.add_argument("--session-logits", type=Path, required=True)
    parser.add_argument("--session-trace", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = compare_session(args.session_log, args.session_logits, args.session_trace, args.oracle_dir)
    if args.output is not None:
        try:
            destination = args.output.resolve()
            _require(destination not in (args.session_log.resolve(), args.session_logits.resolve()) and
                     not destination.is_relative_to(args.session_trace.resolve()) and
                     not destination.is_relative_to(args.oracle_dir.resolve()), "output would overwrite an input artifact")
            _write_output(args.output, json.dumps(report, allow_nan=False, indent=2) + "\n")
        except (OSError, ValueError) as exc:
            _error(report, "output", exc)
            report["passed"] = False
    print(json.dumps(report, allow_nan=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
