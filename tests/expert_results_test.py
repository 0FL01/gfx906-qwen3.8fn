import importlib.util
import json
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("record_expert", Path(__file__).parents[1] / "tools/record_expert.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Log:
    def __init__(self, records):
        self.records = records

    def read_text(self):
        return "\n".join(json.dumps(item) for item in self.records)

    def resolve(self):
        return Path("/fixture/r1-expert.jsonl")


def fixture():
    records = [{"kind": "expert_source", "layer": 0, "expert": 0,
                "weight_bytes": 2867200, "model": "qwen38-keep1-Q4_0.gguf",
                "revision": "fixture", "dirty": 1, "avx2": 1, "cpu_affinity": 0}]
    for columns in (1, 2, 3, 128):
        repeats = 5 if columns == 128 else 50
        for kernel in ("block_calls", "inlined_f16c", "block_calls"):
            records.append({"kind": "expert_cpu", "columns": columns, "kernel": kernel,
                            "workers": 1, "repeats": repeats, "ms": 1.0,
                            "max_abs_error": 0, "rms_error": 0})
        for device in (0, 1):
            for layout in ("canonical", "planar", "canonical"):
                records.append({"kind": "expert_gpu", "device": device, "layout": layout,
                                "columns": columns, "repeats": repeats,
                                "upload_pack_complete_ms": 0.25, "resident_ms": 0.03,
                                "completed_wall_ms": 0.04, "gate_error": 0, "up_error": 0,
                                "down_error": 0, "max_abs_error": 0, "rms_error": 0})
    records.append({"kind": "expert_complete", "passed": True})
    return records


class ExpertResultsTest(unittest.TestCase):
    def test_completed_sequence_and_scope(self):
        record = module.collect(Log(fixture()))
        self.assertEqual(len(record["measurements"]), 36)
        self.assertEqual(record["gpu_default"], "canonical")
        self.assertEqual(record["cpu_default"], "inlined_f16c")
        self.assertIn("not full-model inference", record["scope"])

    def test_invalid_results_never_become_completed_record(self):
        changes = [(0, "weight_bytes", 2764800), (0, "cpu_affinity", -1),
                   (1, "kernel", "inlined_f16c"), (1, "columns", 3),
                   (1, "max_abs_error", 0.5), (1, "ms", 0), (1, "ms", float("nan")),
                   (4, "device", 1), (4, "resident_ms", float("inf")),
                   (4, "completed_wall_ms", 0.001), (4, "down_error", -1),
                   (-1, "passed", False)]
        for index, field, value in changes:
            with self.subTest(index=index, field=field, value=value):
                records = fixture()
                records[index][field] = value
                with self.assertRaises(ValueError):
                    module.collect(Log(records))
        with self.assertRaises(ValueError):
            module.collect(Log(fixture()[:-1]))


if __name__ == "__main__":
    unittest.main()
