import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("record_kv", Path(__file__).parents[1] / "tools/record_kv.py")
record_kv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record_kv)


class KvResultsTest(unittest.TestCase):
    def fixture(self):
        kv = [{"kind": "kv_source", "revision": "a" * 40, "dirty": 1, "kv": "Q4_0"}]
        for device in (0, 1):
            for columns in (1, 2, 3, 128):
                for role in ("q", "k", "v"):
                    kv.append({"kind": "kv_gpu", "device": device, "columns": columns, "role": role,
                               "group": 64 if role == "v" else 256,
                               "elements": (24 if role == "q" else 2) * 256 * columns,
                               "repeats": 100, "resident_ms": .01, "byte_parity": True, "max_abs_error": 0})
                    if role == "k":
                        for layout in ("mx_serial", "cooperative", "mx_serial"):
                            kv.append({"kind": "kv_pack", "device": device, "columns": columns,
                                       "layout": layout, "repeats": 100, "resident_ms": .01, "byte_parity": True})
        kv.append({"kind": "kv_complete", "passed": True})
        counts = [2047, 2048, 2049, 2050, 2051, 2048, 2049, 2050, 2051, 2048]
        extras = [[], [], [], [], [], [0, 1, 2], [0, 2], [0], [], [4, 5, 6]]
        qsa = [{"diagnostic": "mx-expanded-position", "visible": v,
                "n_kv": (v + 255) // 256 * 256, "official_count": counts[i], "fork_width": min((v + 255) // 256 * 256, 2051),
                "fork_valid_count": min(v, 2051), "fork_extra_valid_ids": len(extras[i]), "extra_token_ids": extras[i]}
               for i, v in enumerate(range(2047, 2057))]
        qsa.append({"test": "qsa", "checks": 10, "rejections": 2, "chunk_prefix_queries": 3, "passed": True})
        return kv, qsa

    def collect(self, kv, qsa):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "kv.jsonl", Path(directory) / "qsa.jsonl"
            a.write_text("".join(json.dumps(x) + "\n" for x in kv))
            b.write_text("".join(json.dumps(x) + "\n" for x in qsa))
            return record_kv.collect(a, b)

    def test_complete(self):
        result = self.collect(*self.fixture())
        self.assertEqual(len(result["measurements"]), 48)
        self.assertTrue(result["dirty"])
        self.assertIn("not attention", result["scope"])

    def test_invalid(self):
        mutations = [lambda a, b: a.pop(), lambda a, b: b.pop(),
                     lambda a, b: a[1].update(byte_parity=False),
                     lambda a, b: a[1].update(resident_ms=float("nan")),
                     lambda a, b: a[1].update(resident_ms=float("inf")),
                     lambda a, b: a[1].update(resident_ms=-1),
                     lambda a, b: a[1].update(max_abs_error=.1),
                     lambda a, b: a[1].update(columns=4),
                     lambda a, b: a[1].update(group=128),
                     lambda a, b: a[3].update(layout="cooperative"),
                     lambda a, b: a[0].update(revision="unknown"),
                     lambda a, b: b[5].update(official_count=2051),
                     lambda a, b: b[5].update(extra_token_ids=[1, 2, 3]),
                     lambda a, b: b[-1].update(passed=False)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                a, b = copy.deepcopy(self.fixture())
                mutate(a, b)
                with self.assertRaises(ValueError):
                    self.collect(a, b)


if __name__ == "__main__":
    unittest.main()
