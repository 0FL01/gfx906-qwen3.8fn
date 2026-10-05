"""Protocol2 head-capacity evidence; protocol1 floors remain unchanged."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prefill_wide_results_test as wide
import prefill_long_results_test as long
import prefill_attention_results_test as attention

HEAD = 128 * 248320 * 4
DELTA = (1024 - 128) * 248320 * 4


def objects(value, seen=None):
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list)):
        if id(value) in seen:
            return
        seen.add(id(value))
    if isinstance(value, dict):
        yield value
        for x in value.values():
            yield from objects(x, seen)
    elif isinstance(value, list):
        for x in value:
            yield from objects(x, seen)


def devices(rows):
    return [x for x in objects(rows) if {'device', 'workspace', 'owned_bytes'} <= x.keys()]


def upgrade(rows):
    rows = copy.deepcopy(rows)
    for row in rows:
        row['protocol'] = 2
    for x in objects(rows):
        if 'workspace_aggregate_min_bytes_per_device' in x:
            x['workspace_aggregate_min_bytes_per_device'] -= DELTA
        if 'individual_buffer_capacity_qualification' in x:
            x['individual_buffer_capacity_qualification'] = 'head_logits_reported_other_buffers_aggregate_floor'
    for d in devices(rows):
        d['head_logits_bytes'] = HEAD
        for key in ['workspace', 'owned_bytes', 'owned_peak_bytes']:
            d[key] -= DELTA
        d['free_vram'] += DELTA
    return rows


class HeadLogitsProtocolTest(unittest.TestCase):
    families = (wide, long, attention)

    def collect(self, family, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            return family.MODULE.collect(path)

    def test_both_versions(self):
        for family in self.families:
            with self.subTest(family=family.__name__):
                old = family.fixture()
                self.assertEqual(self.collect(family, old)['protocol'], 1)
                new = upgrade(old)
                result = self.collect(family, new)
                self.assertEqual(result['protocol'], 2)
                self.assertEqual(result['head_logit_byte_proof'],
                                 {'rows': 128, 'bytes_per_device': HEAD, 'included_in_workspace': True})
                self.assertEqual(new[0]['protocol'], 2)

    def test_missing_or_false_head_evidence(self):
        for family in self.families:
            for value in [None, 0, HEAD - 4, 1024 * 248320 * 4, str(HEAD)]:
                with self.subTest(family=family.__name__, value=value):
                    rows = upgrade(family.fixture())
                    d = devices(rows)[0]
                    if value is None:
                        del d['head_logits_bytes']
                    else:
                        d['head_logits_bytes'] = value
                    with self.assertRaises(ValueError):
                        self.collect(family, rows)

    def test_version_two_workspace_floor(self):
        for family in self.families:
            with self.subTest(family=family.__name__):
                rows = upgrade(family.fixture())
                d = devices(rows)[0]
                before = d['workspace']
                d['workspace'] = family.MODULE.workspace_floor(2) - 1
                d['owned_bytes'] += d['workspace'] - before
                with self.assertRaises(ValueError):
                    self.collect(family, rows)

    def test_legacy_floor_not_relaxed(self):
        for family in self.families:
            with self.subTest(family=family.__name__):
                rows = family.fixture()
                d = devices(rows)[0]
                d['workspace'] -= DELTA
                d['owned_bytes'] -= DELTA
                with self.assertRaises(ValueError):
                    self.collect(family, rows)
                rows = family.fixture()
                devices(rows)[0]['head_logits_bytes'] = HEAD
                with self.assertRaises(ValueError):
                    self.collect(family, rows)

    def test_mixed_unknown_boolean_versions(self):
        for family in self.families:
            for version in [1, 3, True]:
                with self.subTest(family=family.__name__, version=version):
                    rows = upgrade(family.fixture())
                    rows[-1]['protocol'] = version
                    with self.assertRaises(ValueError):
                        self.collect(family, rows)


if __name__ == '__main__':
    unittest.main()
