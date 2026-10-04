"""Synthetic faithful 31-record emitter fixtures; no model/HIP or real journal."""

import contextlib
import copy
import datetime
import io
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import record_memory
from tools import record_prefill_attention as MODULE

SCRIPT = ROOT / 'tools/record_prefill_attention.py'
REVISION = '0123456789abcdef0123456789abcdef01234567'
VOCAB, Q40, Q41 = 248320, 2764800, 2867200
PHASES = ('reference_old_n1', 'enabled1024', 'occupied5_enabled997')
ATTENTION = ('batch_calls', 'query_rows', 'multiquery_calls', 'multiquery_rows',
             'singleton_tail_calls', 'max_query_rows')
CATEGORIES = ('weights', 'expert_slots', 'qsa_kv', 'qsa_index', 'gdn_state', 'ple_state', 'workspace')


def empty_state(enabled):
    return {'stats': {'consumed_tokens': 0, 'expert_hits': 0, 'expert_misses': 0,
                      'expert_upload_bytes': 0, 'last_completed_ms': 0, 'last_completed_ms_bits': 0},
            'route_stats': {'last_max_expert_group_assignments': 0, 'expert_groups_gt128': 0},
            'attention_stats': dict.fromkeys(ATTENTION, 0), 'attention_batch': enabled,
            'hybrid_disabled_stats_all_zero': True, 'probe_off_publication_empty': True}


def error_fixture(offset, n, compared):
    return {'values': n * VOCAB, 'finite': n * VOCAB, 'compared': n * VOCAB if compared else 0,
            'finite_pairs': n * VOCAB if compared else 0, 'nonfinite_actual': 0, 'nonfinite_reference': 0,
            'violations': 0, 'bit_mismatches': 0, 'argmax_rows': n if compared else 0,
            'argmax_agree': n if compared else 0, 'maxabs': 0, 'rms': 0, 'maxboundratio': 0,
            'maxabs_row': offset if compared else 0, 'maxabs_column': 0,
            'maxratio_row': offset if compared else 0, 'maxratio_column': 0,
            'first_violation_row': None, 'first_violation_column': None}


def memory_fixture():
    """Synthetic variable weights/counts, not claimed measured device allocations."""
    devices = []
    for i in (0, 1):
        d = {'device': i, 'first_layer': i * 24, 'last_layer': i * 24 + 23,
             'gdn_layers': 18, 'qsa_layers': 6, 'weights': 1100000000 + i * 12345678,
             'expert_slots': [66969600, 66355200][i], 'qsa_kv': 7216128, 'qsa_index': 1612800,
             'gdn_state': 58834944, 'ple_state': 368640 if i == 0 else 0,
             'workspace': 2500000000 + i * 43210, 'owned_buffers': 1001 + i * 37,
             'total_vram': 17163091968,
             'attention': {'gathered_key_bytes': 16801792, 'gathered_value_bytes': 16801792,
                           'partial_output_bytes': 6488064, 'partial_max_sum_bytes': 50688,
                           'staged_buffer_bytes': 50331648, 'output_buffer_bytes': 50331648,
                           'selected_id_bytes': 65632, 'selected_block_bytes': 16384,
                           'selected_count_bytes': 64, 'private_workspace_bytes': 40338944}}
        d['owned_bytes'] = sum(d[k] for k in CATEGORIES)
        d['owned_peak_bytes'] = d['owned_bytes'] + 4096
        d['free_vram'] = d['total_vram'] - d['owned_bytes'] - 67108864
        devices.append(d)
    return {'capacity': 2088, 'expert_slots': 1, 'attention_query_tile': 8, 'ownership_verified': True,
            'ram_expert_capacity': 68262301696, 'ram_expert_payload': 68262297600,
            'host_embedding_capacity': 521472000, 'host_logit_capacity': 1017118720,
            'pinned_handoff': 41943040, 'expert_payload_reads': 144,
            'expert_payload_bytes_read': 68262297600, 'pinned_expert_staging': 183500800,
            'pinned_hybrid_input': 0, 'pinned_hybrid_output': 0, 'pinned_hybrid_error': 0,
            'host_hybrid_plans': 0, 'host_cpu_expert_views': 0, 'cpu_pool_metadata': 0,
            'cpu_pool_scratch': 0, 'cpu_workers': 0, 'host_hybrid_probe': 0,
            'host_routing_capacity': 4358144, 'host_route_group_payload': 340064,
            'pinned_route_metadata': 163840, 'host_hybrid_input_probe': 0, 'pinned_hybrid_middle': 0,
            'hybrid_contribution_bytes': [0, 0], 'expert_stage_capacity_bytes': [45875200, 45875200],
            'route_metadata_bytes': [81920, 81920], 'hybrid_gate_up_bytes': [0, 0],
            'hybrid_middle_float_bytes': [0, 0], 'hybrid_middle_q8_bytes': [0, 0],
            'hybrid_middle_error_bytes': [0, 0], 'devices': devices}


def fixture():
    """Independent transcription of C++ main and its emitted scalar summaries.

    Routing is a valid synthetic slot1 schedule: ten repeated experts/layer
    within a multirow call; all first acquisitions miss. N1 summaries inspect
    each call immediately and publish only their final state. No tensors exist.
    """
    rows = []

    def record(kind, index=None, **fields):
        row = {'protocol': 1, 'kind': 'prefill_attention_' + kind, 'record_index': len(rows),
               'phase_index': index, 'phase': PHASES[index] if index is not None else None,
               **copy.deepcopy(fields), 'passed': True}
        rows.append(row)
        return row

    record('source', revision=REVISION, dirty=1,
           model='/models/данные "quoted"/qwen38-keep1-Q4_0.gguf', model_bytes=75399121792,
           config={'capacity': 2088, 'expert_slots': 1, 'max_batch_tokens': 1024, 'cpu_workers': 0,
                   'hybrid_probe': False, 'trace_directory': '', 'attention_query_tile': 8, 'initial_attention_batch': False},
           source_ids={'bos': 248044, 'formula_after_bos': '99+absolute_position', 'teacher_rows': 2056,
                       'teacher_last': 2154, 'continuation_rows': 32, 'continuation_first': 2155, 'continuation_last': 2186},
           schedules=[{'phase': PHASES[0], 'teacher_calls': 2056, 'teacher_call_rows': 1, 'continuation_calls': 32},
                      {'phase': PHASES[1], 'teacher_chunks': [1024, 1024, 8], 'continuation_calls': 32},
                      {'phase': PHASES[2], 'prefix_n1_calls': 5, 'teacher_chunks': [997, 997, 57], 'continuation_calls': 32}],
           gate={'absolute': .02, 'relative': .002, 'formula': 'abs(actual-reference)<=.02+.002*abs(reference)',
                 'all_vocabulary': True, 'immediate': True, 'bit_identity_required': False, 'argmax_identity_required': False},
           reference_bytes=2073968640, preservation_bytes=1017118720, guard_bytes=256,
           private_workspace_bytes_per_device=40338944, selection_bytes_per_device=82080, expected_success_records=31,
           scope={'reference': 'same-session old N1', 'component_gate_required_separately': True,
                  'actual_selected_id_trace': False, 'actual_visibility_trace': False, 'hf_oracle': False,
                  'performance_qualification': False, 'long4k16k': False,
                  'logical_row_coordinate': 'zero-based absolute source position',
                  'expected_visibility': 'position+1; computed from accepted rows, not GPU-observed'})
    snapshots, preserved = 0, 0
    loaded = memory_fixture()

    def observe(calls=1, values=0):
        nonlocal snapshots, preserved
        snapshots += calls
        preserved += values

    def ledger():
        m = copy.deepcopy(loaded)
        for d in m['devices']:
            d['free_vram'] -= snapshots * 1024
        return {'memory_snapshot_index': snapshots - 1, 'memory': m}

    def mem(event, index, enabled):
        observe()
        record('memory', index, event=event, state=empty_state(enabled), **ledger())

    mem('loaded', None, False)
    phases = []
    schedules = [[(0, 2056, True, 'teacher'), (2056, 32, True, 'continuation')],
                 [(0, 1024, False, 'teacher'), (1024, 1024, False, 'teacher'), (2048, 8, False, 'teacher'),
                  (2056, 31, True, 'continuation'), (2087, 1, True, 'continuation_final')],
                 [(0, 5, True, 'occupied_prefix'), (5, 997, False, 'teacher'), (1002, 997, False, 'teacher'),
                  (1999, 57, False, 'teacher'), (2056, 31, True, 'continuation'), (2087, 1, True, 'continuation_final')]]
    for index, schedule in enumerate(schedules):
        mem('reset', index, bool(index))
        previous, teacher, calls, windows = empty_state(bool(index)), None, 0, []
        for position, (offset, n, singles, segment) in enumerate(schedule):
            call_count = n if singles else 1
            misses, hits = (480 * n if singles else 480), (0 if singles else 480 * (n - 1))
            upload = (misses - misses // 8) * Q40 + misses // 8 * Q41
            after = copy.deepcopy(previous)
            s = after['stats']
            s.update(consumed_tokens=offset + n, expert_hits=s['expert_hits'] + hits,
                     expert_misses=s['expert_misses'] + misses, expert_upload_bytes=s['expert_upload_bytes'] + upload,
                     last_completed_ms=12.5 + position + index)
            s['last_completed_ms_bits'] = struct.unpack('!Q', struct.pack('!d', s['last_completed_ms']))[0]
            after['route_stats']['last_max_expert_group_assignments'] = 1 if singles else n
            after['route_stats']['expert_groups_gt128'] += 480 if not singles and n > 128 else 0
            if index and not singles:
                increments = {1024: (1536, 12288, 1536, 12288, 0), 8: (12, 96, 12, 96, 0),
                              997: (1500, 11964, 1500, 11964, 0), 57: (96, 684, 84, 672, 12)}[n]
                for k, inc in zip(ATTENTION[:-1], increments):
                    after['attention_stats'][k] += inc
                after['attention_stats']['max_query_rows'] = 8
            mod4 = [sum((p + 1) % 4 == r for p in range(offset + n)) for r in range(4)]
            boundary = sum(2047 <= p <= 2056 for p in range(offset + n))
            observe(call_count, n * VOCAB)
            w = record('window', index, window_index=position, offset=offset, rows=n, completed_rows=n,
                       completed_calls=call_count, segment=segment, expected_routes=480 * n,
                       route_hits_delta=hits, route_misses_delta=misses, route_upload_bytes_delta=upload,
                       state_before=previous, state_after=after, errors=error_fixture(offset, n, bool(index)),
                       accepted_rows_expected_visible_mod4=mod4,
                       accepted_rows_logical2047_through2056=boundary, **ledger())
            previous = after
            calls += call_count
            windows.append(w)
            if offset + n == 2056:
                teacher = copy.deepcopy(previous)
            if index and offset == (0 if index == 1 else 5) and segment == 'teacher':
                observe(3, 3 * n * VOCAB)
                cases = [('late_negative_id', 1024, -1), ('late_vocab_id', 1024, VOCAB), ('oversized_1025', 1025, 100)]
            elif index and segment == 'continuation':
                cases = [('near_capacity_2_with_1_remaining', 2, 100)]
            else:
                cases = []
            for reason, attempted, token in cases:
                values = (1 if singles else n) * VOCAB
                observe(2, 2 * values)
                record('rejection', index, reason=reason, attempted_rows=attempted, last_token_id=token,
                       remaining_capacity=2088 - offset - n, preserved_full_span_values=values,
                       state_before=previous, state_after=previous, rejected=True, full_span_bit_preserved=True, **ledger())
        phases.append(record('phase', index, accepted_rows=2088, completed_calls=calls, windows=len(windows),
                             finite_values=518492160, compared_values=518492160 if index else 0,
                             bit_mismatches=0, argmax_agree=2088 if index else 0,
                             accepted_rows_expected_visible_mod4=[522] * 4, accepted_rows_logical2047_through2056=10,
                             rejections=4 if index else 0, toggle_checks=3 if index else 0,
                             toggle_preserved_full_span_rows=[0, 1024, 997][index], teacher_state=teacher,
                             final_state=previous, cache_retained_by_reset=True, cold_or_warm_claim='unknown'))
    mem('final_reset', None, True)
    minimum = [min(r['memory']['devices'][i]['free_vram'] for r in rows if 'memory' in r) - 1 for i in (0, 1)]
    record('complete', records=31, phases=3, windows=13, completed_calls=2163, accepted_rows=6264,
           finite_values=1555476480, compared_values=1036984320, rejections=8, toggle_checks=6,
           accepted_rows_logical2047_through2056=30,
           attention_stats=dict(zip(ATTENTION, (6180, 49284, 6168, 49272, 12, 8))),
           memory_snapshots=snapshots, preserved_values=preserved, minimum_free_vram=minimum,
           session_constructed=True, all_owners_unwound=True, reference='same-session old N1',
           physical_kernel_count_claim=False, selected_id_trace_claim=False,
           hf_oracle_claim=False, performance_qualification=False)
    return rows


def encoded(rows):
    return ''.join(json.dumps(r, allow_nan=False, separators=(',', ':')) + '\n' for r in rows).encode()


def target(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def objects(obj, path=()):
    if type(obj) is dict:
        yield path, obj
        for k, v in obj.items():
            yield from objects(v, (*path, k))
    elif type(obj) is list:
        for i, v in enumerate(obj):
            yield from objects(v, (*path, i))


def leaves(obj, path=()):
    if type(obj) is dict:
        for k, v in obj.items():
            yield from leaves(v, (*path, k))
    elif type(obj) is list:
        for i, v in enumerate(obj):
            yield from leaves(v, (*path, i))
    else:
        yield path, obj


class PrefillAttentionResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = fixture()
        cls.original_raw = encoded(cls.original)

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='prefill-attention-results-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw, self.results = self.root / 'attention.jsonl', self.root / 'results.jsonl'
        self.rows = copy.deepcopy(self.original)
        self.raw.write_bytes(self.original_raw)

    def indices(self, kind):
        return [i for i, r in enumerate(self.rows) if r['kind'] == 'prefill_attention_' + kind]

    def representatives(self):
        return [0, 1, *self.indices('phase'), self.indices('window')[0],
                self.indices('window')[2], self.indices('rejection')[0], 29, 30]

    def collect(self, rows=None):
        if rows is not None:
            self.raw.write_bytes(encoded(rows))
        return MODULE.collect(self.raw)

    def reject(self, rows):
        self.raw.write_bytes(encoded(rows))
        with self.assertRaises(ValueError):
            self.collect()
        self.assertEqual(self.raw.read_bytes(), encoded(rows))

    def change(self, i, path, bad):
        obj = target(self.rows[i], path[:-1])
        old = obj[path[-1]]
        obj[path[-1]] = bad
        try:
            with self.subTest(record=i, path=path, bad=bad):
                self.reject(self.rows)
        finally:
            obj[path[-1]] = old

    def cli(self, destination=None, args=None):
        command = [sys.executable, '-B', str(SCRIPT)]
        command += args if args is not None else ['--raw', self.raw.name, '--results', str(destination or self.results)]
        return subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)

    def test_valid_exact_counts_provenance_compact_aggregate_and_no_writes(self):
        result = self.collect()
        self.assertEqual(len(self.rows), 31)
        self.assertEqual(result['kind'], 'r4_attention_prefill')
        self.assertEqual(result['protocol'], 1)
        self.assertEqual(result['revision'], REVISION)
        self.assertIs(result['dirty'], True)
        self.assertEqual(result['source']['dirty'], 1)
        self.assertEqual(result['source'], self.rows[0])
        self.assertEqual(result['phases'], [self.rows[i] for i in self.indices('phase')])
        self.assertEqual(result['complete'], self.rows[-1])
        self.assertNotIn('windows', result)
        self.assertNotIn('records', result)
        self.assertEqual(len(result['memory']), 5)
        self.assertEqual(len(result['rejections']), 8)
        self.assertEqual(result['raw_logs'], {'prefill_attention': str(self.raw.resolve())})
        self.assertEqual(datetime.datetime.fromisoformat(result['timestamp']).utcoffset(), datetime.timedelta(0))
        self.assertEqual([result['complete'][k] for k in ('windows', 'completed_calls', 'accepted_rows',
                          'rejections', 'toggle_checks', 'memory_snapshots', 'preserved_values')],
                         [13, 2163, 6264, 8, 6, 2190, 6073162240])
        self.assertEqual(result['window_summary']['finite_values'], 1555476480)
        self.assertEqual(result['window_summary']['compared_values'], 1036984320)
        self.assertEqual(result['memory_evidence_limits']['serialized_ledgers'], 26)
        self.assertEqual(self.raw.read_bytes(), self.original_raw)
        self.assertFalse(self.results.exists())
        json.dumps(result, allow_nan=False)

    def test_every_top_and_nested_schema_closed_missing_unknown(self):
        for i in self.representatives():
            for path, obj in list(objects(self.rows[i])):
                obj['unknown'] = 1
                try:
                    self.reject(self.rows)
                finally:
                    del obj['unknown']
                for k in list(obj):
                    old = obj.pop(k)
                    try:
                        with self.subTest(record=i, path=path, missing=k):
                            self.reject(self.rows)
                    finally:
                        obj[k] = old

    def test_typed_flags_nulls_integers_and_unsigned_overflow(self):
        for i in self.representatives():
            for path, value in leaves(self.rows[i]):
                if type(value) is bool:
                    self.change(i, path, int(value))
                    self.change(i, path, not value)
                elif value is None:
                    self.change(i, path, 0)
                elif type(value) is int and path[-1] not in ('last_completed_ms', 'maxabs', 'rms', 'maxboundratio'):
                    self.change(i, path, float(value))
                    self.change(i, path, True)
                    self.change(i, path, 1 << 64)

    def test_source_revision_dirty_model_ids_fixed_gate_and_no_invented_metadata(self):
        for k, bad in (('revision', 'g' * 40), ('revision', 'f' * 39), ('revision', 1),
                       ('dirty', True), ('dirty', 1.), ('dirty', 2), ('dirty', -1),
                       ('model', 'qwen38-keep1-Q4_0.gguf'), ('model', '/models/full-PLE.gguf'),
                       ('model', '/models/qwen38-keep1-Q8_0.gguf'),
                       ('model', '/models/\x00/qwen38-keep1-Q4_0.gguf'), ('model_bytes', 0), ('model_bytes', 1 << 63)):
            self.change(0, (k,), bad)
        for path, value in leaves(self.rows[0]):
            if path[0] not in ('revision', 'dirty', 'model', 'model_bytes'):
                bad = value + 1 if type(value) in (int, float) else ('changed' if type(value) is str else 1)
                self.change(0, path, bad)
        self.rows[0].update(revision='ABCDEF01' * 5, dirty=0)
        result = self.collect(self.rows)
        self.assertEqual(result['revision'], 'ABCDEF01' * 5)
        self.assertIs(result['dirty'], False)
        self.assertNotIn('actual_metadata', result['source'])
        self.assertIn('actual_model_metadata_tensor_dimensions_offsets_and_types',
                      result['memory_evidence_limits']['not_serialized'])

    def test_record_order_index_phase_window_chronology_truncation_duplicate(self):
        for i in range(31):
            self.change(i, ('record_index',), i + 1)
            self.reject(self.rows[:i] + self.rows[i + 1:])
            self.reject(self.rows[:i] + [self.rows[i]] + self.rows[i:])
            if i:
                reordered = copy.deepcopy(self.rows)
                reordered[i], reordered[i - 1] = reordered[i - 1], reordered[i]
                # Also relabel indices: kind/phase/order still must fail.
                reordered[i]['record_index'], reordered[i - 1]['record_index'] = i, i - 1
                self.reject(reordered)
        for i in self.indices('window'):
            for k in ('window_index', 'offset', 'rows', 'completed_rows', 'completed_calls'):
                self.change(i, (k,), self.rows[i][k] + 1)
            self.change(i, ('phase',), 'wrong')
            self.change(i, ('phase_index',), 3)
            self.change(i, ('segment',), 'wrong')
        for n in range(31):
            self.reject(self.rows[:n])

    def test_full_cell_finite_counts_nonfinite_violation_first_error_and_coordinates(self):
        for i in self.indices('window'):
            for k in ('values', 'finite', 'compared', 'finite_pairs', 'nonfinite_actual',
                      'nonfinite_reference', 'violations', 'argmax_rows'):
                self.change(i, ('errors', k), self.rows[i]['errors'][k] + 1)
            self.change(i, ('errors', 'first_violation_row'), self.rows[i]['offset'])
            self.change(i, ('errors', 'first_violation_column'), 0)
            self.change(i, ('errors', 'bit_mismatches'), self.rows[i]['errors']['compared'] + 1)
            if self.rows[i]['phase_index']:
                for k in ('maxabs_row', 'maxratio_row'):
                    self.change(i, ('errors', k), self.rows[i]['offset'] - 1)
                    self.change(i, ('errors', k), self.rows[i]['offset'] + self.rows[i]['rows'])
                self.change(i, ('errors', 'maxabs_column'), VOCAB)
                self.change(i, ('errors', 'maxratio_column'), -1)

    def positive_error(self, i, maximum=.015625, ratio=.78125, bits=1, agree=None):
        e = self.rows[i]['errors']
        e.update(maxabs=maximum, maxboundratio=ratio, rms=maximum / math.sqrt(e['finite_pairs']), bit_mismatches=bits)
        if agree is not None:
            e['argmax_agree'] = agree
        phase = next(p for p in self.rows if p['kind'].endswith('_phase') and p['phase_index'] == self.rows[i]['phase_index'])
        windows = [r for r in self.rows if r['kind'].endswith('_window') and r['phase_index'] == phase['phase_index']]
        phase['bit_mismatches'] = sum(w['errors']['bit_mismatches'] for w in windows)
        phase['argmax_agree'] = sum(w['errors']['argmax_agree'] for w in windows)

    def test_bits_and_argmax_diagnostic_nonzero_errors_and_signed_zero_allowed(self):
        i = self.indices('window')[2]
        self.positive_error(i, agree=1023)
        result = self.collect(self.rows)
        self.assertEqual(result['window_summary']['bit_mismatches_diagnostic'], 1)
        self.assertFalse(result['window_summary']['bit_identity_required'])
        self.assertFalse(result['window_summary']['argmax_identity_required'])
        self.assertEqual(result['window_summary']['argmax_agree_diagnostic'], 4175)
        self.positive_error(i, maximum=0, ratio=0, bits=100, agree=1024)
        self.assertEqual(self.collect(self.rows)['window_summary']['bit_mismatches_diagnostic'], 100)

    def test_frozen_ratio_one_boundary_and_nonzero_envelope_no_invented_abs_only_gate(self):
        i = self.indices('window')[2]
        self.positive_error(i, maximum=.02, ratio=1)
        self.assertTrue(self.collect(self.rows)['passed'])
        self.change(i, ('errors', 'maxboundratio'), math.nextafter(1., math.inf))
        # A reference100 permits .22 error. maxabs alone cannot imply .02.
        self.positive_error(i, maximum=.125, ratio=.125 / .22)
        self.assertTrue(self.collect(self.rows)['passed'])
        for k, bad in (('maxabs', True), ('maxabs', -1), ('rms', 0), ('rms', .2),
                       ('maxboundratio', 0), ('bit_mismatches', 0), ('argmax_agree', 1025)):
            self.change(i, ('errors', k), bad)
        self.positive_error(i)
        self.change(i, ('errors', 'maxboundratio'), .8)
        self.change(i, ('errors', 'rms'), .015625 / math.sqrt(1024 * VOCAB) / 2)

    def test_reference_uncompared_metrics_are_zero_coordinates_not_null(self):
        for i in self.indices('window')[:2]:
            for k in ('maxabs', 'rms', 'maxboundratio', 'maxabs_row', 'maxabs_column', 'maxratio_row', 'maxratio_column'):
                self.change(i, ('errors', k), None)
                self.change(i, ('errors', k), 1)

    def test_phase_sums_teacher_final_and_complete_exact_totals(self):
        for i in self.indices('phase'):
            for k in ('accepted_rows', 'completed_calls', 'windows', 'finite_values', 'compared_values',
                      'bit_mismatches', 'argmax_agree', 'rejections', 'toggle_checks', 'toggle_preserved_full_span_rows'):
                self.change(i, (k,), self.rows[i][k] + 1)
            self.change(i, ('teacher_state', 'stats', 'consumed_tokens'), 2055)
            self.change(i, ('final_state', 'stats', 'consumed_tokens'), 2087)
        for k, value in self.rows[-1].items():
            if type(value) is int:
                self.change(30, (k,), value + 1)

    def test_attention_completed_API_counters_every_window_rejection_phase_reset(self):
        for i in range(1, 31):
            row = self.rows[i]
            for path, obj in list(objects(row)):
                if path and path[-1] == 'attention_stats':
                    for k in ATTENTION:
                        self.change(i, (*path, k), obj[k] + 1)
        phases = [self.rows[i] for i in self.indices('phase')]
        self.assertEqual([list(p['final_state']['attention_stats'].values()) for p in phases],
                         [[0] * 6, [3084, 24672, 3084, 24672, 0, 8], [3096, 24612, 3084, 24600, 12, 8]])
        self.assertEqual(self.rows[-1]['attention_stats'], dict(zip(ATTENTION, (6180, 49284, 6168, 49272, 12, 8))))

    def test_stats_route480_payload_monotone_timing_bits_group_threshold(self):
        for i in self.indices('window'):
            for k in ('route_hits_delta', 'route_misses_delta', 'route_upload_bytes_delta'):
                self.change(i, (k,), self.rows[i][k] + 1)
            for k in ('expert_hits', 'expert_misses', 'expert_upload_bytes', 'last_completed_ms_bits'):
                self.change(i, ('state_after', 'stats', k), self.rows[i]['state_after']['stats'][k] + 1)
            for k in ('last_max_expert_group_assignments', 'expert_groups_gt128'):
                self.change(i, ('state_after', 'route_stats', k), 1 << 64)
            self.change(i, ('state_after', 'stats', 'last_completed_ms'), 0)
        i = self.indices('window')[2]
        self.change(i, ('state_after', 'route_stats', 'last_max_expert_group_assignments'), 128)
        self.change(i, ('state_after', 'route_stats', 'expert_groups_gt128'), 0)
        misses = self.rows[i]['route_misses_delta']
        self.change(i, ('state_after', 'stats', 'expert_upload_bytes'), misses * Q40 - 1)
        self.change(i, ('state_after', 'stats', 'expert_upload_bytes'), misses * Q41 + 1)

    def test_rejections_preserve_all_publication_and_last_N1_span_not31rows(self):
        for i in self.indices('rejection'):
            for k in ('attempted_rows', 'last_token_id', 'remaining_capacity', 'preserved_full_span_values'):
                self.change(i, (k,), self.rows[i][k] + 1)
            self.change(i, ('reason',), 'wrong')
            for path in (('state_after', 'attention_batch'), ('state_after', 'hybrid_disabled_stats_all_zero')):
                self.change(i, path, False)
            self.change(i, ('state_after', 'stats', 'last_completed_ms_bits'), 0)
        late = [self.rows[i] for i in self.indices('rejection') if self.rows[i]['reason'].startswith('near_capacity')]
        self.assertEqual([r['preserved_full_span_values'] for r in late], [VOCAB, VOCAB])
        self.assertEqual([r['remaining_capacity'] for r in late], [1, 1])
        for i in self.indices('memory'):
            enabled = self.rows[i]['state']['attention_batch']
            self.change(i, ('state', 'attention_batch'), not enabled)

    def test_memory_snapshot_indices_hidden_toggles_and_preserved_values(self):
        for i, row in enumerate(self.rows):
            if 'memory_snapshot_index' in row:
                self.change(i, ('memory_snapshot_index',), row['memory_snapshot_index'] + 1)
        self.assertEqual([self.rows[i]['memory_snapshot_index'] for i in self.indices('memory')],
                         [0, 1, 2090, 2137, 2189])
        self.change(30, ('memory_snapshots',), 2163 + 5 + 16)  # Omits6 toggles.
        self.change(30, ('preserved_values',), 6264 * VOCAB)  # Omits preservation reads.

    def test_memory_all_actual_attention_backing_private_prefix_selection_bytes(self):
        result = self.collect()
        b = result['attention_byte_proof']
        self.assertEqual(b['private_workspace_bytes_per_query'], 5042368)
        self.assertEqual(b['private_workspace_bytes_per_device'], 40338944)
        self.assertEqual(b['private_buffer_allocations_bytes_per_device'], 40142336)
        self.assertEqual(b['staged_reused_prefix_bytes_per_device'], 196608)
        self.assertEqual(b['selection_bytes_per_device'], 82080)
        self.assertEqual(b['staged_actual_backing_bytes_per_device'], 50331648)
        self.assertEqual(b['output_actual_backing_bytes_per_device'], 50331648)
        for i, row in enumerate(self.rows):
            if 'memory' in row:
                for d in (0, 1):
                    for k, v in row['memory']['devices'][d]['attention'].items():
                        self.change(i, ('memory', 'devices', d, 'attention', k), v + 1)
        self.change(1, ('memory', 'devices', 0, 'attention', 'staged_buffer_bytes'), 196608)
        self.change(1, ('memory', 'devices', 0, 'attention', 'private_workspace_bytes'), 40338944 + 82080)

    def test_memory_owner_category_geometry_capacity_reads_off_resources(self):
        for i in (1, self.indices('window')[2], 29):
            m = self.rows[i]['memory']
            for k, bad in (('ram_expert_capacity', 68262297599), ('host_logit_capacity', 1017118719),
                           ('host_embedding_capacity', 0), ('host_route_group_payload', 0), ('host_routing_capacity', 4358143),
                           ('expert_payload_reads', 145), ('expert_payload_bytes_read', 68262297599),
                           ('pinned_expert_staging', 183500799), ('pinned_handoff', 41943039), ('cpu_workers', 1)):
                self.change(i, ('memory', k), bad)
            for k in ('pinned_hybrid_input', 'pinned_hybrid_output', 'pinned_hybrid_error', 'host_hybrid_plans',
                      'host_cpu_expert_views', 'cpu_pool_metadata', 'cpu_pool_scratch', 'host_hybrid_probe',
                      'host_hybrid_input_probe', 'pinned_hybrid_middle'):
                self.change(i, ('memory', k), 1)
            self.change(i, ('memory', 'devices'), m['devices'][::-1])
            for d in (0, 1):
                for k in ('device', 'first_layer', 'last_layer', 'qsa_layers', 'gdn_layers', 'qsa_kv',
                          'qsa_index', 'gdn_state', 'ple_state', 'expert_slots', 'owned_bytes'):
                    self.change(i, ('memory', 'devices', d, k), m['devices'][d][k] + 1)
                self.change(i, ('memory', 'devices', d, 'workspace'), MODULE.WORKSPACE_FLOOR - 1)
                self.change(i, ('memory', 'devices', d, 'owned_peak_bytes'), m['devices'][d]['owned_bytes'] - 1)
                self.change(i, ('memory', 'devices', d, 'owned_buffers'), 0)

    def test_cpu_off_single_host_logit_owner_floor_and_steady_capacity(self):
        # The GPU-only Session allocates max_batch * vocabulary floats once.
        # Its transactional second vector is empty without CPU workers. Keep
        # capacity a lower bound, since vector allocator slack is not payload.
        self.assertEqual(MODULE.PRESERVATION_BYTES, 1017118720)
        self.assertTrue(self.collect()['passed'])
        for capacity in (1017118720 + 64, 2 * 1017118720):
            rows = copy.deepcopy(self.rows)
            for row in rows:
                if 'memory' in row:
                    row['memory']['host_logit_capacity'] = capacity
            self.assertTrue(self.collect(rows)['passed'])
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if 'memory' in row:
                row['memory']['host_logit_capacity'] = 1017118719
        self.reject(rows)
        self.change(29, ('memory', 'host_logit_capacity'), 1017118720 + 64)

    def test_all_steady_fields_reject_changes_even_consistent_sums(self):
        for i, row in enumerate(self.rows):
            if 'memory' not in row or i == 1:
                continue
            self.change(i, ('memory', 'ram_expert_capacity'), row['memory']['ram_expert_capacity'] + 1)
            self.change(i, ('memory', 'devices', 0, 'owned_buffers'), 1002)
            changed = copy.deepcopy(self.rows)
            d = changed[i]['memory']['devices'][0]
            d['weights'] += 1
            d['owned_bytes'] += 1
            self.reject(changed)

    def test_variable_observed_weights_buffers_capacities_not_guessed_constants(self):
        for row in self.rows:
            if 'memory' in row:
                row['memory']['ram_expert_capacity'] += 12345
                row['memory']['host_embedding_capacity'] = 1
                row['memory']['host_route_group_payload'] += 6789
                for d in row['memory']['devices']:
                    d['weights'] += 4321
                    d['workspace'] += 54321
                    d['owned_bytes'] += 58642
                    d['owned_peak_bytes'] += 58642
                    d['owned_buffers'] += 17
                    d['free_vram'] -= 58642
        self.rows[-1]['minimum_free_vram'] = [v - 58642 for v in self.rows[-1]['minimum_free_vram']]
        self.assertTrue(self.collect(self.rows)['passed'])

    def test_vram_bounds_floor_can_include_unserialized_observations_not_exact_peak(self):
        for i in (1, 29):
            d = self.rows[i]['memory']['devices'][0]
            for bad in (0, d['total_vram'] + 1, d['total_vram'] - d['owned_bytes'] + 1):
                self.change(i, ('memory', 'devices', 0, 'free_vram'), bad)
        for bad in (0, 1 << 64, self.rows[29]['memory']['devices'][0]['free_vram'] + 1):
            self.change(30, ('minimum_free_vram', 0), bad)
        self.rows[-1]['minimum_free_vram'] = [1, 1]
        self.assertTrue(self.collect(self.rows)['passed'])

    def test_logical_boundary_coordinate_and_mod4_not_GPU_trace(self):
        for i in self.indices('window'):
            self.change(i, ('accepted_rows_logical2047_through2056',),
                        self.rows[i]['accepted_rows_logical2047_through2056'] + 1)
            self.change(i, ('accepted_rows_expected_visible_mod4', 0),
                        self.rows[i]['accepted_rows_expected_visible_mod4'][0] + 1)
        result = self.collect(self.rows)
        self.assertEqual(result['logical_coverage']['visible2047_through2056_mask'], 1023)
        self.assertEqual(result['logical_coverage']['visible_mod4_mask'], 15)
        self.assertIn('not GPU-observed', result['logical_coverage']['scope'])
        # C++ position2047..2056 includes the FIRST continuation position2056;
        # it must not be mislabeled as the visible2047..2056 teacher set.
        p1 = [self.rows[i] for i in self.indices('window') if self.rows[i]['phase_index'] == 1]
        self.assertEqual([r['accepted_rows_logical2047_through2056'] for r in p1], [0, 1, 9, 10, 10])
        self.assertEqual([p['multiquery_logical_visible2047_through2056_mask']
                          for p in result['logical_coverage']['phases']], [0, 1023, 1023])

    def test_failure_footer_schema_and_failed_window_never_return_aggregate(self):
        failure = {'protocol': 1, 'kind': 'prefill_attention_failure', 'record_index': 0,
                   'phase_index': None, 'phase': None, 'passed': False, 'session_constructed': False,
                   'all_owners_unwound': True, 'failure': 'synthetic failure'}
        self.reject([failure])
        for k in tuple(failure):
            changed = failure.copy()
            del changed[k]
            self.reject([changed])
        self.reject([{**failure, 'unknown': 1}])
        self.reject([{**failure, 'session_constructed': 0}])
        self.reject([{**failure, 'failure': 'x' * 2049}])
        candidate = self.indices('window')[2]
        failed = copy.deepcopy(self.rows[candidate])
        for k in ('expected_routes', 'route_hits_delta', 'route_misses_delta', 'route_upload_bytes_delta',
                  'accepted_rows_expected_visible_mod4', 'accepted_rows_logical2047_through2056',
                  'memory_snapshot_index', 'memory'):
            del failed[k]
        failed.update(passed=False, failure='synthetic window gate failure')
        failed['errors'].update(violations=1, first_violation_row=0, first_violation_column=0,
                                maxabs=1, maxboundratio=50, rms=1 / math.sqrt(1024 * VOCAB), bit_mismatches=1)
        failure.update(record_index=candidate + 1, session_constructed=True)
        raw_failure = [*self.rows[:candidate], failed, failure]
        self.raw.write_bytes(encoded(raw_failure))
        with self.assertRaisesRegex(ValueError, 'window failed: synthetic window gate failure'):
            self.collect()
        self.reject(raw_failure)
        for k in tuple(failed):
            changed = copy.deepcopy(raw_failure)
            del changed[candidate][k]
            self.reject(changed)
        self.reject([*raw_failure[:candidate], {**failed, 'unknown': 1}, failure])
        self.change(30, ('passed',), False)

    def test_nonclaims_and_missing_observations_truthfully_explicit(self):
        result = self.collect()
        for k in ('performance_claim', 'speedup_claim', 'independent_hf_claim', 'R4_complete_claim',
                  'long4k16k_qualified', 'physical_128_column_tile_proven', 'physical_kernel_count_claim',
                  'GPU_selected_ID_trace_qualified', 'full_ram_accounting_qualified', 'occupied_128k_qualified',
                  'exact_peak_vram_claim', 'owned_buffer_release_measured', 'cold_cache_equality_claim', 'mtp'):
            self.assertIs(result[k], False)
        self.assertIn('not physical kernel counts', result['attention_counter_scope'])
        self.assertIn('no N1-group total emitted', result['fixture_contract']['timing'])
        limits = result['memory_evidence_limits']
        self.assertEqual(limits['in_process_observations'], 2190)
        self.assertEqual(limits['serialized_ledgers'], 26)
        self.assertIn('not physical SSD/syscall trace', limits['read_scope'])
        self.assertIn('no measured', limits['cleanup_scope'])

    def test_duplicate_JSON_keys_nonfinite_constants_UTF8_invalid_objects(self):
        raw = self.original_raw
        for k in ('protocol', 'dirty', 'teacher_rows', 'all_vocabulary', 'weights', 'gathered_key_bytes',
                  'batch_calls', 'last_completed_ms_bits', 'finite', 'first_violation_row', 'passed'):
            key = ('"' + k + '":').encode()
            self.raw.write_bytes(raw.replace(key, key + b'0,' + key, 1))
            with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                self.collect()
        for token in (b'NaN', b'Infinity', b'-Infinity', b'1e999', b'-1e999'):
            self.raw.write_bytes(raw.replace(b'"maxabs":0', b'"maxabs":' + token, 1))
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                self.collect()
        lines = raw.splitlines()
        bads = [raw[:-1], raw + b'\n', b'\xef\xbb\xbf' + raw, raw.replace(b'prefill_attention_source', b'\xff', 1)]
        for line in (b'', b' ', b'[]', b'null', b'1', b'true', b'{} {}', b'{', b'failure',
                     b'{"x":' + b'[' * 2000 + b'0' + b']' * 2000 + b'}'):
            bads.append(b'\n'.join([*lines[:3], line, *lines[4:]]) + b'\n')
        for bad in bads:
            self.raw.write_bytes(bad)
            with self.assertRaises(ValueError):
                self.collect()

    def test_raw_file_record_bounds_growing_read_nonregular_and_CRLF(self):
        self.assertEqual(MODULE.MAX_RAW_BYTES, 16 * 1024 * 1024)
        self.assertEqual(MODULE.MAX_RECORD_BYTES, 64 * 1024)
        self.assertLess(max(map(len, self.original_raw.splitlines())), MODULE.MAX_RECORD_BYTES)
        lines = self.original_raw.splitlines()
        extra = MODULE.MAX_RECORD_BYTES - len(lines[0])
        self.raw.write_bytes(b' ' * extra + self.original_raw)
        self.assertTrue(self.collect()['passed'])
        self.raw.write_bytes(b' ' * (extra + 1) + self.original_raw)
        with self.assertRaisesRegex(ValueError, 'oversized JSON record'):
            self.collect()
        with self.raw.open('wb') as output:
            output.truncate(MODULE.MAX_RAW_BYTES + 1)
        with self.assertRaisesRegex(ValueError, 'oversized input'):
            self.collect()
        self.raw.write_bytes(self.original_raw)
        real_open = Path.open

        def grown_open(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            proxy = mock.MagicMock(wraps=stream)
            proxy.__enter__.return_value = proxy
            proxy.__exit__.side_effect = lambda *_: stream.close()
            proxy.read.return_value = b'x' * (MODULE.MAX_RAW_BYTES + 1)
            return proxy

        with mock.patch.object(Path, 'open', autospec=True, side_effect=grown_open):
            with self.assertRaisesRegex(ValueError, 'oversized input'):
                self.collect()
        for nonregular in (self.root, self.root / 'fifo'):
            if nonregular != self.root:
                os.mkfifo(nonregular)
            with self.assertRaisesRegex(ValueError, 'regular file'):
                MODULE.collect(nonregular)
        self.raw.write_bytes(self.original_raw.replace(b'\n', b'\r\n'))
        self.assertTrue(self.collect()['passed'])

    def test_cli_append_explicit_history_shared_locked_helper_and_failed_nonappend(self):
        history = b'{ "kind": "old", "values":[1,2] }\n{"kind":"second"}\n'
        self.results.write_bytes(history)
        process = self.cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, '')
        saved = self.results.read_bytes()
        self.assertTrue(saved.startswith(history))
        self.assertEqual(len(saved.splitlines()), 3)
        self.assertEqual(json.loads(saved.splitlines()[-1])['complete'], self.rows[-1])
        self.assertEqual([json.loads(r) for r in saved.splitlines()[:2]], [json.loads(r) for r in history.splitlines()])
        self.assertEqual(self.raw.read_bytes(), self.original_raw)
        self.assertIs(MODULE.append_record, record_memory.append_result)
        with mock.patch.object(MODULE, 'append_record', wraps=record_memory.append_result) as append:
            self.assertEqual(MODULE.main(['--raw', str(self.raw), '--results', str(self.results)]), 0)
            append.assert_called_once()
        self.assertEqual(self.cli(Path('explicit-relative.jsonl')).returncode, 0)
        for args in ([], ['--raw', str(self.raw)], ['--results', str(self.results)],
                     ['--raw', str(self.raw), '--results', str(self.results), '--absolute', '1']):
            self.assertEqual(self.cli(args=args).returncode, 2)
        self.results.write_bytes(history)
        self.raw.write_bytes(self.original_raw[:-1])
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)
        self.results.unlink()
        self.assertEqual(self.cli().returncode, 1)
        self.assertFalse(self.results.exists())
        failed_footer = copy.deepcopy(self.rows)
        failed_footer[-1]['passed'] = False
        self.raw.write_bytes(encoded(failed_footer))
        self.assertEqual(self.cli().returncode, 1)
        self.assertFalse(self.results.exists())
        self.results.write_bytes(history)
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)
        with mock.patch.object(MODULE, 'collect', side_effect=OSError('input failure')):
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(MODULE.main(['--raw', str(self.raw), '--results', str(self.results)]), 1)
                self.assertIn('record_prefill_attention: input failure', errors.getvalue())

    def test_raw_model_symlink_hardlink_aliases_unfinished_nonregular_journal(self):
        model = self.root / 'qwen38-keep1-Q4_0.gguf'
        model.write_bytes(b'synthetic declared model filename, never loaded')
        self.rows[0]['model'] = str(model)
        self.raw.write_bytes(encoded(self.rows))
        for artifact in (self.raw, model):
            before = artifact.read_bytes()
            sym, hard = self.root / (artifact.name + '.sym'), self.root / (artifact.name + '.hard')
            sym.symlink_to(artifact)
            os.link(artifact, hard)
            for destination in (artifact, sym, hard):
                self.assertEqual(self.cli(destination).returncode, 1)
                self.assertEqual(artifact.read_bytes(), before)
        history = b'{"kind":"first"}\n{"kind":"unfinished"}'
        self.results.write_bytes(history)
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)
        fifo = self.root / 'journal-fifo'
        os.mkfifo(fifo)
        for destination in (self.root, fifo, self.root / 'absent/results.jsonl', Path('/dev/stdout'), Path('/dev/stderr')):
            self.assertEqual(self.cli(destination).returncode, 1)

    def test_serialization_flock_shortwrite_write_flush_rollback_and_alias_race(self):
        record = self.collect()
        history = b'{"kind":"old"}\n'
        self.results.write_bytes(history)
        with mock.patch.object(record_memory.fcntl, 'flock', wraps=record_memory.fcntl.flock) as lock:
            MODULE.append_record(self.results, record)
            self.assertEqual(lock.call_args.args[1], record_memory.fcntl.LOCK_EX)
        with mock.patch.object(record_memory.json, 'dumps', side_effect=ValueError('serialization failure')):
            with mock.patch.object(Path, 'open') as opened:
                with self.assertRaisesRegex(ValueError, 'serialization failure'):
                    MODULE.append_record(self.results, record)
                opened.assert_not_called()
        real_open = Path.open
        for failure in ('short', 'write', 'flush'):
            self.results.write_bytes(history)

            def faulty_open(path, *args, **kwargs):
                stream = real_open(path, *args, **kwargs)
                proxy = mock.MagicMock(wraps=stream)
                proxy.__enter__.return_value = proxy
                proxy.__exit__.side_effect = lambda *_: stream.close()

                def write(value):
                    if failure == 'flush':
                        return stream.write(value)
                    stream.write(value[:17])
                    if failure == 'write':
                        raise OSError('injected write failure')
                    return 17

                proxy.write.side_effect = write
                if failure == 'flush':
                    proxy.flush.side_effect = OSError('injected flush failure')
                return proxy

            with mock.patch.object(Path, 'open', autospec=True, side_effect=faulty_open):
                with self.assertRaises((ValueError, OSError)):
                    MODULE.append_record(self.results, record)
            self.assertEqual(self.results.read_bytes(), history)

        def replace_before_open(path, *args, **kwargs):
            self.results.unlink()
            os.link(self.raw, self.results)
            return real_open(path, *args, **kwargs)

        with mock.patch.object(Path, 'open', autospec=True, side_effect=replace_before_open):
            with self.assertRaisesRegex(ValueError, 'input artifact'):
                MODULE.append_record(self.results, record)
        self.assertEqual(self.raw.read_bytes(), self.original_raw)

    def test_parallel_appends_whole_records_history_and_actual_blocking_lock(self):
        history = b'{"kind":"first"}\n{"kind":"second"}\n'
        self.results.write_bytes(history)
        command = [sys.executable, '-B', str(SCRIPT), '--raw', str(self.raw), '--results', str(self.results)]
        with self.results.open('ab') as locked:
            record_memory.fcntl.flock(locked.fileno(), record_memory.fcntl.LOCK_EX)
            processes = [subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                         for _ in range(2)]
            try:
                for process in processes:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        process.wait(timeout=.25)
                self.assertEqual(self.results.read_bytes(), history)
                record_memory.fcntl.flock(locked.fileno(), record_memory.fcntl.LOCK_UN)
                for process in processes:
                    output, errors = process.communicate(timeout=30)
                    self.assertEqual(process.returncode, 0, errors)
                    self.assertEqual(output, '')
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
        saved = self.results.read_bytes()
        self.assertTrue(saved.startswith(history))
        self.assertEqual(len(saved.splitlines()), 4)
        for line in saved.splitlines()[2:]:
            self.assertEqual(json.loads(line)['complete'], self.rows[-1])


if __name__ == '__main__':
    unittest.main()
