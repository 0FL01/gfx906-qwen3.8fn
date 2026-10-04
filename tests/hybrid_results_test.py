"""Standalone exact hybrid protocol fixtures: no model, build, HIP or real journal."""

import contextlib
import copy
import datetime
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
from tools import record_hybrid as MODULE
from tools import record_memory

SCRIPT = ROOT / 'tools/record_hybrid.py'
REVISION = '0123456789abcdef0123456789abcdef01234567'
VOCAB, DOWN, FFN = 248320, 1228800, 122880
DISPATCH = ('cumulative_cpu_assignments', 'cumulative_gpu_hit_assignments',
            'cumulative_gpu_miss_assignments')
STAGES = ('cpu_gate_up_jobs', 'cpu_down_jobs', 'GPU_middle_columns',
          'GPU_middle_batches', 'paired_gate_up_H2D_bytes', 'middle_Q8_D2H_bytes')
CUMULATIVE_STAGES = ('cumulative_cpu_gate_up_jobs', 'cumulative_cpu_down_jobs',
                     'cumulative_GPU_middle_columns', 'cumulative_GPU_middle_batches',
                     'cumulative_paired_gate_up_H2D_bytes', 'cumulative_middle_Q8_D2H_bytes')


def metrics(values):
    return {'values': values, 'violations': 0, 'max_abs': 0,
            'max_bound_ratio': 0, 'diagnostic_bit_mismatches': 0}


def fixture():
    """Independent transcription of C++ main: explicit all case/owner sequences.

    Every token selects the same ten unique experts per layer in this synthetic
    routing example. One is READY; quota misses go to GPU, remaining to CPU.
    Repeated-token groups have C=N, giving exactly 480*(N-1) reuse assignments.
    Fresh replay has zero READY groups. Forced modes only alter dispatch.
    """
    result = [{
        'kind': 'session_hybrid_header', 'protocol': 1, 'revision': REVISION, 'dirty': True,
        'model_variant': 'qwen38-keep1-Q4_0', 'sequential_Sessions': 3, 'simultaneous_Sessions': 1,
        'runtime_scope': 'CPUlinear_GPUmiddle',
        'force_cpu_enum_scope': 'all_routed_linear_CPU_with_canonical_GPU_middle', 'pure_CPU_only_claim': False,
        'max_batch_tokens_by_Session': [3, 1, 4], 'cpu_workers_by_Session': [1, 1, 4],
        'reference_rows': 40, 'teacher': 'BOS248044,100..130', 'continuation': '131..138',
        'full_logit_gate': '.02+.002*abs(ref)', 'intermediate_gate': '.002+.002*abs(ref)',
        'reference': 'retained_full_vocabulary_GPU_N1_self_reference', 'independent_HF_claim': False,
        'performance_claim': False, 'R5_complete_claim': False, 'MTP_claim': False,
    }]

    def memory(instance, batch, workers, slots):
        stage = 45875200 if batch == 4 else 2867200
        result.append({
            'kind': 'session_hybrid_memory', 'session_instance': instance, 'slots': slots,
            'max_batch_tokens': batch, 'cpu_workers': workers, 'pinned_input_bytes': 17280,
            'pinned_output_bytes': 614400, 'pinned_error_bytes': 8, 'pinned_stage_bytes': 4 * stage,
            'pinned_middle_Q8_bytes': 43200,
            # Synthetic metadata baseline, including the 480-byte LP64 headers;
            # actual sizeof(HybridDevice) is observed, not guessed from this fixture.
            'host_plan_bytes': 16864, 'host_cpu_views_and_pending_bytes': 2359296 + slots * 192,
            'host_probe_bytes': 33295968, 'host_input_probe_bytes': 855648, 'pool_metadata_bytes': 4096,
            'pool_scratch_bytes': workers * 65536, 'host_routing_capacity_bytes': batch * 4256,
            'RouteGroups_requested_payload_bytes': 12384 + batch * 320,
            'pinned_route_metadata_bytes': batch * 160, 'route_metadata_bytes_per_GPU': [batch * 80] * 2,
            'paired_gate_up_bytes_per_GPU': [153600, 153600], 'middle_float_bytes_per_GPU': [76800, 76800],
            'middle_Q8_bytes_per_GPU': [21600, 21600], 'middle_error_bytes_per_GPU': [4, 4],
            'contribution_bytes_per_GPU': [409600 if batch == 4 else 307200] * 2,
            'stage_buffer_bytes_per_GPU': [stage] * 2, 'full_RSS_or_stacks_claim': False,
            'ownership_verified': True,
        })

    def window(name, offset, n, totals, mode, quota=2, fresh=False):
        if mode == 'off' or n == 4:
            groups = (0, 0, 0)
        elif mode == 'cpu':
            groups = (480, 0, 0)
        elif mode == 'gpu':
            groups = (0, 0, 480)
        elif mode == 'allhit':
            groups = (0, 480, 0)
        else:
            hit = 0 if fresh else 48
            groups = (480 - hit - 48 * quota, hit, 48 * quota)
        for j in range(3):
            totals[j] += n * groups[j]
        # TWO accepted pool jobs per logical CPU group, not twice cpu_groups.
        # Canonical GPU middle is batched per CPU-bearing layer, not per job.
        cpu = n * groups[0]
        projections = (groups[0], groups[0], cpu, 48 if cpu else 0, cpu * 5120, cpu * 720)
        for j, value in enumerate(projections, 3):
            totals[j] += value
        result.append({
            'kind': 'session_hybrid_window', 'phase': name, 'offset': offset, 'rows': n,
            'continuation': offset >= 32, 'logits': metrics(n * VOCAB),
            'intermediate_probe_available': n <= 3,
            'unweighted_down': metrics(n * DOWN) if n <= 3 else None,
            'ffn_output': metrics(n * FFN) if n <= 3 else None,
            **dict(zip((*DISPATCH, *CUMULATIVE_STAGES), totals)),
        })
        return groups

    def phase(name, n, mode, fresh=False):
        offset, totals, groups, reuse, calls = 0, [0] * 9, [0, 0, 0], 0, 0
        while offset < 40:
            width = min(n, 32 - offset) if offset < 32 else 1
            current = window(name, offset, width, totals, mode, fresh=fresh and offset == 0)
            for j in range(3):
                groups[j] += current[j]
            if mode != 'off':
                reuse += 480 * (width - 1)
            offset += width
            calls += 1
        result.append({
            'kind': 'session_hybrid_phase', 'phase': name, 'teacher_rows': 32, 'continuation_rows': 8,
            'cpu_groups': groups[0], 'gpu_hit_groups': groups[1], 'gpu_miss_groups': groups[2],
            'group_reuse_assignments': reuse,
            'admission_copies': 96 * calls if mode in ('mixed', 'gpu') else 0,
            'evicted_ready_slots': 48 * (calls - int(fresh)) if mode in ('mixed', 'gpu') else 0,
            'input_extractions': 48 * calls if mode in ('mixed', 'cpu') else 0,
            'CPU_input_bytes_checked': totals[0] * 2880, 'fresh_cache_first_call_checked': fresh, 'passed': True,
            **dict(zip(STAGES, totals[3:])),
        })

    memory(1, 3, 1, 1)
    result.append({'kind': 'session_hybrid_reference', 'rows': 40, 'full_vocab_values': 9932800,
                   'routed_down_values': 49152000, 'ffn_output_values': 4915200, 'all_finite': True})
    for family, mode in (('hybrid_OFF_GPU_only', 'off'), ('diagnostic_force_CPU_LINEAR_GPU_middle', 'cpu'), ('mixed_slot1', 'mixed')):
        for n in (1, 2, 3):
            phase(f'{family}_N{n}', n, mode)
    phase('diagnostic_force_GPU_misses_N3', 3, 'gpu')
    window('mixed_slot1_warm_BOS_quota1', 0, 1, [0] * 9, 'mixed', quota=1)
    window('mixed_slot1_captured_reader_then_eviction_BOS', 0, 1, [0] * 9, 'mixed', quota=1)
    result.append({'kind': 'session_hybrid_captured_reader_reuse', 'slots': 1, 'GPU_hit_groups': 48,
                   'CPU_groups': 384, 'GPU_miss_groups': 48, 'evicted_READY_slots': 48, 'quota': 1, 'passed': True})
    window('synthetic_failure_setup_N3', 0, 3, [0] * 9, 'mixed')
    phase('full_replay_after_submitted_admission_failure', 3, 'mixed', fresh=True)
    result.append({
        'kind': 'session_hybrid_submitted_admission_failure', 'synthetic': True, 'failure_layer': 0,
        'accepted_CPU_job_stage': 'gate_up_before_GPU_middle_and_down',
        'accepted_CPU_jobs': 7, 'queued_admission_copies': 2, 'pending_host_slots_at_throw': 1,
        'CPU_return_bytes_before_throw': 0, 'pool_and_both_GPU_streams_drained': True,
        'all_READY_and_pending_IDs_invalidated': True, 'published_stats_route_hybrid_logits_probes_preserved': True,
        'reset_full_replay_rows': 40, 'fresh_physical_misses_first_replay_assignments': 1440,
        'inflight_timing_claim': False, 'passed': True,
    })
    totals = [0] * 9
    for offset in range(0, 39, 3):
        window('sticky_failure_setup', offset, 3, totals, 'mixed')
    window('reset_after_sticky_failure', 0, 1, [0] * 9, 'mixed')
    result.append({'kind': 'session_hybrid_sticky_failure', 'injection': 'owned_trace_IO_before_CPU_submission',
                   'published_stats_logits_intermediates_preserved': True, 'reset_recovery': True, 'passed': True})
    memory(2, 1, 1, 112)
    window('all_hit_warm_GPU_BOS', 0, 1, [0] * 9, 'off')
    totals = [0] * 9
    window('real_all_hit_reset_BOS', 0, 1, totals, 'allhit')
    window('all_hit_session_continuation', 1, 1, totals, 'allhit')
    result.append({'kind': 'session_hybrid_real_all_hit', 'warm_reset_replay': 'BOS', 'ready_assignments': 480,
                   'CPU_extractions': 0, 'CPU_jobs': 0, 'GPU_uploads': 0, 'passed': True})
    memory(3, 4, 4, 1)
    offset, totals = 0, [0] * 9
    for n in [1, 4, 2, 4, 3, 4, 1, 4, 2, 4, 3] + [1] * 8:
        name = 'max4_GPU_fallback_N4' if n == 4 else f'max4_short_hybrid_N{n}'
        window(name, offset, n, totals, 'mixed')
        offset += n
    result.append({
        'kind': 'session_hybrid_short_wide_short', 'session_instance': 3, 'max_batch_tokens': 4, 'cpu_workers': 4,
        'teacher_window_sizes': [1, 4, 2, 4, 3, 4, 1, 4, 2, 4, 3], 'teacher_rows': 32,
        'continuation_N1_rows': 8, 'short_calls': 14, 'wide_N4_calls': 5, 'short_layers': 672,
        'gpu_only_wide_layers': 240, 'wide_CPU_counters_and_transfer_bytes_unchanged': True,
        'wide_intermediate_probe_available': False, 'stage_capacity_bytes_each': 45875200,
        'total_pinned_stage_bytes': 183500800, 'contribution_bytes_each_GPU': 409600,
        'full_40_row_replay': True, 'passed': True,
    })
    result.append({
        'kind': 'session_hybrid_footer', 'passed': True, 'sequential_Sessions': 3, 'simultaneous_Sessions': 1,
        'runtime_scope': 'CPUlinear_GPUmiddle', 'pure_CPU_only_claim': False,
        'bounded_GPU_quota': 2, 'physical_residency_not_dispatch': True,
        'sticky_failure_scopes': ['trace_IO_before_CPU', 'synthetic_accepted_CPU_plus_queued_GPU_admission'],
        'retained_reference_rows': 40, 'compared_windows': 326, 'compared_full_vocab_rows': 528,
        'compared_full_vocab_values': 131112960, 'intermediate_compared_rows': 508,
        'compared_routed_down_values': 624230400, 'compared_ffn_output_values': 62423040,
        'wide_N4_windows': 5, 'wide_N4_rows': 20, 'expected_execution_failures': 2,
        'CPU_descriptor_exact_original_GPU_Q8_bytes': True, 'CPU_descriptor_matches_GPU_route_DTO': True,
        'numeric_failure_fixture': 'separate_CPU_stages_canonical_GPU_middle_and_GPU_shadow',
        'individual_added_GPU_buffers_observable': True, 'physical_128_tile_claim': False,
        'performance_claim': False, 'R5_complete_claim': False,
    })
    return result


def encoded(rows):
    return (''.join(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n' for row in rows)).encode()


def target(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def objects(obj, path=()):
    if type(obj) is dict:
        yield path, obj
        for key, value in obj.items():
            yield from objects(value, (*path, key))
    elif type(obj) is list:
        for i, value in enumerate(obj):
            yield from objects(value, (*path, i))


class HybridResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = fixture()
        cls.original_raw = encoded(cls.original)

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='hybrid-results-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw, self.results = self.root / 'hybrid.jsonl', self.root / 'results.jsonl'
        self.rows = copy.deepcopy(self.original)
        self.raw.write_bytes(self.original_raw)

    def indices(self, kind):
        return [i for i, row in enumerate(self.rows) if row['kind'] == 'session_hybrid_' + kind]

    def named(self, phase):
        return [i for i, row in enumerate(self.rows) if row.get('phase') == phase and row['kind'].endswith('_window')]

    def collect(self, rows=None):
        if rows is not None:
            self.raw.write_bytes(encoded(rows))
        return MODULE.collect(self.raw)

    def reject(self, rows):
        self.raw.write_bytes(encoded(rows))
        with self.assertRaises(ValueError):
            self.collect()
        self.assertEqual(self.raw.read_bytes(), encoded(rows))
        self.assertFalse(self.results.exists())

    def change(self, index, path, bad):
        obj = target(self.rows[index], path[:-1])
        old = obj[path[-1]]
        obj[path[-1]] = bad
        try:
            with self.subTest(index=index, path=path, bad=bad):
                self.reject(self.rows)
        finally:
            obj[path[-1]] = old

    def cli(self, destination=None, args=None):
        command = [sys.executable, '-B', str(SCRIPT)]
        command += args if args is not None else ['--raw', self.raw.name, '--results', str(destination or self.results)]
        return subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)

    def test_valid_all_sequences_exact_aggregate_compact_provenance_and_nonclaims(self):
        result = self.collect()
        self.assertEqual(len(self.rows), 348)
        self.assertEqual((MODULE.RECORD_COUNT, MODULE.WINDOW_COUNT, MODULE.FULL_ROWS, MODULE.SHORT_ROWS),
                         (348, 326, 528, 508))
        self.assertEqual(result['kind'], 'r5_hybrid')
        self.assertEqual(result['protocol'], 1)
        self.assertEqual(result['revision'], REVISION)
        self.assertIs(result['dirty'], True)
        self.assertEqual(result['source'], self.rows[0])
        self.assertEqual(result['phases'], [self.rows[i] for i in self.indices('phase')])
        self.assertEqual(result['memory'], [self.rows[i] for i in self.indices('memory')])
        self.assertEqual(len(result['case_summary']), 5)
        self.assertEqual(result['reference'], self.rows[2])
        self.assertEqual(result['complete'], self.rows[-1])
        self.assertNotIn('windows', result)
        self.assertNotIn('records', result)
        self.assertEqual(result['raw_logs'], {'hybrid': str(self.raw.resolve())})
        self.assertEqual(datetime.datetime.fromisoformat(result['timestamp']).utcoffset(), datetime.timedelta(0))
        summary = result['window_summary']
        for key, expected in (('logits', 131112960), ('unweighted_down', 624230400), ('ffn_output', 62423040)):
            self.assertEqual(summary[key]['values'], expected)
            self.assertEqual(summary[key]['violations'], 0)
        for key in ('performance_claim', 'speedup_claim', 'dispatch_threshold_qualified', 'R5_complete_claim',
                    'occupied_128k_qualified', 'independent_hf_claim', 'mtp', 'full_ram_accounting_qualified',
                    'exact_peak_vram_claim', 'owned_buffer_release_measured', 'model_path_observable', 'pure_CPU_only_claim'):
            self.assertIs(result[key], False)
        self.assertEqual(result['runtime_scope'], 'CPUlinear_GPUmiddle')
        self.assertEqual(summary['projection_totals'], {
            'cpu_gate_up_jobs': 84672, 'cpu_down_jobs': 84672,
            'GPU_middle_columns': 133440, 'GPU_middle_batches': 10368,
            'paired_gate_up_H2D_bytes': 683212800, 'middle_Q8_D2H_bytes': 96076800,
        })
        self.assertEqual(summary['projection_totals_scope'],
                         'sum_accepted_compared_window_deltas_excludes_reference_and_failed_call')
        self.assertEqual(self.raw.read_bytes(), self.original_raw)

    def test_exact_source_and_reference_gates_ids_geometry_provenance(self):
        for key, bad in (('revision', 'g' * 40), ('revision', 'f' * 39), ('revision', 123), ('dirty', 1),
                         ('model_variant', 'qwen38-full-PLE-Q4_0'), ('protocol', 2), ('sequential_Sessions', 2),
                         ('simultaneous_Sessions', 3), ('teacher', 'BOS248044,100..131'),
                         ('continuation', '130..137'), ('full_logit_gate', '.021+.002*abs(ref)'),
                         ('intermediate_gate', '.003+.002*abs(ref)'), ('reference_rows', 39)):
            self.change(0, (key,), bad)
        for path, bad in ((('max_batch_tokens_by_Session', 2), 3), (('cpu_workers_by_Session', 2), 1)):
            self.change(0, path, bad)
        reference = self.indices('reference')[0]
        for key in ('rows', 'full_vocab_values', 'routed_down_values', 'ffn_output_values'):
            self.change(reference, (key,), self.rows[reference][key] - 1)
        self.change(reference, ('all_finite',), False)
        self.rows[0].update(revision='ABCDEF01' * 5, dirty=False)
        result = self.collect(self.rows)
        self.assertEqual(result['revision'], 'ABCDEF01' * 5)
        self.assertIs(result['dirty'], False)

    def test_every_kind_and_nested_object_has_closed_schema(self):
        representatives = list(dict.fromkeys(self.indices(k)[0] for k in (
            'header', 'reference', 'memory', 'window', 'phase', 'captured_reader_reuse',
            'submitted_admission_failure', 'sticky_failure', 'real_all_hit', 'short_wide_short', 'footer')))
        representatives += [self.named('max4_GPU_fallback_N4')[0], self.indices('memory')[-1]]
        for index in representatives:
            for path, obj in list(objects(self.rows[index])):
                obj['unknown'] = 1
                try:
                    self.reject(self.rows)
                finally:
                    del obj['unknown']
                for key in list(obj):
                    value = obj.pop(key)
                    try:
                        self.reject(self.rows)
                    finally:
                        obj[key] = value

    def test_bool_null_integer_types_and_failure_claims_all_kinds(self):
        # Every literal boolean in the raw is typed and frozen except dirty.
        for index, row in enumerate(self.rows):
            for path, obj in objects(row):
                for key, value in obj.items():
                    if type(value) is bool:
                        self.change(index, (*path, key), int(value))
                        if key != 'dirty':
                            self.change(index, (*path, key), not value)
        for index in self.indices('memory') + self.indices('phase') + [0, 2, len(self.rows) - 1]:
            for path, obj in objects(self.rows[index]):
                for key, value in obj.items():
                    if type(value) is int:
                        self.change(index, (*path, key), float(value))
                        self.change(index, (*path, key), True)
        wide = self.named('max4_GPU_fallback_N4')[0]
        for key in ('unweighted_down', 'ffn_output'):
            for bad in (False, {}, metrics(4 * DOWN)):
                self.change(wide, (key,), bad)

    def test_chronology_n123_tail_continuations_sticky_setup_and_n4_fallback(self):
        windows = self.indices('window')
        for i in windows:
            self.change(i, ('offset',), self.rows[i]['offset'] + 1)
        for name, sizes in (('hybrid_OFF_GPU_only_N1', [1] * 40),
                            ('diagnostic_force_CPU_LINEAR_GPU_middle_N2', [2] * 16 + [1] * 8),
                            ('mixed_slot1_N3', [3] * 10 + [2] + [1] * 8),
                            ('sticky_failure_setup', [3] * 13)):
            self.assertEqual([self.rows[i]['rows'] for i in self.named(name)], sizes)
        for i in (windows[0], windows[52], self.named('mixed_slot1_N3')[-9],
                  self.named('sticky_failure_setup')[-1], self.named('max4_GPU_fallback_N4')[0]):
            self.change(i, ('rows',), self.rows[i]['rows'] + 1)
            self.change(i, ('phase',), 'wrong_phase')
        representative = [0, 1, 2, *self.indices('phase'), *self.indices('memory')[1:],
                          *self.indices('sticky_failure'), *self.indices('submitted_admission_failure'), len(self.rows) - 1]
        for i in representative:
            self.reject(self.rows[:i] + self.rows[i + 1:])
            self.reject(self.rows[:i] + [self.rows[i]] + self.rows[i:])
            if i:
                self.rows[i], self.rows[i - 1] = self.rows[i - 1], self.rows[i]
                try:
                    self.reject(self.rows)
                finally:
                    self.rows[i], self.rows[i - 1] = self.rows[i - 1], self.rows[i]
        self.reject(self.rows + [{'kind': 'session_hybrid_failure', 'passed': False}])

    def test_frozen_metrics_all_values_finite_gate_ratio_bits_and_intermediate_extents(self):
        window = self.named('mixed_slot1_N3')[0]
        for key in ('logits', 'unweighted_down', 'ffn_output'):
            for field, bad in (('values', 1), ('values', True), ('violations', 1),
                               ('max_abs', -1), ('max_abs', True), ('max_abs', None),
                               ('max_bound_ratio', math.nextafter(1., math.inf)),
                               ('max_bound_ratio', -1), ('max_bound_ratio', .1),
                               ('diagnostic_bit_mismatches', -1),
                               ('diagnostic_bit_mismatches', self.rows[window][key]['values'] + 1)):
                self.change(window, (key, field), bad)
            self.change(window, (key, 'max_abs'), .0001)  # Missing positive ratio/bit count.
        self.rows[window]['logits'].update(max_abs=.015625, max_bound_ratio=.78125, diagnostic_bit_mismatches=1)
        self.rows[window]['unweighted_down'].update(max_abs=.001, max_bound_ratio=.5, diagnostic_bit_mismatches=1)
        self.rows[window]['ffn_output'].update(max_abs=.001, max_bound_ratio=.5, diagnostic_bit_mismatches=1)
        self.assertIs(self.collect(self.rows)['passed'], True)
        self.change(window, ('logits', 'max_bound_ratio'), .8)  # > max_abs/.02 at any reference value
        self.change(window, ('unweighted_down', 'max_bound_ratio'), .6)
        self.rows[window]['logits'].update(max_abs=0, max_bound_ratio=0, diagnostic_bit_mismatches=200)
        result = self.collect(self.rows)
        self.assertEqual(result['window_summary']['logits']['diagnostic_bit_mismatches'], 200)
        self.assertFalse(result['window_summary']['bit_equality_required'])  # +/-0 differences are allowed.
        self.assertFalse(result['window_summary']['argmax_observable'])

    def test_dispatch_conservation_monotonicity_disabled_and_forced_modes(self):
        for name in ('hybrid_OFF_GPU_only_N3', 'diagnostic_force_CPU_LINEAR_GPU_middle_N3',
                     'diagnostic_force_GPU_misses_N3', 'mixed_slot1_N3'):
            window = self.named(name)[1]
            for key in DISPATCH:
                for bad in (self.rows[window][key] + 1, -1, True, 1., 1 << 64):
                    self.change(window, (key,), bad)
        window = self.named('mixed_slot1_N3')[1]
        self.rows[window][DISPATCH[0]] = 0
        self.rows[window][DISPATCH[2]] += 2016
        self.reject(self.rows)  # Sum cannot hide a decreasing CPU counter.

    def test_mixed_slot1_quota_real_miss_reuse_descriptor_and_group_summaries(self):
        for i in self.indices('phase'):
            for field in ('cpu_groups', 'gpu_hit_groups', 'gpu_miss_groups', 'group_reuse_assignments',
                          'admission_copies', 'input_extractions', 'CPU_input_bytes_checked'):
                self.change(i, (field,), self.rows[i][field] + 1)
        phase = next(i for i in self.indices('phase') if self.rows[i]['phase'] == 'mixed_slot1_N3')
        self.change(phase, ('evicted_ready_slots',), 0)
        self.change(phase, ('evicted_ready_slots',), 48 * 19 + 1)
        w = self.named('mixed_slot1_N3')[0]
        # Preserve conservation while falsely labeling all routing as CPU.
        self.rows[w].update(dict(zip(DISPATCH, (1440, 0, 0))))
        self.reject(self.rows)

    def test_CPU_linear_stages_match_every_phase_group_once_and_last_window(self):
        for i in self.indices('phase'):
            for key in STAGES:
                self.change(i, (key,), self.rows[i][key] + 1)
            name = self.rows[i]['phase']
            final = self.rows[self.named(name)[-1]]
            for key, cumulative in zip(STAGES, CUMULATIVE_STAGES):
                self.assertEqual(self.rows[i][key], final[cumulative])
            self.assertEqual(self.rows[i]['cpu_gate_up_jobs'], self.rows[i]['cpu_groups'])
            self.assertEqual(self.rows[i]['cpu_down_jobs'], self.rows[i]['cpu_groups'])
            self.assertEqual(self.rows[i]['GPU_middle_columns'], final[DISPATCH[0]])
            self.assertEqual(self.rows[i]['GPU_middle_batches'], self.rows[i]['input_extractions'])
            self.assertEqual(self.rows[i]['paired_gate_up_H2D_bytes'], final[DISPATCH[0]] * 5120)
            self.assertEqual(self.rows[i]['middle_Q8_D2H_bytes'], final[DISPATCH[0]] * 720)
        forced = next(i for i in self.indices('phase') if
                      self.rows[i]['phase'] == 'diagnostic_force_CPU_LINEAR_GPU_middle_N3')
        self.assertEqual(self.rows[forced]['cpu_groups'], 9120)
        self.assertEqual(self.rows[forced]['GPU_middle_columns'], 19200)
        self.change(forced, ('cpu_groups',), 18240)  # Two pool phases do not double logical groups.

    def test_projection_counter_differences_and_exact_types_every_window(self):
        # Cover every accepted lifetime, reset boundary, tail and wide call.
        for i in self.indices('window'):
            for key in CUMULATIVE_STAGES:
                self.change(i, (key,), self.rows[i][key] + 1)
        for name in ('hybrid_OFF_GPU_only_N3', 'diagnostic_force_CPU_LINEAR_GPU_middle_N3',
                     'mixed_slot1_N3', 'diagnostic_force_GPU_misses_N3',
                     'real_all_hit_reset_BOS', 'max4_GPU_fallback_N4'):
            i = self.named(name)[0]
            for key in CUMULATIVE_STAGES:
                for bad in (True, float(self.rows[i][key]), -1, None, '0', 1 << 64):
                    self.change(i, (key,), bad)

    def test_projection_jobs_per_group_allow_measured_reuse_but_reject_bad_deltas(self):
        name = 'diagnostic_force_CPU_LINEAR_GPU_middle_N3'
        first, second = self.named(name)[:2]
        # Equal gate/down totals and a correct final phase cannot conceal fewer
        # than ceil(CPU assignments/N) jobs in either accepted call.
        for key in CUMULATIVE_STAGES[:2]:
            self.rows[second][key] -= 1
        self.reject(self.rows)
        self.rows = copy.deepcopy(self.original)
        for key in CUMULATIVE_STAGES[:2]:
            self.rows[first][key] += 1
        self.reject(self.rows)  # Next delta is now479, although both totals rise.
        self.rows = copy.deepcopy(self.original)
        # Different expert overlap is allowed: three extra logical groups in
        # first N3, three fewer C-1 reuse assignments, same assignment count.
        for i in self.named(name):
            for key in CUMULATIVE_STAGES[:2]:
                self.rows[i][key] += 3
        phase = next(i for i in self.indices('phase') if self.rows[i]['phase'] == name)
        for key in ('cpu_groups', 'cpu_gate_up_jobs', 'cpu_down_jobs'):
            self.rows[phase][key] += 3
        self.rows[phase]['group_reuse_assignments'] -= 3
        result = self.collect(self.rows)
        self.assertEqual(result['window_summary']['projection_totals']['cpu_gate_up_jobs'], 84675)
        self.assertEqual(result['window_summary']['projection_totals']['GPU_middle_columns'], 133440)

    def test_GPU_middle_batches_are_CPU_layer_batches_not_jobs_columns_or_workers(self):
        for name in ('diagnostic_force_CPU_LINEAR_GPU_middle_N1',
                     'diagnostic_force_CPU_LINEAR_GPU_middle_N3', 'mixed_slot1_N3',
                     'mixed_slot1_captured_reader_then_eviction_BOS', 'max4_short_hybrid_N1'):
            i = self.named(name)[0]
            for bad in (0, 47, 49, self.rows[i]['cumulative_cpu_gate_up_jobs'],
                        self.rows[i][DISPATCH[0]], 48 * 4):
                self.change(i, ('cumulative_GPU_middle_batches',), bad)
        bos = self.named('real_all_hit_reset_BOS')[0]
        for key in CUMULATIVE_STAGES:
            self.change(bos, (key,), 1)
        # Only BOS is guaranteed all-hit; continuation may use five CPU-bearing
        # layers (nine CPU groups each) with no GPU misses. Do not invent a second
        # all-hit proof or require48 extractions in this112-slot owner.
        continuation = self.named('all_hit_session_continuation')[0]
        self.rows[continuation].update(dict(zip((*DISPATCH, *CUMULATIVE_STAGES),
                                              (45, 915, 0, 45, 45, 45, 5, 230400, 32400))))
        result = self.collect(self.rows)
        self.assertEqual(result['window_summary']['projection_totals']['GPU_middle_batches'], 10373)
        for bad in (0, 4, 46, 49):
            self.change(continuation, ('cumulative_GPU_middle_batches',), bad)

    def test_runtime_scope_force_phase_names_failure_stage_and_old_protocol_rejected(self):
        for i in (0, len(self.rows) - 1):
            for bad in ('CPU_only', 'whole_CPU_expert', '', True):
                self.change(i, ('runtime_scope',), bad)
        self.change(0, ('force_cpu_enum_scope',), 'whole_CPU_expert')
        failure = self.indices('submitted_admission_failure')[0]
        for bad in ('whole_CPU_before_return', 'down_before_return', 'actual_inflight_gate_up', None, True):
            self.change(failure, ('accepted_CPU_job_stage',), bad)
        for n in (1, 2, 3):
            for i in self.named(f'diagnostic_force_CPU_LINEAR_GPU_middle_N{n}')[:1]:
                self.change(i, ('phase',), f'diagnostic_force_CPU_N{n}')
        # Old logs are unaccepted candidates; no backwards-compatible success
        # fallback for a protocol1 producer lacking the new closed typed fields.
        old = copy.deepcopy(self.rows)
        for row in old:
            for key in (*CUMULATIVE_STAGES, *STAGES, 'runtime_scope', 'force_cpu_enum_scope',
                        'pure_CPU_only_claim', 'accepted_CPU_job_stage', 'pinned_middle_Q8_bytes',
                        'paired_gate_up_bytes_per_GPU', 'middle_float_bytes_per_GPU',
                        'middle_Q8_bytes_per_GPU', 'middle_error_bytes_per_GPU'):
                row.pop(key, None)
            if 'phase' in row:
                row['phase'] = row['phase'].replace('diagnostic_force_CPU_LINEAR_GPU_middle_',
                                                  'diagnostic_force_CPU_')
        old[-1]['numeric_failure_fixture'] = 'separate_CPU_expert_and_actual_GPU_shadow'
        self.reject(old)
        self.raw.write_bytes(encoded(old))
        history = b'{"kind":"prior_synthetic_test_record"}\n'
        self.results.write_bytes(history)
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)

    def test_GPU_middle_new_buffer_capacities_all_owners_closed_typed_payload_only(self):
        sizes = {'paired_gate_up_bytes_per_GPU': 153600, 'middle_float_bytes_per_GPU': 76800,
                 'middle_Q8_bytes_per_GPU': 21600, 'middle_error_bytes_per_GPU': 4}
        self.assertEqual(30 * 2 * 640 * 4, 153600)
        self.assertEqual(30 * 640 * 4, 76800)
        self.assertEqual(30 * 20 * 36, 21600)
        self.assertEqual(sum(sizes.values()), 252004)
        for i in self.indices('memory'):
            self.assertEqual(self.rows[i]['pinned_middle_Q8_bytes'], 2 * 21600)
            for bad in (0, 21600, 43199, 43201, 43200 + 8, 86400, True, 43200., None):
                self.change(i, ('pinned_middle_Q8_bytes',), bad)
            for key, value in sizes.items():
                self.assertEqual(self.rows[i][key], [value, value])
                for owner in (0, 1):
                    for bad in (0, value - 1, value + 1, True, float(value), None, '4'):
                        self.change(i, (key, owner), bad)
                for bad in ([value], [value] * 3, value, {}, None):
                    self.change(i, (key,), bad)
                missing = self.rows[i].pop(key)
                try:
                    self.reject(self.rows)
                finally:
                    self.rows[i][key] = missing
            # Middle capacity does not enlarge the original constructor probes.
            self.change(i, ('host_probe_bytes',), 33295968 + 43200)
        limits = self.collect(self.rows)['evidence_limits']
        self.assertIn('252004B_per_GPU', limits['GPU_middle_memory_scope'])
        self.assertIn('memory.steady_categories_devices_owned_ledger', limits['not_serialized'])
        self.assertIn('not_production_qualified', limits['whole_CPU_expert_scope'])

    def test_synthetic_failure_drains_preserves_publication_invalidates_ids_and_full_replay(self):
        failure = self.indices('submitted_admission_failure')[0]
        for key in ('failure_layer', 'queued_admission_copies', 'pending_host_slots_at_throw',
                    'CPU_return_bytes_before_throw', 'reset_full_replay_rows',
                    'fresh_physical_misses_first_replay_assignments'):
            self.change(failure, (key,), self.rows[failure][key] + 1)
        for value in (0, 6, 29, True, 7.):
            self.change(failure, ('accepted_CPU_jobs',), value)
        for value in (7, 28):
            self.rows[failure]['accepted_CPU_jobs'] = value
            self.assertIs(self.collect(self.rows)['passed'], True)
        replay = self.named('full_replay_after_submitted_admission_failure')[0]
        self.rows[replay][DISPATCH[0]] -= 1
        self.rows[replay][DISPATCH[1]] += 1
        self.reject(self.rows)  # Otherwise conserved replay must still have zero READY dispatch.

    def test_sticky_trace_io_scope_recovery_and_all_hit_no_extraction_upload(self):
        sticky = self.indices('sticky_failure')[0]
        self.change(sticky, ('injection',), 'numeric_or_post_CPU_failure')
        hit = self.indices('real_all_hit')[0]
        for key in ('ready_assignments', 'CPU_extractions', 'CPU_jobs', 'GPU_uploads'):
            self.change(hit, (key,), self.rows[hit][key] + 1)
        bos = self.named('real_all_hit_reset_BOS')[0]
        self.rows[bos].update(dict(zip(DISPATCH, (1, 479, 0))))
        self.reject(self.rows)  # A conserved but non-all-hit BOS cannot pass.

    def test_captured_reader_real_slot_eviction_quota_and_n1_group_assignments(self):
        case = self.indices('captured_reader_reuse')[0]
        for key in ('slots', 'quota', 'GPU_hit_groups', 'CPU_groups', 'GPU_miss_groups'):
            for bad in (self.rows[case][key] + 1, True, 1.):
                self.change(case, (key,), bad)
        self.change(case, ('evicted_READY_slots',), 0)
        self.change(case, ('evicted_READY_slots',), 49)
        window = self.named('mixed_slot1_captured_reader_then_eviction_BOS')[0]
        self.rows[window][DISPATCH[0]] -= 1
        self.rows[window][DISPATCH[1]] += 1
        self.reject(self.rows)  # A summary cannot silently describe a different dispatch.

    def test_n4_short_wide_short_counters_probe_lifetime_return_and_geometry(self):
        for i in self.named('max4_GPU_fallback_N4'):
            for key in (*DISPATCH, *CUMULATIVE_STAGES):
                self.change(i, (key,), self.rows[i][key] + 1)
            self.change(i, ('intermediate_probe_available',), True)
            self.change(i, ('unweighted_down',), metrics(4 * DOWN))
        wide_summary = self.indices('short_wide_short')[0]
        for key in ('short_calls', 'wide_N4_calls', 'short_layers', 'gpu_only_wide_layers',
                    'stage_capacity_bytes_each', 'total_pinned_stage_bytes', 'contribution_bytes_each_GPU'):
            self.change(wide_summary, (key,), self.rows[wide_summary][key] + 1)
        self.change(wide_summary, ('teacher_window_sizes', 1), 3)
        short = next(i for i in self.named('max4_short_hybrid_N2'))
        self.change(short, ('unweighted_down',), None)

    def test_memory_workers_pinned_shared_buffers_views_pool_headers_and_payload_limits(self):
        for i in self.indices('memory'):
            for key in ('session_instance', 'slots', 'max_batch_tokens', 'cpu_workers', 'pinned_input_bytes',
                         'pinned_output_bytes', 'pinned_error_bytes', 'pinned_stage_bytes', 'pinned_route_metadata_bytes'):
                self.change(i, (key,), self.rows[i][key] + 1)
            for key in ('route_metadata_bytes_per_GPU', 'contribution_bytes_per_GPU', 'stage_buffer_bytes_per_GPU'):
                for owner in (0, 1):
                    self.change(i, (key, owner), self.rows[i][key][owner] + 1)
                self.change(i, (key,), self.rows[i][key][:1])
            for key in ('host_plan_bytes', 'host_cpu_views_and_pending_bytes', 'host_probe_bytes',
                        'host_input_probe_bytes',
                        'pool_metadata_bytes', 'pool_scratch_bytes', 'host_routing_capacity_bytes',
                        'RouteGroups_requested_payload_bytes'):
                for bad in (0, -1, True, 1., 1 << 64):
                    self.change(i, (key,), bad)
        # Larger observed host/pool capacities are valid, not allocator-size guesses.
        for i in self.indices('memory'):
            for key in ('host_plan_bytes', 'host_cpu_views_and_pending_bytes',
                        'pool_metadata_bytes', 'pool_scratch_bytes', 'host_routing_capacity_bytes'):
                self.rows[i][key] += 1024
        self.assertIs(self.collect(self.rows)['passed'], True)
        self.change(self.indices('memory')[1], ('RouteGroups_requested_payload_bytes',), 13344)
        self.change(self.indices('memory')[1], ('host_plan_bytes',), 1)

    def test_constructor_input_probe_exact_double_buffer_capacity_and_total_all_batch_classes(self):
        # Independently sum the five payloads; metadata is NOT an extra payload.
        q8 = 2 * 3 * 48 * 80 * 36
        route_ids = route_weights = 2 * 3 * 48 * 10 * 4
        cpu_flags, available = 2 * 3 * 48 * 10, 2 * 3 * 48
        self.assertEqual((q8, route_ids, route_weights, cpu_flags, available),
                         (829440, 11520, 11520, 2880, 288))
        self.assertEqual(q8 + route_ids + route_weights + cpu_flags + available, 855648)
        intermediate = 2 * 3 * (48 * 10 * 2560 + 48 * 2560) * 4
        self.assertEqual(intermediate, 32440320)
        self.assertEqual(intermediate + 855648, 33295968)
        self.assertEqual(2 * 2 * 5 * 24, 480)
        self.assertEqual([self.rows[i]['max_batch_tokens'] for i in self.indices('memory')], [3, 1, 4])
        for i in self.indices('memory'):
            self.assertEqual(self.rows[i]['host_input_probe_bytes'], 855648)
            self.assertEqual(self.rows[i]['host_probe_bytes'], 33295968)
            for bad in (0, 427824, 855647, 855649, 855648 * 2, True, 855648., None, '855648'):
                self.change(i, ('host_input_probe_bytes',), bad)
            for bad in (32440320, 33295967, 33295969, 33295968 + 855648,
                        33295968 + 480, 33295968., True):
                self.change(i, ('host_probe_bytes',), bad)
            old = self.rows[i].pop('host_input_probe_bytes')
            try:
                self.reject(self.rows)
            finally:
                self.rows[i]['host_input_probe_bytes'] = old
        result = self.collect(self.rows)
        self.assertEqual(result['fixture_contract']['constructor_host_input_probe_payload_bytes_required'], 855648)
        self.assertEqual(result['fixture_contract']['constructor_host_total_probe_payload_bytes_required'], 33295968)
        self.assertEqual(result['fixture_contract']['input_probe_plan_metadata_added_bytes_LP64'], 480)
        self.assertIn('window.input_probe_original_route_dispatch_payloads',
                      result['evidence_limits']['not_serialized'])

    def test_memory_owner_record_order_and_plan_headers_remain_strict(self):
        first, second, third = self.indices('memory')
        for left, right in ((first, second), (second, third), (first, third)):
            self.rows[left], self.rows[right] = self.rows[right], self.rows[left]
            try:
                self.reject(self.rows)
            finally:
                self.rows[left], self.rows[right] = self.rows[right], self.rows[left]
        for i in (second, third):
            self.change(i, ('host_plan_bytes',), self.rows[i]['host_plan_bytes'] - 480)
        # A different reported sizeof-based positive baseline can be consistent;
        # this log has no old-size field with which to assert a 480-byte delta.
        for i in (first, second, third):
            self.rows[i]['host_plan_bytes'] += 1024
        self.assertIs(self.collect(self.rows)['passed'], True)

    def test_force_cpu_offset1_frozen_failure_or_incomplete_log_cannot_append_success(self):
        window = self.named('diagnostic_force_CPU_LINEAR_GPU_middle_N1')[1]
        self.assertEqual(self.rows[window]['offset'], 1)
        for key in ('logits', 'unweighted_down', 'ffn_output'):
            self.change(window, (key, 'violations'), 1)
            self.change(window, (key, 'max_bound_ratio'), math.nextafter(1., math.inf))
        # A failed driver can print the offending window, then exits with no
        # success footer. Exercise both a new journal and existing temp history.
        self.rows[window]['logits'].update(violations=1, max_abs=.021,
                                          max_bound_ratio=1.05, diagnostic_bit_mismatches=1)
        self.raw.write_bytes(encoded(self.rows[:window + 1]))
        process = self.cli()
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertEqual(process.stdout, '')
        self.assertFalse(self.results.exists())
        history = b'{"kind":"prior_synthetic_test_record"}\n'
        self.results.write_bytes(history)
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)
        # Even a fabricated complete footer cannot hide the frozen violation.
        self.raw.write_bytes(encoded(self.rows))
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)

    def test_footer_every_fixed_field_counts_descriptors_and_qualification_limits(self):
        footer = len(self.rows) - 1
        for key, value in self.rows[footer].items():
            if type(value) is int:
                self.change(footer, (key,), value + 1)
            elif type(value) is str:
                self.change(footer, (key,), 'changed')
        self.change(footer, ('sticky_failure_scopes',), list(reversed(self.rows[footer]['sticky_failure_scopes'])))
        limits = self.collect(self.rows)['evidence_limits']
        for field in ('source.model_path', 'window.stats_before_after', 'window.route_stats',
                      'window.ready_hit_assignments', 'window.physical_miss_assignments',
                      'phase.atomic_rejection_proofs', 'memory.expert_payload_reads', 'complete.owned_buffer_release'):
            self.assertIn(field, limits['not_serialized'])
        contract = self.collect()['fixture_contract']
        self.assertEqual(contract['constructor_payload_reads_required'], 144)
        self.assertEqual(contract['constructor_payload_bytes_required'], 68262297600)
        self.assertEqual(contract['atomic_reject_driver_checks_required'], 175)
        self.assertEqual(contract['atomic_reject_active_logit_values_required'], 110999040)
        self.assertIn('not_inflight_timing_proof', contract['synthetic_failure'])

    def test_duplicate_keys_nonfinite_malformed_nonobject_utf8_deep_and_unfinished_logs(self):
        raw = self.original_raw
        for field in ('revision', 'dirty', 'values', 'max_abs', 'phase', 'rows', 'accepted_CPU_jobs',
                      'CPU_input_bytes_checked', 'pinned_input_bytes', 'passed'):
            key = ('"' + field + '":').encode()
            self.raw.write_bytes(raw.replace(key, key + b'0,' + key, 1))
            with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                self.collect()
        for bad in (b'NaN', b'Infinity', b'-Infinity', b'1e999'):
            self.raw.write_bytes(raw.replace(b'"max_abs":0', b'"max_abs":' + bad, 1))
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                self.collect()
        lines = raw.splitlines()
        for bad in (raw[:-1], raw[:-20], raw + b'\n', raw + b'failure\n', b'\xef\xbb\xbf' + raw,
                    raw.replace(b'session_hybrid_header', b'\xff', 1)):
            self.raw.write_bytes(bad)
            with self.assertRaises(ValueError):
                self.collect()
        for obj in (b'[]', b'null', b'true', b'1', b'{} {}', b'{',
                    b'{"x":' + b'[' * 2000 + b'0' + b']' * 2000 + b'}',
                    b'{"x":' + b'9' * 10000 + b'}'):
            self.raw.write_bytes(b'\n'.join([*lines[:2], obj, *lines[3:]]) + b'\n')
            with self.assertRaises(ValueError):
                self.collect()
        self.raw.write_bytes(raw.replace(b'\n', b'\r\n'))
        self.assertIs(self.collect()['passed'], True)

    def test_whole_16mib_regular_preopen_and_growing_read_limits(self):
        self.assertEqual(MODULE.MAX_RAW_BYTES, 16 * 1024 * 1024)
        raw, limit = self.original_raw, MODULE.MAX_RAW_BYTES
        # Exercise the whole-file limit without an object expansion attack.
        lines = raw.splitlines(keepends=True)
        quotient, remainder = divmod(limit - len(raw), len(lines))
        padded = b''.join(b' ' * (quotient + (i < remainder)) + line for i, line in enumerate(lines))
        self.assertEqual(len(padded), limit)
        self.raw.write_bytes(padded)
        self.assertIs(self.collect()['passed'], True)
        self.raw.write_bytes(b' ' + padded)
        with self.assertRaisesRegex(ValueError, 'oversized'):
            self.collect()
        self.raw.write_bytes(b' ' * MODULE.MAX_RECORD_BYTES + raw)
        with self.assertRaisesRegex(ValueError, 'oversized JSON record'):
            self.collect()
        self.raw.write_bytes(raw)
        real_open = Path.open

        def growing_open(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            proxy = mock.MagicMock(wraps=stream)
            proxy.__enter__.return_value = proxy
            proxy.__exit__.side_effect = lambda *_: stream.close()
            proxy.read.return_value = b'x' * (limit + 1)
            return proxy

        with mock.patch.object(Path, 'open', autospec=True, side_effect=growing_open):
            with self.assertRaisesRegex(ValueError, 'oversized'):
                self.collect()
        with self.assertRaisesRegex(ValueError, 'regular file'):
            MODULE.collect(self.root)
        fifo = self.root / 'fifo'
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, 'regular file'):
            MODULE.collect(fifo)

    def test_cli_append_once_exact_history_relative_explicit_and_shared_locked_helper(self):
        history = b'{ "kind": "old", "values":[1,2] }\n{"kind":"second","dirty":1}\n'
        self.results.write_bytes(history)
        process = self.cli()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, '')
        saved = self.results.read_bytes()
        self.assertTrue(saved.startswith(history))
        self.assertEqual(len(saved.splitlines()), 3)
        self.assertEqual([json.loads(line) for line in saved.splitlines()[:2]],
                         [json.loads(line) for line in history.splitlines()])
        result = json.loads(saved.splitlines()[-1])
        self.assertEqual(result['source'], self.rows[0])
        self.assertEqual(result['complete'], self.rows[-1])
        self.assertEqual(self.raw.read_bytes(), self.original_raw)
        self.assertIs(MODULE.append_record, record_memory.append_result)
        with mock.patch.object(MODULE, 'append_record', wraps=record_memory.append_result) as append:
            self.assertEqual(MODULE.main(['--raw', str(self.raw), '--results', str(self.results)]), 0)
            append.assert_called_once()
        self.assertEqual(self.cli(Path('explicit-relative.jsonl')).returncode, 0)

    def test_cli_requires_raw_results_no_overrides_and_validation_before_open(self):
        for args in ([], ['--raw', str(self.raw)], ['--results', str(self.results)],
                     ['--raw', str(self.raw), '--results', str(self.results), '--absolute', '1']):
            self.assertEqual(self.cli(args=args).returncode, 2)
            self.assertFalse(self.results.exists())
        history = b'{"kind":"first"}\n'
        self.results.write_bytes(history)
        self.raw.write_bytes(self.original_raw[:-1])
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)
        self.results.unlink()
        self.assertEqual(self.cli().returncode, 1)
        self.assertFalse(self.results.exists())
        with mock.patch.object(MODULE, 'collect', side_effect=OSError('input failure')):
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(MODULE.main(['--raw', str(self.raw), '--results', str(self.results)]), 1)
                self.assertIn('record_hybrid: input failure', errors.getvalue())

    def test_alias_symlink_hardlink_declared_filename_guard_unfinished_and_nonregular_journal(self):
        for artifact in (self.raw, self.root / 'qwen38-keep1-Q4_0.gguf'):
            if artifact != self.raw:
                artifact.write_bytes(b'synthetic declared model filename, never opened as weights')
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
        for destination in (self.root, fifo, self.root / 'absent' / 'results.jsonl', Path('/dev/stdout'), Path('/dev/stderr')):
            process = self.cli(destination)
            self.assertEqual(process.returncode, 1)
            self.assertEqual(process.stdout, '')

    def test_serialization_lock_short_write_exception_flush_rollback(self):
        record = self.collect()
        history = b'{"kind":"first"}\n{"kind":"second","proof":[1,2]}\n'
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

    def test_alias_guard_repeated_after_locked_open(self):
        record = self.collect()
        self.results.write_bytes(b'{"kind":"old"}\n')
        before = self.raw.read_bytes()
        real_open = Path.open

        def replace_before_open(path, *args, **kwargs):
            self.results.unlink()
            os.link(self.raw, self.results)
            return real_open(path, *args, **kwargs)

        with mock.patch.object(Path, 'open', autospec=True, side_effect=replace_before_open):
            with self.assertRaisesRegex(ValueError, 'input artifact'):
                MODULE.append_record(self.results, record)
        self.assertEqual(self.raw.read_bytes(), before)

    def test_parallel_history_appends_are_whole_individual_records(self):
        history = b'{"kind":"first"}\n{"kind":"second"}\n'
        self.results.write_bytes(history)
        command = [sys.executable, '-B', str(SCRIPT), '--raw', str(self.raw), '--results', str(self.results)]
        processes = [subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True) for _ in range(2)]
        try:
            for process in processes:
                _, errors = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, errors)
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
