"""Exact synthetic restore emitter fixtures; no model/HIP or repository journal."""

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
from tools import record_memory
from tools import record_restore as MODULE

SCRIPT = ROOT / 'tools/record_restore.py'
REVISION = '0123456789abcdef0123456789abcdef01234567'


def public(forward, cursor, rows, position, windows=0, verify=0, k=None, pending=False,
           attention=False, pointer=4096, unavailable=False, recovery=False):
    """Synthetic, independently transcribed getter schema (not GPU evidence)."""
    hybrid = dict.fromkeys(('short_layers gpu_only_wide_layers ready_hit_assignments physical_miss_assignments '
                            'group_reuse_assignments cpu_groups cpu_assignments gpu_hit_groups gpu_hit_assignments '
                            'gpu_miss_groups gpu_miss_assignments admitted_groups evicted_ready_slots input_extractions '
                            'input_bytes cpu_return_bytes cpu_input_bytes_checked all_hit_layers forced_cpu_layers '
                            'forced_gpu_layers cpu_gate_up_jobs cpu_down_jobs gpu_middle_columns gpu_middle_batches '
                            'paired_gate_up_bytes middle_q8_bytes').split(), 0)
    if recovery:
        hybrid.update(short_layers=288, physical_miss_assignments=2880,
                      cpu_groups=2304, cpu_assignments=2304, gpu_miss_groups=576, gpu_miss_assignments=576,
                      admitted_groups=576, evicted_ready_slots=528, input_extractions=288, input_bytes=829440,
                      cpu_return_bytes=23592960, cpu_input_bytes_checked=6635520,
                      cpu_gate_up_jobs=2304, cpu_down_jobs=2304, gpu_middle_columns=2304,
                      gpu_middle_batches=288, paired_gate_up_bytes=11796480, middle_q8_bytes=1658880)
    obj = {
        'stats': {'consumed_tokens': cursor, 'expert_hits': 0, 'expert_misses': 480 * forward,
                  'expert_upload_bytes': 1327104000 * forward,
                  'last_completed_ms_bits': 4607182418800017408 if forward else 0},
        'route': {'last_max_expert_group_assignments': 1 if forward else 0, 'expert_groups_gt128': 0},
        'hybrid': hybrid,
        'attention': {'batch_calls': 36 if attention else 0, 'query_rows': 36 if attention else 0,
                      'multiquery_calls': 0, 'multiquery_rows': 0, 'singleton_tail_calls': 36 if attention else 0,
                      'max_query_rows': 1 if attention else 0},
        'speculative': {'target_forward_rows': forward, 'verify_windows': windows, 'verify_rows': verify,
                        'restore_calls': int(k is not None), 'retained_inputs': 0 if k is None else k,
                        'gdn_prefix_calls': 36 * windows, 'ple_prefix_calls': windows,
                        'qsa_tail_prefix_calls': 12 * windows,
                        'restored_prefixes': [int(k == i) for i in range(4)]},
        'target_tap': {'device': 1, 'rows': rows, 'width': 10240, 'first_position': position,
                       'pointer_address': pointer if rows else 0, 'pointer_present': bool(rows)},
        'checkpoint_available': not unavailable,
        'pending_window': None if unavailable else {'enabled': True, 'pending': pending,
                           'start_position': position if pending else 0, 'inputs': rows if pending else 0,
                           'valid_slots': rows + 1 if pending else 0},
    }
    if unavailable:
        obj['checkpoint_unavailable_reason'] = 'execution_failure_requires_reset'
    return obj


def window(start, n, k, attention=False):
    return {'preverify': public(start, start, 1, start - 1),
            'postverify': public(start + n, start + n, n, start, 1, n, pending=True, attention=attention, pointer=8192),
            'postrestore': public(start + n, start + k, n, start, 1, n, k, attention=attention, pointer=8192),
            'postcontinuation': public(start + n + 3, start + k + 3, 1, start + k + 2, 1, n, k, attention=attention)}


def rejected(group, count, snapshot):
    return {'group': group, 'count': count, 'before': copy.deepcopy(snapshot), 'after': copy.deepcopy(snapshot)}


def phase(rows, taps, restores, rejects, memory, sticky=0):
    return {'compared_full_vocab_values': rows * 248320, 'compared_full_vocab_rows': rows,
            'tap_exact_values': taps, 'successful_restores': restores, 'argument_rejections': rejects,
            'steady_memory_observations': memory, 'sticky_rejections': sticky}


def loaded(owner, index, batch, workers):
    """Actual schema, with explicitly synthetic capacities/sizeof/pointers/VRAM."""
    capacity = 2 * batch if index == 1 else 40
    tap = batch * 40960
    private = [236851200, 235376640 + 2 * tap]
    stage = 45875200 if batch > 3 else 2867200 if workers else 0
    memory = dict.fromkeys(('capacity expert_slots cpu_workers attention_query_tile ram_expert_capacity '
                           'ram_expert_payload host_embedding_capacity host_logit_capacity pinned_handoff '
                           'expert_payload_reads expert_payload_bytes_read pinned_expert_staging pinned_hybrid_input '
                           'pinned_hybrid_output pinned_hybrid_error pinned_hybrid_middle host_hybrid_plans '
                           'host_cpu_expert_views cpu_pool_metadata cpu_pool_scratch host_hybrid_probe '
                           'host_routing_capacity host_route_group_payload pinned_route_metadata host_hybrid_input_probe '
                           'host_session_impl_bytes host_session_config_bytes host_speculative_owner_bytes '
                           'host_speculative_hash_payload_bytes').split(), 0)
    memory.update(capacity=capacity, expert_slots=1, cpu_workers=workers, attention_query_tile=1,
                  speculative_checkpoints=True, ownership_verified=True,
                  ram_expert_capacity=68262297600, ram_expert_payload=68262297600,
                  host_embedding_capacity=1430323200, host_logit_capacity=2 * batch * 248320 * 4,
                  pinned_handoff=tap, expert_payload_reads=144, expert_payload_bytes_read=68262297600,
                  pinned_expert_staging=4 * stage, host_routing_capacity=2 * batch * 532 * 4,
                  host_route_group_payload=32768 + 160 * batch, pinned_route_metadata=160 * batch,
                  host_session_impl_bytes=8192, host_session_config_bytes=96,
                  host_speculative_owner_bytes=4096, host_speculative_hash_payload_bytes=1024)
    memory.update(checkpoint_recurrent_bytes=[226492416] * 2, checkpoint_history_bytes=[8847360] * 2,
                  checkpoint_qsa_tail_bytes=[36864] * 2, checkpoint_ple_history_bytes=[1474560, 0],
                  target_tap_bytes=[0, tap], target_tap_staging_bytes=[0, tap],
                  expert_stage_capacity_bytes=[stage] * 2, route_metadata_bytes=[batch * 80] * 2)
    for key, value in {'hybrid_contribution_bytes': 307200, 'hybrid_gate_up_bytes': 153600,
                       'hybrid_middle_float_bytes': 76800, 'hybrid_middle_q8_bytes': 21600,
                       'hybrid_middle_error_bytes': 4}.items():
        memory[key] = [value if workers else 0] * 2
    if workers:
        memory.update(pinned_hybrid_input=17280, pinned_hybrid_output=614400, pinned_hybrid_error=8,
                      pinned_hybrid_middle=43200, host_hybrid_plans=32768, host_cpu_expert_views=3000000,
                      cpu_pool_metadata=4096, cpu_pool_scratch=524288, host_hybrid_input_probe=855648,
                      host_hybrid_probe=855648 + 32440320)
    scratch = max(32768, batch * 12288) * 4
    attention = {'gathered_key_bytes': 2100224, 'gathered_value_bytes': 2100224,
                 'partial_output_bytes': 811008, 'partial_max_sum_bytes': 6336,
                 'staged_buffer_bytes': scratch, 'output_buffer_bytes': scratch,
                 'selected_id_bytes': 8204, 'selected_block_bytes': 2048,
                 'selected_count_bytes': 8, 'private_workspace_bytes': 5042368}
    memory['attention'] = [copy.deepcopy(attention) for _ in range(2)]
    memory['devices'] = []
    for id in range(2):
        d = {'device': id, 'first_layer': id * 24, 'last_layer': id * 24 + 23,
             'gdn_layers': 18, 'qsa_layers': 6, 'weights': 2000000000,
             'expert_slots': 66969600 if id == 0 else 66355200,
             'qsa_kv': 6 * capacity * 2 * 16 * 18, 'qsa_index': 6 * (capacity // 4 * 128 + 384) * 4,
             'gdn_state': 58834944, 'ple_state': 368640 if id == 0 else 0,
             'workspace': private[id] + 24 * scratch + 2000000000,
             'owned_peak_bytes': 12000000000, 'owned_buffers': 512,
             'total_vram': 17163091968, 'free_vram': 1000000000}
        d['category_sum_bytes'] = sum(d[key] for key in ('weights', 'expert_slots', 'qsa_kv', 'qsa_index', 'gdn_state', 'ple_state', 'workspace'))
        d['owned_bytes'] = d['category_sum_bytes']
        memory['devices'].append(d)
    return {'protocol': 1, 'kind': 'session_restore_loaded_memory', 'owner': owner, 'owner_index': index,
            'live_session_owners': 1,
            'config': {'capacity': capacity, 'expert_slots': 1, 'max_batch_tokens': batch, 'cpu_workers': workers,
                       'hybrid_probe': bool(workers), 'attention_query_tile': 1, 'trace_first_token': 0,
                       'trace_directory': '', 'speculative_checkpoints': True},
            'tap_capacity': {'device': 1, 'rows': batch, 'width': 10240, 'published_bytes': tap, 'staging_bytes': tap},
            'speculative_buffer_counts_from_validated_geometry': [43, 44],
            'speculative_buffer_count_total_from_geometry': 87, 'speculative_buffer_bytes_by_device': private,
            'host_logit_buffer_count_required': 2, 'host_each_logit_requested_bytes': batch * 248320 * 4,
            'host_logit_combined_capacity_floor_bytes': 2 * batch * 248320 * 4,
            'memory': memory, 'initial_public': public(0, 0, 0, 0),
            'ledger_scope': 'all_live_Session_Buffer_pointer_tree_and_reported_host_capacities',
            'host_config_included_in_impl': True, 'host_input_probe_included_in_hybrid_probe': True,
            'speculative_GPU_bytes_included_in_workspace': True,
            'constructor_read_scope': 'expert_payload_API_reads_not_physical_SSD_trace',
            'peak_scope': 'Buffer_process_peak_not_reset_between_sequential_owners',
            'evidence_limits': ['separate_working_logit_capacity_not_exposed', 'snapshot_Buffer_count_geometry_derived',
                                'private_hash_history_and_route_allocator_slack_excluded',
                                'allocator_overhead_and_thread_stacks_excluded', 'HIP_rocBLAS_context_allocations_not_owned_ledger'],
            'full_RSS_measured': False, 'owned_buffer_release_measured': False, 'passed': True}


def fixture(wide=32, model='/synthetic/qwen38-keep1-Q4_0.gguf'):
    """All 56 current protocol-1 records, with no model/HIP execution."""
    rows = [{
        'protocol': 1, 'kind': 'session_restore_source', 'revision': REVISION, 'dirty': True,
        'capacity': 40, 'slots': 1, 'max_batch': 3, 'ple_eos': 248044,
        'cpu_workers': 0, 'absolute_gate': .02, 'relative_gate': .002,
        'speculative_checkpoints': True, 'model_path': model, 'model_argument': model,
        'expert_inventory': {'scope': 'tensor_inventory_no_payload_reads', 'matrices': 144,
                             'bytes': 68262297600, 'q4_0_down_layers': 42, 'q4_1_down_layers': 6},
        'runtime_geometry': {'vocabulary': 248320, 'layers': 48, 'gdn_layers': 36, 'qsa_layers': 12,
            'kv_type': 'Q4_0', 'kv_width': 512, 'kv_blocks_per_position': 16, 'kv_block_elements': 32,
            'kv_block_bytes': 18, 'pooled_index_type': 'F32', 'pooled_index_width': 128,
            'qsa_block_tokens': 4, 'qsa_raw_tail_elements': 384, 'gdn_recurrent_elements': 786432,
            'gdn_history_elements': 30720, 'ple_conv_history_elements': 92160,
            'checkpoint_slots': 4, 'max_verify_inputs': 3, 'tap_width': 10240, 'tap_device': 1,
            'tap_location': 'layer47_combine_before_root_HC'},
        'source_required': {'records': 56, 'loaded_memory_owners': 3,
            'primary': {'successful_restores': 50, 'argument_rejections': 83, 'steady_memory_observations': 133},
            'CPU_sticky_rejections': 6, 'expert_payload_reads_per_owner': 144,
            'expert_payload_bytes_per_owner': 68262297600,
            'expert_geometry': {'layers': 48, 'experts_per_layer': 512, 'top_k': 10, 'input': 2560, 'middle': 640,
                               'gate_up_type': 'Q4_0', 'q4_0_down_layers': 42, 'q4_1_down_layers': 6,
                               'q4_0_triplet_bytes_per_expert': 2764800, 'q4_1_triplet_bytes_per_expert': 2867200}},
        'wide_PP_mode': 'default_32' if wide == 32 else 'explicit_1024', 'wide_PP_rows': wide,
        'reference': 'retained_N1_full_vocabulary_same_Session', 'performance_claim': False,
    }, loaded('primary_restoration', 0, 3, 0)]
    for start in (2, 3, 4, 5, 8):
        for n in (1, 2, 3):
            for retained in range(n + 1):
                rows.append({
                    'protocol': 1, 'kind': 'session_restore_case', 'owner': 'primary_restoration', 'start': start, 'inputs': n,
                    'retained': retained, 'restored_mod4': (start + retained) % 4,
                    'restored_blocks': (start + retained) // 4,
                    'attention_batch_enabled': (start, n, retained) == (5, 3, 1),
                    'continuation_rows': 3, 'public_snapshots': window(start, n, retained, (start, n, retained) == (5, 3, 1)),
                    'argument_rejections': [], 'published_full_span_and_tap_bits_preserved': True, 'passed': True,
                })
                r = rows[-1]
                if start == 3 and retained == 0:
                    r['argument_rejections'].append(rejected('invalid_pending_arguments', 11, r['public_snapshots']['postverify']))
                r['argument_rejections'].append(rejected('consumed_window_restore', 1, r['public_snapshots']['postrestore']))
    for retained in (0, 1, 2, 3):
        rows.append({'protocol': 1, 'kind': 'session_restore_divergent_suffix', 'owner': 'primary_restoration', 'start': 3, 'inputs': 3,
                     'retained': retained, 'continuation_rows': 3, 'public_snapshots': window(3, 3, retained), 'passed': True})
    rows.append(loaded('wide_PP_' + str(wide), 1, wide, 0))
    wide_phase = phase(2 * wide, 2 * wide * 10240, 1, 1, 2)
    cpu_phase = phase(9, 0, 0, 0, 2, 6)
    prereject = public(1 + wide, wide, wide, 0, 1, 1, 0)
    rows.append({'protocol': 1, 'kind': 'session_restore_wide_PP', 'owner': 'wide_PP_' + str(wide), 'rows': wide,
                  'public_snapshots': {'preverify': public(0, 0, 0, 0),
                        'postverify': public(1, 1, 1, 0, 1, 1, pending=True, pointer=8192),
                        'postrestore': public(1, 0, 1, 0, 1, 1, 0, pointer=8192),
                        'pre_late_ID_reject': prereject, 'post_late_ID_reject': copy.deepcopy(prereject),
                        'post_reset_replay': public(wide, wide, wide, 0, pointer=8192)},
                  'phase_totals': wide_phase,
                  'hidden_N1_numerical_comparison': {'values': wide * 10240, 'max_error': 0, 'max_bound_ratio': 0},
                  'all_rows_D2H_before_reuse': True, 'full_vocabulary_N1_reference': True,
                  'tap_replay_exact': True, 'passed': True})
    rows.append(loaded('CPU_failure_recovery', 2, 3, 1))
    rows.append({
        'protocol': 1, 'kind': 'session_restore_CPU_failure_recovery', 'owner': 'CPU_failure_recovery', 'cpu_workers': 1,
        'initial_policy': 'disabled',
        'fault': 'after_gate_up_submission_and_GPU_admission_before_join_middle_down',
        'old_tap_physical_bytes_preserved': True, 'sticky_rejections': 6,
        'recovery_full_vocab_rows': 6, 'inflight_timing_claim': False, 'passed': True,
        'fault_diagnostics': {'accepted_cpu_jobs': 24, 'queued_admission_copies': 2, 'pending_slots_at_failure': 1,
                             'cpu_return_bytes_before_failure': 0, 'ready_cache_ids': 0, 'pending_cache_ids': 0,
                             'failure_layer': 0, 'synthetic_failure': True, 'cpu_pool_drained': True, 'gpu_streams_drained': True},
        'public_snapshots': {'prefault': public(3, 3, 3, 0), 'postfault': public(3, 3, 3, 0, unavailable=True),
                             'poststicky': public(3, 3, 3, 0, unavailable=True), 'postreset': public(0, 0, 0, 0),
                             'postrecovery': public(6, 6, 1, 5, recovery=True)}, 'phase_totals': cpu_phase,
    })
    rows.append({
        'protocol': 1, 'kind': 'session_restore_complete', 'restores': 50, 'rejections': 83,
        'compared_full_vocab_values': 151723520 if wide == 32 else 644390400,
        'diagnostic_bit_differences': 0, 'max_error': 0, 'max_bound_ratio': 0,
        'tap_exact_values': 6103040 if wide == 32 else 26419200,
        'memory_observations': 133, 'counts_scope': 'primary_restoration_fixture',
        'comparison_counts_scope': 'all_three_owners', 'compared_full_vocab_rows': 611 if wide == 32 else 2595,
        'phase_totals': {'primary_restoration': phase(538, 5447680, 50, 83, 133),
                         'wide_PP': wide_phase, 'CPU_failure_recovery': cpu_phase},
        'primary_extra_argument_rejections': [rejected('ordinary_PP_invalidated_pending', 1, public(9, 9, 3, 6, 1, 3)),
                                              rejected('capacity_and_prefix_arguments', 3, public(40, 40, 1, 39, 1, 1, pending=True)),
                                              rejected('restore_after_reset_without_window', 1, public(1, 1, 1, 0))],
        'aggregate_counts': {'successful_restores': 51, 'argument_rejections': 84, 'sticky_rejections': 6,
                             'steady_memory_observations': 137}, 'records': 56, 'loaded_memory_records': 3,
        'snapshot_counters_scope': 'completed_public_work_since_last_reset_not_kernel_counts',
        'tap_exact_count_scope': 'numerical_reference_and_replay_comparisons_excludes_rejection_and_failure_preservation_checks',
        'wide_PP_rows': wide, 'CPU_failure_recovery_passed': True, 'session_instances': 3,
        'raii_cleanup_completed': True, 'owned_buffer_release_measured': False,
        'trained_MTP_qualified': False, 'HF_qualified': False, 'R6_complete_claim': False,
        'performance_claim': False, 'passed': True,
    })
    return rows


def encoded(rows):
    return ''.join(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n' for row in rows).encode()


class RestoreResultsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='restore-results-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw, self.results = self.root / 'restore.jsonl', self.root / 'results.jsonl'
        self.model = self.root / 'qwen38-keep1-Q4_0.gguf'
        self.rows = fixture(model=str(self.model))
        self.original = encoded(self.rows)
        self.raw.write_bytes(self.original)

    def collect(self, rows=None):
        if rows is not None:
            self.raw.write_bytes(encoded(rows))
        return MODULE.collect(self.raw)

    def reject(self, rows):
        raw = encoded(rows)
        self.raw.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.collect()
        self.assertEqual(self.raw.read_bytes(), raw)
        self.assertFalse(self.results.exists())

    def change(self, index, key, value):
        old = self.rows[index][key]
        self.rows[index][key] = value
        try:
            with self.subTest(index=index, key=key, value=value):
                self.reject(self.rows)
        finally:
            self.rows[index][key] = old

    def nested_change(self, index, path, value):
        obj = self.rows[index]
        for key in path[:-1]:
            obj = obj[key]
        old = obj[path[-1]]
        obj[path[-1]] = value
        try:
            with self.subTest(index=index, path=path, value=value):
                self.reject(self.rows)
        finally:
            obj[path[-1]] = old

    def cli(self, destination=None, args=None):
        command = [sys.executable, '-B', str(SCRIPT)]
        command += args if args is not None else ['--raw', self.raw.name, '--results', str(destination or self.results)]
        return subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)

    def test_exact_default_counts_compact_aggregate_readonly_and_provenance(self):
        result = self.collect()
        self.assertEqual(len(self.rows), 56)
        self.assertEqual((MODULE.RECORD_COUNT, len(MODULE.CASES), MODULE.RESTORES,
                          MODULE.REJECTIONS, MODULE.MEMORY_OBSERVATIONS), (56, 45, 50, 83, 133))
        self.assertEqual(result['kind'], 'r6_target_restore')
        self.assertEqual(result['protocol'], 1)
        self.assertEqual(result['revision'], REVISION)
        self.assertIs(result['dirty'], True)
        self.assertEqual(result['source'], self.rows[0])
        self.assertEqual(result['complete'], self.rows[-1])
        self.assertEqual(result['wide_PP'], self.rows[52])
        self.assertEqual(result['CPU_failure_recovery'], self.rows[54])
        self.assertEqual(result['loaded_memory'], [self.rows[i] for i in (1, 51, 53)])
        self.assertEqual(result['comparison_summary']['full_vocab_rows'], 611)
        self.assertEqual(result['comparison_summary']['full_vocab_values'], 151723520)
        self.assertEqual(result['comparison_summary']['primary_compared_rows'], 538)
        self.assertEqual(result['comparison_summary']['exact_tap_values'], 6103040)
        self.assertEqual(result['raw_logs'], {'restore': str(self.raw.resolve())})
        self.assertEqual(datetime.datetime.fromisoformat(result['timestamp']).utcoffset(), datetime.timedelta(0))
        for key in ('windows', 'records', 'memory', 'cases'):
            self.assertNotIn(key, result)
        self.assertLess(len(json.dumps(result)), 65536)
        self.assertEqual(self.raw.read_bytes(), self.original)
        self.assertFalse(self.results.exists())

    def test_optional_wide1024_exact_extents_and_unchanged_primary_counts(self):
        result = self.collect(fixture(1024))
        self.assertEqual(result['comparison_summary']['full_vocab_rows'], 2595)
        self.assertEqual(result['complete']['compared_full_vocab_values'], 644390400)
        self.assertEqual(result['complete']['tap_exact_values'], 26419200)
        self.assertEqual(result['complete']['restores'], 50)
        self.assertEqual(result['complete']['rejections'], 83)
        self.assertEqual(result['complete']['memory_observations'], 133)
        self.rows = fixture(1024)
        self.change(52, 'rows', 32)  # Footer cannot silently describe another scope.
        for width in (0, 1, 3, 31, 33, 128, 1023, 1025, True, 32., 1 << 64):
            self.change(52, 'rows', width)

    def test_all_rows_closed_missing_extra_schema(self):
        for i, row in enumerate(self.rows):
            row['unexpected'] = 1
            try:
                self.reject(self.rows)
            finally:
                del row['unexpected']
            for key in list(row):
                old = row.pop(key)
                try:
                    with self.subTest(index=i, missing=key):
                        self.reject(self.rows)
                finally:
                    row[key] = old

    def test_version_required_no_old_null_or_mixed_versions(self):
        for i in range(len(self.rows)):
            for version in (None, 0, 2, True, '1', 1.):
                self.change(i, 'protocol', version)
        for row in self.rows:
            del row['protocol']
        self.reject(self.rows)

    def test_missing_duplicate_reordered_and_partial_timeline_every_row(self):
        for i in range(56):
            self.reject(self.rows[:i] + self.rows[i + 1:])
            self.reject(self.rows[:i] + [self.rows[i]] + self.rows[i:])
            self.reject(self.rows[:i])
            if i:
                alternate = copy.deepcopy(self.rows)
                alternate[i - 1], alternate[i] = alternate[i], alternate[i - 1]
                self.reject(alternate)
        self.reject(self.rows + [{'kind': 'session_restore_failure', 'passed': False}])

    def test_all_primary_case_coordinates_N123_k0_to_N_mod4_attention(self):
        cases = self.rows[2:47]
        self.assertEqual(len(cases), 45)
        self.assertEqual([sum(row['inputs'] == n for row in cases) for n in (1, 2, 3)], [10, 15, 20])
        self.assertEqual({row['start'] % 4 for row in cases}, {0, 1, 2, 3})
        self.assertEqual(sum(row['attention_batch_enabled'] for row in cases), 1)
        for i in range(2, 47):
            for key in ('start', 'inputs', 'retained', 'restored_mod4', 'restored_blocks', 'continuation_rows'):
                self.change(i, key, self.rows[i][key] + 1)
            self.change(i, 'attention_batch_enabled', not self.rows[i]['attention_batch_enabled'])
        for i in range(47, 51):
            for key in ('start', 'inputs', 'retained', 'continuation_rows'):
                self.change(i, key, self.rows[i][key] + 1)
        # Forged contiguous order with plausible aggregate coverage still fails.
        changed = copy.deepcopy(self.rows)
        changed[2:47] = sorted(changed[2:47], key=lambda row: (row['inputs'], row['start'], row['retained']))
        self.reject(changed)

    def test_types_bool_vs_integer_uint64_strings_null_every_record(self):
        for i, row in enumerate(self.rows):
            for key, value in list(row.items()):
                if type(value) is bool:
                    self.change(i, key, int(value))
                    if key != 'dirty':
                        self.change(i, key, not value)
                elif type(value) is int and key not in ('max_error', 'max_bound_ratio'):
                    for bad in (True, float(value), -1, None, str(value), 1 << 64):
                        self.change(i, key, bad)
        for key in ('max_error', 'max_bound_ratio'):
            for bad in (True, None, '0', -1):
                self.change(55, key, bad)

    def test_source_exact_compiled_revision_dirty_and_metadata_EOS(self):
        for revision in ('f' * 39, 'f' * 41, 'g' * 40, 'ABCDEF01' * 5, '', 1, None):
            self.change(0, 'revision', revision)  # main itself requires lowercase hex.
        for key in ('capacity', 'slots', 'max_batch', 'cpu_workers'):
            self.change(0, key, self.rows[0][key] + 1)
        for key, value in (('absolute_gate', .021), ('relative_gate', .003),
                           ('absolute_gate', True), ('reference', 'HF'), ('ple_eos', 248320)):
            self.change(0, key, value)
        self.rows[0].update(dirty=False, ple_eos=7)
        result = self.collect(self.rows)
        self.assertIs(result['dirty'], False)
        self.assertEqual(result['fixture_contract']['ple_source_ids']['ple_eos'], 7)
        self.assertEqual(result['fixture_contract']['ple_source_ids']['metadata_EOS_positions'], [0, 4, 8, 10, 37])
        # Metadata EOS is not frozen to tokenizer EOS or the known model's ID.
        self.assertEqual(result['source']['model_path'], str(self.model))
        self.assertEqual(result['model'], str(self.model))
        self.assertTrue(result['model_path_observable'])
        self.assertFalse(result['model_variant_attested'])

    def test_footer_every_count_and_scope_source_derived_not_just_positive(self):
        for key in ('restores', 'rejections', 'compared_full_vocab_values', 'tap_exact_values',
                    'memory_observations', 'wide_PP_rows', 'session_instances'):
            self.change(55, key, self.rows[55][key] + 1)
        self.change(55, 'counts_scope', 'all_three_Sessions')
        self.change(55, 'compared_full_vocab_values', 538 * 248320)  # Omits wide+CPU.
        self.change(55, 'tap_exact_values', 532 * 10240)  # Omits wide reject/replay.
        self.change(54, 'sticky_rejections', 5)
        self.change(54, 'recovery_full_vocab_rows', 3)
        self.change(54, 'fault', 'before_CPU_submission')
        self.change(54, 'initial_policy', 'mixed')

    def test_numerical_gate_finite_FP32_envelope_and_diagnostic_bits(self):
        for key, bad in (('max_error', .001), ('max_bound_ratio', .1),
                         ('max_bound_ratio', math.nextafter(1., math.inf)),
                         ('diagnostic_bit_differences', 151723521)):
            self.change(55, key, bad)
        self.rows[-1].update(max_error=.02, max_bound_ratio=1., diagnostic_bit_differences=1)
        self.assertTrue(self.collect(self.rows)['passed'])
        self.change(55, 'max_bound_ratio', math.nextafter(1., math.inf))
        self.rows[-1].update(max_error=.125, max_bound_ratio=.125 / .22)
        self.assertTrue(self.collect(self.rows)['passed'])  # Reference100 allows .22.
        self.change(55, 'max_error', .02 + .002 * MODULE.FP32_MAX + 1e35)
        self.change(55, 'max_bound_ratio', 0)
        self.change(55, 'diagnostic_bit_differences', 0)
        self.rows[-1].update(max_error=.015625, max_bound_ratio=.8)
        self.reject(self.rows)  # Ratio cannot exceed max_error/.02 at any reference.
        self.rows[-1].update(max_error=0, max_bound_ratio=0, diagnostic_bit_differences=400)
        result = self.collect(self.rows)  # Signed-zero differences are diagnostic.
        self.assertFalse(result['comparison_summary']['bit_identity_required'])

    def test_geometry_math_snapshots_double_taps_two_host_logit_owners_CPU_staging(self):
        result = self.collect()
        primary, wide, cpu = result['required_geometry']
        for geometry in (primary, wide, cpu):
            self.assertEqual(geometry['checkpoint_recurrent_bytes'], [226492416, 226492416])
            self.assertEqual(geometry['checkpoint_history_bytes'], [8847360, 8847360])
            self.assertEqual(geometry['checkpoint_qsa_tail_bytes'], [36864, 36864])
            self.assertEqual(geometry['checkpoint_ple_history_bytes'], [1474560, 0])
            self.assertEqual(geometry['additional_GPU_buffers'], [43, 44])
            self.assertIn('checked_against_loaded_getters', geometry['scope'])
        self.assertEqual(primary['target_tap_bytes'], [0, 122880])
        self.assertEqual(primary['target_tap_staging_bytes'], [0, 122880])
        self.assertEqual(primary['checkpoint_and_tap_GPU_bytes'], [236851200, 235622400])
        self.assertEqual(primary['working_logit_payload_bytes'], 2979840)
        self.assertEqual(primary['host_logit_capacity_floor_bytes'], 5959680)
        self.assertEqual(primary['expert_stage_capacity_bytes'], [0, 0])
        self.assertEqual(primary['pinned_expert_staging_bytes'], 0)
        self.assertEqual(cpu['expert_stage_capacity_bytes'], [2867200, 2867200])
        self.assertEqual(cpu['pinned_expert_staging_bytes'], 11468800)
        self.assertEqual(wide['target_tap_bytes'], [0, 1310720])
        self.assertEqual(wide['pinned_expert_staging_bytes'], 183500800)
        maximum = self.collect(fixture(1024))['required_geometry'][1]
        self.assertEqual(maximum['target_tap_bytes'], [0, 41943040])
        self.assertEqual(maximum['target_tap_staging_bytes'], [0, 41943040])
        self.assertEqual(maximum['checkpoint_and_tap_GPU_bytes'], [236851200, 319262720])
        self.assertEqual(maximum['working_logit_payload_bytes'], 1017118720)
        self.assertEqual(maximum['host_logit_capacity_floor_bytes'], 2034237440)
        for batch, workers in ((2, 0), (1025, 0), (True, 0), (3., 0), (3, True), (3, 2)):
            with self.assertRaises(ValueError):
                MODULE.required_geometry(batch, workers)

    def test_source_counter_contract_not_fabricated_observations_limits_explicit(self):
        result = self.collect()
        contract = result['fixture_contract']
        self.assertEqual(contract['primary_successful_forward_rows_required'], 596)
        self.assertEqual(contract['primary_route_assignments_required'], 286080)
        self.assertEqual(contract['primary_verify_windows_required'], 52)
        self.assertEqual(contract['primary_verify_rows_required'], 117)
        self.assertEqual(contract['restored_prefix_counts_required'], [17, 16, 11, 6])
        self.assertEqual(contract['retained_inputs_required'], 56)
        self.assertEqual(contract['prefix_APIs_per_successful_verify'], {'GDN': 36, 'PLE': 1, 'QSA_tail': 12})
        self.assertIn('not_physical_kernels', contract['prefix_count_scope'])
        self.assertIn('not_logical_cursor', contract['route_accounting'])
        self.assertEqual(contract['target_tap']['row_bytes'], 40960)
        self.assertEqual(contract['target_tap']['carry'], 'k>0_row_k_minus_1_k0_no_carry')
        self.assertEqual(len(contract['CPU_sticky_checks']), 6)
        missing = result['evidence_limits']['not_serialized']
        for field in ('full_raw_snapshot_buffers', 'individual_rejection_snapshots_within_group_and_full_span_bytes',
                       'individual_CPU_vector_capacities_and_allocator_slack',
                       'detailed_individual_owned_release', 'independent_HF_reference', 'trained_MTP',
                       'performance_measurements', 'full_R6', 'per_step_memory_ledgers'):
            self.assertIn(field, missing)
        for observed in ('protocol', 'SessionMemory_public_records_and_individual_GPU_buffers',
                         'per_case_stats_route_hybrid_attention_and_speculative_counters',
                         'host_object_sizeof_and_vector_capacities', 'CPU_capacity_observations'):
            self.assertNotIn(observed, missing)
        self.assertNotIn('counts_scope_defect', result['evidence_limits'])
        self.assertIn('OR_speculative_checkpoints', result['evidence_limits']['working_logits'])
        for key in ('trained_MTP_qualified', 'HF_qualified', 'R6_complete_claim', 'performance_claim', 'speedup_claim',
                    'physical_kernel_count_claim', 'full_ram_accounting_qualified', 'exact_peak_vram_claim',
                     'owned_buffer_release_measured', 'mtp', 'model_variant_attested'):
            self.assertIs(result[key], False)

    def test_every_nested_object_closed_and_array_extents(self):
        # Every nested schema family, including both pending rejection groups,
        # all three memory owners, the invalidated getter variant and footer.
        # Other chronological cases use the same validators.
        indices = (0, 1, 2, 11, 47, 51, 52, 53, 54, 55)

        def dictionaries(value):
            if type(value) is dict:
                yield value
                for child in list(value.values()):
                    yield from dictionaries(child)
            elif type(value) is list:
                for child in value:
                    yield from dictionaries(child)

        for index in indices:
            for obj in list(dictionaries(self.rows[index])):
                obj['unexpected_nested_field'] = 0
                try:
                    self.reject(self.rows)
                finally:
                    del obj['unexpected_nested_field']
                for key in list(obj):
                    previous = obj.pop(key)
                    try:
                        self.reject(self.rows)
                    finally:
                        obj[key] = previous
        for index, path in ((1, ('memory', 'devices')), (1, ('memory', 'attention')),
                            (1, ('memory', 'checkpoint_recurrent_bytes')),
                            (2, ('public_snapshots', 'postverify', 'speculative', 'restored_prefixes')),
                            (11, ('argument_rejections',)), (55, ('primary_extra_argument_rejections',))):
            obj = self.rows[index]
            for key in path:
                obj = obj[key]
            for bad in ([], obj[:-1], obj + [copy.deepcopy(obj[0])], {}, None):
                self.nested_change(index, path, bad)

    def test_nested_uint64_and_bool_types_all_getter_memory_and_counter_fields(self):
        def leaves(value, path=()):
            if type(value) is dict:
                for key, child in value.items():
                    yield from leaves(child, path + (key,))
            elif type(value) is list:
                for i, child in enumerate(value):
                    yield from leaves(child, path + (i,))
            else:
                yield path, value

        for index in (1, 2, 53, 54):
            for path, value in list(leaves(self.rows[index])):
                if type(value) is int:
                    # The strict schema must not accept Python bool/int equality.
                    self.nested_change(index, path, bool(value))
                    self.nested_change(index, path, float(value))
                elif type(value) is bool:
                    self.nested_change(index, path, int(value))
        for path in (('stats', 'last_completed_ms_bits'), ('stats', 'expert_hits'),
                     ('hybrid', 'short_layers'), ('target_tap', 'pointer_address'),
                     ('speculative', 'target_forward_rows')):
            for bad in (-1, 1 << 64, '1', None):
                self.nested_change(2, ('public_snapshots', 'postverify') + path, bad)

    def test_restore_and_verify_work_counters_cursor_and_pending_slots_tampering(self):
        for snapshot in ('preverify', 'postverify', 'postrestore', 'postcontinuation'):
            for field in ('target_forward_rows', 'verify_windows', 'verify_rows', 'restore_calls', 'retained_inputs',
                          'gdn_prefix_calls', 'ple_prefix_calls', 'qsa_tail_prefix_calls'):
                old = self.rows[4]['public_snapshots'][snapshot]['speculative'][field]
                self.nested_change(4, ('public_snapshots', snapshot, 'speculative', field), old + 1)
            old = self.rows[4]['public_snapshots'][snapshot]['stats']['consumed_tokens']
            self.nested_change(4, ('public_snapshots', snapshot, 'stats', 'consumed_tokens'), old + 1)
        for k in range(4):
            self.nested_change(4, ('public_snapshots', 'postrestore', 'speculative', 'restored_prefixes', k), 9)
        for key, bad in (('pending', False), ('inputs', 3), ('valid_slots', 1), ('start_position', 0), ('enabled', False)):
            self.nested_change(2, ('public_snapshots', 'postverify', 'pending_window', key), bad)
        for key, bad in (('pending', True), ('inputs', 1), ('valid_slots', 2)):
            self.nested_change(2, ('public_snapshots', 'postrestore', 'pending_window', key), bad)
        # Logical rollback does not erase completed physical rows/assignments.
        changed = copy.deepcopy(self.rows)
        p = changed[2]['public_snapshots']['postrestore']
        p['speculative']['target_forward_rows'] = p['stats']['consumed_tokens']
        p['stats']['expert_misses'] = 480 * p['stats']['consumed_tokens']
        self.reject(changed)
        # Keep the aggregate assignment sum valid, but rewind a physical counter.
        for snapshot in ('postrestore', 'postcontinuation'):
            changed = copy.deepcopy(self.rows)
            before = changed[2]['public_snapshots']['postverify']
            before['stats']['expert_hits'] = 20
            before['stats']['expert_misses'] -= 20
            self.reject(changed)
        for section, field in (('stats', 'expert_upload_bytes'), ('stats', 'last_completed_ms_bits'),
                               ('route', 'last_max_expert_group_assignments'), ('route', 'expert_groups_gt128'),
                               ('hybrid', 'gpu_middle_columns'), ('attention', 'query_rows')):
            previous = self.rows[2]['public_snapshots']['postrestore'][section][field]
            self.nested_change(2, ('public_snapshots', 'postrestore', section, field), previous + 1)
        for bits in (0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000001, 0xbff0000000000000):
            self.nested_change(2, ('public_snapshots', 'preverify', 'stats', 'last_completed_ms_bits'), bits)

    def test_grouped_rejections_preserve_all_public_bits_and_metadata(self):
        for index, group in ((2, 0), (11, 0), (11, 1)):
            self.nested_change(index, ('argument_rejections', group, 'count'), 0)
            for section, field in (('stats', 'last_completed_ms_bits'), ('stats', 'expert_upload_bytes'),
                                   ('speculative', 'restore_calls'), ('pending_window', 'valid_slots'),
                                   ('target_tap', 'pointer_address'), ('target_tap', 'first_position')):
                old = self.rows[index]['argument_rejections'][group]['after'][section][field]
                self.nested_change(index, ('argument_rejections', group, 'after', section, field), old + 1)
        for group in range(3):
            self.nested_change(55, ('primary_extra_argument_rejections', group, 'after', 'stats', 'last_completed_ms_bits'), 0)
        self.nested_change(52, ('public_snapshots', 'post_late_ID_reject', 'stats', 'last_completed_ms_bits'), 0)

    def test_tap_F32_extent_owner_first_position_restore_and_kcarry_metadata(self):
        for snapshot in ('preverify', 'postverify', 'postrestore', 'postcontinuation'):
            tap = self.rows[2]['public_snapshots'][snapshot]['target_tap']
            for key in ('device', 'rows', 'width', 'first_position'):
                self.nested_change(2, ('public_snapshots', snapshot, 'target_tap', key), tap[key] + 1)
            for pointer in (0, 3, 4097):
                self.nested_change(2, ('public_snapshots', snapshot, 'target_tap', 'pointer_address'), pointer)
            self.nested_change(2, ('public_snapshots', snapshot, 'target_tap', 'pointer_present'), False)
        self.nested_change(2, ('public_snapshots', 'postrestore', 'target_tap', 'pointer_address'), 12288)
        self.nested_change(2, ('public_snapshots', 'postverify', 'target_tap', 'pointer_address'), 4096)
        self.nested_change(2, ('public_snapshots', 'postcontinuation', 'target_tap', 'pointer_address'), 8192)
        self.nested_change(52, ('public_snapshots', 'postrestore', 'target_tap', 'pointer_address'), 12288)
        self.nested_change(54, ('public_snapshots', 'postfault', 'target_tap', 'pointer_address'), 8192)
        # Bytes/carry are asserted by the driver, not fabricated as tensor evidence.
        limits = self.collect(self.rows)['evidence_limits']['not_serialized']
        self.assertIn('tap_tensor_carry_bytes_and_independent_pre_root_HC_provenance', limits)

    def test_memory_observed_getter_geometry_double_count_and_actual_unknown_sizeof(self):
        for index in (1, 51, 53):
            for field in ('checkpoint_recurrent_bytes', 'checkpoint_history_bytes', 'checkpoint_qsa_tail_bytes',
                          'checkpoint_ple_history_bytes', 'target_tap_bytes', 'target_tap_staging_bytes',
                          'expert_stage_capacity_bytes', 'route_metadata_bytes'):
                old = self.rows[index]['memory'][field][0]
                self.nested_change(index, ('memory', field, 0), old + 4)
            for device in (0, 1):
                for field in ('owned_bytes', 'category_sum_bytes', 'qsa_kv', 'qsa_index', 'gdn_state', 'ple_state'):
                    old = self.rows[index]['memory']['devices'][device][field]
                    self.nested_change(index, ('memory', 'devices', device, field), old + 1)
                self.nested_change(index, ('speculative_buffer_counts_from_validated_geometry', device), 86)
                self.nested_change(index, ('memory', 'devices', device, 'owned_buffers'), 43 if device == 0 else 44)
                self.nested_change(index, ('memory', 'devices', device, 'owned_peak_bytes'), 0)
            self.change(index, 'speculative_buffer_count_total_from_geometry', 174)
            self.nested_change(index, ('speculative_buffer_bytes_by_device', 1), 235622400 + 122880)
            self.nested_change(index, ('memory', 'host_logit_capacity'), self.rows[index]['host_each_logit_requested_bytes'])
            self.nested_change(index, ('memory', 'ram_expert_capacity'), 68262297599)
        # These emitted values are actual sizeof/vector aggregate capacities, not
        # fixed LP64 guesses. Preserve valid changes verbatim in compact output.
        self.rows[1]['memory'].update(host_session_impl_bytes=16384, host_session_config_bytes=128,
                                      host_speculative_owner_bytes=8192, host_speculative_hash_payload_bytes=2048,
                                      host_logit_capacity=6000000, host_embedding_capacity=1430323201)
        self.rows[53]['memory'].update(host_hybrid_plans=65536, cpu_pool_metadata=8192, cpu_pool_scratch=1048576,
                                       host_cpu_expert_views=6000000)
        result = self.collect(self.rows)
        self.assertEqual(result['loaded_memory'][0]['memory'], self.rows[1]['memory'])
        self.assertEqual(result['loaded_memory'][2]['memory'], self.rows[53]['memory'])
        self.assertEqual(result['aggregate_counts'], {'successful_restores': 51, 'argument_rejections': 84,
                                                      'sticky_rejections': 6, 'steady_memory_observations': 137})

    def test_phase_totals_hidden_gate_fault_sticky_reset_recovery_fail_closed(self):
        for owner in ('primary_restoration', 'wide_PP', 'CPU_failure_recovery'):
            for field, value in self.rows[55]['phase_totals'][owner].items():
                self.nested_change(55, ('phase_totals', owner, field), value + 1)
        for index in (52, 54):
            for field, value in self.rows[index]['phase_totals'].items():
                self.nested_change(index, ('phase_totals', field), value + 1)
        for field, value in self.rows[55]['aggregate_counts'].items():
            self.nested_change(55, ('aggregate_counts', field), value + 1)
        for field, bad in (('values', 327681), ('max_error', .001), ('max_bound_ratio', 1.01)):
            self.nested_change(52, ('hidden_N1_numerical_comparison', field), bad)
        self.rows[52]['hidden_N1_numerical_comparison'].update(max_error=.125, max_bound_ratio=.125 / .22)
        self.assertTrue(self.collect(self.rows)['passed'])
        for key in ('accepted_cpu_jobs', 'queued_admission_copies', 'pending_slots_at_failure'):
            self.nested_change(54, ('fault_diagnostics', key), 0)
        for key in ('ready_cache_ids', 'pending_cache_ids', 'cpu_return_bytes_before_failure', 'failure_layer'):
            self.nested_change(54, ('fault_diagnostics', key), 1)
        for key in ('synthetic_failure', 'cpu_pool_drained', 'gpu_streams_drained'):
            self.nested_change(54, ('fault_diagnostics', key), False)
        for key in ('postfault', 'poststicky'):
            self.nested_change(54, ('public_snapshots', key, 'checkpoint_available'), True)
            self.nested_change(54, ('public_snapshots', key, 'pending_window'), {'enabled': True, 'pending': False,
                                                                                 'start_position': 0, 'inputs': 0, 'valid_slots': 0})
            self.nested_change(54, ('public_snapshots', key, 'checkpoint_unavailable_reason'), 'no_pending_window')
            self.nested_change(54, ('public_snapshots', key, 'stats', 'last_completed_ms_bits'), 0)
        self.nested_change(54, ('public_snapshots', 'postreset', 'speculative', 'target_forward_rows'), 3)
        self.nested_change(54, ('public_snapshots', 'postrecovery', 'hybrid', 'cpu_gate_up_jobs'), 0)
        self.nested_change(54, ('public_snapshots', 'postrecovery', 'hybrid', 'short_layers'), 0)

    def test_actual_source_path_argument_and_model_alias_guard_not_filename_guess(self):
        for key, value in (('model_path', 'relative.gguf'), ('model_path', '/a/../b.gguf'),
                           ('model_path', ''), ('model_path', '\x00'), ('model_argument', ''),
                           ('model_argument', '/another/model.gguf')):
            self.change(0, key, value)
        self.rows[0]['model_argument'] = './qwen38-keep1-Q4_0.gguf'
        self.assertEqual(self.collect(self.rows)['model'], str(self.model))
        other = self.root / 'actual model filename.gguf'
        other.write_bytes(b'synthetic actual path inventory')
        self.rows[0].update(model_path=str(other), model_argument=str(other))
        self.raw.write_bytes(encoded(self.rows))
        before = other.read_bytes()
        for destination in (other, self.root / 'model-symlink', self.root / 'model-hardlink'):
            if destination.name == 'model-symlink':
                destination.symlink_to(other)
            elif destination.name == 'model-hardlink':
                os.link(other, destination)
            self.assertEqual(self.cli(destination).returncode, 1)
            self.assertEqual(other.read_bytes(), before)

    def test_invalid_JSON_duplicate_keys_nonfinite_utf8_nonobject_recursion_incomplete(self):
        for field in ('revision', 'dirty', 'retained', 'passed', 'max_error', 'rows', 'memory_observations'):
            key = ('"' + field + '":').encode()
            self.raw.write_bytes(self.original.replace(key, key + b'0,' + key, 1))
            with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                self.collect()
        for bad in (b'NaN', b'Infinity', b'-Infinity', b'1e999'):
            self.raw.write_bytes(self.original.replace(b'"max_error":0', b'"max_error":' + bad))
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                self.collect()
        for raw in (self.original[:-1], self.original[:-20], self.original + b'\n',
                    self.original + b'failure\n', b'\xef\xbb\xbf' + self.original,
                    self.original.replace(b'session_restore_source', b'\xff')):
            self.raw.write_bytes(raw)
            with self.assertRaises(ValueError):
                self.collect()
        lines = self.original.splitlines()
        for bad in (b'[]', b'null', b'true', b'1', b'{} {}', b'{',
                    b'{"x":' + b'[' * 2000 + b'0' + b']' * 2000 + b'}',
                    b'{"x":' + b'9' * 10000 + b'}'):
            self.raw.write_bytes(b'\n'.join([lines[0], bad, *lines[2:]]) + b'\n')
            with self.assertRaises(ValueError):
                self.collect()
        self.raw.write_bytes(self.original.replace(b'\n', b'\r\n'))
        self.assertTrue(self.collect()['passed'])

    def test_raw_file_and_row_bounds_preopen_fstat_and_growing_read(self):
        self.assertEqual((MODULE.MAX_RAW_BYTES, MODULE.MAX_RECORD_BYTES), (16777216, 65536))
        lines = self.original.splitlines()
        padded = b' ' * (65536 - len(lines[0])) + lines[0]
        self.raw.write_bytes(b'\n'.join([padded, *lines[1:]]) + b'\n')
        self.assertTrue(self.collect()['passed'])
        self.raw.write_bytes(b' ' + self.raw.read_bytes())
        with self.assertRaisesRegex(ValueError, 'oversized JSON record'):
            self.collect()
        self.raw.write_bytes(b'x' * (16777216 + 1))
        with mock.patch.object(MODULE.os, 'open') as opened:
            with self.assertRaisesRegex(ValueError, 'oversized input'):
                self.collect()
            opened.assert_not_called()
        self.raw.write_bytes(self.original)
        real_fstat = os.fstat

        def huge_stat(fd):
            fields = list(real_fstat(fd))
            fields[6] = 16777217
            return os.stat_result(fields)

        with mock.patch.object(MODULE.os, 'fstat', side_effect=huge_stat):
            with self.assertRaisesRegex(ValueError, 'oversized input'):
                self.collect()
        real_fdopen = os.fdopen

        def growing(fd, *args, **kwargs):
            stream = real_fdopen(fd, *args, **kwargs)
            proxy = mock.MagicMock(wraps=stream)
            proxy.__enter__.return_value = proxy
            proxy.__exit__.side_effect = lambda *_: stream.close()
            proxy.read.return_value = b'x' * 16777217
            return proxy

        with mock.patch.object(MODULE.os, 'fdopen', side_effect=growing):
            with self.assertRaisesRegex(ValueError, 'oversized input'):
                self.collect()

    def test_no_follow_nonblocking_regular_file_checks_and_stat_open_races(self):
        alias, fifo = self.root / 'raw-symlink', self.root / 'raw-fifo'
        alias.symlink_to(self.raw)
        os.mkfifo(fifo)
        for path in (alias, fifo, self.root, Path('/dev/null')):
            with self.assertRaisesRegex(ValueError, 'regular file'):
                MODULE.collect(path)
        with mock.patch.object(MODULE.os, 'open', wraps=os.open) as opened:
            self.collect()
            flags = opened.call_args.args[1]
            self.assertEqual(flags & (os.O_NOFOLLOW | os.O_NONBLOCK), os.O_NOFOLLOW | os.O_NONBLOCK)
        real_open = os.open

        def swap_to_fifo(path, flags):
            self.raw.unlink()
            os.mkfifo(self.raw)
            return real_open(path, flags)

        with mock.patch.object(MODULE.os, 'open', side_effect=swap_to_fifo):
            with self.assertRaisesRegex(ValueError, 'regular file'):
                self.collect()  # O_NONBLOCK ensures the replacement never hangs.
        self.raw.unlink()
        self.raw.write_bytes(self.original)

        def swap_to_symlink(path, flags):
            self.raw.unlink()
            self.raw.symlink_to(fifo)
            return real_open(path, flags)

        with mock.patch.object(MODULE.os, 'open', side_effect=swap_to_symlink):
            with self.assertRaises(OSError):
                self.collect()
        self.raw.unlink()
        self.raw.write_bytes(self.original)
        replacement = self.root / 'replacement'
        replacement.write_bytes(self.original)

        def swap_regular(path, flags):
            os.replace(replacement, self.raw)
            return real_open(path, flags)

        with mock.patch.object(MODULE.os, 'open', side_effect=swap_regular):
            with self.assertRaisesRegex(ValueError, 'changed during open'):
                self.collect()

    def test_cli_required_explicit_paths_once_history_shared_helper(self):
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
        self.assertEqual(result['protocol'], 1)
        self.assertIs(MODULE.append_record, record_memory.append_result)
        with mock.patch.object(MODULE, 'append_record', wraps=record_memory.append_result) as append:
            self.assertEqual(MODULE.main(['--raw', str(self.raw), '--results', str(self.results)]), 0)
            append.assert_called_once()
        self.assertEqual(self.raw.read_bytes(), self.original)
        self.assertEqual(self.cli(Path('explicit-relative.jsonl')).returncode, 0)

    def test_cli_arguments_validation_before_open_failed_complete_never_appended(self):
        for args in ([], ['--raw', str(self.raw)], ['--results', str(self.results)],
                     ['--raw', str(self.raw), '--results', str(self.results), '--absolute', '1']):
            self.assertEqual(self.cli(args=args).returncode, 2)
            self.assertFalse(self.results.exists())
        history = b'{"kind":"old"}\n'
        self.results.write_bytes(history)
        for index in (1, 2, 47, 51, 52, 53, 54, 55):
            failed = fixture(model=str(self.model))
            failed[index]['passed'] = False
            self.raw.write_bytes(encoded(failed))
            self.assertEqual(self.cli().returncode, 1)
            self.assertEqual(self.results.read_bytes(), history)
        self.raw.write_bytes(self.original[:-1])
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), history)
        self.results.unlink()
        self.assertEqual(self.cli().returncode, 1)
        self.assertFalse(self.results.exists())
        with mock.patch.object(MODULE, 'collect', side_effect=OSError('injected input failure')):
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(MODULE.main(['--raw', str(self.raw), '--results', str(self.results)]), 1)
                self.assertIn('record_restore: injected input failure', errors.getvalue())

    def test_result_aliases_raw_declared_model_and_unfinished_nonregular_history(self):
        for artifact in (self.raw, self.model):
            if artifact != self.raw:
                artifact.write_bytes(b'synthetic emitted model path not opened as a model')
            before = artifact.read_bytes()
            sym, hard = self.root / (artifact.name + '.sym'), self.root / (artifact.name + '.hard')
            sym.symlink_to(artifact)
            os.link(artifact, hard)
            for destination in (artifact, sym, hard):
                self.assertEqual(self.cli(destination).returncode, 1)
                self.assertEqual(artifact.read_bytes(), before)
        unfinished = b'{"kind":"old"}\n{"kind":"unfinished"}'
        self.results.write_bytes(unfinished)
        self.assertEqual(self.cli().returncode, 1)
        self.assertEqual(self.results.read_bytes(), unfinished)
        fifo = self.root / 'journal-fifo'
        os.mkfifo(fifo)
        for destination in (self.root, fifo, self.root / 'absent' / 'results.jsonl', Path('/dev/stdout'), Path('/dev/stderr')):
            process = self.cli(destination)
            self.assertEqual(process.returncode, 1)
            self.assertEqual(process.stdout, '')

    def test_serialization_lock_partial_write_error_flush_rollback(self):
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

            def faulty(path, *args, **kwargs):
                stream = real_open(path, *args, **kwargs)
                proxy = mock.MagicMock(wraps=stream)
                proxy.__enter__.return_value = proxy
                proxy.__exit__.side_effect = lambda *_: stream.close()

                def write(payload):
                    if failure == 'flush':
                        return stream.write(payload)
                    stream.write(payload[:17])
                    if failure == 'write':
                        raise OSError('injected write failure')
                    return 17

                proxy.write.side_effect = write
                if failure == 'flush':
                    proxy.flush.side_effect = OSError('injected flush failure')
                return proxy

            with mock.patch.object(Path, 'open', autospec=True, side_effect=faulty):
                with self.assertRaises((ValueError, OSError)):
                    MODULE.append_record(self.results, record)
            self.assertEqual(self.results.read_bytes(), history)

    def test_alias_check_after_locked_open_still_preserves_input(self):
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

    def test_concurrent_append_preserves_history_and_individual_records(self):
        history = b'{"kind":"first"}\n{"kind":"second"}\n'
        self.results.write_bytes(history)
        command = [sys.executable, '-B', str(SCRIPT), '--raw', str(self.raw), '--results', str(self.results)]
        processes = [subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True) for _ in range(2)]
        try:
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
