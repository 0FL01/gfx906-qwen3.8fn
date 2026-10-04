#!/usr/bin/env python3
"""Validate the exact core-session-restore-test stream; append ONE r6_target_restore.

The protocol-1 emitter has 56 successful records and three sequential owners.
Public snapshots and memory ledgers are validated, not inferred from totals.
This is a target-restoration prerequisite, not trained MTP or R6
qualification. Check the executable's exit status before running this collector.
--results explicitly names the caller's canonical ROOT/results.jsonl.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

if __package__:
    from .record_memory import (append_result as append_record, exact, expect,
                                finite_float, integer, invalid_constant, number,
                                require, unique_object)
    from .record_prefill import json_integer
else:
    from record_memory import (append_result as append_record, exact, expect,
                               finite_float, integer, invalid_constant, number,
                               require, unique_object)
    from record_prefill import json_integer


MAX_RAW_BYTES, MAX_RECORD_BYTES = 16 * 1024 * 1024, 64 * 1024
CAPACITY, BATCH, WIDTH, VOCAB = 40, 3, 10240, 248320
STARTS = (2, 3, 4, 5, 8)
CASES = tuple((start, n, k) for start in STARTS for n in range(1, 4) for k in range(n + 1))
DIVERGENT_PREFIXES = tuple(range(4))
RECORD_COUNT = 1 + 3 + len(CASES) + len(DIVERGENT_PREFIXES) + 1 + 1 + 1
# session_restore_test.cpp::main, Fixture::invalid_pending/reject/memory.
RESTORES = len(CASES) + len(DIVERGENT_PREFIXES) + 1  # Exhausted-capacity k=0.
REJECTIONS = len(CASES) + 11 * 3 + 1 + 3 + 1
MEMORY_OBSERVATIONS = REJECTIONS + len(CASES) + len(DIVERGENT_PREFIXES) + 1
# Full compare counts include prefix setup/branch oracles; initial retained
# references and raw read_tap calls are NOT Comparisons::compare invocations.
CASE_ROWS = sum(start + n + 3 for start, n, _ in CASES)
PRIMARY_COMPARED_ROWS = (3 + sum(3 + k for k in DIVERGENT_PREFIXES) + CASE_ROWS +
                         3 + 3 + len(DIVERGENT_PREFIXES) * 9 + 39 + 1 + 1 + 1)
PRIMARY_EXACT_TAP_ROWS = PRIMARY_COMPARED_ROWS - sum(DIVERGENT_PREFIXES)
PRIMARY_FORWARD_ROWS = (CAPACITY + 3 + 3 + sum(3 + k + 3 for k in DIVERGENT_PREFIXES) +
                        CASE_ROWS + 3 + 3 + 3 + len(DIVERGENT_PREFIXES) * 9 + 39 + 1 + 1 + 1)
PRIMARY_VERIFY_WINDOWS = len(CASES) + 1 + len(DIVERGENT_PREFIXES) + 2
PRIMARY_VERIFY_ROWS = sum(n for _, n, _ in CASES) + 3 + 3 * len(DIVERGENT_PREFIXES) + 2
FP32_MAX = float.fromhex('0x1.fffffep+127')
Q40, Q41 = 2764800, 2867200
RAM_PAYLOAD = (42 * Q40 + 6 * Q41) * 512
HYBRID_FIELDS = ('short_layers gpu_only_wide_layers ready_hit_assignments physical_miss_assignments '
                 'group_reuse_assignments cpu_groups cpu_assignments gpu_hit_groups gpu_hit_assignments '
                 'gpu_miss_groups gpu_miss_assignments admitted_groups evicted_ready_slots input_extractions '
                 'input_bytes cpu_return_bytes cpu_input_bytes_checked all_hit_layers forced_cpu_layers '
                 'forced_gpu_layers cpu_gate_up_jobs cpu_down_jobs gpu_middle_columns gpu_middle_batches '
                 'paired_gate_up_bytes middle_q8_bytes').split()
ATTENTION_FIELDS = ('batch_calls query_rows multiquery_calls multiquery_rows singleton_tail_calls max_query_rows').split()
SPEC_FIELDS = ('target_forward_rows verify_windows verify_rows restore_calls retained_inputs gdn_prefix_calls '
               'ple_prefix_calls qsa_tail_prefix_calls').split()
MEMORY_SCALARS = ('capacity expert_slots cpu_workers attention_query_tile ram_expert_capacity ram_expert_payload '
                  'host_embedding_capacity host_logit_capacity pinned_handoff expert_payload_reads '
                  'expert_payload_bytes_read pinned_expert_staging pinned_hybrid_input pinned_hybrid_output '
                  'pinned_hybrid_error pinned_hybrid_middle host_hybrid_plans host_cpu_expert_views cpu_pool_metadata '
                  'cpu_pool_scratch host_hybrid_probe host_routing_capacity host_route_group_payload '
                  'pinned_route_metadata host_hybrid_input_probe host_session_impl_bytes host_session_config_bytes '
                  'host_speculative_owner_bytes host_speculative_hash_payload_bytes').split()
MEMORY_ARRAYS = ('hybrid_contribution_bytes expert_stage_capacity_bytes route_metadata_bytes hybrid_gate_up_bytes '
                 'hybrid_middle_float_bytes hybrid_middle_q8_bytes hybrid_middle_error_bytes '
                 'checkpoint_recurrent_bytes checkpoint_history_bytes checkpoint_qsa_tail_bytes '
                 'checkpoint_ple_history_bytes target_tap_bytes target_tap_staging_bytes').split()
ATTENTION_MEMORY_FIELDS = ('gathered_key_bytes gathered_value_bytes partial_output_bytes partial_max_sum_bytes '
                           'staged_buffer_bytes output_buffer_bytes selected_id_bytes selected_block_bytes '
                           'selected_count_bytes private_workspace_bytes').split()
CATEGORIES = ('weights expert_slots qsa_kv qsa_index gdn_state ple_state workspace').split()
DEVICE_FIELDS = ('device first_layer last_layer gdn_layers qsa_layers owned_bytes owned_peak_bytes '
                 'owned_buffers total_vram free_vram category_sum_bytes').split() + CATEGORIES
LOADED_LIMITS = ['separate_working_logit_capacity_not_exposed', 'snapshot_Buffer_count_geometry_derived',
                 'private_hash_history_and_route_allocator_slack_excluded', 'allocator_overhead_and_thread_stacks_excluded',
                 'HIP_rocBLAS_context_allocations_not_owned_ledger']
RUNTIME_GEOMETRY = {
    'vocabulary': VOCAB, 'layers': 48, 'gdn_layers': 36, 'qsa_layers': 12, 'kv_type': 'Q4_0',
    'kv_width': 512, 'kv_blocks_per_position': 16, 'kv_block_elements': 32, 'kv_block_bytes': 18,
    'pooled_index_type': 'F32', 'pooled_index_width': 128, 'qsa_block_tokens': 4,
    'qsa_raw_tail_elements': 384, 'gdn_recurrent_elements': 786432, 'gdn_history_elements': 30720,
    'ple_conv_history_elements': 92160, 'checkpoint_slots': 4, 'max_verify_inputs': 3,
    'tap_width': WIDTH, 'tap_device': 1, 'tap_location': 'layer47_combine_before_root_HC',
}
SOURCE_REQUIRED = {
    'records': RECORD_COUNT, 'loaded_memory_owners': 3,
    'primary': {'successful_restores': RESTORES, 'argument_rejections': REJECTIONS,
                'steady_memory_observations': MEMORY_OBSERVATIONS},
    'CPU_sticky_rejections': 6, 'expert_payload_reads_per_owner': 144,
    'expert_payload_bytes_per_owner': RAM_PAYLOAD,
    'expert_geometry': {'layers': 48, 'experts_per_layer': 512, 'top_k': 10, 'input': 2560, 'middle': 640,
                        'gate_up_type': 'Q4_0', 'q4_0_down_layers': 42, 'q4_1_down_layers': 6,
                        'q4_0_triplet_bytes_per_expert': Q40, 'q4_1_triplet_bytes_per_expert': Q41},
}


def records(path):
    """No symlink following or blocking FIFO opens, including a stat/open race."""
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), str(path) + ': not a regular file')
    require(info.st_size <= MAX_RAW_BYTES, str(path) + ': oversized input')
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode), str(path) + ': not a regular file')
        require(os.path.samestat(info, opened), str(path) + ': input changed during open')
        require(opened.st_size <= MAX_RAW_BYTES, str(path) + ': oversized input')
        with os.fdopen(fd, 'rb') as stream:
            fd = None
            raw = stream.read(MAX_RAW_BYTES + 1)
    finally:
        if fd is not None:
            os.close(fd)
    require(len(raw) <= MAX_RAW_BYTES, str(path) + ': oversized input')
    require(raw.endswith(b'\n'), str(path) + ': missing completed final line')
    lines = raw.split(b'\n')[:-1]
    require(len(lines) == RECORD_COUNT, f'{path}: expected exactly {RECORD_COUNT} completed JSONL rows')
    rows = []
    for i, line in enumerate(lines, 1):
        require(len(line) <= MAX_RECORD_BYTES, f'{path}:row {i}: oversized JSON record')
        try:
            obj = json.loads(line.decode('utf-8'), object_pairs_hook=unique_object,
                             parse_int=json_integer, parse_float=finite_float,
                             parse_constant=invalid_constant)
            require(type(obj) is dict, 'expected an object')
        except (ValueError, RecursionError) as error:
            raise ValueError(f'{path}:row {i}: {error}') from error
        rows.append(obj)
    return rows


def source(obj):
    expect(obj, {
        'protocol': 1, 'kind': 'session_restore_source', 'capacity': CAPACITY, 'slots': 1, 'max_batch': BATCH,
        'cpu_workers': 0, 'absolute_gate': .02, 'relative_gate': .002,
        'reference': 'retained_N1_full_vocabulary_same_Session', 'performance_claim': False,
        'speculative_checkpoints': True, 'runtime_geometry': RUNTIME_GEOMETRY, 'source_required': SOURCE_REQUIRED,
        'expert_inventory': {'scope': 'tensor_inventory_no_payload_reads', 'matrices': 144,
                             'bytes': RAM_PAYLOAD, 'q4_0_down_layers': 42, 'q4_1_down_layers': 6},
    }, ('revision', 'dirty', 'ple_eos', 'model_path', 'model_argument', 'wide_PP_mode', 'wide_PP_rows'), 'source')
    revision = obj['revision']
    require(type(revision) is str and len(revision) == 40 and
            all(c in '0123456789abcdef' for c in revision), 'source.revision: invalid compiled revision')
    # Unlike the prefill-attention driver, this emitter explicitly prints booleans.
    require(type(obj['dirty']) is bool, 'source.dirty: expected boolean')
    integer(obj['ple_eos'], 'source.ple_eos', 0, VOCAB - 1)
    for key in ('model_path', 'model_argument'):
        value = obj[key]
        require(type(value) is str and value and '\x00' not in value and
                not any(0xd800 <= ord(c) <= 0xdfff for c in value), 'source.' + key + ': invalid path')
    require(Path(obj['model_path']).is_absolute() and os.path.normpath(obj['model_path']) == obj['model_path'],
            'source.model_path: expected lexical absolute normalized path')
    if Path(obj['model_argument']).is_absolute():
        exact(os.path.normpath(obj['model_argument']), obj['model_path'], 'source.model_argument')
    wide = integer(obj['wide_PP_rows'], 'source.wide_PP_rows', 32, 1024)
    require(wide in (32, 1024), 'source.wide_PP_rows: unsupported scope')
    exact(obj['wide_PP_mode'], 'default_32' if wide == 32 else 'explicit_1024', 'source.wide_PP_mode')


def metrics(obj, values, label='complete', diagnostic=True):
    bits = integer(obj['diagnostic_bit_differences'], label + '.diagnostic_bit_differences', 0, values) if diagnostic else None
    maximum = number(obj['max_error'], label + '.max_error')
    ratio = number(obj['max_bound_ratio'], label + '.max_bound_ratio')
    require(ratio <= 1 and maximum <= .02 + .002 * FP32_MAX,
             label + ': frozen numerical gate failed')
    if maximum == 0:
        require(ratio == 0, label + ': zero error with nonzero bound ratio')
    else:
        require((bits is None or bits > 0) and ratio > 0, label + ': positive error missing diagnostics')
        require(maximum / (.02 + .002 * FP32_MAX) <= ratio <= maximum / .02,
                 label + ': inconsistent bound ratio')


def required_geometry(max_batch, cpu_workers=0):
    """Source-required payloads, NEVER observations or guessed sizeof/capacities.

    Buffer/requested float payload math in session.hip::TargetCheckpoints/Impl,
    separately checked against the actual serialized getter categories. Host
    vector capacity/sizeof values are retained from the emitter, never guessed.
    """
    integer(max_batch, 'geometry.max_batch', 3, 1024)
    integer(cpu_workers, 'geometry.cpu_workers', 0, 1)
    recurrent, history, tail = 18 * 4 * 786432 * 4, 18 * 4 * 30720 * 4, 6 * 4 * 384 * 4
    ple, tap = 4 * 92160 * 4, max_batch * WIDTH * 4
    common = recurrent + history + tail
    working = max_batch * VOCAB * 4
    # cpu_workers>0 prepares short expert stages; wide prepares band16 stages
    # independently. Snapshot ownership does not masquerade as CPU preparation.
    stage = 16 * 2867200 if max_batch > 3 else (2867200 if cpu_workers else 0)
    return {
        'max_batch_tokens': max_batch, 'cpu_workers': cpu_workers,
        'checkpoint_recurrent_bytes': [recurrent, recurrent],
        'checkpoint_history_bytes': [history, history],
        'checkpoint_qsa_tail_bytes': [tail, tail], 'checkpoint_ple_history_bytes': [ple, 0],
        'target_tap_bytes': [0, tap], 'target_tap_staging_bytes': [0, tap],
        'checkpoint_and_tap_GPU_bytes': [common + ple, common + 2 * tap],
        'additional_GPU_buffers': [18 * 2 + 6 + 1, 18 * 2 + 6 + 2],
        'published_logit_payload_bytes': working, 'working_logit_payload_bytes': working,
        'host_logit_capacity_floor_bytes': 2 * working,
        'expert_stage_capacity_bytes': [stage, stage], 'pinned_expert_staging_bytes': 4 * stage,
        'scope': 'source_required_payload_geometry_checked_against_loaded_getters_not_individual_allocations',
    }


def uint_fields(obj, fields, label):
    expect(obj, {}, fields, label)
    for key in fields:
        integer(obj[key], label + '.' + key)


def uint_array(value, size, label):
    require(type(value) is list and len(value) == size, label + ': incorrect array extent')
    for i, item in enumerate(value):
        integer(item, f'{label}[{i}]')


def speculative(forward, windows=0, verify_rows=0, prefixes=(0, 0, 0, 0)):
    return dict(zip(SPEC_FIELDS, (forward, windows, verify_rows, sum(prefixes),
                                  sum(k * count for k, count in enumerate(prefixes)),
                                  36 * windows, windows, 12 * windows)), restored_prefixes=list(prefixes))


def public(obj, forward, cursor, rows, position, *, windows=0, verify_rows=0,
           prefixes=(0, 0, 0, 0), pending=False, attention=False, hybrid=False,
           available=True, max_batch=BATCH, capacity=CAPACITY, label='public'):
    """Closed getter schema, exact logical schedule and independent physical work."""
    extra = () if available else ('checkpoint_unavailable_reason',)
    expect(obj, {'checkpoint_available': available},
           ('stats', 'route', 'hybrid', 'attention', 'speculative', 'target_tap', 'pending_window', *extra), label)
    stats = obj['stats']
    uint_fields(stats, ('consumed_tokens', 'expert_hits', 'expert_misses', 'expert_upload_bytes',
                        'last_completed_ms_bits'), label + '.stats')
    exact(stats['consumed_tokens'], cursor, label + '.cursor')
    require(cursor <= capacity, label + ': cursor outside capacity')
    require(stats['expert_hits'] + stats['expert_misses'] == 480 * forward, label + ': physical assignment accounting')
    if not hybrid:
        # Every disabled-path non-READY triplet acquisition (including wide
        # staged bands) adds exactly its Q4_0/Q4_1 payload, not an assignment.
        residual = stats['expert_upload_bytes'] - Q40 * stats['expert_misses']
        require(0 <= residual <= (Q41 - Q40) * stats['expert_misses'] and residual % (Q41 - Q40) == 0,
                label + ': expert miss/upload payload accounting')
    elapsed = struct.unpack('<d', struct.pack('<Q', stats['last_completed_ms_bits']))[0]
    require(math.isfinite(elapsed) and elapsed >= 0, label + ': invalid last_completed_ms bits')
    if not forward:
        exact(stats, {key: 0 for key in stats}, label + '.reset_stats')
    uint_fields(obj['route'], ('last_max_expert_group_assignments', 'expert_groups_gt128'), label + '.route')
    group = obj['route']['last_max_expert_group_assignments']
    require((group == 0 if not forward else 1 <= group <= rows), label + ': last-call expert group bounds')
    threshold = obj['route']['expert_groups_gt128']
    # In this fixture every earlier call since reset is N1; only the current
    # wide PP call can have >128 groups. Do not use a logical restored cursor.
    require(threshold <= 480 * rows // 129 and bool(threshold) == (group > 128),
            label + ': >128 group accounting')
    uint_fields(obj['hybrid'], HYBRID_FIELDS, label + '.hybrid')
    if not hybrid:
        exact(obj['hybrid'], dict.fromkeys(HYBRID_FIELDS, 0), label + '.disabled_hybrid')
    else:
        h = obj['hybrid']
        exact(h['short_layers'], 48 * forward, label + '.short_layers')  # Recovery is six N1 calls.
        require(h['ready_hit_assignments'] + h['physical_miss_assignments'] == 480 * forward and
                h['cpu_assignments'] + h['gpu_hit_assignments'] + h['gpu_miss_assignments'] == 480 * forward,
                label + ': hybrid physical/dispatch assignment accounting')
        require(h['cpu_gate_up_jobs'] > 0 and h['cpu_down_jobs'] > 0 and h['gpu_middle_columns'] > 0,
                label + ': recovered CPU/GPU work missing')
    expected_attention = dict.fromkeys(ATTENTION_FIELDS, 0)
    if attention:
        expected_attention.update(batch_calls=36, query_rows=36, singleton_tail_calls=36, max_query_rows=1)
    exact(obj['attention'], expected_attention, label + '.attention')
    exact(obj['speculative'], speculative(forward, windows, verify_rows, prefixes), label + '.speculative')
    tap = obj['target_tap']
    expect(tap, {'device': 1, 'width': WIDTH, 'rows': rows, 'first_position': position,
                 'pointer_present': bool(rows)}, ('pointer_address',), label + '.target_tap')
    address = integer(tap['pointer_address'], label + '.tap.pointer_address')
    require((address > 0 and address % 4 == 0) if rows else address == 0, label + ': typed tap pointer')
    require(0 <= rows <= max_batch and position + rows <= capacity, label + ': tap extent')
    if available:
        exact(obj['pending_window'], {'enabled': True, 'pending': pending,
              'start_position': position if pending else 0, 'inputs': rows if pending else 0,
              'valid_slots': rows + 1 if pending else 0}, label + '.pending_window')
    else:
        exact(obj['pending_window'], None, label + '.pending_window')
        exact(obj['checkpoint_unavailable_reason'], 'execution_failure_requires_reset', label + '.unavailable_reason')


def monotone(before, after, label):
    for key in ('expert_hits', 'expert_misses', 'expert_upload_bytes'):
        require(after['stats'][key] >= before['stats'][key], label + ': rewound ' + key)
    for key in HYBRID_FIELDS:
        require(after['hybrid'][key] >= before['hybrid'][key], label + ': rewound hybrid.' + key)
    require(after['route']['expert_groups_gt128'] >= before['route']['expert_groups_gt128'], label + ': rewound route work')


def restored(before, after, retained, label):
    for key in ('route', 'hybrid', 'attention', 'target_tap'):
        exact(after[key], before[key], label + '.' + key)
    for key in before['stats']:
        if key != 'consumed_tokens':
            exact(after['stats'][key], before['stats'][key], label + '.stats.' + key)


def window_snapshots(obj, start, n, k, attention=False, label='window'):
    fields = ('preverify', 'postverify', 'postrestore', 'postcontinuation')
    expect(obj, {}, fields, label)
    prefix = tuple(int(i == k) for i in range(4))
    public(obj['preverify'], start, start, 1, start - 1, label=label + '.preverify')
    public(obj['postverify'], start + n, start + n, n, start, windows=1, verify_rows=n,
           pending=True, attention=attention, label=label + '.postverify')
    public(obj['postrestore'], start + n, start + k, n, start, windows=1, verify_rows=n,
           prefixes=prefix, attention=attention, label=label + '.postrestore')
    public(obj['postcontinuation'], start + n + 3, start + k + 3, 1, start + k + 2,
           windows=1, verify_rows=n, prefixes=prefix, attention=attention, label=label + '.postcontinuation')
    monotone(obj['preverify'], obj['postverify'], label)
    restored(obj['postverify'], obj['postrestore'], k, label + '.restore')
    monotone(obj['postrestore'], obj['postcontinuation'], label)
    # The two taps alternate once per successful call, not once per row.
    pre = obj['preverify']['target_tap']['pointer_address']
    verified = obj['postverify']['target_tap']['pointer_address']
    require(pre != verified, label + ': tap publication did not alternate buffers')
    exact(obj['postcontinuation']['target_tap']['pointer_address'], pre, label + '.continuation_tap_owner')


def rejection(obj, group, count, before, label):
    expect(obj, {'group': group, 'count': count}, ('before', 'after'), label)
    exact(obj['before'], before, label + '.before')
    exact(obj['after'], before, label + '.after')


def phase(compared_rows, tap_rows, restores, rejects, memory, sticky=0):
    return {'compared_full_vocab_values': compared_rows * VOCAB, 'compared_full_vocab_rows': compared_rows,
            'tap_exact_values': tap_rows * WIDTH, 'successful_restores': restores, 'argument_rejections': rejects,
            'steady_memory_observations': memory, 'sticky_rejections': sticky}


def loaded(obj, owner, index, max_batch, workers):
    label = 'loaded.' + owner
    capacity = max(40, 2 * max_batch) if index == 1 else CAPACITY
    geometry = required_geometry(max_batch, workers)
    logit = max_batch * VOCAB * 4
    expect(obj, {
        'protocol': 1, 'kind': 'session_restore_loaded_memory', 'owner': owner, 'owner_index': index,
        'live_session_owners': 1, 'config': {'capacity': capacity, 'expert_slots': 1, 'max_batch_tokens': max_batch,
            'cpu_workers': workers, 'hybrid_probe': bool(workers), 'attention_query_tile': 1,
            'trace_first_token': 0, 'trace_directory': '', 'speculative_checkpoints': True},
        'tap_capacity': {'device': 1, 'rows': max_batch, 'width': WIDTH,
                         'published_bytes': max_batch * WIDTH * 4, 'staging_bytes': max_batch * WIDTH * 4},
        'speculative_buffer_counts_from_validated_geometry': [43, 44],
        'speculative_buffer_count_total_from_geometry': 87,
        'speculative_buffer_bytes_by_device': geometry['checkpoint_and_tap_GPU_bytes'],
        'host_logit_buffer_count_required': 2, 'host_each_logit_requested_bytes': logit,
        'host_logit_combined_capacity_floor_bytes': 2 * logit,
        'ledger_scope': 'all_live_Session_Buffer_pointer_tree_and_reported_host_capacities',
        'host_config_included_in_impl': True, 'host_input_probe_included_in_hybrid_probe': True,
        'speculative_GPU_bytes_included_in_workspace': True,
        'constructor_read_scope': 'expert_payload_API_reads_not_physical_SSD_trace',
        'peak_scope': 'Buffer_process_peak_not_reset_between_sequential_owners',
        'evidence_limits': LOADED_LIMITS, 'full_RSS_measured': False, 'owned_buffer_release_measured': False,
        'passed': True,
    }, ('memory', 'initial_public'), label)
    m = obj['memory']
    expect(m, {'speculative_checkpoints': True, 'ownership_verified': True},
           (*MEMORY_SCALARS, *MEMORY_ARRAYS, 'attention', 'devices'), label + '.memory')
    for key in MEMORY_SCALARS:
        integer(m[key], label + '.memory.' + key)
    for key in MEMORY_ARRAYS:
        uint_array(m[key], 2, label + '.memory.' + key)
    fixed = {'capacity': capacity, 'expert_slots': 1, 'cpu_workers': workers, 'attention_query_tile': 1,
             'ram_expert_payload': RAM_PAYLOAD, 'expert_payload_reads': 144,
             'expert_payload_bytes_read': RAM_PAYLOAD, 'pinned_handoff': max_batch * WIDTH * 4,
             'pinned_expert_staging': geometry['pinned_expert_staging_bytes'],
             'pinned_route_metadata': 2 * max_batch * 10 * 8, 'route_metadata_bytes': [max_batch * 80] * 2}
    for key in MEMORY_ARRAYS:
        if key in geometry:
            fixed[key] = geometry[key]
    for key, value in fixed.items():
        exact(m[key], value, label + '.memory.' + key)
    require(m['ram_expert_capacity'] >= RAM_PAYLOAD and m['host_logit_capacity'] >= 2 * logit and
            m['host_logit_capacity'] % 4 == 0 and m['host_embedding_capacity'] > 0,
            label + ': host payload capacity floor')
    require(m['host_session_impl_bytes'] >= m['host_session_config_bytes'] > 0 and
            m['host_speculative_owner_bytes'] > 0 and m['host_speculative_hash_payload_bytes'] > 0,
            label + ': reported host sizeof/owner evidence')
    require(m['host_routing_capacity'] >= 2 * max_batch * (512 + 10 + 10) * 4 and
            m['host_route_group_payload'] > 0, label + ': routing payload capacity')
    cpu_only = ('host_hybrid_plans host_cpu_expert_views cpu_pool_metadata cpu_pool_scratch '
                'host_hybrid_probe host_hybrid_input_probe').split()
    for key in cpu_only:
        require(m[key] > 0 if workers else m[key] == 0, label + ': CPU ownership.' + key)
    for key, value in {'pinned_hybrid_input': 2 * 8640, 'pinned_hybrid_output': 2 * 307200,
                       'pinned_hybrid_error': 8, 'pinned_hybrid_middle': 2 * 21600}.items():
        exact(m[key], value if workers else 0, label + '.memory.' + key)
    for key, value in {'hybrid_gate_up_bytes': 153600, 'hybrid_middle_float_bytes': 76800,
                       'hybrid_middle_q8_bytes': 21600, 'hybrid_middle_error_bytes': 4,
                       'hybrid_contribution_bytes': 3 * 10 * 2560 * 4}.items():
        exact(m[key], [value if workers else 0] * 2, label + '.memory.' + key)
    if workers:
        probe_input = 2 * 3 * 48 * (80 * 36 + 10 * 9 + 1)
        require(m['host_hybrid_input_probe'] >= probe_input and
                m['host_hybrid_probe'] >= m['host_hybrid_input_probe'] + 2 * 3 * 48 * 11 * 2560 * 4,
                label + ': reported probe capacities')
    for key in ('attention', 'devices'):
        require(type(m[key]) is list and len(m[key]) == 2, label + ': device array extent')
    for id, (a, d) in enumerate(zip(m['attention'], m['devices'])):
        uint_fields(a, ATTENTION_MEMORY_FIELDS, label + f'.attention[{id}]')
        scratch = max(32768, max_batch * 12288) * 4
        exact(a, {'gathered_key_bytes': 1050112 * 2, 'gathered_value_bytes': 1050112 * 2,
                  'partial_output_bytes': 202752 * 4, 'partial_max_sum_bytes': 792 * 8,
                  'staged_buffer_bytes': scratch, 'output_buffer_bytes': scratch,
                  'selected_id_bytes': 2051 * 4, 'selected_block_bytes': 512 * 4,
                  'selected_count_bytes': 8, 'private_workspace_bytes': 5042368}, label + f'.attention[{id}]')
        uint_fields(d, DEVICE_FIELDS, label + f'.devices[{id}]')
        for key, value in {'device': id, 'first_layer': id * 24, 'last_layer': id * 24 + 23,
                           'gdn_layers': 18, 'qsa_layers': 6,
                           'qsa_kv': 6 * capacity * 2 * 16 * 18,
                           'qsa_index': 6 * (capacity // 4 * 128 + 384) * 4,
                           'gdn_state': 18 * (786432 + 30720) * 4,
                           'ple_state': 92160 * 4 if id == 0 else 0}.items():
            exact(d[key], value, label + f'.device[{id}].' + key)
        category_sum = sum(d[key] for key in CATEGORIES)
        require(category_sum == d['category_sum_bytes'] == d['owned_bytes'], label + ': category/owned double counting')
        require(d['owned_bytes'] <= d['owned_peak_bytes'] and d['free_vram'] <= d['total_vram'] and
                d['owned_bytes'] <= d['total_vram'] - d['free_vram'], label + ': GPU owned/peak/free bounds')
        require(d['weights'] > 0 and d['workspace'] >= geometry['checkpoint_and_tap_GPU_bytes'][id] and
                d['owned_buffers'] > geometry['additional_GPU_buffers'][id] and
                24 * Q40 <= d['expert_slots'] <= 24 * Q41, label + ': owned category/count floor')
    require(sum(d['expert_slots'] for d in m['devices']) * 512 == RAM_PAYLOAD,
            label + ': expert inventory/slots payload mismatch')
    public(obj['initial_public'], 0, 0, 0, 0, max_batch=max_batch, capacity=capacity, label=label + '.initial')


def collect(raw_path):
    """Read-only compact aggregate of every emitted field and closed chronology."""
    # Do not resolve the input before O_NOFOLLOW: that would silently follow it.
    raw_path = Path(os.path.abspath(os.fspath(raw_path)))
    rows = records(raw_path)
    origin = rows[0]
    source(origin)
    cursor = 1

    def take(fixed, variable=()):
        nonlocal cursor
        obj = rows[cursor]
        expect(obj, {'protocol': 1, **fixed}, variable, f'timeline[{cursor}]')
        cursor += 1
        return obj

    memories = []

    def take_memory(owner, index, max_batch, workers):
        nonlocal cursor
        obj = rows[cursor]
        loaded(obj, owner, index, max_batch, workers)
        cursor += 1
        memories.append(obj)

    take_memory('primary_restoration', 0, 3, 0)
    for start, n, k in CASES:
        attention = start == 5 and n == 3 and k == 1
        case = take({'kind': 'session_restore_case', 'owner': 'primary_restoration', 'start': start, 'inputs': n, 'retained': k,
              'restored_mod4': (start + k) % 4, 'restored_blocks': (start + k) // 4,
              'attention_batch_enabled': attention, 'continuation_rows': 3,
              'published_full_span_and_tap_bits_preserved': True, 'passed': True}, ('public_snapshots', 'argument_rejections'))
        snapshots = case['public_snapshots']
        window_snapshots(snapshots, start, n, k, attention, f'case.{start}.{n}.{k}')
        rejects = case['argument_rejections']
        pending = start == 3 and k == 0
        require(type(rejects) is list and len(rejects) == 1 + int(pending), 'case: rejection extent')
        if pending:
            rejection(rejects[0], 'invalid_pending_arguments', 11, snapshots['postverify'], 'case.pending_rejects')
        rejection(rejects[-1], 'consumed_window_restore', 1, snapshots['postrestore'], 'case.consumed_reject')
    for k in DIVERGENT_PREFIXES:
        divergent = take({'kind': 'session_restore_divergent_suffix', 'owner': 'primary_restoration',
                          'start': 3, 'inputs': 3, 'retained': k,
                          'continuation_rows': 3, 'passed': True}, ('public_snapshots',))
        window_snapshots(divergent['public_snapshots'], 3, 3, k, label=f'divergent.{k}')
    wide_rows = origin['wide_PP_rows']
    wide_owner = 'wide_PP_' + str(wide_rows)
    take_memory(wide_owner, 1, wide_rows, 0)
    primary_phase = phase(PRIMARY_COMPARED_ROWS, PRIMARY_EXACT_TAP_ROWS, RESTORES, REJECTIONS, MEMORY_OBSERVATIONS)
    wide_phase = phase(2 * wide_rows, 2 * wide_rows, 1, 1, 2)
    cpu_phase = phase(BATCH + 6, 0, 0, 0, 2, 6)
    wide = take({'kind': 'session_restore_wide_PP', 'owner': wide_owner, 'rows': wide_rows,
                 'all_rows_D2H_before_reuse': True, 'full_vocabulary_N1_reference': True,
                 'tap_replay_exact': True, 'phase_totals': wide_phase, 'passed': True},
                ('public_snapshots', 'hidden_N1_numerical_comparison'))
    ws = wide['public_snapshots']
    expect(ws, {}, ('preverify', 'postverify', 'postrestore', 'pre_late_ID_reject', 'post_late_ID_reject', 'post_reset_replay'), 'wide.snapshots')
    limits = {'max_batch': wide_rows, 'capacity': 2 * wide_rows}
    public(ws['preverify'], 0, 0, 0, 0, **limits)
    public(ws['postverify'], 1, 1, 1, 0, windows=1, verify_rows=1, pending=True, **limits)
    public(ws['postrestore'], 1, 0, 1, 0, windows=1, verify_rows=1, prefixes=(1, 0, 0, 0), **limits)
    restored(ws['postverify'], ws['postrestore'], 0, 'wide.restore')
    public(ws['pre_late_ID_reject'], 1 + wide_rows, wide_rows, wide_rows, 0,
           windows=1, verify_rows=1, prefixes=(1, 0, 0, 0), **limits)
    monotone(ws['postrestore'], ws['pre_late_ID_reject'], 'wide.PP')
    require(ws['postrestore']['target_tap']['pointer_address'] != ws['pre_late_ID_reject']['target_tap']['pointer_address'], 'wide.PP: tap did not alternate')
    exact(ws['post_late_ID_reject'], ws['pre_late_ID_reject'], 'wide.rejection')
    public(ws['post_reset_replay'], wide_rows, wide_rows, wide_rows, 0, **limits)
    hidden = wide['hidden_N1_numerical_comparison']
    expect(hidden, {'values': wide_rows * WIDTH}, ('max_error', 'max_bound_ratio'), 'wide.hidden')
    metrics(hidden, wide_rows * WIDTH, 'wide.hidden', diagnostic=False)
    take_memory('CPU_failure_recovery', 2, 3, 1)
    cpu = take({'kind': 'session_restore_CPU_failure_recovery', 'owner': 'CPU_failure_recovery', 'cpu_workers': 1,
                 'initial_policy': 'disabled',
                 'fault': 'after_gate_up_submission_and_GPU_admission_before_join_middle_down',
                 'old_tap_physical_bytes_preserved': True, 'sticky_rejections': 6,
                 'recovery_full_vocab_rows': 6, 'inflight_timing_claim': False, 'passed': True,
                 'phase_totals': cpu_phase}, ('public_snapshots', 'fault_diagnostics'))
    fault = cpu['fault_diagnostics']
    expect(fault, {'failure_layer': 0, 'synthetic_failure': True, 'cpu_pool_drained': True,
                  'gpu_streams_drained': True, 'cpu_return_bytes_before_failure': 0,
                  'ready_cache_ids': 0, 'pending_cache_ids': 0},
           ('accepted_cpu_jobs', 'queued_admission_copies', 'pending_slots_at_failure'), 'CPU.fault')
    for key in ('accepted_cpu_jobs', 'queued_admission_copies', 'pending_slots_at_failure'):
        integer(fault[key], 'CPU.fault.' + key, 1)
    cs = cpu['public_snapshots']
    expect(cs, {}, ('prefault', 'postfault', 'poststicky', 'postreset', 'postrecovery'), 'CPU.snapshots')
    public(cs['prefault'], 3, 3, 3, 0)
    for key in ('postfault', 'poststicky'):
        public(cs[key], 3, 3, 3, 0, available=False, label='CPU.' + key)
        for field in ('stats', 'route', 'hybrid', 'attention', 'speculative', 'target_tap'):
            exact(cs[key][field], cs['prefault'][field], 'CPU.' + key + '.' + field)
    exact(cs['poststicky'], cs['postfault'], 'CPU.sticky')
    public(cs['postreset'], 0, 0, 0, 0)
    public(cs['postrecovery'], 6, 6, 1, 5, hybrid=True, label='CPU.recovery')
    compared_rows = PRIMARY_COMPARED_ROWS + 2 * wide_rows + BATCH + 6
    exact_tap_rows = PRIMARY_EXACT_TAP_ROWS + 2 * wide_rows
    footer = take({
        'kind': 'session_restore_complete', 'restores': RESTORES, 'rejections': REJECTIONS,
        'compared_full_vocab_values': compared_rows * VOCAB, 'tap_exact_values': exact_tap_rows * WIDTH,
        'memory_observations': MEMORY_OBSERVATIONS, 'counts_scope': 'primary_restoration_fixture',
        'wide_PP_rows': wide_rows, 'CPU_failure_recovery_passed': True, 'session_instances': 3,
        'comparison_counts_scope': 'all_three_owners', 'compared_full_vocab_rows': compared_rows,
        'phase_totals': {'primary_restoration': primary_phase, 'wide_PP': wide_phase, 'CPU_failure_recovery': cpu_phase},
        'aggregate_counts': {'successful_restores': RESTORES + 1, 'argument_rejections': REJECTIONS + 1,
                             'sticky_rejections': 6, 'steady_memory_observations': MEMORY_OBSERVATIONS + 4},
        'records': RECORD_COUNT, 'loaded_memory_records': 3,
        'snapshot_counters_scope': 'completed_public_work_since_last_reset_not_kernel_counts',
        'tap_exact_count_scope': 'numerical_reference_and_replay_comparisons_excludes_rejection_and_failure_preservation_checks',
        'raii_cleanup_completed': True, 'owned_buffer_release_measured': False,
        'trained_MTP_qualified': False, 'HF_qualified': False, 'R6_complete_claim': False,
        'performance_claim': False, 'passed': True,
    }, ('diagnostic_bit_differences', 'max_error', 'max_bound_ratio', 'primary_extra_argument_rejections'))
    extras = footer['primary_extra_argument_rejections']
    require(type(extras) is list and len(extras) == 3, 'complete: extra rejection extent')
    for obj, group, count, args in zip(extras, ('ordinary_PP_invalidated_pending', 'capacity_and_prefix_arguments',
                                              'restore_after_reset_without_window'), (1, 3, 1),
                                      ((9, 9, 3, 6), (40, 40, 1, 39), (1, 1, 1, 0))):
        expect(obj, {'group': group, 'count': count}, ('before', 'after'), 'complete.extra_reject')
        kw = {'windows': 1, 'verify_rows': 3} if args[0] == 9 else (
              {'windows': 1, 'verify_rows': 1, 'pending': True} if args[0] == 40 else {})
        public(obj['before'], *args, **kw, label='complete.' + group)
        rejection(obj, group, count, obj['before'], 'complete.' + group)
    exact(cursor, RECORD_COUNT, 'complete.record_count')
    metrics(footer, compared_rows * VOCAB)
    retained_histogram = [sum(k == prefix for _, _, k in CASES) + 1 + int(prefix == 0)
                          for prefix in range(4)]
    mod4 = [sum((start + k) % 4 == phase for start, _, k in CASES) for phase in range(4)]
    return {
        'kind': 'r6_target_restore', 'protocol': 1,
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'revision': origin['revision'], 'dirty': origin['dirty'], 'runtime': 'own_48_layer_HIP',
        'model': origin['model_path'], 'model_variant': 'qwen38-keep1-Q4_0',
        'model_path_observable': True, 'model_variant_attested': False,
        'raw_logs': {'restore': str(raw_path)}, 'source': origin, 'complete': footer,
        'wide_PP': wide, 'CPU_failure_recovery': cpu,
        'loaded_memory': memories,
        'scope': 'three sequential target Sessions; retained same-Session N1 full-vocabulary restore/tap correctness and synthetic CPU failure recovery',
        'record_count': RECORD_COUNT,
        'aggregate_counts': footer['aggregate_counts'],
        'snapshot_validation': {'primary_windows': 45, 'divergent_windows': 4,
                                'public_counters_and_tap_metadata_checked': True,
                                'restore_physical_work_and_grouped_rejection_snapshots_preserved': True,
                                'physical_rows_distinct_from_logical_cursor': True},
        'case_summary': {'primary_cases': len(CASES), 'starts': list(STARTS), 'input_sizes': [1, 2, 3],
                         'every_retained_prefix': True, 'restored_mod4_cases': mod4,
                         'divergent_suffix_cases': len(DIVERGENT_PREFIXES), 'attention_enabled_cases': 1},
        'comparison_summary': {'full_vocab_rows': compared_rows, 'full_vocab_values': compared_rows * VOCAB,
                               'primary_compared_rows': PRIMARY_COMPARED_ROWS,
                               'wide_compared_rows': 2 * wide_rows, 'CPU_compared_rows': BATCH + 6,
                               'exact_tap_values': exact_tap_rows * WIDTH,
                               'absolute_gate': .02, 'relative_gate': .002,
                               'formula': 'abs(actual-reference)<=.02+.002*abs(reference)',
                               'finite_scope': 'driver_compare_requires_both_operands_finite_all_vocabulary',
                               'bit_identity_required': False, 'argmax_observable': False},
        'required_geometry': [required_geometry(3), required_geometry(wide_rows), required_geometry(3, 1)],
        'fixture_contract': {
            'scope': 'source_verified_requirements_not_additional_serialized_observations',
            'ple_source_ids': {'formula': '100+position', 'metadata_EOS_positions': [0, 4, 8, 10, 37],
                               'ple_eos': origin['ple_eos'], 'alternate_suffix_ids': [150, 151, 152]},
            'pending_window': 'verify_N1_2_3_publishes_PRE_and_all_N_chronological_slots_only_after_success',
            'invalid_pending_checks_per_N_at_start3_k0': 11,
            'restore': 'one_shot_k0_through_N_cursor_start_plus_k_no_physical_diagnostic_rewind',
            'restored_prefix_counts_required': retained_histogram,
            'retained_inputs_required': sum(k * count for k, count in enumerate(retained_histogram)),
            'prefix_APIs_per_successful_verify': {'GDN': 36, 'PLE': 1, 'QSA_tail': 12},
            'prefix_count_scope': 'prefix_enabled_API_invocations_not_physical_kernels',
            'primary_successful_forward_rows_required': PRIMARY_FORWARD_ROWS,
            'primary_route_assignments_required': 480 * PRIMARY_FORWARD_ROWS,
            'route_accounting': 'hits_plus_misses_equal_480_times_successful_forward_rows_since_reset_not_logical_cursor',
            'primary_verify_windows_required': PRIMARY_VERIFY_WINDOWS,
            'primary_verify_rows_required': PRIMARY_VERIFY_ROWS,
            'forward_row_counter': 'monotonic_completed_work_since_reset_restore_never_decreases_it',
            'target_tap': {'device': 1, 'width': WIDTH, 'dtype': 'F32', 'row_bytes': WIDTH * 4,
                           'location': 'layer47_final_widened_combine_before_root_HC',
                           'position': 'latest_successful_forward_first_absolute_position',
                           'carry': 'k>0_row_k_minus_1_k0_no_carry',
                           'lifetime': 'D2H_before_next_successful_forward_reset_destruction',
                           'restore_and_reject': 'metadata_and_all_published_physical_bytes_preserved'},
            'wide_tap': 'separate_default32_or_explicit1024_PP_all_rows_N1_numeric_gate_exact_reject_and_replay_bytes',
            'CPU_failure': 'synthetic_post_gate_up_acceptance_and_GPU_admission_before_join_middle_down_not_inflight_timing',
            'CPU_sticky_checks': ['step', 'verify_window', 'restore_prefix', 'checkpoint_state',
                                 'set_hybrid_policy', 'set_attention_batch'],
            'cleanup': 'completion_emitted_after_all_three_sequential_RAII_Session_scopes',
            'decision_math': 'stochastic_MTP_decision_math_is_a_separate_fixture_not_joined_here',
        },
        'evidence_limits': {
            'not_serialized': ['model_variant_metadata_and_model_file_bytes', 'full_raw_snapshot_buffers',
                               'individual_rejection_snapshots_within_group_and_full_span_bytes',
                               'per_step_memory_ledgers', 'individual_GPU_buffer_addresses_and_sizes',
                               'separate_working_logit_capacity', 'individual_CPU_vector_capacities_and_allocator_slack',
                               'tap_tensor_carry_bytes_and_independent_pre_root_HC_provenance',
                               'CPU_probe_payloads_and_individual_sticky_check_snapshots',
                               'post_destruction_owned_ledger_and_free_VRAM', 'detailed_individual_owned_release',
                               'independent_HF_reference', 'trained_MTP', 'full_model_precision_qualification',
                               'performance_measurements', 'full_R6'],
            'memory': 'three_constructor_ledgers_137_in_process_steady_checks_not_137_serialized_ledgers',
            'host_capacity': 'SessionMemory_reports_actual_sizeof_and_vector_capacity_not_RSS_config_included_in_Impl_not_added_twice',
            'working_logits': 'session.hip::Impl_allocates_for_cpu_workers_OR_speculative_checkpoints_memory_sums_both_capacities',
            'tap': 'source_location_and_driver_readback_assertions_not_serialized_tensor_or_independent_provenance',
            'physical_counters': 'serialized_completed_API_work_since_reset_not_physical_kernel_trace',
            'rejections': 'before_after_serialized_per_group_not_per_individual_rejected_call',
            'snapshot_buffer_counts': '43_44_geometry_derived_actual_getter_reports_all_live_owned_Buffer_counts_and_category_bytes',
            'CPU_capacity': 'reported_aggregate_capacities_and_sizeof_only_no_individual_vector_capacity_or_RSS_measurement',
            'cleanup': 'RAII_completion_claim_only_owned_buffer_release_measured_false_no_recovered_bytes_inferred',
            'reference': 'retained_same_Session_N1_not_independent_HF',
            'CPU_failure': 'synthetic_correctness_recovery_not_observed_overlap_or_inflight_timing',
        },
        'trained_MTP_qualified': False, 'HF_qualified': False, 'R6_complete_claim': False,
        'performance_claim': False, 'speedup_claim': False, 'physical_kernel_count_claim': False,
        'full_ram_accounting_qualified': False, 'exact_peak_vram_claim': False,
        'owned_buffer_release_measured': False, 'mtp': False, 'passed': True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True, help='explicit canonical ROOT/results.jsonl')
    args = parser.parse_args(argv)
    try:
        append_record(args.results, collect(args.raw))
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print('record_restore: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
