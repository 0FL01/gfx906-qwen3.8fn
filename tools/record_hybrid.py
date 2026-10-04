#!/usr/bin/env python3
"""Validate every core-session-hybrid-test protocol-1 record; append ONE r5_hybrid.

--results explicitly names the caller's canonical ROOT/results.jsonl. Caller
checks the executable's exit status first. This collector has no model/GPU IO.
The driver's assertions are not additional serialized observations: in
particular the protocol omits the model path, public/physical routing snapshots,
atomic-rejection records and steady GPU/read ledgers. Keep those limits explicit.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import stat
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


MAX_RAW_BYTES = 16 * 1024 * 1024
MAX_RECORD_BYTES = 64 * 1024  # Driver records are scalar summaries, never tensor arrays.
VOCAB, HIDDEN, LAYERS, ROWS, TEACHER = 248320, 2560, 48, 40, 32
DOWN_ROW, FFN_ROW = LAYERS * 10 * HIDDEN, LAYERS * HIDDEN
INTERMEDIATE_PROBE_BYTES = 2 * 3 * (DOWN_ROW + FFN_ROW) * 4
# Constructor-only, device0 payloads: two frames, three rows, all 48 layers.
# Q8_1 is 36 bytes; each route retains int32 ID, float weight, uint8 CPU flag,
# plus one uint8 availability flag per row/layer. These are pageable capacities.
INPUT_PROBE_BYTES = 2 * 3 * LAYERS * (80 * 36 + 10 * (4 + 4 + 1) + 1)
HOST_PROBE_BYTES = INTERMEDIATE_PROBE_BYTES + INPUT_PROBE_BYTES
# Each of two HybridDevice owners embeds two frames with five LP64 std::vector
# headers (24 bytes each), even though only device0 allocates their payloads.
INPUT_PROBE_PLAN_METADATA_BYTES = 2 * 2 * 5 * 24
Q40, Q41 = 2764800, 2867200
RAM_PAYLOAD = (6 * Q41 + 42 * Q40) * 512
FP32_MAX = float.fromhex('0x1.fffffep+127')
DISPATCH = ('cumulative_cpu_assignments', 'cumulative_gpu_hit_assignments',
            'cumulative_gpu_miss_assignments')
STAGES = ('cpu_gate_up_jobs', 'cpu_down_jobs', 'GPU_middle_columns',
          'GPU_middle_batches', 'paired_gate_up_H2D_bytes', 'middle_Q8_D2H_bytes')
CUMULATIVE_STAGES = tuple('cumulative_' + key for key in STAGES)
COUNTERS = (*DISPATCH, *CUMULATIVE_STAGES)
ZERO_COUNTERS = (0,) * len(COUNTERS)
FAMILIES = ('hybrid_OFF_GPU_only', 'diagnostic_force_CPU_LINEAR_GPU_middle', 'mixed_slot1')
PHASE_SPECS = tuple((f'{family}_N{n}', n, mode) for family, mode in
                    zip(FAMILIES, ('off', 'cpu', 'mixed')) for n in (1, 2, 3)) + (
    ('diagnostic_force_GPU_misses_N3', 3, 'gpu'),
    ('full_replay_after_submitted_admission_failure', 3, 'mixed'))
WIDE_TEACHER = (1, 4, 2, 4, 3, 4, 1, 4, 2, 4, 3)


def shapes(n):
    offset, result = 0, []
    while offset < ROWS:
        width = min(n, TEACHER - offset) if offset < TEACHER else 1
        result.append((offset, width))
        offset += width
    return result


# Derived from main/phase/ref.check, not inferred from a footer supplied by a log.
WINDOW_COUNT = sum(len(shapes(n)) for _, n, _ in PHASE_SPECS) + 2 + 1 + 13 + 1 + 3 + 19
FULL_ROWS = len(PHASE_SPECS) * ROWS + 2 + 3 + 39 + 1 + 3 + 40
WIDE_ROWS, WIDE_WINDOWS = sum(n for n in WIDE_TEACHER if n > 3), WIDE_TEACHER.count(4)
SHORT_ROWS = FULL_ROWS - WIDE_ROWS
RECORD_COUNT = WINDOW_COUNT + len(PHASE_SPECS) + 3 + 5 + 3  # memory, cases, header/reference/footer


def records(path):
    """Bound before/open/during reading; never load an unbounded or growing log."""
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), str(path) + ': not a regular file')
    require(info.st_size <= MAX_RAW_BYTES, str(path) + ': oversized input')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        require(stat.S_ISREG(opened.st_mode), str(path) + ': not a regular file')
        require(opened.st_size <= MAX_RAW_BYTES, str(path) + ': oversized input')
        raw = stream.read(MAX_RAW_BYTES + 1)
    require(len(raw) <= MAX_RAW_BYTES, str(path) + ': oversized input')
    require(raw.endswith(b'\n'), str(path) + ': missing completed final line')
    lines = raw.split(b'\n')[:-1]
    require(len(lines) == RECORD_COUNT, f'{path}: expected exactly {RECORD_COUNT} JSONL rows')
    result = []
    for i, line in enumerate(lines, 1):
        require(len(line) <= MAX_RECORD_BYTES, f'{path}:row {i}: oversized JSON record')
        try:
            obj = json.loads(line.decode('utf-8'), object_pairs_hook=unique_object,
                             parse_int=json_integer, parse_float=finite_float,
                             parse_constant=invalid_constant)
            require(type(obj) is dict, 'expected an object')
        except (ValueError, RecursionError) as error:
            raise ValueError(f'{path}:row {i}: {error}') from error
        result.append(obj)
    return result


def source(obj):
    expect(obj, {
        'kind': 'session_hybrid_header', 'protocol': 1,
        'model_variant': 'qwen38-keep1-Q4_0', 'sequential_Sessions': 3,
        'runtime_scope': 'CPUlinear_GPUmiddle',
        'force_cpu_enum_scope': 'all_routed_linear_CPU_with_canonical_GPU_middle',
        'pure_CPU_only_claim': False,
        'simultaneous_Sessions': 1, 'max_batch_tokens_by_Session': [3, 1, 4],
        'cpu_workers_by_Session': [1, 1, 4], 'reference_rows': 40,
        'teacher': 'BOS248044,100..130', 'continuation': '131..138',
        'full_logit_gate': '.02+.002*abs(ref)', 'intermediate_gate': '.002+.002*abs(ref)',
        'reference': 'retained_full_vocabulary_GPU_N1_self_reference',
        'independent_HF_claim': False, 'performance_claim': False,
        'R5_complete_claim': False, 'MTP_claim': False,
    }, ('revision', 'dirty'), 'source')
    rev = obj['revision']
    require(type(rev) is str and len(rev) == 40 and
            all(c in '0123456789abcdefABCDEF' for c in rev), 'source.revision: invalid compiled revision')
    require(type(obj['dirty']) is bool, 'source.dirty: expected boolean')


def metrics(obj, count, absolute, label):
    expect(obj, {'values': count, 'violations': 0},
           ('max_abs', 'max_bound_ratio', 'diagnostic_bit_mismatches'), label)
    maximum = number(obj['max_abs'], label + '.max_abs')
    ratio = number(obj['max_bound_ratio'], label + '.max_bound_ratio')
    bits = integer(obj['diagnostic_bit_mismatches'], label + '.diagnostic_bit_mismatches', 0, count)
    require(ratio <= 1, label + ': frozen numerical gate failed')
    # Reference maxima/coordinates are not emitted. Use only the finite-FP32
    # envelope and the ratio's mathematical bounds, not an invented ref norm.
    require(maximum <= absolute + .002 * FP32_MAX, label + ': impossible finite-FP32 gated error')
    if maximum == 0:
        require(ratio == 0, label + ': zero error with nonzero bound ratio')
    else:
        require(bits > 0 and ratio > 0, label + ': positive error missing diagnostics')
        require(ratio <= maximum / absolute and
                maximum / (absolute + .002 * FP32_MAX) <= ratio,
                label + ': inconsistent bound ratio')


def memory(obj, instance, previous=None):
    batch, workers, slots = ((3, 1, 1), (1, 1, 112), (4, 4, 1))[instance - 1]
    stage = Q41 * (16 if batch > 3 else 1)
    label = f'memory[{instance}]'
    expect(obj, {
        'kind': 'session_hybrid_memory', 'session_instance': instance,
        'slots': slots, 'max_batch_tokens': batch, 'cpu_workers': workers,
        'pinned_input_bytes': 17280, 'pinned_output_bytes': 614400, 'pinned_error_bytes': 8,
        'pinned_middle_Q8_bytes': 2 * 30 * 20 * 36,
        'pinned_stage_bytes': 4 * stage, 'pinned_route_metadata_bytes': 2 * batch * 10 * 8,
        'route_metadata_bytes_per_GPU': [batch * 10 * 8] * 2,
        'paired_gate_up_bytes_per_GPU': [30 * 2 * 640 * 4] * 2,
        'middle_float_bytes_per_GPU': [30 * 640 * 4] * 2,
        'middle_Q8_bytes_per_GPU': [30 * 20 * 36] * 2,
        'middle_error_bytes_per_GPU': [4] * 2,
        'contribution_bytes_per_GPU': [max(3, batch) * 10 * HIDDEN * 4] * 2,
        'stage_buffer_bytes_per_GPU': [stage] * 2,
        'host_input_probe_bytes': INPUT_PROBE_BYTES, 'host_probe_bytes': HOST_PROBE_BYTES,
        'full_RSS_or_stacks_claim': False, 'ownership_verified': True,
    }, ('host_plan_bytes', 'host_cpu_views_and_pending_bytes',
        'pool_metadata_bytes', 'pool_scratch_bytes', 'host_routing_capacity_bytes',
        'RouteGroups_requested_payload_bytes'), label)
    for key in ('host_plan_bytes', 'pool_metadata_bytes'):
        integer(obj[key], label + '.' + key, 1)
    # LP64 target: QMatrix is32 bytes, CpuExpert contains three; pending int/slot.
    integer(obj['host_cpu_views_and_pending_bytes'], label + '.host_cpu_views_and_pending_bytes',
            48 * 512 * 96 + 48 * slots * 4)
    # CpuExpertScratch: 64560 payload bytes rounded to its alignas(64).
    integer(obj['pool_scratch_bytes'], label + '.pool_scratch_bytes', workers * 64576)
    integer(obj['host_routing_capacity_bytes'], label + '.host_routing_capacity_bytes',
            2 * batch * (512 * 4 + 10 * 4 + 10 * 4))
    # RouteGroups object header includes vector/size_t; do not pretend its ABI
    # header is payload-free. Exact requested assignment payload is 2*N*10*16;
    # the unchanged object headers must agree between all three constructors.
    payload = integer(obj['RouteGroups_requested_payload_bytes'],
                      label + '.RouteGroups_requested_payload_bytes', 2 * 6148 + 320 * batch)
    header = payload - 320 * batch
    require(header % 2 == 0, label + ': unequal RouteGroups object headers')
    if previous is not None:
        exact(header, previous['RouteGroups_requested_payload_bytes'] -
              320 * previous['max_batch_tokens'], label + '.RouteGroups_header_steady')
        exact(obj['host_plan_bytes'], previous['host_plan_bytes'], label + '.host_plan_steady')


def window(obj, phase, offset, n, mode, previous=ZERO_COUNTERS, quota=2, slots=1, fresh=False):
    label = f'{phase}[{offset}]'
    expect(obj, {
        'kind': 'session_hybrid_window', 'phase': phase, 'offset': offset, 'rows': n,
        'continuation': offset >= TEACHER, 'intermediate_probe_available': n <= 3,
    }, ('logits', 'unweighted_down', 'ffn_output', *COUNTERS), label)
    metrics(obj['logits'], n * VOCAB, .02, label + '.logits')
    for key, extent in (('unweighted_down', DOWN_ROW), ('ffn_output', FFN_ROW)):
        if n <= 3:
            metrics(obj[key], n * extent, .002, label + '.' + key)
        else:
            exact(obj[key], None, label + '.' + key)
    limits = (480 * ROWS,) * 6 + (48 * ROWS, 5120 * 480 * ROWS, 720 * 480 * ROWS)
    current = tuple(integer(obj[k], label + '.' + k, 0, high) for k, high in zip(COUNTERS, limits))
    delta = tuple(a - b for a, b in zip(current, previous))
    require(all(d >= 0 for d in delta), label + ': dispatch/projection counters decreased')
    if mode == 'off':
        exact(list(current), list(ZERO_COUNTERS), label + '.disabled_scheduler')
    elif n > 3:
        exact(list(current), list(previous), label + '.wide_CPU_and_short_dispatch_unchanged')
    else:
        exact(sum(delta[:3]), 480 * n, label + '.dispatch_conservation')
        cpu, hit, miss = delta[:3]
        if mode == 'cpu':
            exact(list(delta[:3]), [480 * n, 0, 0], label + '.forced_CPU_LINEAR_dispatch')
        elif mode == 'gpu':
            exact(list(delta[:3]), [0, 0, 480 * n], label + '.forced_GPU_dispatch')
        else:
            require(miss <= 48 * quota * n, label + ': per-layer GPU quota exceeded')
            if slots == 1:
                require(hit <= 48 * n and cpu >= 48 * (9 - quota) * n and
                        miss >= 48 * quota, label + ': slot1 physical miss dispatch not exercised')
            if fresh:
                exact(hit, 0, label + '.fresh_replay_no_READY_dispatch')
        jobs, down, columns, batches, paired, middle = delta[3:]
        exact(down, jobs, label + '.two_projection_jobs_per_logical_CPU_group')
        integer(jobs, label + '.CPU_groups_from_accepted_gate_up_jobs', (cpu + n - 1) // n, cpu)
        exact(columns, cpu, label + '.GPU_middle_columns_per_CPU_assignment')
        exact(paired, 5120 * cpu, label + '.paired_gate_up_H2D_bytes_per_CPU_assignment')
        exact(middle, 720 * cpu, label + '.middle_Q8_D2H_bytes_per_CPU_assignment')
        # One canonical GPU middle batch per CPU-bearing layer, not per expert
        # group or worker. Slots1 necessarily dispatches CPU work in all48 layers;
        # the slots112 continuation may naturally use only some (or none).
        integer(batches, label + '.GPU_middle_batches_CPU_bearing_layers',
                (cpu + 10 * n - 1) // (10 * n), min(48, jobs))
        if mode == 'cpu' or (mode == 'mixed' and slots == 1):
            exact(batches, 48, label + '.GPU_middle_batches_equal_input_extractions')
        # Check cumulative identities as well as each accepted window delta.
        exact(current[4], current[3], label + '.cumulative_projection_job_pair')
        exact(current[5], current[0], label + '.cumulative_GPU_middle_columns')
        exact(current[7], current[0] * 5120, label + '.cumulative_paired_gate_up_H2D_bytes')
        exact(current[8], current[0] * 720, label + '.cumulative_middle_Q8_D2H_bytes')
    return current, delta


def phase(obj, name, n, mode, windows, fresh=False):
    label = name + '.phase'
    keys = ('cpu_groups', 'gpu_hit_groups', 'gpu_miss_groups', 'group_reuse_assignments',
            'admission_copies', 'evicted_ready_slots', 'input_extractions', 'CPU_input_bytes_checked', *STAGES)
    expect(obj, {
        'kind': 'session_hybrid_phase', 'phase': name, 'teacher_rows': 32, 'continuation_rows': 8,
        'fresh_cache_first_call_checked': fresh, 'passed': True,
    }, keys, label)
    for key in keys:
        integer(obj[key], label + '.' + key, 0, 480 * ROWS * (5120 if key == 'paired_gate_up_H2D_bytes' else 2880))
    calls, active = len(windows), mode != 'off'
    final = tuple(windows[-1][k] for k in DISPATCH)
    deltas, previous = [], (0, 0, 0)
    for w in windows:
        now = tuple(w[k] for k in DISPATCH)
        deltas.append(tuple(a - b for a, b in zip(now, previous)))
        previous = now
    for j, key in enumerate(keys[:3]):
        low = sum((d[j] + w['rows'] - 1) // w['rows'] for d, w in zip(deltas, windows))
        integer(obj[key], label + '.' + key, low, final[j])
    groups = sum(obj[k] for k in keys[:3])
    exact(groups + obj['group_reuse_assignments'], 480 * ROWS if active else 0,
          label + '.group_reuse_conservation')
    if active:
        require(groups >= 480 * calls, label + ': fewer than ten unique experts per token/layer')
    exact(obj['CPU_input_bytes_checked'], final[0] * 2880, label + '.original_Q8_descriptor_bytes')
    exact(obj['input_extractions'], 48 * calls if mode in ('cpu', 'mixed') else 0,
          label + '.input_extractions')
    if mode in ('off', 'cpu'):
        exact(obj['admission_copies'], 0, label + '.no_admission')
        exact(obj['evicted_ready_slots'], 0, label + '.no_eviction')
    else:
        exact(obj['admission_copies'], 96 * calls, label + '.bounded_actual_admission')
        integer(obj['evicted_ready_slots'], label + '.evicted_ready_slots', 1, 48 * calls)
    if mode == 'mixed':
        exact(obj['gpu_miss_groups'], 96 * calls, label + '.mixed_GPU_miss_groups')
    for key, cumulative in zip(STAGES, CUMULATIVE_STAGES):
        exact(obj[key], windows[-1][cumulative], label + '.' + key + '_matches_last_window')
    exact(obj['cpu_gate_up_jobs'], obj['cpu_groups'], label + '.gate_up_jobs_once_per_CPU_group')
    exact(obj['cpu_down_jobs'], obj['cpu_groups'], label + '.down_jobs_once_per_CPU_group')
    exact(obj['GPU_middle_columns'], final[0], label + '.GPU_middle_columns_per_CPU_assignment')
    exact(obj['GPU_middle_batches'], obj['input_extractions'], label + '.GPU_middle_batches_per_extraction')
    exact(obj['paired_gate_up_H2D_bytes'], final[0] * 5120, label + '.paired_gate_up_transfer_bytes')
    exact(obj['middle_Q8_D2H_bytes'], final[0] * 720, label + '.middle_Q8_transfer_bytes')


def collect(raw_path):
    """Return a compact validated record; neither append nor mutate input files."""
    raw_path = Path(raw_path).resolve()
    rows = records(raw_path)
    source(rows[0])
    cursor, windows, phases, memories, cases = 1, [], [], [], []
    stage_totals = dict.fromkeys(STAGES, 0)

    def take():
        nonlocal cursor
        require(cursor < len(rows), 'timeline: missing record')
        obj = rows[cursor]
        cursor += 1
        return obj

    def mem(instance):
        obj = take()
        memory(obj, instance, memories[0] if memories else None)
        memories.append(obj)

    def observe(obj, name, offset, n, mode, previous=ZERO_COUNTERS, quota=2, slots=1, fresh=False):
        current, delta = window(obj, name, offset, n, mode, previous, quota, slots, fresh)
        for key, value in zip(STAGES, delta[3:]):
            stage_totals[key] += value
        windows.append(obj)
        return current

    def calls(name, schedule, mode, quota=2, slots=1, fresh=False):
        previous, current = ZERO_COUNTERS, []
        for index, (offset, n) in enumerate(schedule):
            obj = take()
            previous = observe(obj, name, offset, n, mode, previous, quota, slots, fresh and index == 0)
            current.append(obj)
        return current

    def run_phase(spec, fresh=False):
        name, n, mode = spec
        current = calls(name, shapes(n), mode, fresh=fresh)
        obj = take()
        phase(obj, name, n, mode, current, fresh)
        phases.append(obj)

    mem(1)
    reference = take()
    expect(reference, {'kind': 'session_hybrid_reference', 'rows': ROWS,
                       'full_vocab_values': ROWS * VOCAB, 'routed_down_values': ROWS * DOWN_ROW,
                       'ffn_output_values': ROWS * FFN_ROW, 'all_finite': True}, label='reference')
    for spec in PHASE_SPECS[:-1]:
        run_phase(spec)
    calls('mixed_slot1_warm_BOS_quota1', [(0, 1)], 'mixed', quota=1)
    captured = calls('mixed_slot1_captured_reader_then_eviction_BOS', [(0, 1)], 'mixed', quota=1)[0]
    obj = take()
    expect(obj, {'kind': 'session_hybrid_captured_reader_reuse', 'slots': 1, 'quota': 1, 'passed': True},
           ('GPU_hit_groups', 'CPU_groups', 'GPU_miss_groups', 'evicted_READY_slots'), 'captured_reader_reuse')
    for key, value in zip(('CPU_groups', 'GPU_hit_groups', 'GPU_miss_groups'), (captured[k] for k in DISPATCH)):
        exact(obj[key], value, 'captured_reader_reuse.' + key)  # N1 assignments == groups
        integer(obj[key], 'captured_reader_reuse.' + key, 1, 480)
    integer(obj['evicted_READY_slots'], 'captured_reader_reuse.evicted_READY_slots', 1, 48)
    cases.append(obj)
    calls('synthetic_failure_setup_N3', [(0, 3)], 'mixed')
    run_phase(PHASE_SPECS[-1], fresh=True)
    obj = take()
    expect(obj, {
        'kind': 'session_hybrid_submitted_admission_failure', 'synthetic': True, 'failure_layer': 0,
        'accepted_CPU_job_stage': 'gate_up_before_GPU_middle_and_down',
        'queued_admission_copies': 2, 'pending_host_slots_at_throw': 1, 'CPU_return_bytes_before_throw': 0,
        'pool_and_both_GPU_streams_drained': True, 'all_READY_and_pending_IDs_invalidated': True,
        'published_stats_route_hybrid_logits_probes_preserved': True, 'reset_full_replay_rows': ROWS,
        'fresh_physical_misses_first_replay_assignments': 1440, 'inflight_timing_claim': False, 'passed': True,
    }, ('accepted_CPU_jobs',), 'submitted_admission_failure')
    integer(obj['accepted_CPU_jobs'], 'submitted_admission_failure.accepted_CPU_jobs', 7, 28)
    cases.append(obj)
    calls('sticky_failure_setup', [(i, 3) for i in range(0, 39, 3)], 'mixed')
    calls('reset_after_sticky_failure', [(0, 1)], 'mixed')
    obj = take()
    expect(obj, {'kind': 'session_hybrid_sticky_failure', 'injection': 'owned_trace_IO_before_CPU_submission',
                 'published_stats_logits_intermediates_preserved': True, 'reset_recovery': True, 'passed': True},
           label='sticky_failure')
    cases.append(obj)
    mem(2)
    calls('all_hit_warm_GPU_BOS', [(0, 1)], 'off', slots=112)
    allhit, previous = [], ZERO_COUNTERS
    for offset, name in enumerate(('real_all_hit_reset_BOS', 'all_hit_session_continuation')):
        obj = take()
        previous = observe(obj, name, offset, 1, 'mixed', previous, slots=112)
        allhit.append(obj)
    obj = take()
    expect(obj, {'kind': 'session_hybrid_real_all_hit', 'warm_reset_replay': 'BOS', 'ready_assignments': 480,
                 'CPU_extractions': 0, 'CPU_jobs': 0, 'GPU_uploads': 0, 'passed': True}, label='real_all_hit')
    exact([allhit[0][k] for k in DISPATCH], [0, 480, 0], 'real_all_hit.BOS_dispatch')
    exact([allhit[0][k] for k in CUMULATIVE_STAGES], [0] * 6, 'real_all_hit.BOS_no_CPU_linear_or_GPU_middle_work')
    cases.append(obj)
    mem(3)
    offset, previous = 0, ZERO_COUNTERS
    for n in (*WIDE_TEACHER, *([1] * 8)):
        name = 'max4_GPU_fallback_N4' if n == 4 else f'max4_short_hybrid_N{n}'
        obj = take()
        previous = observe(obj, name, offset, n, 'mixed', previous)
        offset += n
    obj = take()
    expect(obj, {
        'kind': 'session_hybrid_short_wide_short', 'session_instance': 3, 'max_batch_tokens': 4, 'cpu_workers': 4,
        'teacher_window_sizes': list(WIDE_TEACHER), 'teacher_rows': TEACHER, 'continuation_N1_rows': 8,
        'short_calls': 14, 'wide_N4_calls': 5, 'short_layers': 672, 'gpu_only_wide_layers': 240,
        'wide_CPU_counters_and_transfer_bytes_unchanged': True, 'wide_intermediate_probe_available': False,
        'stage_capacity_bytes_each': 16 * Q41, 'total_pinned_stage_bytes': 4 * 16 * Q41,
        'contribution_bytes_each_GPU': 4 * 10 * HIDDEN * 4, 'full_40_row_replay': True, 'passed': True,
    }, label='short_wide_short')
    cases.append(obj)
    footer = take()
    short = [w for w in windows if w['rows'] <= 3]
    wide = [w for w in windows if w['rows'] > 3]
    compared_rows, short_rows = sum(w['rows'] for w in windows), sum(w['rows'] for w in short)
    expect(footer, {
        'kind': 'session_hybrid_footer', 'passed': True, 'sequential_Sessions': 3, 'simultaneous_Sessions': 1,
        'runtime_scope': 'CPUlinear_GPUmiddle', 'pure_CPU_only_claim': False,
        'bounded_GPU_quota': 2, 'physical_residency_not_dispatch': True,
        'sticky_failure_scopes': ['trace_IO_before_CPU', 'synthetic_accepted_CPU_plus_queued_GPU_admission'],
        'retained_reference_rows': ROWS, 'compared_windows': len(windows),
        'compared_full_vocab_rows': compared_rows, 'compared_full_vocab_values': compared_rows * VOCAB,
        'intermediate_compared_rows': short_rows, 'compared_routed_down_values': short_rows * DOWN_ROW,
        'compared_ffn_output_values': short_rows * FFN_ROW, 'wide_N4_windows': len(wide),
        'wide_N4_rows': sum(w['rows'] for w in wide), 'expected_execution_failures': 2,
        'CPU_descriptor_exact_original_GPU_Q8_bytes': True, 'CPU_descriptor_matches_GPU_route_DTO': True,
        'numeric_failure_fixture': 'separate_CPU_stages_canonical_GPU_middle_and_GPU_shadow',
        'individual_added_GPU_buffers_observable': True, 'physical_128_tile_claim': False,
        'performance_claim': False, 'R5_complete_claim': False,
    }, label='complete')
    exact(cursor, RECORD_COUNT, 'complete.record_count')
    exact([len(windows), compared_rows, short_rows, len(wide)],
          [WINDOW_COUNT, FULL_ROWS, SHORT_ROWS, WIDE_WINDOWS], 'complete.derived_counts')

    def summary(key, selected):
        return {'values': sum(w[key]['values'] for w in selected), 'violations': 0,
                'max_abs': max(w[key]['max_abs'] for w in selected),
                'max_bound_ratio': max(w[key]['max_bound_ratio'] for w in selected),
                'diagnostic_bit_mismatches': sum(w[key]['diagnostic_bit_mismatches'] for w in selected)}

    return {
        'kind': 'r5_hybrid', 'protocol': 1,
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'revision': rows[0]['revision'], 'dirty': rows[0]['dirty'], 'runtime': 'own_48_layer_HIP',
        'runtime_scope': 'CPUlinear_GPUmiddle', 'pure_CPU_only_claim': False,
        'model_variant': rows[0]['model_variant'],
        # append_result requires a model artifact identifier. This is the declared
        # variant filename ONLY: no absent path is invented or read from weights.
        'model': rows[0]['model_variant'] + '.gguf', 'model_path_observable': False,
        'raw_logs': {'hybrid': str(raw_path)},
        'scope': 'CPU linear projections with canonical GPU middle; short same-runtime retained GPU-N1 self-parity and synthetic lifetime recovery; correctness only',
        'source': rows[0], 'reference': reference, 'phases': phases, 'memory': memories,
        'case_summary': cases, 'complete': footer,
        'window_summary': {'windows': len(windows), 'full_vocab_rows': compared_rows,
                           'intermediate_rows': short_rows, 'wide_N4_windows': len(wide),
                            'wide_N4_rows': sum(w['rows'] for w in wide),
                            'projection_totals': stage_totals,
                            'projection_totals_scope': 'sum_accepted_compared_window_deltas_excludes_reference_and_failed_call',
                           'logits': summary('logits', windows),
                           'unweighted_down': summary('unweighted_down', short),
                           'ffn_output': summary('ffn_output', short),
                           'finite_counts_scope': 'reference.all_finite_and_driver_compare_requires_finite_both_operands',
                           'bit_equality_required': False, 'argmax_equality_required': False,
                           'argmax_observable': False},
        'fixture_contract': {
            'weight_values': 'unchanged_loaded_GGUF', 'weight_precision': 'unchanged_loaded_tensor_types',
            'kv': 'Q4_0_K_and_V', 'sampling': 'teacher_forced',
            'contract_scope': 'source-verified_driver_and_Session_API_not_additional_raw_fields',
            'original_rank_fold': 'token_major_original_rank_all_ten_unweighted_expert_contributions_plus_shared',
            'physical_residency': 'READY_assignments_distinct_from_CPU_GPU_dispatch',
            'historical_stats': 'first_READY_acquisition_or_miss_plus_C_minus_1_within_window_reuse',
            'reset': 'clears_GDN_QSA_HC_PLE_logical_state_and_published_stats_probes_retains_warm_READY_cache',
            'span_lifetime': 'last_success_active_extent_checked_before_next_accepted_call_never_stale_span',
            'atomic_reject_driver_checks_required': 175,
            'atomic_reject_active_logit_values_required': 447 * VOCAB,
            'atomic_reject_driver_cases': ['empty_window', 'oversized_window', 'negative_last_ID',
                                           'out_of_vocabulary_last_ID', 'invalid_mode', 'mixed_quota_zero',
                                           'mixed_quota_three', 'disabled_negative_quota',
                                           'force_CPU_synthetic_fault_policy', 'exhausted_capacity'],
            'atomic_reject_counts_scope': 'source-derived_required_checks_not_serialized_observations',
            'constructor_payload_reads_required': 144, 'constructor_payload_bytes_required': RAM_PAYLOAD,
            'constructor_host_input_probe_payload_bytes_required': INPUT_PROBE_BYTES,
            'constructor_host_intermediate_probe_payload_bytes_required': INTERMEDIATE_PROBE_BYTES,
            'constructor_host_total_probe_payload_bytes_required': HOST_PROBE_BYTES,
            'input_probe_plan_metadata_added_bytes_LP64': INPUT_PROBE_PLAN_METADATA_BYTES,
            'input_probe_memory_scope': 'constructor_pageable_double_buffer_payload_in_host_probe_bytes_vector_headers_in_host_plan_bytes_no_double_count',
            'CPU_runtime_scope': 'gate_up_and_down_projections_with_canonical_GPU_SiLU_Q8_no_host_middle',
            'CPU_group_accounting': 'one_logical_expert_group_two_accepted_projection_jobs_gate_up_and_down',
            'GPU_middle_accounting': 'one_batch_per_CPU_bearing_layer_one_column_per_CPU_assignment_5120B_paired_H2D_720B_Q8_D2H',
            'synthetic_failure': 'accepted_gate_up_jobs_and_queued_admission_before_GPU_middle_and_down_and_CPU_return_not_inflight_timing_proof',
        },
        'evidence_limits': {
            'not_serialized': ['source.model_path', 'source.kv', 'source.weight_values',
                               'window.stats_before_after', 'window.route_stats',
                               'window.ready_hit_assignments', 'window.physical_miss_assignments',
                               'phase.atomic_rejection_proofs', 'phase.policy_probe_preservation',
                               'window.input_probe_original_route_dispatch_payloads',
                               'memory.steady_categories_devices_owned_ledger',
                               'memory.expert_payload_reads', 'memory.expert_payload_bytes_read',
                               'complete.owned_buffer_release'],
            'driver_assertions_only': 'atomic_reject_all_active_SPAN_stats_route_hybrid_policy_probe_and_fault_evidence; steady_GPU_categories_host_capacities_constructor_reads',
            'model_alias_guard': 'declared_filename_only_actual_model_path_not_emitted',
            'memory_scope': 'reported_payload_capacities_and_explicit_metadata_pool_headers_excludes_allocator_slack_stacks_full_RSS',
            'cleanup_scope': 'three_sequential_RAII_owners_footer_after_destruction_no_measured_owned_release',
            'forced_dispatch_scope': 'diagnostic_all_routed_CPU_linear_canonical_GPU_middle_or_GPU_misses_not_actual_cold_physical_cache',
            'whole_CPU_expert_scope': 'host_libm_middle_retained_oracle_not_production_qualified',
            'numeric_failure_scope': 'separate_CPU_stages_canonical_GPU_middle_and_actual_GPU_shadow_not_joined_or_attested_by_this_log',
            'GPU_middle_memory_scope': 'four_independent_buffers_252004B_per_GPU_in_workspace_owned_ledger_pinned_Q8_43200B_capacity_not_RSS_no_extra_ledger_attestation',
        },
        'record_count': RECORD_COUNT, 'performance_claim': False, 'speedup_claim': False,
        'dispatch_threshold_qualified': False, 'R5_complete_claim': False,
        'occupied_128k_qualified': False, 'independent_hf_claim': False, 'mtp': False,
        'full_ram_accounting_qualified': False, 'exact_peak_vram_claim': False,
        'owned_buffer_release_measured': False, 'passed': True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True, help='explicit canonical ROOT/results.jsonl')
    args = parser.parse_args(argv)
    try:
        append_record(args.results, collect(args.raw))
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print('record_hybrid: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
