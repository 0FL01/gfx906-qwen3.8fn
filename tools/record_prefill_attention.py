#!/usr/bin/env python3
"""Validate core-prefill-attention-test protocols 1/2; append ONE r4_attention_prefill.

The frozen C++ emitter has 31 successful records. This is retained same-Session
old-N1 full-vocabulary correctness, not a component gate, HF oracle or speed test.
--results is the caller's explicit canonical ROOT/results.jsonl; collect never
writes. The caller must check the executable's exit status before using main.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import stat
import sys

if __package__:
    from .record_memory import (append_result as append_record, exact, expect,
                                finite_float, integer, invalid_constant, number,
                                require, text, unique_object)
    from .record_prefill import json_integer, leq, stats
else:
    from record_memory import (append_result as append_record, exact, expect,
                               finite_float, integer, invalid_constant, number,
                               require, text, unique_object)
    from record_prefill import json_integer, leq, stats


MAX_RAW_BYTES, MAX_RECORD_BYTES = 16 * 1024 * 1024, 64 * 1024
RECORD_COUNT, VOCAB, CAPACITY, TEACHER, MAX_BATCH, TILE = 31, 248320, 2088, 2056, 1024, 8
Q40, Q41 = 2764800, 2867200
RAM_PAYLOAD = (6 * Q41 + 42 * Q40) * 512
REFERENCE_BYTES, PRESERVATION_BYTES = CAPACITY * VOCAB * 4, MAX_BATCH * VOCAB * 4
PHASES = ('reference_old_n1', 'enabled1024', 'occupied5_enabled997')
COUNTERS = ('expert_hits', 'expert_misses', 'expert_upload_bytes')
GROUP_FIELDS = ('last_max_expert_group_assignments', 'expert_groups_gt128')
ATTENTION_FIELDS = ('batch_calls', 'query_rows', 'multiquery_calls', 'multiquery_rows',
                    'singleton_tail_calls', 'max_query_rows')
CATEGORIES = ('weights', 'expert_slots', 'qsa_kv', 'qsa_index', 'gdn_state', 'ple_state', 'workspace')
ZERO_HOST = ('pinned_hybrid_input', 'pinned_hybrid_output', 'pinned_hybrid_error',
             'host_hybrid_plans', 'host_cpu_expert_views', 'cpu_pool_metadata', 'cpu_pool_scratch',
             'host_hybrid_probe', 'host_hybrid_input_probe', 'pinned_hybrid_middle')
ZERO_ARRAYS = ('hybrid_contribution_bytes', 'hybrid_gate_up_bytes', 'hybrid_middle_float_bytes',
               'hybrid_middle_q8_bytes', 'hybrid_middle_error_bytes')
ATTENTION_MEMORY = {
    'gathered_key_bytes': TILE * 1050112 * 2, 'gathered_value_bytes': TILE * 1050112 * 2,
    'partial_output_bytes': TILE * 202752 * 4, 'partial_max_sum_bytes': TILE * 792 * 8,
    'staged_buffer_bytes': MAX_BATCH * 12288 * 4, 'output_buffer_bytes': MAX_BATCH * 12288 * 4,
    'selected_id_bytes': TILE * 2051 * 4, 'selected_block_bytes': TILE * 512 * 4,
    'selected_count_bytes': TILE * 2 * 4, 'private_workspace_bytes': 40338944,
}
# The staged tile prefix REUSES f(15), so it is not another allocation. Selection
# is separate from the four private buffers and their staged prefix.
STAGED_PREFIX_BYTES = TILE * 6144 * 4
PRIVATE_BUFFER_BYTES = sum(ATTENTION_MEMORY[k] for k in (
    'gathered_key_bytes', 'gathered_value_bytes', 'partial_output_bytes', 'partial_max_sum_bytes'))
SELECTION_BYTES = sum(ATTENTION_MEMORY[k] for k in (
    'selected_id_bytes', 'selected_block_bytes', 'selected_count_bytes'))
# Source-derived necessary floor, not a reported per-buffer measurement. Session
# owns 24 scratch buffers, Q8, logits, routed contributions, two expert stages,
# the four attention buffers, selection and indexed route metadata on each GPU.
WORKSPACE_FLOOR = (24 * MAX_BATCH * 12288 * 4 + MAX_BATCH * 320 * 36 + PRESERVATION_BYTES +
                   MAX_BATCH * 10 * 2560 * 4 + 2 * 16 * Q41 + PRIVATE_BUFFER_BYTES +
                   SELECTION_BYTES + MAX_BATCH * 10 * 8)
FP32_MAX = float.fromhex('0x1.fffffep+127')
ERROR_METRICS = ('maxabs', 'rms', 'maxboundratio')
ERROR_COORDS = ('maxabs_row', 'maxabs_column', 'maxratio_row', 'maxratio_column')


def protocol(obj):
    return integer(obj.get("protocol"), "protocol", 1, 2)


def workspace_floor(version):
    # Version1 keeps its ORIGINAL full-frame device-output floor. Version2
    # explicitly reports each bounded128 Buffer; host output capacity is unchanged.
    return WORKSPACE_FLOOR - ((MAX_BATCH - 128) * VOCAB * 4 if version == 2 else 0)


def records(path):
    """Bound stat/open/read and each scalar-summary record; no tensor arrays."""
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
    require(1 <= len(lines) <= RECORD_COUNT + 1, str(path) + ': incorrect record count')
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


def base(kind, record_index, phase_index=None, passed=True, version=1):
    return {'protocol': version, 'kind': 'prefill_attention_' + kind, 'record_index': record_index,
            'phase_index': phase_index, 'phase': PHASES[phase_index] if phase_index is not None else None,
            'passed': passed}


def failure(obj, index):
    # main's failure footer always has null phase, even when a phase failed.
    expect(obj, {**base('failure', index, passed=False, version=protocol(obj)), 'all_owners_unwound': True},
           ('session_constructed', 'failure'), 'failure')
    require(type(obj['session_constructed']) is bool, 'failure.session_constructed: expected boolean')
    message = text(obj['failure'], 'failure.failure')
    require(len(message.encode('utf-8')) <= 2048, 'failure: oversized message')
    raise ValueError('prefill attention execution failed: ' + message)


def failed_window(obj, record_index):
    """Recognize the catch-path schema, but never turn partial work into evidence."""
    phase_index = integer(obj.get('phase_index'), 'failed_window.phase_index', 0, 2)
    label = 'failed_window'
    expect(obj, base('window', record_index, phase_index, passed=False, version=protocol(obj)),
           ('window_index', 'offset', 'rows', 'completed_rows', 'completed_calls', 'segment',
            'errors', 'state_before', 'state_after', 'failure'), label)
    integer(obj['window_index'], label + '.window_index', 0, 5)
    offset = integer(obj['offset'], label + '.offset', 0, CAPACITY - 1)
    count = integer(obj['rows'], label + '.rows', 1, CAPACITY - offset)
    completed = integer(obj['completed_rows'], label + '.completed_rows', 0, count)
    integer(obj['completed_calls'], label + '.completed_calls', 0, completed)
    require(obj['segment'] in ('teacher', 'occupied_prefix', 'continuation', 'continuation_final'),
            label + ': invalid segment')
    state(obj['state_before'], offset, bool(phase_index), label + '.state_before')
    after = obj['state_after']
    require(type(after) is dict and type(after.get('stats')) is dict, label + ': missing state')
    consumed = integer(after['stats'].get('consumed_tokens'), label + '.consumed', offset, offset + completed)
    state(after, consumed, bool(phase_index), label + '.state_after')
    e = obj['errors']
    expect(e, {}, ('values', 'finite', 'compared', 'finite_pairs', 'nonfinite_actual', 'nonfinite_reference',
                  'violations', 'bit_mismatches', 'argmax_rows', 'argmax_agree', *ERROR_METRICS, *ERROR_COORDS,
                  'first_violation_row', 'first_violation_column'), label + '.errors')
    for k in e.keys() - set(ERROR_METRICS) - {'first_violation_row', 'first_violation_column'}:
        integer(e[k], label + '.errors.' + k)
    for k in ERROR_METRICS:
        number(e[k], label + '.errors.' + k)
    if e['violations']:
        integer(e['first_violation_row'], label + '.first_row', offset, offset + count - 1)
        integer(e['first_violation_column'], label + '.first_column', 0, VOCAB - 1)
    else:
        exact(e['first_violation_row'], None, label + '.first_row')
        exact(e['first_violation_column'], None, label + '.first_column')
    message = text(obj['failure'], label + '.failure')
    require(len(message.encode('utf-8')) <= 1024, label + ': oversized message')
    raise ValueError('prefill attention window failed: ' + message)


def source(obj):
    expect(obj, {
        **base('source', 0, version=protocol(obj)),
        'config': {'capacity': 2088, 'expert_slots': 1, 'max_batch_tokens': 1024,
                   'cpu_workers': 0, 'hybrid_probe': False, 'trace_directory': '',
                   'attention_query_tile': 8, 'initial_attention_batch': False},
        'source_ids': {'bos': 248044, 'formula_after_bos': '99+absolute_position',
                       'teacher_rows': 2056, 'teacher_last': 2154, 'continuation_rows': 32,
                       'continuation_first': 2155, 'continuation_last': 2186},
        'schedules': [
            {'phase': PHASES[0], 'teacher_calls': 2056, 'teacher_call_rows': 1, 'continuation_calls': 32},
            {'phase': PHASES[1], 'teacher_chunks': [1024, 1024, 8], 'continuation_calls': 32},
            {'phase': PHASES[2], 'prefix_n1_calls': 5, 'teacher_chunks': [997, 997, 57], 'continuation_calls': 32}],
        'gate': {'absolute': .02, 'relative': .002,
                 'formula': 'abs(actual-reference)<=.02+.002*abs(reference)',
                 'all_vocabulary': True, 'immediate': True,
                 'bit_identity_required': False, 'argmax_identity_required': False},
        'reference_bytes': REFERENCE_BYTES, 'preservation_bytes': PRESERVATION_BYTES, 'guard_bytes': 256,
        'private_workspace_bytes_per_device': 40338944, 'selection_bytes_per_device': 82080,
        'expected_success_records': 31,
        'scope': {'reference': 'same-session old N1', 'component_gate_required_separately': True,
                  'actual_selected_id_trace': False, 'actual_visibility_trace': False, 'hf_oracle': False,
                  'performance_qualification': False, 'long4k16k': False,
                  'logical_row_coordinate': 'zero-based absolute source position',
                  'expected_visibility': 'position+1; computed from accepted rows, not GPU-observed'},
    }, ('revision', 'dirty', 'model', 'model_bytes'), 'source')
    revision = obj['revision']
    require(type(revision) is str and len(revision) == 40 and
            all(c in '0123456789abcdefABCDEF' for c in revision), 'source: invalid compiled revision')
    # Frozen source streams CORE_DIRTY directly (integer macro), not boolalpha.
    # Preserve it in source; expose a boolean ONLY in the aggregate envelope.
    integer(obj['dirty'], 'source.dirty', 0, 1)
    model = Path(text(obj['model'], 'source.model'))
    require(model.is_absolute() and model.name == 'qwen38-keep1-Q4_0.gguf',
            'source: expected absolute keep1-Q4_0 model path')
    integer(obj['model_bytes'], 'source.model_bytes', 1, (1 << 63) - 1)


def zero_state(enabled):
    return {'stats': {'consumed_tokens': 0, **dict.fromkeys(COUNTERS, 0),
                      'last_completed_ms': 0, 'last_completed_ms_bits': 0},
            'route_stats': dict.fromkeys(GROUP_FIELDS, 0),
            'attention_stats': dict.fromkeys(ATTENTION_FIELDS, 0), 'attention_batch': enabled,
            'hybrid_disabled_stats_all_zero': True, 'probe_off_publication_empty': True}


def state(obj, consumed, enabled, label):
    expect(obj, {'attention_batch': enabled, 'hybrid_disabled_stats_all_zero': True,
                 'probe_off_publication_empty': True}, ('stats', 'route_stats', 'attention_stats'), label)
    stats(obj['stats'], consumed, label + '.stats')
    expect(obj['route_stats'], {}, GROUP_FIELDS, label + '.routes')
    for key in GROUP_FIELDS:
        integer(obj['route_stats'][key], label + '.routes.' + key)
    expect(obj['attention_stats'], {}, ATTENTION_FIELDS, label + '.attention')
    for key in ATTENTION_FIELDS:
        integer(obj['attention_stats'][key], label + '.attention.' + key)


def same_state(obj, previous, label):
    state(obj, previous['stats']['consumed_tokens'], previous['attention_batch'], label)
    # Compare timing bits, since C++ includes bit_cast<double> in state_key.
    exact(obj, previous, label)


def expected_attention(previous, enabled, rows, singles):
    result = previous.copy()
    if enabled and not singles and rows > 1:
        tiles, tails = (rows + TILE - 1) // TILE, int(rows % TILE == 1)
        for key, n in zip(ATTENTION_FIELDS[:-1], (12 * tiles, 12 * rows, 12 * (tiles - tails),
                                                12 * (rows - tails), 12 * tails)):
            result[key] += n
        result['max_query_rows'] = max(result['max_query_rows'], min(rows, TILE))
    return result


def completed_state(obj, previous, offset, rows, singles, label):
    state(obj, offset + rows, previous['attention_batch'], label)
    deltas = [integer(obj['stats'][k] - previous['stats'][k], label + '.delta.' + k) for k in COUNTERS]
    hits, misses, uploaded = deltas
    require(hits + misses == 480 * rows and misses * Q40 <= uploaded <= misses * Q41,
            label + ': unchanged route/payload accounting failed')
    maximum = integer(obj['route_stats'][GROUP_FIELDS[0]], label + '.max_group', 1, 1 if singles else rows)
    groups = integer(obj['route_stats'][GROUP_FIELDS[1]] - previous['route_stats'][GROUP_FIELDS[1]],
                     label + '.groups_delta', 0, 0 if singles else 480 * rows // 129)
    exact(groups > 0, maximum > 128, label + '.threshold')
    exact(obj['attention_stats'], expected_attention(previous['attention_stats'],
          previous['attention_batch'], rows, singles), label + '.completed_API_counters')
    return deltas


def memory(obj, label, loaded=None, version=1):
    expect(obj, {
        'capacity': CAPACITY, 'expert_slots': 1, 'attention_query_tile': TILE, 'ownership_verified': True,
        'ram_expert_payload': RAM_PAYLOAD, 'expert_payload_reads': 144, 'expert_payload_bytes_read': RAM_PAYLOAD,
        'pinned_handoff': 41943040, 'pinned_expert_staging': 183500800, 'cpu_workers': 0,
        **dict.fromkeys(ZERO_HOST, 0), **{k: [0, 0] for k in ZERO_ARRAYS},
        'expert_stage_capacity_bytes': [16 * Q41] * 2, 'route_metadata_bytes': [MAX_BATCH * 80] * 2,
        'pinned_route_metadata': MAX_BATCH * 160,
    }, ('ram_expert_capacity', 'host_embedding_capacity', 'host_logit_capacity',
        'host_routing_capacity', 'host_route_group_payload', 'devices'), label)
    integer(obj['ram_expert_capacity'], label + '.ram_capacity', RAM_PAYLOAD)
    integer(obj['host_embedding_capacity'], label + '.embedding_capacity', 1)
    # SessionMemory sums host_logits and the opt-in working_logits capacities.
    # This CPU-workers-off fixture owns only host_logits; the transactional
    # second owner is allocated only when cpu_workers > 0 (Session::Impl).
    integer(obj['host_logit_capacity'], label + '.logit_capacity', PRESERVATION_BYTES)
    integer(obj['host_routing_capacity'], label + '.routing_capacity', 2 * MAX_BATCH * (512 + 20) * 4)
    integer(obj['host_route_group_payload'], label + '.route_group_payload', 1)
    devices = obj['devices']
    require(type(devices) is list and len(devices) == 2, label + ': expected both owners')
    for i, d in enumerate(devices):
        where = f'{label}.devices[{i}]'
        expect(d, {**({'head_logits_bytes': 128 * VOCAB * 4} if version == 2 else {}), 'device': i, 'first_layer': 24 * i, 'last_layer': 24 * i + 23,
                   'gdn_layers': 18, 'qsa_layers': 6,
                   'expert_slots': (6 * Q41 + 18 * Q40) if i == 0 else 24 * Q40,
                   'qsa_kv': 6 * CAPACITY * 32 * 18, 'qsa_index': 6 * (CAPACITY // 4 * 128 + 384) * 4,
                   'gdn_state': 18 * (786432 + 30720) * 4, 'ple_state': 368640 if i == 0 else 0,
                   'attention': ATTENTION_MEMORY},
               ('weights', 'workspace', 'owned_bytes', 'owned_peak_bytes', 'owned_buffers',
                'total_vram', 'free_vram'), where)
        integer(d['weights'], where + '.weights', 1)
        integer(d['workspace'], where + '.workspace', workspace_floor(version))
        owned = integer(d['owned_bytes'], where + '.owned_bytes', 1)
        exact(owned, integer(sum(d[k] for k in CATEGORIES), where + '.category_sum'), where + '.category_sum')
        integer(d['owned_peak_bytes'], where + '.peak', owned)
        integer(d['owned_buffers'], where + '.buffers', 1)
        total = integer(d['total_vram'], where + '.total', 1)
        free = integer(d['free_vram'], where + '.free', 1, total)
        require(owned <= total - free, where + ': owned bytes exceed observed used VRAM')
        if loaded is not None:
            for key in d.keys() - {'free_vram'}:
                exact(d[key], loaded['devices'][i][key], where + '.steady.' + key)
    if loaded is not None:
        for key in obj.keys() - {'devices'}:
            exact(obj[key], loaded[key], label + '.steady.' + key)


def errors(obj, offset, rows, compared, label):
    values, pairs = rows * VOCAB, rows * VOCAB if compared else 0
    expect(obj, {'values': values, 'finite': values, 'compared': pairs, 'finite_pairs': pairs,
                 'nonfinite_actual': 0, 'nonfinite_reference': 0, 'violations': 0,
                 'argmax_rows': rows if compared else 0,
                 'first_violation_row': None, 'first_violation_column': None},
           ('bit_mismatches', 'argmax_agree', *ERROR_METRICS, *ERROR_COORDS), label)
    bits = integer(obj['bit_mismatches'], label + '.bits', 0, pairs)
    integer(obj['argmax_agree'], label + '.argmax_agree', 0, rows if compared else 0)
    maximum, rms, ratio = [number(obj[k], label + '.' + k) for k in ERROR_METRICS]
    if not compared:
        for k in (*ERROR_METRICS, *ERROR_COORDS):
            require(type(obj[k]) in (int, float) and obj[k] == 0, label + ': reference error fields must be zero')
        for k in ERROR_COORDS:
            integer(obj[k], label + '.' + k, 0, 0)
        return
    require(ratio <= 1 and maximum <= .02 + .002 * FP32_MAX, label + ': frozen full-vocabulary gate failed')
    leq(rms, maximum, label + '.rms<=max', pairs)
    leq(maximum / math.sqrt(pairs), rms, label + '.rms>=one_max', pairs)
    leq(rms, maximum * math.sqrt(bits / pairs), label + '.rms_vs_bits', pairs)
    leq(maximum / (.02 + .002 * FP32_MAX), ratio, label + '.ratio_lower')
    leq(ratio, maximum / .02, label + '.ratio_upper')
    for k in ('maxabs_row', 'maxratio_row'):
        integer(obj[k], label + '.' + k, offset, offset + rows - 1)
    for k in ('maxabs_column', 'maxratio_column'):
        integer(obj[k], label + '.' + k, 0, VOCAB - 1)
    if maximum == 0:
        require(rms == ratio == 0, label + ': inconsistent zero error')
        exact(obj['argmax_agree'], rows, label + '.zero_error_argmax')
        for k, value in zip(ERROR_COORDS, (offset, 0, offset, 0)):
            exact(obj[k], value, label + '.first_zero_max.' + k)
    else:
        require(bits > 0 and rms > 0 and ratio > 0, label + ': positive error missing diagnostics')


def logical_coverage(end):
    # C++ boundary field is positions 2047..2056 (visibility 2048..2057).
    # The separate requested visibility2047..2056 coverage is derived below;
    # neither calculation is an observed GPU selected-ID/visibility trace.
    mod4 = [sum((p + 1) % 4 == remainder for p in range(end)) for remainder in range(4)]
    boundary = max(0, min(end, 2057) - 2047)
    return mod4, boundary


def layout(index):
    if index == 0:
        return [(0, 2056, True, 'teacher'), (2056, 32, True, 'continuation')]
    prefix, widths = ([], [1024, 1024, 8]) if index == 1 else ([(0, 5, True, 'occupied_prefix')], [997, 997, 57])
    offset = 0 if index == 1 else 5
    result = list(prefix)
    for width in widths:
        result.append((offset, width, False, 'teacher'))
        offset += width
    return result + [(2056, 31, True, 'continuation'), (2087, 1, True, 'continuation_final')]


def visibility_coverage(windows):
    """Compute logical coverage only from the already checked window timeline."""
    masks = []
    for index in range(3):
        mask, mod4, multi = 0, 0, 0
        for w in windows:
            if w['phase_index'] != index:
                continue
            # C++ summaries of many N1 calls are not a multi-query call.
            batched = w['completed_calls'] == 1 and w['rows'] > 1 and bool(index)
            for position in range(w['offset'], w['offset'] + w['completed_rows']):
                visible = position + 1
                if 2047 <= visible <= 2056:
                    bit = 1 << (visible - 2047)
                    mask |= bit
                    mod4 |= 1 << (visible % 4)
                    if batched:
                        multi |= bit
        exact([mask, mod4, multi], [1023, 15, 1023 if index else 0], 'logical_visibility_coverage')
        masks.append({'phase': PHASES[index], 'visible2047_through2056_mask': mask,
                      'visible_mod4_mask': mod4, 'multiquery_logical_visible2047_through2056_mask': multi})
    return {'visible2047_through2056_mask': masks[0]['visible2047_through2056_mask'],
            'visible_mod4_mask': masks[0]['visible_mod4_mask'], 'phases': masks,
            'emitted_position2047_through2056_rows_per_phase': 10,
            'scope': 'computed position+1 from accepted logical rows; not GPU-observed visibility or selected IDs'}


def collect(raw_path):
    """Return a fully validated compact aggregate; no append, model IO or GPU IO."""
    raw_path = Path(raw_path).resolve()
    rows = records(raw_path)
    version = protocol(rows[0])
    require(all(protocol(row) == version for row in rows), "mixed protocol versions")
    for i, obj in enumerate(rows):
        if obj.get('kind') == 'prefill_attention_window' and obj.get('passed') is False:
            failed_window(obj, i)
        if obj.get('kind') == 'prefill_attention_failure':
            failure(obj, i)
    require(len(rows) == RECORD_COUNT, 'expected exactly 31 completed records')
    origin = rows[0]
    source(origin)
    cursor, snapshots, preserved, loaded = 1, 0, 0, None
    memories, memory_events, windows, phases, rejections = [], [], [], [], []

    def take(kind, phase_index, fixed, variable):
        nonlocal cursor
        require(cursor < len(rows), 'timeline: missing record')
        obj = rows[cursor]
        expect(obj, {**base(kind, cursor, phase_index, version=version), **fixed}, variable, f'{kind}[{cursor}]')
        cursor += 1
        return obj

    def ledger(obj):
        memory(obj['memory'], 'memory', loaded, version)
        exact(obj['memory_snapshot_index'], snapshots - 1, 'memory_snapshot_index')
        memories.append(obj['memory'])

    def memory_event(event, phase_index, enabled):
        nonlocal snapshots
        snapshots += 1
        obj = take('memory', phase_index, {'event': event}, ('state', 'memory_snapshot_index', 'memory'))
        same_state(obj['state'], zero_state(enabled), event + '.state')
        ledger(obj)
        memory_events.append(obj)
        return obj['state']

    memory_event('loaded', None, False)
    loaded = memories[0]
    for index in range(3):
        previous = memory_event('reset', index, bool(index))
        current, teacher_state, calls = [], None, 0
        for position, (offset, n, singles, segment) in enumerate(layout(index)):
            call_count = n if singles else 1
            obj = take('window', index, {'window_index': position, 'offset': offset, 'rows': n,
                       'completed_rows': n, 'completed_calls': call_count, 'segment': segment,
                       'expected_routes': 480 * n},
                       ('route_hits_delta', 'route_misses_delta', 'route_upload_bytes_delta',
                        'state_before', 'state_after', 'errors', 'accepted_rows_expected_visible_mod4',
                        'accepted_rows_logical2047_through2056', 'memory_snapshot_index', 'memory'))
            same_state(obj['state_before'], previous, 'window.state_before')
            deltas = completed_state(obj['state_after'], previous, offset, n, singles, 'window.state_after')
            for key, delta in zip(('route_hits_delta', 'route_misses_delta', 'route_upload_bytes_delta'), deltas):
                exact(obj[key], delta, 'window.' + key)
            errors(obj['errors'], offset, n, bool(index), 'window.errors')
            mod4, boundary = logical_coverage(offset + n)
            exact(obj['accepted_rows_expected_visible_mod4'], mod4, 'window.logical_mod4')
            exact(obj['accepted_rows_logical2047_through2056'], boundary, 'window.logical_boundary')
            snapshots += call_count
            preserved += n * VOCAB
            ledger(obj)
            previous = obj['state_after']
            current.append(obj)
            windows.append(obj)
            calls += call_count
            if offset + n == TEACHER:
                teacher_state = previous
            if index and segment == 'teacher' and offset == (0 if index == 1 else 5):
                # Three toggles have no individual records. Their preservation
                # is driver-asserted; subsequent state and snapshot indices agree.
                snapshots += 3
                preserved += 3 * n * VOCAB
                cases = [('late_negative_id', 1024, -1), ('late_vocab_id', 1024, VOCAB), ('oversized_1025', 1025, 100)]
            elif index and segment == 'continuation':
                cases = [('near_capacity_2_with_1_remaining', 2, 100)]
            else:
                cases = []
            for reason, attempted, token in cases:
                snapshots += 2
                # singles returns only the latest N1 span, never all31 rows.
                active_values = (1 if singles else n) * VOCAB
                preserved += 2 * active_values
                rejection = take('rejection', index, {'reason': reason, 'attempted_rows': attempted,
                                 'last_token_id': token, 'remaining_capacity': CAPACITY - offset - n,
                                 'preserved_full_span_values': active_values, 'rejected': True,
                                 'full_span_bit_preserved': True},
                                 ('state_before', 'state_after', 'memory_snapshot_index', 'memory'))
                same_state(rejection['state_before'], previous, 'rejection.state_before')
                same_state(rejection['state_after'], previous, 'rejection.state_after')
                ledger(rejection)
                rejections.append(rejection)
        summary = take('phase', index, {'accepted_rows': CAPACITY, 'completed_calls': calls, 'windows': len(current),
                       'finite_values': sum(w['errors']['finite'] for w in current),
                       'compared_values': sum(w['errors']['compared'] for w in current),
                       'bit_mismatches': sum(w['errors']['bit_mismatches'] for w in current),
                       'argmax_agree': sum(w['errors']['argmax_agree'] for w in current),
                       'accepted_rows_expected_visible_mod4': [522] * 4,
                       'accepted_rows_logical2047_through2056': 10,
                       'rejections': 4 if index else 0, 'toggle_checks': 3 if index else 0,
                       'toggle_preserved_full_span_rows': (0, 1024, 997)[index],
                       'cache_retained_by_reset': True, 'cold_or_warm_claim': 'unknown'},
                       ('teacher_state', 'final_state'))
        same_state(summary['teacher_state'], teacher_state, 'phase.teacher_state')
        same_state(summary['final_state'], previous, 'phase.final_state')
        phases.append(summary)
    memory_event('final_reset', None, True)
    attention = {key: (max(p['final_state']['attention_stats'][key] for p in phases) if key == 'max_query_rows' else
                       sum(p['final_state']['attention_stats'][key] for p in phases)) for key in ATTENTION_FIELDS}
    exact(attention, dict(zip(ATTENTION_FIELDS, (6180, 49284, 6168, 49272, 12, 8))), 'complete.attention_totals')
    footer = take('complete', None, {
        'records': RECORD_COUNT, 'phases': 3, 'windows': len(windows),
        'completed_calls': sum(p['completed_calls'] for p in phases),
        'accepted_rows': sum(p['accepted_rows'] for p in phases),
        'finite_values': sum(p['finite_values'] for p in phases),
        'compared_values': sum(p['compared_values'] for p in phases),
        'rejections': len(rejections), 'toggle_checks': sum(p['toggle_checks'] for p in phases),
        'accepted_rows_logical2047_through2056': 30, 'attention_stats': attention,
        'memory_snapshots': snapshots, 'preserved_values': preserved,
        'session_constructed': True, 'all_owners_unwound': True, 'reference': 'same-session old N1',
        'physical_kernel_count_claim': False, 'selected_id_trace_claim': False,
        'hf_oracle_claim': False, 'performance_qualification': False,
    }, ('minimum_free_vram',))
    exact(cursor, RECORD_COUNT, 'complete.actual_records')
    require(type(footer['minimum_free_vram']) is list and len(footer['minimum_free_vram']) == 2,
            'complete: expected both free VRAM minima')
    for i, value in enumerate(footer['minimum_free_vram']):
        # 26 ledgers are serialized; the other2164 observations can lower this
        # floor. Equality to the serialized minimum would invent observations.
        integer(value, f'complete.minimum_free_vram[{i}]', 1, min(m['devices'][i]['free_vram'] for m in memories))
    paired = [w['errors'] for w in windows if w['errors']['compared']]
    backing = loaded['devices'][0]['attention']  # Both owners' capacities were validated equal.
    private = sum(backing[k] for k in ('gathered_key_bytes', 'gathered_value_bytes',
                                      'partial_output_bytes', 'partial_max_sum_bytes'))
    prefix = backing['private_workspace_bytes'] - private
    selection = sum(backing[k] for k in ('selected_id_bytes', 'selected_block_bytes', 'selected_count_bytes'))
    return {
        'kind': 'r4_attention_prefill', 'protocol': version,
        **({'head_logit_byte_proof': {'rows': 128, 'bytes_per_device': 128 * VOCAB * 4, 'included_in_workspace': True}} if version == 2 else {}),
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'revision': origin['revision'], 'dirty': bool(origin['dirty']), 'runtime': 'own_48_layer_HIP',
        'model_variant': 'qwen38-keep1-Q4_0', 'model': origin['model'], 'model_bytes': origin['model_bytes'],
        'raw_logs': {'prefill_attention': str(raw_path)},
        'scope': 'one Session old-N1 retained full-vocabulary self-parity versus enabled B8 attention; correctness only',
        'source': origin, 'phases': phases, 'memory': memory_events, 'rejections': rejections, 'complete': footer,
        'window_summary': {'windows': len(windows), 'completed_calls': footer['completed_calls'],
                           'accepted_rows': footer['accepted_rows'], 'finite_values': footer['finite_values'],
                           'compared_values': footer['compared_values'], 'violations': 0,
                           'bit_mismatches_diagnostic': sum(e['bit_mismatches'] for e in paired),
                           'argmax_agree_diagnostic': sum(e['argmax_agree'] for e in paired),
                           'maxabs': max(e['maxabs'] for e in paired),
                           'maxboundratio': max(e['maxboundratio'] for e in paired),
                           'rms': math.hypot(*(e['rms'] * math.sqrt(e['finite_pairs']) for e in paired)) /
                                  math.sqrt(footer['compared_values']),
                           'bit_identity_required': False, 'argmax_identity_required': False,
                           'routes': {k: sum(p['final_state']['stats'][k] for p in phases) for k in COUNTERS}},
        'attention_counter_scope': 'successfully completed batch API invocations since reset; not physical kernel counts',
        'logical_coverage': visibility_coverage(windows),
        'attention_byte_proof': {'private_workspace_bytes_per_query': backing['private_workspace_bytes'] // TILE,
                                 'private_workspace_bytes_per_device': backing['private_workspace_bytes'],
                                 'private_buffer_allocations_bytes_per_device': private,
                                 'staged_reused_prefix_bytes_per_device': prefix,
                                 'staged_actual_backing_bytes_per_device': backing['staged_buffer_bytes'],
                                 'output_actual_backing_bytes_per_device': backing['output_buffer_bytes'],
                                 'selection_bytes_per_device': selection,
                                 'scope': 'reported actual backing capacities already in workspace/owned bytes; private prefix is not an extra allocation'},
        'memory_evidence_limits': {
            'in_process_observations': snapshots, 'serialized_ledgers': len(memories),
            'not_serialized': ['per_N1_call_state_and_memory', 'individual_toggle_states_and_span_bits',
                               'guard_values_and_fixture_vector_capacities', 'full_logit_values',
                               'actual_model_metadata_tensor_dimensions_offsets_and_types', 'device_properties_BDF',
                               'hybrid_policy_and_full_disabled_stats_probe_diagnostics',
                               'per_query_visibility_and_selected_IDs', 'individual_allocation_release',
                               'post_destruction_owned_bytes_and_free_VRAM', 'RSS_allocator_overhead_and_stacks'],
            'driver_assertions_only': 'per-call immediate inspect/guards/preservation; individual toggle preservation; full disabled policy/probe checks; unreported memory observations',
            'ledger_scope': 'Session category traversal verified against independent live Buffer byte/count counters; HIP context/rocBLAS only visible in total/free VRAM',
            'read_scope': '144 constructor logical payload loads/68262297600 bytes, steady reported counters; not physical SSD/syscall trace',
            'cleanup_scope': 'completed RAII owner scopes; no measured individual or aggregate release',
            'model_scope': 'declared keep1-Q4_0 filename and file size; no serialized GGUF metadata/tensor attestation',
            'dirty_scope': 'source CORE_DIRTY integer0/1 preserved; aggregate converts to boolean',
            'min_free_scope': 'in-process floor bounded by serialized samples; not exact instantaneous peak VRAM',
        },
        'fixture_contract': {'source_verified_not_additional_observations': True, 'session_instances': 1,
                             'sampling': 'teacher_forced', 'kv': 'Q4_0_K_and_V',
                             'weights': 'unchanged_loaded_GGUF_values_and_tensor_types',
                             'timing': 'last_completed_ms is last completed API correctness wall time; no N1-group total emitted'},
        'record_count': RECORD_COUNT, 'performance_claim': False, 'speedup_claim': False,
        'independent_hf_claim': False, 'R4_complete_claim': False, 'long4k16k_qualified': False,
        'physical_128_column_tile_proven': False, 'physical_kernel_count_claim': False,
        'GPU_selected_ID_trace_qualified': False, 'full_ram_accounting_qualified': False,
        'occupied_128k_qualified': False, 'exact_peak_vram_claim': False,
        'owned_buffer_release_measured': False, 'cold_cache_equality_claim': False, 'mtp': False, 'passed': True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True, help='explicit canonical ROOT/results.jsonl')
    args = parser.parse_args(argv)
    try:
        append_record(args.results, collect(args.raw))
    except (ValueError, OSError, OverflowError, RecursionError) as error:
        print('record_prefill_attention: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
