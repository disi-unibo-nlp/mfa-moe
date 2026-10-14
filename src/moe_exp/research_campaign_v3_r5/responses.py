"""Registered, comparison-specific continuation measurements; no new draws."""
from .contracts import CLASSES, require


def semantic_response(row, result, context, annotations, cell, *,
                      analysis_pulse_tokens=None, analysis_windows=None):
    result = result or {}
    row = row or {}
    complete = result.get('status') in ('complete', 'reused')
    generated = int(result.get('generated_tokens') or 0)
    stopped = complete and result.get('finish_reason') == 'stop'
    sentences = []
    for s in (context or {}).get('sentences', []):
        label = (annotations.state(s['annotation_uid']) or {}).get('label')
        sentences.append({**s, 'label':label if label in CLASSES else None})
    sentences.sort(key=lambda s:(s['token_start'], s['sentence_index']))
    first = next((s for s in sentences if s['first_complete']), None)
    label = first['label'] if first and complete else None
    terminated = first is None and stopped
    values = {'first_class':label, 'first_sentence_status':
              'labelled' if label else 'unknown' if first else 'termination_or_no_complete_sentence',
              'first_sentence_tokens':first['n_tokens'] if first else None,
              'termination':result.get('finish_reason'),
              'first_termination':float(terminated) if complete else None,
              'first_unknown':float(label is None and not terminated) if complete else None}
    for c in CLASSES:
        values['first:'+c] = float(label == c) if label else 0. if terminated else None
        values['first_lower:'+c] = float(label == c) if label else 0.
        values['first_upper:'+c] = float(label == c) if label else 0. if terminated else 1.
    windows = analysis_windows if analysis_windows is not None else cell.get('windows')
    windows = list(windows or ([row['window']] if row.get('window') else []))
    pulse = analysis_pulse_tokens
    if pulse is None: pulse = cell.get('analysis_pulse_tokens')
    if pulse is None: pulse = (row.get('action') or {}).get('pulse_tokens', 0)
    pulse = int(pulse)
    require(pulse >= 0 and all(w >= 0 for w in windows), 'invalid analysis horizon')
    if windows and row.get('window'):
        require(max(windows) <= row['window'], 'analysis exceeds saved capture horizon')
    values.update(analysis_pulse_tokens=pulse, analysis_windows=windows)

    def overlap(s, start, end):
        return max(0, min(s['token_end'], end, generated)-max(s['token_start'], start))

    for i, window in enumerate(windows):
        name = 'w'+str(i)
        available = min(window, generated)
        labelled = sum(overlap(s, 0, window) for s in sentences if s['label'])
        require(labelled <= available, 'overlapping annotated token ownership')
        values['coverage:'+name] = labelled/available if complete and available else None
        for c in CLASSES:
            n = sum(overlap(s, 0, window) for s in sentences if s['label'] == c)
            values[f'occupancy:{name}:{c}'] = n/labelled if complete and labelled else None
            empty = complete and stopped and available == 0
            values[f'occupancy_lower:{name}:{c}'] = n/available if complete and available else 0.
            values[f'occupancy_upper:{name}:{c}'] = (n+available-labelled)/available if complete and available else 0. if empty else 1.
        exposure = max(0, available-pulse)
        labelled_post = sum(overlap(s, pulse, window) for s in sentences if s['label'])
        require(labelled_post <= exposure, 'overlapping post-pulse token ownership')
        empty = stopped and exposure == 0
        status = ('unavailable' if not complete else 'natural_termination_before_exposure' if empty
                  else 'truncated_before_exposure' if not exposure else
                  'observed' if labelled_post == exposure else 'partial_annotation')
        values.update({f'recovery_status:{name}':status,
                       f'recovery_available_tokens:{name}':exposure if complete else None,
                       f'recovery_labelled_tokens:{name}':labelled_post if complete else None,
                       f'recovery_coverage:{name}':labelled_post/exposure if complete and exposure else None,
                       f'window_reached:{name}':bool(complete and generated >= window),
                       f'termination_before_window:{name}':bool(stopped and generated < window),
                       f'window_right_censored:{name}':not complete or (generated < window and not stopped)})
        for c in CLASSES:
            n = sum(overlap(s, pulse, window) for s in sentences if s['label'] == c)
            # Conditional descriptive occupancy is explicitly separate from
            # primary bounds over ALL emitted post-pulse tokens.
            values[f'recovery:{name}:{c}'] = n/labelled_post if complete and labelled_post else 0. if empty else None
            values[f'recovery_lower:{name}:{c}'] = n/exposure if complete and exposure else 0.
            values[f'recovery_upper:{name}:{c}'] = (n+exposure-labelled_post)/exposure if complete and exposure else 0. if empty else 1.

    end = min(max(windows, default=0), generated)
    clipped = [{**s, 'n_tokens':overlap(s, 0, end)} for s in sentences if overlap(s, 0, end)>0]
    adjacent = []
    seen = set(); run = 0; longest = 0; reentries = 0; previous = None
    for s in clipped:
        contiguous = bool(previous and previous['label'] and s['label'] and
            s['sentence_index'] == previous['sentence_index']+1 and
            s['segment'] == previous['segment'] and s['token_start'] == previous['token_end'])
        if not contiguous: run = 0; seen = set()
        if s['label']:
            if contiguous:
                adjacent.append((previous, s))
                if previous['label'] != s['label'] and s['label'] in seen: reentries += 1
            seen.add(s['label'])
            run = run+s['n_tokens'] if contiguous and previous['label'] == s['label'] else s['n_tokens']
            longest = max(longest, run)
        previous = s
    target = cell.get('target')
    endpoint = max(windows, default=0)
    last = next((s for s in sentences if s['token_start'] < endpoint <= s['token_end']), None)
    post = next((s for s in clipped if s['token_end'] > pulse), None)
    target_run=target_longest=target_reentries=target_transitions=target_exits=0
    previous=None; seen_target=False; target_gaps=0
    post_clipped=[{**s,'n_tokens':overlap(s,pulse,end)} for s in sentences if overlap(s,pulse,end)>0]
    for s in post_clipped:
        contiguous=bool(previous and previous['label'] and s['label'] and
            s['sentence_index']==previous['sentence_index']+1 and s['segment']==previous['segment'] and
            s['token_start']==previous['token_end'])
        if not contiguous:
            target_run=0;seen_target=False
            if previous:target_gaps+=1
        if s['label']==target:
            if contiguous and previous['label']!=target:
                target_transitions+=1
                if seen_target:target_reentries+=1
            target_run=target_run+s['n_tokens'] if contiguous and previous['label']==target else s['n_tokens']
            target_longest=max(target_longest,target_run);seen_target=True
        elif contiguous and previous['label']==target:
            target_transitions+=1;target_exits+=1;target_run=0
        previous=s
    post_known=complete and any(s['label'] for s in post_clipped)
    values.update(restricted_target_dwell_tokens=target_longest if post_known else 0. if stopped and generated<=pulse else None,
        target_reentry_count=target_reentries if post_known else None,
        target_transition_count=target_transitions if post_known else None,
        postpulse_trajectory_gaps=target_gaps,
        first_postpulse_matches_source=float(post['label']==(cell.get('prefix') or {}).get('source_class'))
            if complete and post and post['label'] else None)
    values.update(class_switches=sum(a['label'] != b['label'] for a,b in adjacent) if complete and adjacent else None,
        observed_boundaries=len(adjacent), reentry_count=reentries if complete and adjacent else None,
        longest_observed_dwell_tokens=longest if complete and any(s['label'] for s in clipped) else None,
        dwell_left_censored=bool(clipped and clipped[0]['token_start'] <= 0),
        dwell_right_censored=bool(clipped and (clipped[-1]['token_end'] >= end or not stopped)),
        target_at_window_end=float(last['label'] == target) if complete and generated >= endpoint and last and last['label'] else None,
        first_labelled_postpulse_matches_source=values['first_postpulse_matches_source'],
        observed_target_exits=target_exits if post_known else 0. if stopped and generated<=pulse else None)
    return values, sentences
