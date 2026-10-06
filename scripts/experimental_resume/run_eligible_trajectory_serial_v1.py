"""Fail-closed placeholder for the future 1,024-token trajectory driver.

The previous file at this path was a byte-identical copy of the 256-token
``run_eligible_micro_serial_v3.py`` driver.  Running it would have re-created
the already-queued positive pilot under a different version label, so it was
renamed to ``run_eligible_trajectory_serial_v1.py.superseded-v3-copy`` and
replaced by this explicit stop until the real trajectory stage is sealed.

Contract the eventual driver must satisfy (from the dense-trajectory audit):

- schema ``routing-eligible-trajectory-serial-v1``, ``max_tokens == 1024``;
- every intervention arm carries a sealed template
  ``{'action_policy_names': [first, second], 'slots': [0, 512], 'horizon': 1024,
  'sha256': ...}``;
- the request policy is ``first`` and the ordered metadata is
  ``[first, second]``; the reversed arm uses the same multiset with
  ``[second, first]`` and policy ``second``;
- both actions are positive .5/1 ``always`` policies, native top-k stays eight
  and the shared expert is unchanged;
- ``maximum_decode_tokens == requests * 1024`` and
  ``maximum_context_tokens == max(prompt + prefix + 1024)``;
- serial/eager execution, per-batch receipts, output directory bound to the
  manifest digest, and same-manifest resume without duplicate or skipped UIDs.

Two inputs do not exist yet and must be frozen before this driver can be
implemented or run: a family-disjoint mechanism-family start pool (the current
exact pool contains only the 13 discovery families) and a sealed template
selection record produced from the discovery readout.  See
``RESUME_STATE_2026-10-02.md`` and ``DENSE_TRAJECTORY_ANALYSIS_INPUT_v1.md``.
"""


def main():
    raise SystemExit(
        'run_eligible_trajectory_serial_v1.py is intentionally fail-closed: '
        'the 1,024-token trajectory manifest, disjoint mechanism pool and '
        'sealed template selection do not exist yet. Do not substitute the '
        '256-token positive-v3 driver or resubmit its queued workload.')


if __name__ == '__main__':
    main()
