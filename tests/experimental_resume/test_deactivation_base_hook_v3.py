"""Bounded CPU checks for the separately sealed negative-routing fallback."""

from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/experimental_resume'))

import run_eligible_deactivation_serial_v3 as screen  # noqa: E402
from moe_steer.ops import DeviceTables, steer_logits  # noqa: E402
from moe_steer.trigger import THINK_END_ID, TriggerFSM  # noqa: E402


def test_negative_base_kernel_and_inactive_neighbor():
    manifest = screen.base.sealed(screen.MANIFEST)
    table = screen.build_policy_table(manifest['actions'])
    screen.validate_negative_compilation(table, manifest['actions'])
    compiled = table.compile()
    layer = 28
    h = list(compiled['layers']).index(layer)
    targets = [9, 189]
    logits = torch.zeros((2, table.n_experts), dtype=torch.float32)
    logits[:, targets] = torch.tensor([10.0, 9.0])
    active = torch.tensor([True, False])
    tables = DeviceTables.from_compiled(compiled, 'cpu', max_tokens=2)

    force = table.index_of('verify_target_force_off')
    actual = steer_logits(logits, active, torch.tensor([force, force]), h, tables)
    assert all(actual[0, expert] < 0 for expert in targets)
    assert torch.equal(actual[1], logits[1])

    bias = table.index_of('verify_target_bias_minus1')
    actual = steer_logits(logits, active, torch.tensor([bias, bias]), h, tables)
    assert all(actual[0, expert] == logits[0, expert] - 1 for expert in targets)
    assert torch.equal(actual[1], logits[1])


def test_always_schedule_starts_at_branch_and_stops_after_closure():
    fsm = TriggerFSM({'kind': 'always'}, None, None, 'negative-base-cpu-check')
    assert fsm.fast_forward([11, 12]) == 2
    assert fsm.is_active(2)
    fsm.feed(2, 13)
    assert fsm.is_active(3)
    fsm.feed(3, THINK_END_ID)
    assert not fsm.is_active(4)
    assert fsm.think_end == 3
