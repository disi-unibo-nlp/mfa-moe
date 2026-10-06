"""Fail-closed launch gate for the 96-family original-prompt utility scout.

The frozen adaptive policy and its qualified generator do not yet exist. This
entrypoint validates their future binding and complete-stage price, then stops
explicitly until the implementation is supplied in a versioned successor.
It cannot silently run a native-only or unqualified proxy for the policy arm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import utility_scout_v1 as U


def preflight(plan_path: Path, policy_path: Path, qualification_path: Path,
              execution_path: Path, price_path: Path) -> dict:
    plan = U.sealed(plan_path)
    U.validate_plan(plan)
    policy = U.sealed(policy_path)
    qualification = U.sealed(qualification_path)
    expected_execution = U.bind_execution(plan, policy, qualification)
    execution = U.sealed(execution_path)
    if execution != expected_execution:
        raise ValueError('utility execution binding differs from exact frozen inputs')
    price = U.sealed(price_path)
    if (price.get('schema') != 'routing-utility-scout-price-v1' or
            price.get('plan_sha256') != plan['sha256'] or
            price.get('execution_sha256') != execution['sha256'] or
            price.get('status') != 'PASS_COMPLETE_STAGE'):
        raise ValueError('complete measured utility price has not passed')
    return {'plan_sha256': plan['sha256'], 'execution_sha256': execution['sha256'],
            'price_sha256': price['sha256'], 'assigned': plan['assignment_count']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'policy', 'qualification', 'execution', 'price'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    preflight(args.plan, args.policy, args.qualification, args.execution, args.price)
    raise SystemExit('utility scout v1 remains fail-closed: versioned adaptive '
                     'controller execution and GPU replay qualification are pending')


if __name__ == '__main__':
    main()
