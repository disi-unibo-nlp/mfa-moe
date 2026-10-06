"""Named-argument recovery entry for unchanged, source-bound paper v3.

The two original attachment attempts inverted the positional stage/phase args.
This adapter translates explicit named args and leaves all v3 children, analysis,
figures, source paths and scientific interpretation unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

FROZEN = {
    'routing_paper_v3.py': '57496a379a2ed9094749bc684fa8e679db306f7c7ffae8249ba75badc54840fe',
    'routing_paper_v3.sbatch': 'e2ca1dd722dfc10a7af59aa00aa6b6a94c0a48319e895fdfb6d7674b11d17583',
}


def validate_sources():
    for name, expected in FROZEN.items():
        path = Path(__file__).with_name(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('frozen paper source changed: ' + name)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('C', 'FRESH'), required=True)
    parser.add_argument('--phase', choices=('attach', 'dispatch', 'build'), required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    validate_sources()
    import routing_paper_v3 as original
    normalized = [str(Path(original.__file__).resolve()), args.stage, args.phase]
    print(json.dumps({'status': 'PAPER_V4_NAMED_ARGUMENTS_VALIDATED',
                      'stage': args.stage, 'phase': args.phase,
                      'frozen_sources': FROZEN, 'scientific_driver_changed': False}), flush=True)
    previous = sys.argv
    try:
        sys.argv = normalized
        original.main()
    finally:
        sys.argv = previous


if __name__ == '__main__': main()
