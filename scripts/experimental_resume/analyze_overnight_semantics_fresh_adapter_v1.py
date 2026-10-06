"""Guard recovery provenance, then run unchanged frozen v2 semantic statistics."""
import sys
from pathlib import Path
from unittest.mock import patch

import analyze_overnight_semantics_v2 as frozen
import fresh_frame_recovery_v1 as recovery


def main():
    recovery.require('--manifest' in sys.argv, 'explicit recovery manifest required')
    m = recovery.base.sealed(Path(sys.argv[sys.argv.index('--manifest') + 1]))
    a = recovery.base.sealed(recovery.AMENDMENT)
    paths = a['paths']
    p = Path(paths['measurement'])
    fields = recovery.validate_measurement(m, *(recovery.base.sealed(p / name) for name in
                    ('ARM_MAP.json', 'BLIND_FRAME.json', 'READER_PRICE.json')), a)
    for option, expected in (('--arm-map', p / 'ARM_MAP.json'), ('--frame', p / 'BLIND_FRAME.json'),
                             ('--price', p / 'READER_PRICE.json'), ('--ratings-out', paths['ratings']),
                             ('--out', paths['analysis'])):
        recovery.require(option in sys.argv and
                         Path(sys.argv[sys.argv.index(option) + 1]).resolve() == Path(expected).resolve(),
                         'recovery analysis CLI path differs')
    previous_save = frozen.rating.save

    def save(path, body, **kwargs):
        if Path(path).name in ('ANALYSIS.json', 'ASSIGNED_RESULTS.json'):
            body = {**body, 'recovery_provenance': fields,
                    'analysis_operational_entry_sha256': recovery.base.file_sha(__file__)}
        return previous_save(path, body, **kwargs)

    with patch.object(frozen.rating, 'save', save):
        frozen.main()


if __name__ == '__main__':
    main()
