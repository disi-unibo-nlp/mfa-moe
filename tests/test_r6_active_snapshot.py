import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('active_paper_snapshot', Path(__file__).resolve().parents[1] / 'scripts/paper/build_campaign_paper_r4.py')
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)


class ActiveSnapshotContracts(unittest.TestCase):
    def test_local_bst_and_active_graphics_included_review_excluded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            (root / 'main.tex').write_text('\\documentclass{local}\\input{section}\\bibliographystyle{local}\\bibliography{refs}')
            (root / 'local.cls').write_text('class')
            (root / 'local.bst').write_text('bibliography style')
            (root / 'refs.bib').write_text('references')
            (root / 'section.tex').write_text('\\includegraphics{chart}')
            (root / 'chart.pdf').write_bytes(b'fixture')
            (root / '.paper-review').mkdir()
            (root / '.paper-review/unrelated.tex').write_text('ignored output')
            paths = {p.relative_to(root).as_posix() for p in builder.active_sources(root)}
            self.assertEqual(paths, {'main.tex','section.tex','local.cls','local.bst','refs.bib','chart.pdf'})
            before = builder.source_state(root)
            (root / '.paper-review/unrelated.tex').write_text('another review iteration')
            self.assertEqual(before, builder.source_state(root))
            (root / 'section.tex').write_text('changed active source')
            self.assertNotEqual(before, builder.source_state(root))

    def test_missing_or_outside_active_source_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            (root / 'main.tex').write_text('\\input{missing}')
            with self.assertRaises(ValueError): builder.active_sources(root)

    def test_macro_icons_are_bound(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary); (root / 'icons').mkdir()
            (root / 'main.tex').write_text('\\includegraphics{icons/#1-icon.png}')
            (root / 'icons/model-icon.png').write_bytes(b'icon')
            self.assertIn(root / 'icons/model-icon.png', builder.active_sources(root))


if __name__ == '__main__': unittest.main()
