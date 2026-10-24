"""Static import check: every module the live entry points reach resolves (stdlib only, nothing is imported).

For each module under src/ and each test, every `import` and `from ... import` that names a repo module (one whose
first component is a top-level folder or module of src/) must name a module that exists, and every imported name must
be defined there or be a submodule. So must every attribute read through a module alias (`driver.binding()` after
`from . import pulse as driver`). Strings that name a repo module (worker extensions, `python -m` commands, mock
patches) are checked the same way. The name `moe_exp` is not a package of src/ but counts as a repo name, so a
reference to it fails unless EXEMPT lists it. The modules reachable from ENTRY_POINTS must also cover all of
src/, so dead code cannot linger.
"""
import ast
from pathlib import Path
import re
import unittest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'

# Modules started by `python -m`, by a job script, by a vLLM engine string or loaded by file path.
ENTRY_POINTS = [
    'steering.pulsed_steering', 'steering.pulsed_outcomes', 'steering.pulsed_effects', 'steering.lfm25_generation',
    'steering.whole_reasoning', 'steering.whole_reasoning_effects', 'steering.release', 'steering.plan_continuation',
    'routing.efficiency_link', 'routing.lexical', 'routing.lfm',
    'steering.stream', 'steering.observer', 'steering.fast_penalties', 'steering.penalties',
    'labeling.renderer_child',
    'steering.outcome.__main__',
    'generation.generate', 'generation.serve_flags', 'models.roster', 'labeling.sample_cohorts',
    'labeling.annotation_batch', 'generation.answer_equivalence',
    'labeling.gepa.run',
    'capture.run', 'capture.gate', 'capture.vllm_capture',
    'routing.readout', 'routing.anticipation', 'routing.next_state', 'routing.selection',
    'routing.dynamics', 'routing.sentence_table', 'routing.transition_pairs',
    'routing.benchmark_accuracy', 'routing.benchmark_inventory',
    'routing.efficiency_link_prereg',
    'routing.lexical_data.build_native', 'routing.lexical_data.build_extras',
    'routing.lexical_data.build_steered', 'routing.lexical_data.make_readme',
    'routing.lexical_data.validate', 'routing.lexical_data.lfm_table',
    'paper.build_cl_figures', 'paper.build_cl_manuscript', 'paper.cl_paths', 'paper.export_cl_exhibits',
    'paper.export_routing_results', 'paper.export_whole_results', 'paper.format_cl_sentences', 'paper.optimize_sections', 'paper.package_cl_manuscript',
    'paper.prepare_cl_review', 'paper.verify_cl_author_style', 'paper.verify_cl_bibliography_offline', 'paper.verify_cl_figure_labels', 'paper.verify_cl_printed_tables', 'paper.verify_cl_tables',
]

# (importer, imported) pairs that are known not to resolve.
EXEMPT = {
    # The efficiency-link pre-registration script is byte-frozen by its hash and imports moe_exp.final_routing.common;
    # routing.efficiency_link registers the dotted names (moe_exp, moe_exp.final_routing,
    # moe_exp.final_routing.common) for it at load.
    ('routing.efficiency_link_prereg', 'moe_exp.final_routing.common'),
    ('routing.efficiency_link', 'moe_exp.final_routing'),
    ('routing.efficiency_link', 'moe_exp.final_routing.common'),
    # The stored lexical fold pickles name the text transform by the module moe_exp.study.readout_text;
    # routing.lexical maps it (and a test).
    ('routing.lexical', 'moe_exp.study.readout_text'),
    ('tests', 'moe_exp.study.readout_text'),
}

UNRESOLVED_PACKAGE = 'moe_exp'


def module_table():
    """Dotted name -> file of every module under src/ (src/ itself is not a package)."""
    table = {}
    for path in sorted(SRC.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        parts = list(path.relative_to(SRC).parts)
        if parts[-1] == '__init__.py':
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        table['.'.join(parts)] = path
    return table


MODULES = module_table()
REPO_NAMES = {module.split('.')[0] for module in MODULES} | {UNRESOLVED_PACKAGE}
# A string names a module when it starts with a repo name (`steering.stream.StreamWorkerExtension`,
# `steering.fast_penalties:FastPrefixPenalties`) or when the repo name follows `-m` in a command. File names such as
# `stats.csv` and prose that mentions `receipts.row_index` are not module references.
DOTTED = re.compile(r'(?:^|(?<=-m ))(?:' + '|'.join(sorted(REPO_NAMES)) + r')(?:\.[A-Za-z_][A-Za-z0-9_]*)+')
FILE_SUFFIXES = {'csv', 'dat', 'gz', 'json', 'log', 'md', 'npz', 'parquet', 'pkl', 'py', 'sbatch', 'sh', 'tex', 'txt', 'yaml'}
DEFINED = {}


def module_strings(text):
    """Dotted module paths that a string names."""
    return [match for match in DOTTED.findall(text) if match.rpartition('.')[2] not in FILE_SUFFIXES]


def parse(path):
    return ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))


def defined_names(module):
    """Top-level names of a module; None means 'anything' (star import or module __getattr__)."""
    if module in DEFINED:
        return DEFINED[module]
    names = set()
    anything = False

    def visit(body):
        nonlocal anything
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                anything = anything or node.name == '__getattr__'
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for leaf in ast.walk(target):
                        if isinstance(leaf, ast.Name):
                            names.add(leaf.id)
            elif isinstance(node, ast.Import):
                names.update((a.asname or a.name).split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name == '*':
                        anything = True
                    names.add(a.asname or a.name)
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for field in ('body', 'orelse', 'finalbody'):
                    visit(getattr(node, field, []))
                for handler in getattr(node, 'handlers', []):
                    visit(handler.body)

    visit(parse(MODULES[module]).body)
    DEFINED[module] = None if anything else names
    return DEFINED[module]


def resolve_from(importer, node):
    if not node.level:
        return node.module or ''
    package = importer if MODULES[importer].name == '__init__.py' else importer.rpartition('.')[0]
    base = package.split('.') if package else []
    base = base[:len(base) - (node.level - 1)]
    return '.'.join(base + ([node.module] if node.module else []))


def references(importer, tree):
    """Yield (module, name or None) for every repo import and every repo module string in a module."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant):
            docstrings.add(id(node.body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in REPO_NAMES:
                    yield alias.name, None
        elif isinstance(node, ast.ImportFrom):
            full = resolve_from(importer, node)
            if full.split('.')[0] in REPO_NAMES:
                for alias in node.names:
                    yield full, alias.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            for match in module_strings(node.value):
                yield match, 'string'


def attribute_references(importer, tree):
    """Yield (module, attribute) for every attribute read through a name bound to a repo module."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and alias.name in MODULES:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            full = resolve_from(importer, node)
            for alias in node.names:
                if f'{full}.{alias.name}' in MODULES:
                    aliases[alias.asname or alias.name] = f'{full}.{alias.name}'
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) \
                and isinstance(node.value, ast.Name) and node.value.id in aliases:
            yield aliases[node.value.id], node.attr


def problem(importer, module, name):
    """Why a reference fails, or None."""
    if (importer, module) in EXEMPT:
        return None
    if name == 'string':
        # longest existing module prefix; the next component, if any, must be a name defined in it
        parts = module.split('.')
        for i in range(len(parts), 0, -1):
            head = '.'.join(parts[:i])
            if head in MODULES:
                if i == len(parts):
                    return None
                names = defined_names(head)
                return None if names is None or parts[i] in names else f'{head} defines no {parts[i]}'
        return f'no module {module}'
    if module not in MODULES:
        return f'no module {module}'
    if name is None or f'{module}.{name}' in MODULES:
        return None
    names = defined_names(module)
    return None if names is None or name in names else f'{module} defines no {name}'


def reachable(entries):
    seen, todo = set(), list(entries)
    while todo:
        module = todo.pop()
        if module in seen or module not in MODULES:
            continue
        seen.add(module)
        parts = module.split('.')
        todo += ['.'.join(parts[:i]) for i in range(1, len(parts))]
        for target, name in references(module, parse(MODULES[module])):
            parts = target.split('.')
            while parts and '.'.join(parts) not in MODULES:
                parts.pop()
            if parts:
                todo.append('.'.join(parts))
                if name and name != 'string' and '.'.join(parts + [name]) in MODULES:
                    todo.append('.'.join(parts + [name]))
    return seen


class ImportsResolve(unittest.TestCase):
    def test_every_source_module_resolves(self):
        failures = []
        for module, path in MODULES.items():
            tree = parse(path)
            for target, name in [*references(module, tree), *attribute_references(module, tree)]:
                reason = problem(module, target, name)
                if reason:
                    failures.append(f'{module}: {reason}')
        self.assertEqual([], sorted(set(failures)))

    def test_every_test_resolves(self):
        failures = []
        for path in sorted((REPO / 'tests').rglob('*.py')):
            if '__pycache__' in path.parts or path == Path(__file__).resolve():
                continue
            for target, name in self._test_references(parse(path)):
                reason = problem('tests', target, name)
                if reason:
                    failures.append(f'{path.relative_to(REPO)}: {reason}')
        self.assertEqual([], sorted(set(failures)))

    @staticmethod
    def _test_references(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in REPO_NAMES:
                        yield alias.name, None
            elif isinstance(node, ast.ImportFrom) and not node.level and (node.module or '').split('.')[0] in REPO_NAMES:
                for alias in node.names:
                    yield node.module, alias.name
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                for match in module_strings(node.value):
                    yield match, 'string'

    def test_entry_points_cover_the_package(self):
        for entry in ENTRY_POINTS:
            self.assertTrue(entry in MODULES, entry)
        unreachable = sorted(set(MODULES) - reachable(ENTRY_POINTS))
        self.assertEqual([], unreachable)


if __name__ == '__main__':
    unittest.main()
