"""Static import check: every module the live entry points reach resolves (stdlib only, nothing is imported).

For each module under src/moe_exp and each test, every `import` and `from ... import` that names a repo module
must name a module that exists, and every imported name must be defined there or be a submodule. So must every
attribute read through a module alias (`driver.binding()` after `from . import pulse as driver`). Strings that name
a `moe_exp` module (worker extensions, `python -m` commands, mock patches) are checked the same way. The modules
reachable from ENTRY_POINTS must also cover all of src/moe_exp, so dead code cannot linger.
"""
import ast
from pathlib import Path
import re
import unittest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'

# Modules started by `python -m`, by a job script, by a vLLM engine string or loaded by file path.
ENTRY_POINTS = [
    'moe_exp.steering.pulse', 'moe_exp.steering.pulse_outcomes', 'moe_exp.steering.pulse_effects', 'moe_exp.steering.whole',
    'moe_exp.steering.whole_effects', 'moe_exp.steering.whole_complement', 'moe_exp.steering.whole_relink',
    'moe_exp.routing.efficiency_link', 'moe_exp.routing.lexical', 'moe_exp.routing.lfm_readout',
    'moe_exp.steering.stream', 'moe_exp.steering.observer', 'moe_exp.steering.fast_penalties', 'moe_exp.steering.penalties',
    'moe_exp.labeling.renderer_child',
    'moe_exp.steering.outcome.__main__',
    'moe_exp.generation.generate', 'moe_exp.labeling.sample_cohorts',
    'moe_exp.labeling.annotation_batch', 'moe_exp.generation.answer_equivalence',
    'moe_exp.generation.model_profiles', 'moe_exp.labeling.gepa.run',
    'moe_exp.capture.campaign_v3_capture', 'moe_exp.capture.capture_gate', 'moe_exp.capture.vllm_capture',
    'moe_exp.routing.readout', 'moe_exp.routing.anticipation', 'moe_exp.routing.next_state', 'moe_exp.routing.selection',
    'moe_exp.routing.dynamics', 'moe_exp.routing.benchmark_accuracy', 'moe_exp.routing.benchmark_inventory',
    'moe_exp.routing.efficiency_link_prereg',
    'moe_exp.routing.lexical_data.build_native', 'moe_exp.routing.lexical_data.build_extras',
    'moe_exp.routing.lexical_data.build_steered', 'moe_exp.routing.lexical_data.make_readme',
    'moe_exp.routing.lexical_data.validate', 'moe_exp.routing.lexical_data.lfm_table',
    'moe_exp.paper.build_cl_figures', 'moe_exp.paper.build_cl_manuscript', 'moe_exp.paper.cl_paths', 'moe_exp.paper.export_cl_exhibits',
    'moe_exp.paper.export_routing_results', 'moe_exp.paper.export_whole_results', 'moe_exp.paper.format_cl_sentences', 'moe_exp.paper.optimize_sections', 'moe_exp.paper.package_cl_manuscript',
    'moe_exp.paper.prepare_cl_review', 'moe_exp.paper.verify_cl_author_style', 'moe_exp.paper.verify_cl_bibliography_offline', 'moe_exp.paper.verify_cl_figure_labels', 'moe_exp.paper.verify_cl_tables',
]

# (importer, imported) pairs that are known not to resolve.
EXEMPT = {
    # Byte-frozen by the efficiency-link pre-registration hash; routing.efficiency_link aliases the module at load.
    ('moe_exp.routing.efficiency_link_prereg', 'moe_exp.final_routing.common'),
    ('moe_exp.routing.efficiency_link', 'moe_exp.final_routing.common'),
}

DOTTED = re.compile(r'\bmoe_exp(?:\.[A-Za-z_][A-Za-z0-9_]*)+')


def module_table():
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
DEFINED = {}


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
    base = package.split('.')
    base = base[:len(base) - (node.level - 1)]
    return '.'.join(base + ([node.module] if node.module else []))


def references(importer, tree):
    """Yield (module, name or None) for every repo import and every moe_exp string in a module."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant):
            docstrings.add(id(node.body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] == 'moe_exp':
                    yield alias.name, None
        elif isinstance(node, ast.ImportFrom):
            full = resolve_from(importer, node)
            if full.split('.')[0] == 'moe_exp':
                for alias in node.names:
                    yield full, alias.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            for match in DOTTED.findall(node.value):
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
                    if alias.name.split('.')[0] == 'moe_exp':
                        yield alias.name, None
            elif isinstance(node, ast.ImportFrom) and not node.level and (node.module or '').split('.')[0] == 'moe_exp':
                for alias in node.names:
                    yield node.module, alias.name
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                for match in DOTTED.findall(node.value):
                    yield match, 'string'

    def test_entry_points_cover_the_package(self):
        for entry in ENTRY_POINTS:
            self.assertTrue(entry in MODULES, entry)
        unreachable = sorted(set(MODULES) - reachable(ENTRY_POINTS))
        self.assertEqual([], unreachable)


if __name__ == '__main__':
    unittest.main()
