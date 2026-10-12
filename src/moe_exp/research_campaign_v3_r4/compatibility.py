"""Scientific AST projection after verification of the complete source seal.

Only the named infrastructure definitions are excluded. Numerical modules,
input reads, execution identity, constants, and serialization remain bound.
"""
import ast
import hashlib
import re
from pathlib import Path
from .contracts import file_sha, require

FULL = ('accuracy.py', 'derive.py', 'design.py', 'numeric.py', 'observations.py',
        'populations.py', 'statistics.py', 'native.py')
SELECT = {
    'inputs.py': ('read_source', 'counts_for'),
    'artifacts.py': ('npz', 'frame'),
    'extraction.py': ('array', 'extract'),
    'contracts.py': ('digest', 'canonical', 'seal', 'execution_id', 'MODELS', 'CLASSES',
                     'BENCHMARKS', 'CONFIGURATIONS', 'METHODS', 'REMOVALS', 'SEED',
                     'REPLICATE', 'EXECUTION_FIELDS'),
}


def projection(text, names=None):
    text = re.sub(r'research_campaign_v3(?:_r[0-9]+)?', 'research_campaign_v3', text)
    tree = ast.parse(text)
    if names is not None:
        nodes = {}
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nodes[n.name] = n
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name): nodes[t.id] = n
        require(set(names) <= set(nodes), 'scientific projection is missing a required definition')
        tree = ast.Module(body=[nodes[n] for n in names], type_ignores=[])
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def analysis_sources(campaign):
    bindings = {}
    for name in (*FULL, *SELECT):
        paths = [Path(p) for p in campaign.get('code_files', {})
                 if '/src/moe_exp/research_campaign_v3' in p and Path(p).name == name]
        if len(paths) != 1: return None
        path = paths[0]
        require(file_sha(path) == campaign['code_files'][str(path)], 'analysis source seal differs')
        bindings[name] = projection(path.read_text(), SELECT.get(name))
    # Imported scientific helpers also remain exact, independent of package name.
    bindings['external_scientific_sources'] = hashlib.sha256(repr(sorted(
        (p, h) for p, h in campaign['code_files'].items()
        if '/src/moe_exp/' in p and '/research_campaign_v3' not in p)).encode()).hexdigest()
    return bindings
