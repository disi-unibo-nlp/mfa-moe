"""Run paper-writer's build command with the existing verified offline compiler.

No installations, scientific computation or agent-runtime changes. Each TeX
invocation has an explicit walltime bound (45 seconds by default). Review artifacts remain
inside paper/manuscript-support/build (ignored).
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import pymupdf

from paper.cl_paths import BUILD, PAPER, REPO  # noqa: E402
SKILL = Path("/leonardo/home/userexternal/lmolfett/.paper-writer-skill/skills/paper-writer")
PYTHON = Path("/leonardo/home/userexternal/lmolfett/.cache/paper-writer/venv/bin/python")
TOOLS = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/verification-paper-tectonic-0.17.0")
EXPECTED = "a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_bounded(snapshot: Path, engine: Path, env: dict, allow_cache_fill: bool,
                    timeout_seconds: int = 45) -> tuple[subprocess.CompletedProcess, list[dict]]:
    """Converge the manuscript's bibliography and references within bounded compiler passes.

    Args:
        snapshot: Self-contained manuscript build directory.
        engine: Verified compiler executable.
        env: Compiler environment with private cache and temporary paths.
        allow_cache_fill: Whether the compiler may fetch uncached TeX resources.
        timeout_seconds: Walltime bound for each compiler process.

    Returns:
        The final process result and evidence for every attempted pass.
    """
    built = snapshot / '.paper-review'
    output = built / 'build'
    output.mkdir(parents=True, exist_ok=True)
    command = [str(engine), '--untrusted', '--keep-logs', '--keep-intermediates',
               '--reruns', '0', '--outdir', str(output), 'main.tex']
    if not allow_cache_fill:
        command.insert(1, '--only-cached')
    passes = []
    for number in range(1, 5):
        for suffix in ('aux', 'bbl', 'out', 'toc'):
            auxiliary = output / ('main.' + suffix)
            if auxiliary.exists():
                shutil.copyfile(auxiliary, snapshot / auxiliary.name)
        started = time.monotonic()
        with (built / f'compiler-pass-{number}.log').open('w') as stream:
            try:
                run = subprocess.run(command, cwd=snapshot, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                run = subprocess.CompletedProcess(command, 124)
        text = (output / 'main.log').read_text(errors='replace') if (output / 'main.log').exists() else ''
        pending = bool(re.search(r'Rerun to get cross-references right|Label\(s\) may have changed|There were undefined (?:references|citations)|Citation .+ undefined', text))
        passes.append({'pass': number, 'exit_code': run.returncode,
                       'elapsed_seconds': time.monotonic() - started, 'rerun_needed': pending})
        print(passes[-1], flush=True)
        if run.returncode or not pending:
            break
    else:
        run = subprocess.CompletedProcess(command, 1)
    if (output / 'main.log').exists():
        shutil.copyfile(output / 'main.log', built / 'main.log')
    return run, passes


def main(allow_cache_fill: bool = False, bounded_passes: bool = False,
         timeout_seconds: int = 45) -> int:
    """Snapshot and build the canonical manuscript without altering its sources.

    Args:
        allow_cache_fill: Whether the existing compiler cache may fetch missing resources.
        bounded_passes: Whether to converge references in at most four separate passes.
        timeout_seconds: Walltime bound for each compiler invocation.

    Returns:
        The compiler exit status; zero also produces current page renders and a build receipt.
    """
    toolchain = json.loads((TOOLS / "TOOLCHAIN.json").read_text())
    engine = Path(toolchain["binary_path"])
    if sha(engine) != EXPECTED or toolchain["binary_sha256"] != EXPECTED:
        raise ValueError("existing compiler does not match its verified receipt")
    review = BUILD
    review.mkdir(parents=True, exist_ok=True)
    (review / ".gitignore").write_text("*\n")
    adapter = review / "offline-toolchain"
    (adapter / "bin").mkdir(parents=True, exist_ok=True)
    (review / "tmp").mkdir(exist_ok=True)
    shim = adapter / "bin/tectonic"
    shim.write_text("#!/bin/sh\nexec timeout " + str(timeout_seconds) + "s " + str(engine) +
                    ' --untrusted ' + ('' if allow_cache_fill else '--only-cached ') + '"$@"\n')
    shim.chmod(0o700)
    # The source folder retains historical AUX/BBL outputs from another class.
    # A fresh dependency snapshot prevents those outputs from entering this build.
    index = len(list(review.glob('source-round-*'))) + 1
    snapshot = review / f'source-round-{index:02d}'
    snapshot.mkdir()
    spec = importlib.util.spec_from_file_location('paper_writer_rules', SKILL / 'scripts/rules.py')
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    origins = rules.active_files(PAPER / 'main.tex')
    origins += [PAPER / 'clv2025.cls', PAPER / 'compling.bst']
    origins += [p for p in (PAPER / 'figures/journal').rglob('*') if p.is_file()]
    origins += list(PAPER.glob('*.md'))
    bindings = {}
    for origin in dict.fromkeys(origins):
        target = snapshot / origin.relative_to(PAPER)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        bindings[str(origin.relative_to(PAPER))] = sha(origin)
    # Paper-writer discovers local rules beside main.tex. Keep the canonical notes outside
    # Overleaf, but include them in the review snapshot so those rules are actually checked.
    for name in ('paper-rules.md', 'writing_brief.md'):
        shutil.copyfile(REPO / 'paper/manuscript-support/notes' / name, snapshot / name)
    # The bibliography is unchanged; retain its existing verification evidence in the snapshot.
    (snapshot / '.paper-review').mkdir()
    shutil.copyfile(review / 'bib-verify.log', snapshot / '.paper-review/bib-verify.log')
    fonts = TOOLS / "FONT_ASSETS.json"
    for name, record in json.loads(fonts.read_text()).get("files", {}).items():
        origin = Path(record["path"])
        if sha(origin) != record["sha256"]:
            raise ValueError("existing math font changed")
        target = snapshot / name
        if target.exists() and sha(target) != record["sha256"]:
            raise ValueError("paper-local math font differs")
        if not target.exists():
            shutil.copyfile(origin, target)
    env = dict(os.environ)
    env.update(PAPER_WRITER_CACHE=str(adapter), XDG_CACHE_HOME=str(TOOLS / "cache"),
               XDG_CONFIG_HOME=str(review / "config"), TMPDIR=str(review / "tmp"),
               TMP=str(review / "tmp"), TEMP=str(review / "tmp"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for key in ("TEXINPUTS", "TEXMFHOME", "TEXMFVAR", "TEXMFCONFIG"):
        env.pop(key, None)
    command = [str(PYTHON), str(SKILL / "scripts/rules.py"), "build", str(snapshot / "main.tex")]
    started = time.monotonic()
    passes = []
    auxiliary_seed = None
    if bounded_passes:
        previous_path = review / 'BUILD.json'
        if previous_path.exists():
            previous = json.loads(previous_path.read_text())
            unchanged = all(previous.get('source_hashes', {}).get(name) == bindings.get(name)
                            for name in ('references.bib', 'clv2025.cls', 'compling.bst'))
            if previous.get('exit_code') == 0 and unchanged:
                old_output = Path(previous['source_snapshot']) / '.paper-review/build'
                auxiliary_seed = {}
                for suffix in ('aux', 'bbl', 'out', 'toc'):
                    source = old_output / ('main.' + suffix)
                    if source.exists():
                        shutil.copyfile(source, snapshot / source.name)
                        auxiliary_seed[str(source)] = sha(source)
        run, passes = compile_bounded(snapshot, engine, env, allow_cache_fill, timeout_seconds)
        command = run.args
    else:
        run = subprocess.run(command, cwd=snapshot, env=env, timeout=timeout_seconds + 15)
    built = snapshot / '.paper-review'
    if run.returncode == 0:
        (review / 'build').mkdir(exist_ok=True)
        for name in ('main.pdf', 'main.log'):
            shutil.copyfile(built / 'build' / name, review / 'build' / name)
        shutil.copyfile(built / 'main.log', review / 'main.log')
        (review / 'pages').mkdir(exist_ok=True)
        for old in (review / 'pages').glob('page-*.png'):
            old.unlink()
        # Render pages independently to make each page proof self-contained.
        # Reviewers must inspect original-detail PNG delivery: the resized image
        # preview can misleadingly crop tables whose PDF bounds are correct.
        with pymupdf.open(review / 'build/main.pdf') as document:
            page_count = len(document)
        fresh_pages = review / f'pages-{index:02d}'
        fresh_pages.mkdir()
        for number in range(page_count):
            with pymupdf.open(review / 'build/main.pdf') as document:
                target = fresh_pages / f'page-{number + 1:02d}.png'
                document[number].get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(target)
                shutil.copyfile(target, review / 'pages' / target.name)
    elif (built / 'main.log').exists():
        shutil.copyfile(built / 'main.log', review / 'main.log')
    pdf = review / "build/main.pdf"
    evidence = {"compiler_sha256": EXPECTED, "command": command,
                "source_snapshot": str(snapshot), "source_hashes": bindings,
                "elapsed_seconds": time.monotonic() - started, "exit_code": run.returncode,
                "compiler_timeout_seconds": timeout_seconds,
                "offline": not allow_cache_fill, "scientific_computation": False,
                "agent_runtime_modified": False}
    if passes:
        evidence['bounded_compile_passes'] = passes
    if auxiliary_seed:
        evidence['generated_auxiliary_seed'] = auxiliary_seed
    if pdf.exists():
        evidence.update(pdf_sha256=sha(pdf), pdf_path=str(pdf))
    if run.returncode == 0:
        evidence.update(page_count=page_count, page_renders=str(fresh_pages),
                        rendering='144 dpi; independently reopened PDF per page')
    changed = [name for name, expected in bindings.items() if sha(PAPER / name) != expected]
    evidence['changed_sources'] = changed
    if changed:
        raise ValueError('manuscript sources changed during the build')
    (review / "BUILD.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return run.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-cache-fill', action='store_true',
                        help='Permit the verified compiler to fetch missing private TeX-cache resources.')
    parser.add_argument('--bounded-passes', action='store_true',
                        help='Converge references in at most four separately bounded compiler passes.')
    parser.add_argument('--timeout-seconds', type=int, default=45,
                        help='Per-compiler walltime bound; does not change host CPU limits.')
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 600:
        parser.error('--timeout-seconds must be between 1 and 600')
    raise SystemExit(main(args.allow_cache_fill, args.bounded_passes, args.timeout_seconds))
