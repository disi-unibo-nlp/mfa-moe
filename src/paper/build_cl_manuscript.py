"""Build the manuscript PDF with the verified offline tectonic, in passes that each have a time limit.

Copies every source file into a fresh folder BUILD/source-round-NN, compiles it in up to four passes until the
references stop changing, and writes BUILD/build/main.pdf, one image per page (BUILD/pages) and the receipt
BUILD/BUILD.json. The manuscript folder is only read, and the build stops if a source changes while it runs.

  python -m paper.build_cl_manuscript [--timeout-seconds 180]
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time

import pymupdf

from paper.cl_paths import BUILD, NOTES, PAPER, PAPER_WRITER, TECTONIC, TECTONIC_SHA256
from store import file_sha

PASS_SECONDS = 180  # time limit of one compiler pass; the first pass of the full manuscript takes about 175 s
UNSETTLED = re.compile(r"Rerun to get cross-references right|Label\(s\) may have changed|"
                       r"There were undefined (?:references|citations)|Citation .+ undefined")


def tectonic():
    """The compiler, once its hash matches the verified one."""
    engine = TECTONIC / "tectonic"
    if file_sha(engine) != TECTONIC_SHA256:
        raise ValueError("verified compiler changed")
    return engine


def compile_env(tmp):
    """Environment of a compiler run: private cache, config and temporary folders, one thread."""
    env = dict(os.environ)
    env.update(XDG_CACHE_HOME=str(TECTONIC / "cache"), XDG_CONFIG_HOME=str(BUILD / "config"),
               TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for key in ("TEXINPUTS", "TEXMFHOME", "TEXMFVAR", "TEXMFCONFIG"):
        env.pop(key, None)
    return env


def source_files(figures=PAPER / "figures/journal"):
    """Every file a build reads: what main.tex inputs, the class and style, the figures below `figures`, the notes."""
    spec = importlib.util.spec_from_file_location("paper_writer_rules", PAPER_WRITER / "scripts/rules.py")
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    files = rules.active_files(PAPER / "main.tex")
    files += [PAPER / "clv2025.cls", PAPER / "compling.bst", PAPER / "references.bib"]
    files += [p for p in figures.rglob("*") if p.is_file()]
    files += [p for p in PAPER.glob("*.md") if p.is_file()]
    return list(dict.fromkeys(files))


def compile_bounded(snapshot, engine, env, timeout=PASS_SECONDS):
    """Compile `snapshot` until references and the bibliography settle, at most four passes of `timeout` seconds.

    Returns the last process result and a record of every pass; the PDF is snapshot/.paper-review/build/main.pdf.
    """
    built = snapshot / ".paper-review"
    output = built / "build"
    output.mkdir(parents=True, exist_ok=True)
    command = [str(engine), "--only-cached", "--untrusted", "--keep-logs", "--keep-intermediates",
               "--reruns", "0", "--outdir", str(output), "main.tex"]
    passes = []
    for number in range(1, 5):
        for suffix in ("aux", "bbl", "out", "toc"):
            if (output / ("main." + suffix)).exists():
                shutil.copyfile(output / ("main." + suffix), snapshot / ("main." + suffix))
        started = time.monotonic()
        with (built / f"compiler-pass-{number}.log").open("w") as stream:
            try:
                run = subprocess.run(command, cwd=snapshot, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                run = subprocess.CompletedProcess(command, 124)
        log = output / "main.log"
        pending = bool(UNSETTLED.search(log.read_text(errors="replace"))) if log.exists() else False
        passes.append({"pass": number, "exit_code": run.returncode,
                       "elapsed_seconds": time.monotonic() - started, "rerun_needed": pending})
        print(passes[-1], flush=True)
        if run.returncode or not pending:
            break
    else:
        run = subprocess.CompletedProcess(command, 1)
    if (output / "main.log").exists():
        shutil.copyfile(output / "main.log", built / "main.log")
    return run, passes


def seed_from_last_build(snapshot, bindings):
    """Copy the auxiliary files of the last good build when the class and bibliography are unchanged."""
    path = BUILD / "BUILD.json"
    if not path.exists():
        return None
    previous = json.loads(path.read_text())
    unchanged = all(previous.get("source_hashes", {}).get(name) == bindings.get(name)
                    for name in ("references.bib", "clv2025.cls", "compling.bst"))
    if previous.get("exit_code") != 0 or not unchanged:
        return None
    seed = {}
    for suffix in ("aux", "bbl", "out", "toc"):
        source = Path(previous["source_snapshot"]) / (".paper-review/build/main." + suffix)
        if source.exists():
            shutil.copyfile(source, snapshot / source.name)
            seed[str(source)] = file_sha(source)
    return seed


def render_pages(index):
    """One 144 dpi image per page of BUILD/build/main.pdf, in a fresh folder and in BUILD/pages; returns (folder, count)."""
    with pymupdf.open(BUILD / "build/main.pdf") as document:
        count = len(document)
    (BUILD / "pages").mkdir(exist_ok=True)
    for old in (BUILD / "pages").glob("page-*.png"):
        old.unlink()
    fresh = BUILD / f"pages-{index:02d}"
    fresh.mkdir()
    # Each page is rendered from a freshly opened PDF so that every page image is self-contained.
    for number in range(count):
        with pymupdf.open(BUILD / "build/main.pdf") as document:
            target = fresh / f"page-{number + 1:02d}.png"
            document[number].get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(target)
            shutil.copyfile(target, BUILD / "pages" / target.name)
    return fresh, count


def main(timeout=PASS_SECONDS):
    engine = tectonic()
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / ".gitignore").write_text("*\n")
    (BUILD / "tmp").mkdir(exist_ok=True)
    # A fresh snapshot keeps old auxiliary files of the source folder out of the build.
    index = len(list(BUILD.glob("source-round-*"))) + 1
    snapshot = BUILD / f"source-round-{index:02d}"
    snapshot.mkdir()
    bindings = {}
    for origin in source_files():
        target = snapshot / origin.relative_to(PAPER)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        bindings[str(origin.relative_to(PAPER))] = file_sha(origin)
    # paper-writer looks for the rules beside main.tex; they live outside the Overleaf folder.
    for name in ("paper-rules.md", "writing_brief.md"):
        shutil.copyfile(NOTES / name, snapshot / name)
    (snapshot / ".paper-review").mkdir()
    shutil.copyfile(BUILD / "bib-verify.log", snapshot / ".paper-review/bib-verify.log")
    started = time.monotonic()
    seed = seed_from_last_build(snapshot, bindings)
    run, passes = compile_bounded(snapshot, engine, compile_env(BUILD / "tmp"), timeout)
    built = snapshot / ".paper-review"
    pages = count = None
    if run.returncode == 0:
        (BUILD / "build").mkdir(exist_ok=True)
        for name in ("main.pdf", "main.log"):
            shutil.copyfile(built / "build" / name, BUILD / "build" / name)
        pages, count = render_pages(index)
    if (built / "main.log").exists():
        shutil.copyfile(built / "main.log", BUILD / "main.log")
    pdf = BUILD / "build/main.pdf"
    evidence = {"compiler_sha256": TECTONIC_SHA256, "command": run.args, "source_snapshot": str(snapshot),
                "source_hashes": bindings, "elapsed_seconds": time.monotonic() - started,
                "exit_code": run.returncode, "compiler_timeout_seconds": timeout, "offline": True,
                "scientific_computation": False, "agent_runtime_modified": False,
                "bounded_compile_passes": passes}
    if seed:
        evidence["generated_auxiliary_seed"] = seed
    if pdf.exists():
        evidence.update(pdf_sha256=file_sha(pdf), pdf_path=str(pdf))
    if pages:
        evidence.update(page_count=count, page_renders=str(pages), rendering="144 dpi; independently reopened PDF per page")
    evidence["changed_sources"] = [name for name, expected in bindings.items() if file_sha(PAPER / name) != expected]
    if evidence["changed_sources"]:
        raise ValueError("manuscript sources changed during the build")
    (BUILD / "BUILD.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return run.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--timeout-seconds", type=int, default=PASS_SECONDS, help="time limit of each compiler pass")
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 600:
        parser.error("--timeout-seconds must be between 1 and 600")
    raise SystemExit(main(args.timeout_seconds))
