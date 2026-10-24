"""Build the manuscript PDF with the verified offline tectonic, in passes that each have a time limit.

Copies every source file into a fresh folder BUILD/source-round-NN, compiles it in up to four passes until the
references stop changing, and writes BUILD/build/main.pdf, one image per page (BUILD/pages) and the receipt
BUILD/BUILD.json. The manuscript folder is only read, and the build stops if a source changes while it runs.

  python -m paper.build_cl_manuscript [--timeout-seconds 300]
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time

import pymupdf

from paper.cl_paths import BUILD, FIGURES, PAPER, PAPER_WRITER, TECTONIC, TECTONIC_SHA256
from store import file_sha

PASS_SECONDS = 300  # time limit of one compiler pass; the first pass of the full manuscript takes 70 to 175 s
AUXILIARY = ("aux", "bbl", "out", "toc")  # files that carry references and the bibliography from one pass to the next
UNSETTLED = re.compile(r"Rerun to get cross-references right|Label\(s\) may have changed|"
                       r"There were undefined (?:references|citations)|Citation .+ undefined")


def seconds(text):
    """argparse type of --timeout-seconds: a whole number from 1 to 600."""
    value = int(text)
    if not 0 < value <= 600:
        raise argparse.ArgumentTypeError("must be between 1 and 600")
    return value


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


def source_files(figures=FIGURES):
    """Source files of a build: what main.tex inputs, the class and style, everything below `figures`, the *.md notes."""
    spec = importlib.util.spec_from_file_location("paper_writer_rules", PAPER_WRITER / "scripts/rules.py")
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    files = rules.active_files(PAPER / "main.tex")
    files += [PAPER / "clv2025.cls", PAPER / "compling.bst", PAPER / "references.bib"]
    files += [p for p in figures.rglob("*") if p.is_file()]
    files += [p for p in PAPER.glob("*.md") if p.is_file()]
    return list(dict.fromkeys(files))


def copy_sources(files, folder):
    """Copy manuscript files into `folder` (same relative paths); returns their hashes by relative path."""
    hashes = {}
    for origin in files:
        name = str(origin.relative_to(PAPER))
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, folder / name)
        hashes[name] = file_sha(origin)
    return hashes


def copy_auxiliary(source, folder):
    """Copy the auxiliary files found in `source` as main.* into `folder`; returns their hashes by source path."""
    copied = {}
    for suffix in AUXILIARY:
        if (source / ("main." + suffix)).exists():
            shutil.copyfile(source / ("main." + suffix), folder / ("main." + suffix))
            copied[str(source / ("main." + suffix))] = file_sha(source / ("main." + suffix))
    return copied


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
        copy_auxiliary(output, snapshot)
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
    """Copy the auxiliary files of the newest completed build whose class, bibliography style and references still match.

    A failed latest build does not hide an earlier compatible bibliography cache, because every source-round folder is searched.
    """
    for earlier in sorted(BUILD.glob("source-round-*"), key=lambda folder: int(folder.name.rsplit("-", 1)[1]), reverse=True):
        output = earlier / ".paper-review/build"
        if not all((output / name).is_file() for name in ("main.pdf", "main.aux", "main.bbl")):
            continue
        if all((earlier / name).is_file() and file_sha(earlier / name) == bindings.get(name)
               for name in ("references.bib", "clv2025.cls", "compling.bst")):
            return copy_auxiliary(output, snapshot)
    return None


def render_pages(index):
    """One 144 dpi image per page of BUILD/build/main.pdf, in a fresh folder and in BUILD/pages; returns (folder, count)."""
    (BUILD / "pages").mkdir(exist_ok=True)
    for stale in (BUILD / "pages").glob("page-*.png"):
        stale.unlink()
    fresh = BUILD / f"pages-{index:02d}"
    fresh.mkdir()
    with pymupdf.open(BUILD / "build/main.pdf") as document:
        for number, page in enumerate(document, 1):
            page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(fresh / f"page-{number:02d}.png")
            shutil.copyfile(fresh / f"page-{number:02d}.png", BUILD / "pages" / f"page-{number:02d}.png")
        return fresh, len(document)


def main(timeout):
    engine = tectonic()
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / ".gitignore").write_text("*\n")
    (BUILD / "tmp").mkdir(exist_ok=True)
    # A fresh snapshot keeps the auxiliary files of the source folder out of the build.
    index = len(list(BUILD.glob("source-round-*"))) + 1
    snapshot = BUILD / f"source-round-{index:02d}"
    snapshot.mkdir()
    bindings = copy_sources(source_files(), snapshot)
    started = time.monotonic()
    seed = seed_from_last_build(snapshot, bindings)
    run, passes = compile_bounded(snapshot, engine, compile_env(BUILD / "tmp"), timeout)
    if [name for name, expected in bindings.items() if file_sha(PAPER / name) != expected]:
        raise ValueError("manuscript sources changed during the build")
    built = snapshot / ".paper-review"
    evidence = {"compiler_sha256": TECTONIC_SHA256, "command": run.args, "source_snapshot": str(snapshot),
                "source_hashes": bindings, "elapsed_seconds": time.monotonic() - started,
                "exit_code": run.returncode, "compiler_timeout_seconds": timeout, "offline": True,
                "scientific_computation": False, "agent_runtime_modified": False, "bounded_compile_passes": passes}
    if seed:
        evidence["generated_auxiliary_seed"] = seed
    if run.returncode == 0:
        (BUILD / "build").mkdir(exist_ok=True)
        for name in ("main.pdf", "main.log"):
            shutil.copyfile(built / "build" / name, BUILD / "build" / name)
        pages, count = render_pages(index)
        evidence.update(pdf_sha256=file_sha(BUILD / "build/main.pdf"), pdf_path=str(BUILD / "build/main.pdf"),
                        page_count=count, page_renders=str(pages), rendering="144 dpi")
    if (built / "main.log").exists():
        shutil.copyfile(built / "main.log", BUILD / "main.log")
    (BUILD / "BUILD.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return run.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--timeout-seconds", type=seconds, default=PASS_SECONDS, help="time limit of each compiler pass")
    args = parser.parse_args()
    raise SystemExit(main(args.timeout_seconds))
