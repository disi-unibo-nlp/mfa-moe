"""Bounded full-class manuscript build using a private TeX package overlay.

Scientific sources and installed software are never modified. Public CTAN
runtime archives are retained, hashed and reused. TeX runs without shell escape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper/6a510c6db0d9b08aa438c142"
STAGING = Path("/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp")
OVERLAY = STAGING / "verification-paper-tex-overlay-r1"
PACKAGES = {
    "xstring.sty": "xstring", "totpages.sty": "totpages",
    "environ.sty": "environ", "trimspaces.sty": "trimspaces",
    "framed.sty": "framed", "hyperxmp.sty": "hyperxmp",
    "draftwatermark.sty": "draftwatermark", "everypage.sty": "everypage",
    "manyfoot.sty": "ncctools", "nccfoots.sty": "ncctools",
    "comment.sty": "comment", "balance.sty": "preprint",
    "pbalance.sty": "pbalance", "doclicense.sty": "doclicense",
    "xifthen.sty": "xifthen", "ifmtarg.sty": "ifmtarg",
    "ifdraft.sty": "ifdraft", "ifplatform.sty": "ifplatform",
    "catchfile.sty": "catchfile", "pdfescape.sty": "pdfescape",
    "kvdefinekeys.sty": "kvdefinekeys", "kvsetkeys.sty": "kvsetkeys",
    "ltxcmds.sty": "ltxcmds", "pdftexcmds.sty": "pdftexcmds",
    "ifluatex.sty": "ifluatex", "ifpdf.sty": "ifpdf",
    "iftex.sty": "iftex", "kvoptions.sty": "kvoptions",
    "stringenc.sty": "stringenc", "intcalc.sty": "intcalc",
    "bitset.sty": "bitset", "bigintcalc.sty": "bigintcalc",
    "atbegshi.sty": "atbegshi", "atveryend.sty": "atveryend",
    "zref-base.sty": "zref", "infwarerr.sty": "infwarerr",
}
MAX_DOWNLOAD = 8 * 1024 * 1024
MAX_TOTAL = 40 * 1024 * 1024


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private(path):
    path = path.resolve()
    if STAGING not in path.parents:
        raise ValueError("build writes must stay in the approved private staging tree")
    return path


def publish(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def install(package):
    archive = OVERLAY / "archives" / (package + ".tar.xz")
    legacy_receipt = archive.with_suffix(archive.suffix + ".json")
    receipt = archive.with_suffix(archive.suffix + ".runtime-r2.json")
    previous = receipt if receipt.exists() else legacy_receipt
    if previous.exists():
        item = json.loads(previous.read_text())
        if sha(archive) != item["archive_sha256"]:
            raise ValueError("retained TeX archive changed")
        for relative, expected in item["runtime_files"].items():
            if sha(OVERLAY / relative) != expected:
                raise ValueError("retained TeX runtime changed")
        if item["runtime_files"]:
            return item
    elif archive.exists():
        raise ValueError("unreceipted download needs explicit reconciliation")
    archive.parent.mkdir(parents=True, exist_ok=True)
    url = "https://mirrors.ctan.org/systems/texlive/tlnet/archive/" + archive.name
    if not archive.exists():
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
            "--max-time", "20", "--max-filesize", str(MAX_DOWNLOAD), "--output", str(archive), url],
            check=True, timeout=25)
    if sum(p.stat().st_size for p in archive.parent.glob("*.tar.xz")) > MAX_TOTAL:
        raise ValueError("bounded TeX package budget exceeded")
    runtime = {}
    with tarfile.open(archive, "r:xz") as bundle:
        members = list(bundle.getmembers())
        if sum(m.size for m in members) > 80 * 1024 * 1024:
            raise ValueError("TeX runtime extraction exceeds bound")
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                raise ValueError("unsafe TeX archive member")
            if relative.parts and relative.parts[0] == "texmf-dist":
                relative = Path(*relative.parts[1:])
            if not member.isfile() or not relative.parts or relative.parts[0] not in {"tex", "fonts"}:
                continue
            target = private(OVERLAY / relative)
            data = bundle.extractfile(member).read()
            if target.exists() and target.read_bytes() != data:
                raise ValueError("TeX runtime file collision")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            runtime[str(relative)] = hashlib.sha256(data).hexdigest()
    if not runtime:
        raise ValueError("package archive has no recognized runtime")
    item = {"package": package, "url": url, "archive_sha256": sha(archive), "runtime_files": runtime,
            "layout_revision": 2}
    if legacy_receipt.exists():
        item["previous_receipt_sha256"] = sha(legacy_receipt)
    publish(receipt, item)
    print("retained CTAN runtime: " + package, flush=True)
    return item


def run(command, snapshot, output, environment, label, seconds=40):
    with (output / (label + ".stdout")).open("w") as stdout:
        result = subprocess.run(command, cwd=snapshot, env=environment, stdout=stdout,
            stderr=subprocess.STDOUT, timeout=seconds)
    return result.returncode


def build(directory):
    directory = private(directory)
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "source"
    snapshot.mkdir()
    sources = {}
    for path in PAPER.rglob("*"):
        if not path.is_file() or not (path.suffix in {".tex", ".cls", ".bib", ".bbx", ".cbx", ".dbx"}
                or "figures" in path.relative_to(PAPER).parts):
            continue
        sources[str(path)] = sha(path)
        target = snapshot / path.relative_to(PAPER)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(path) != sources[str(path)] or sha(target) != sources[str(path)]:
            raise ValueError("paper source changed during snapshot")
    output = directory / "output"
    output.mkdir()
    environment = dict(os.environ)
    environment.update(TEXMFHOME=str(OVERLAY), TEXMFVAR=str(directory / "texmf-var"),
        TEXMFCONFIG=str(directory / "texmf-config"), TMPDIR=str(directory / "tmp"),
        TEXINPUTS=str(OVERLAY / "tex") + "//:" + str(snapshot) + "//:")
    (directory / "tmp").mkdir()
    command = ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
               "-recorder", "-output-directory=" + str(output), "main.tex"]
    installed = []; attempts = []; start = time.time(); state = "BUILD_FAILED"
    try:
        for index in range(1, 19):
            code = run(command, snapshot, output, environment, "latex-" + str(index))
            log = (output / "main.log").read_text(errors="replace")
            attempts.append({"pass": index, "exit_code": code, "stdout": str(output / ("latex-" + str(index) + ".stdout"))})
            if code == 0:
                bib = run(["bibtex", str(output / "main")], snapshot, output, environment, "bibtex", 15)
                attempts.append({"role": "bibtex", "exit_code": bib})
                if bib != 0:
                    state = "BIBLIOGRAPHY_FAILED"; break
                for final in (1, 2):
                    final_code = run(command, snapshot, output, environment, "latex-final-" + str(final))
                    attempts.append({"role": "final-" + str(final), "exit_code": final_code})
                    if final_code != 0:
                        raise ValueError("final TeX pass failed")
                state = "COMPILED_FULL_CLASS"; break
            match = re.search(r"! LaTeX Error: File `([^']+)' not found", log)
            if not match or match.group(1) not in PACKAGES:
                break
            package = PACKAGES[match.group(1)]
            if package in {item['package'] for item in installed}:
                raise ValueError("installed runtime still missing")
            installed.append(install(package))
        changed = [path for path, expected in sources.items() if sha(Path(path)) != expected]
        if changed:
            state = "CANONICAL_SOURCE_CHANGED_DURING_BUILD"
    except Exception as error:
        state = "BUILD_ERROR"
        attempts.append({"error": type(error).__name__ + ": " + str(error)})
    log_path = output / "main.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    result = {"schema": "verification-full-paper-build-v3", "state": state,
        "observed_epoch": time.time(), "elapsed_seconds": time.time() - start,
        "canonical_source_files": sources, "source_snapshot": str(snapshot),
        "build_script_sha256": sha(Path(__file__)), "attempts": attempts,
        "retained_packages": installed,
        "engine": subprocess.check_output(["pdflatex", "--version"], text=True).splitlines()[0],
        "log_path": str(log_path), "missing_file": re.findall(r"File `([^']+)' not found", log),
        "diagnostics": [line.strip() for line in log.splitlines() if line.startswith("!") or "Warning:" in line],
        "installed_runtime_modified": False, "scientific_dispatch": "NONE"}
    pdf = output / "main.pdf"
    if state == "COMPILED_FULL_CLASS":
        result.update(pdf_path=str(pdf), pdf_sha256=sha(pdf), pdf_bytes=pdf.stat().st_size)
    publish(directory / "BUILD.json", result)
    print(json.dumps({key:result[key] for key in ['state','elapsed_seconds','missing_file','diagnostics']}, indent=2))
    return 0 if state == "COMPILED_FULL_CLASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(build(args.output))
