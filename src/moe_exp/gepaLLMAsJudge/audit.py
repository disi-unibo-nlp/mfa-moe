from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import annotation_audit, load_documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Schoenfeld annotation units without calling a model."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = annotation_audit(load_documents(args.dataset_dir))
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
