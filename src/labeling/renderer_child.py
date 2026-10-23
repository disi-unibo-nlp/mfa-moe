"""Bounded worker child for the exact original DSPy program, renderer and parser.

Run as `python -m labeling.renderer_child PROGRAM` by `OriginalRenderer`; moved unchanged from
`scripts/research_campaign_v3_r6/renderer.py` except the repository depth of `__file__`.
"""
import json
import sys
from pathlib import Path


def main():
    repo=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo/".research-runtime/original-renderer"))
    sys.path.insert(0,str(repo/"src"))
    from labeling.batch_predictor import (
        load_program_and_adapter,
        render_conversation,
        parse_completion,
        parse_failure_types,
    )
    program,predict,adapter=load_program_and_adapter(sys.argv[1])
    for line in sys.stdin:
        item=json.loads(line)
        try:
            if item["operation"]=="render":
                value={"conversation":render_conversation(adapter,predict,item["inputs"])}
            elif item["operation"]=="parse":
                value={"label":parse_completion(adapter,predict,item["content"])}
            else:
                raise ValueError("unknown renderer operation")
        except parse_failure_types() as exc:
            value={"label":None,"parse_error":str(exc)[:1000]}
        print(json.dumps(value,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
