from scripts.experimental_resume.rate_transition_v22_fullprefix_starts_v2 import (
    BATCH, FRAME, digest, messages, parse_rating,
)


def test_all_372_frozen_reader_inputs_are_prefix_only_and_assignments_are_complete():
    import json

    frame = json.loads(FRAME.read_text())
    assert frame["sha256"] == digest({k: v for k, v in frame.items() if k != "sha256"})
    rows = frame["records"]
    assert len(rows) == 372 and len({r["uid"] for r in rows}) == 372
    assert len({r["family"] for r in rows}) == 48
    assert len([(r["uid"], reader) for r in rows for reader in (0, 1)]) == 744
    assert sum(len(rows[start:start+BATCH]) for start in range(0, len(rows), BATCH)) == 372
    for row in rows:
        prompt = messages(row)
        assert row["reader_input"]["triggering_sentence"] in prompt[1]["content"]
        changed = {**row, "analysis_meta": {"future_sentence": "forbidden secret"}}
        assert messages(changed) == prompt
        assert "forbidden secret" not in str(prompt)


def test_full_stage_parser_is_strict():
    assert parse_rating('<think>private</think> {"start": true}') == {"start": True}
    assert parse_rating('{"start": true, "reason": "extra"}') is None
    assert parse_rating('{"start": 1}') is None
