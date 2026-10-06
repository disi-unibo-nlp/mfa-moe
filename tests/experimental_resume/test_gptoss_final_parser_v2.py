from scripts.experimental_resume.gptoss_final_parser_v2 import parse_rating


def test_explicit_and_detokenized_final_channels():
    assert parse_rating('analysisReasoning.assistantfinal{"start": true}') == {"start": True}
    assert parse_rating('<|channel|>final<|message|>{"start":false}<|return|>') == {"start": False}
    assert parse_rating('{"start": true}') == {"start": True}


def test_strict_no_analysis_substring_or_extra_fields():
    assert parse_rating('analysis{"start":true}') is None
    assert parse_rating('analysis...assistantfinal{"start":true} trailing') is None
    assert parse_rating('analysis...assistantfinal{"start":true,"reason":"x"}') is None
    assert parse_rating('analysis...assistantfinal{"start":1}') is None
    assert parse_rating('assistantfinal{"start":true}') is None
    assert parse_rating('analysis assistantfinal{"start":true} assistantfinal{"start":false}') is None
