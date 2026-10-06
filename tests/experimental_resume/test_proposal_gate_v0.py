from moe_exp.routing_control.proposal_gate_v0 import passes


def row(sentence):
    return {"problem": "Find a probability.",
            "emitted_prefix": "Previous work.\n" + sentence,
            "triggering_sentence": sentence,
            "later_sentence": "forbidden future"}


def test_narrow_gate_and_future_invariance():
    assert passes(row("State becomes $B=3,G=3$."))
    assert not passes(row("$2n=74$."))
    assert not passes(row("The problem asks for $N=5$."))
    assert not passes(row("Let's calculate $x^u$."))
    candidate = row("We obtain $x=5$.")
    assert passes(candidate)
    candidate["later_sentence"] = "changed future"
    candidate["gold_answer"] = "secret"
    assert passes(candidate)


def test_gate_rejects_prefix_suffix_change():
    candidate = row("We obtain $x=5$.")
    candidate["emitted_prefix"] += " Future text."
    try:
        passes(candidate)
    except ValueError:
        pass
    else:
        raise AssertionError("future suffix accepted")
