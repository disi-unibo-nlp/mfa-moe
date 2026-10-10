from submit_dense_recovery_v3 import completed_zero


def test_native_sacct_field_parser_accepts_both_formats_and_rejects_ambiguity():
    assert completed_zero("59491564|COMPLETED|0:0\n", "59491564")
    assert completed_zero("59491564|COMPLETED|0:0|\n", "59491564")
    assert not completed_zero("59491564|FAILED|1:0\n", "59491564")
    assert not completed_zero("59491564.batch|COMPLETED|0:0\n", "59491564")
    assert not completed_zero("59491564|COMPLETED|0:0\n59491564|COMPLETED|0:0\n", "59491564")
