# Temporary probe for the #21 verification protocol. Reverted before merge.
def test_deliberate_failure_probe():
    assert 1 == 2, "deliberate CI failure-path probe"
