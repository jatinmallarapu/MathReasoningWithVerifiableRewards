"""Check that reasoning format and final-answer correctness score independently."""
from rewards import correctness, format_reward, prediction


def test_correct_numeric_format_variants():
    assert prediction("<think>Divide in half.</think><answer>1/2</answer>") == prediction("#### 0.5")
    assert prediction("#### 1/0") is None


def test_well_formed_wrong_answer_earns_only_format_reward():
    # Correct tags should never make a mathematically wrong answer count toward accuracy.
    completions = [[{"content": "<think>Three plus three is seven.</think><answer>7</answer>"}]]
    assert correctness(completions, ["6"]) == [0.0]
    assert format_reward(completions) == [0.25]
