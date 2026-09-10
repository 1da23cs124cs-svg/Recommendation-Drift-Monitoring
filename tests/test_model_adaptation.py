import numpy as np

from src.model_adaptation import (
    calculate_change,
    classify_adaptation,
)


def test_calculate_change_positive():
    result = calculate_change(0.10, 0.12)

    assert np.isclose(result, 20.0)


def test_calculate_change_negative():
    result = calculate_change(0.10, 0.08)

    assert np.isclose(result, -20.0)


def test_calculate_change_zero_baseline():
    result = calculate_change(0.0, 0.10)

    assert np.isnan(result)


def test_classify_recovered():
    result = classify_adaptation(
        before_ndcg=0.002,
        after_ndcg=0.003,
        before_recall=0.002,
        after_recall=0.004,
    )

    assert result == "RECOVERED"


def test_classify_partial_recovery_ndcg_only():
    result = classify_adaptation(
        before_ndcg=0.002,
        after_ndcg=0.003,
        before_recall=0.004,
        after_recall=0.003,
    )

    assert result == "PARTIAL_RECOVERY"


def test_classify_partial_recovery_recall_only():
    result = classify_adaptation(
        before_ndcg=0.004,
        after_ndcg=0.003,
        before_recall=0.002,
        after_recall=0.004,
    )

    assert result == "PARTIAL_RECOVERY"


def test_classify_not_recovered():
    result = classify_adaptation(
        before_ndcg=0.004,
        after_ndcg=0.003,
        before_recall=0.004,
        after_recall=0.003,
    )

    assert result == "NOT_RECOVERED"


def test_classify_improved_from_zero():
    result = classify_adaptation(
        before_ndcg=0.0,
        after_ndcg=0.002,
        before_recall=0.0,
        after_recall=0.003,
    )

    assert result == "IMPROVED_FROM_ZERO"


def test_classify_zero_to_zero():
    result = classify_adaptation(
        before_ndcg=0.0,
        after_ndcg=0.0,
        before_recall=0.0,
        after_recall=0.0,
    )

    assert result == "NOT_RECOVERED"
    