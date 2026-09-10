import pandas as pd
import pytest

from src.model_health_monitor import (
    add_health_status,
    add_monitoring_indicators,
    classify_health,
    prepare_health_output,
)


def test_classify_healthy():
    result = classify_health(
        ndcg_change=10.0,
        recall_change=5.0,
        threshold=20.0,
    )

    assert result == "healthy"


def test_classify_warning_for_recall_degradation():
    result = classify_health(
        ndcg_change=10.0,
        recall_change=-25.0,
        threshold=20.0,
    )

    assert result == "warning"


def test_classify_degraded_for_ndcg_degradation():
    result = classify_health(
        ndcg_change=-25.0,
        recall_change=-10.0,
        threshold=20.0,
    )

    assert result == "degraded"


def test_ndcg_degradation_has_priority_over_warning():
    result = classify_health(
        ndcg_change=-30.0,
        recall_change=-40.0,
        threshold=20.0,
    )

    assert result == "degraded"


def test_add_health_status_creates_expected_flags():
    df = pd.DataFrame(
        {
            "ndcg_relative_change_pct": [
                10.0,
                -10.0,
                -30.0,
            ],
            "recall_relative_change_pct": [
                5.0,
                -30.0,
                -40.0,
            ],
        }
    )

    result = add_health_status(
        df,
        threshold=20.0,
    )

    assert result["ndcg_degradation_flag"].tolist() == [
        False,
        False,
        True,
    ]

    assert result["recall_degradation_flag"].tolist() == [
        False,
        True,
        True,
    ]

    assert result["health_status"].tolist() == [
        "healthy",
        "warning",
        "degraded",
    ]

    assert result["health_threshold_pct"].tolist() == [
        20.0,
        20.0,
        20.0,
    ]


def test_custom_threshold_changes_classification():
    df = pd.DataFrame(
        {
            "ndcg_relative_change_pct": [-15.0],
            "recall_relative_change_pct": [5.0],
        }
    )

    result_20 = add_health_status(
        df,
        threshold=20.0,
    )

    result_10 = add_health_status(
        df,
        threshold=10.0,
    )

    assert result_20["health_status"].iloc[0] == "healthy"
    assert result_10["health_status"].iloc[0] == "degraded"


def test_invalid_threshold_is_rejected():
    df = pd.DataFrame(
        {
            "ndcg_relative_change_pct": [0.0],
            "recall_relative_change_pct": [0.0],
        }
    )

    with pytest.raises(ValueError):
        add_health_status(
            df,
            threshold=0,
        )


def test_monitoring_indicators():
    df = pd.DataFrame(
        {
            "product_jsd": [0.3928, 0.5203],
            "user_jsd": [0.7988, 0.8193],
            "cold_user_rate_pct": [99.1, 96.9],
            "new_item_rate_pct": [80.2, 41.2],
        }
    )

    result = add_monitoring_indicators(df)

    assert result["product_drift_level"].tolist() == [
        "moderate",
        "higher_recent",
    ]

    assert result["user_drift_level"].tolist() == [
        "high",
        "high",
    ]

    assert result["cold_start_pressure"].tolist() == [
        "very_high",
        "very_high",
    ]

    assert result["new_item_pressure"].tolist() == [
        "very_high",
        "high",
    ]


def test_prepare_health_output_keeps_monitoring_columns():
    df = pd.DataFrame(
        {
            "period": ["recent"],
            "task": ["engagement"],
            "model": ["svd"],
            "event_jsd": [0.0019],
            "product_jsd": [0.5203],
            "user_jsd": [0.8193],
            "cold_user_rate_pct": [99.1],
            "new_item_rate_pct": [70.7],
            "ndcg_relative_change_pct": [-61.94],
            "recall_relative_change_pct": [-66.15],
            "ndcg_degradation_flag": [True],
            "recall_degradation_flag": [True],
            "product_drift_level": ["higher_recent"],
            "user_drift_level": ["high"],
            "cold_start_pressure": ["very_high"],
            "new_item_pressure": ["very_high"],
            "health_threshold_pct": [20.0],
            "health_status": ["degraded"],
            "association_note": [
                "Temporal association only; not evidence of causation."
            ],
        }
    )

    result = prepare_health_output(df)

    assert len(result) == 1
    assert result["health_status"].iloc[0] == "degraded"
    assert result["model"].iloc[0] == "svd"
    assert "product_jsd" in result.columns
    assert "cold_start_pressure" in result.columns
    assert "association_note" in result.columns