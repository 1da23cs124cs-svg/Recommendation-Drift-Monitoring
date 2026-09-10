import numpy as np
import pandas as pd

from src.drift_performance_analysis import (
    prepare_performance,
    prepare_cold_start,
    build_relationship_table,
    add_model_health,
)


def test_prepare_performance_calculates_relative_changes():
    performance = pd.DataFrame(
        {
            "period": [
                "baseline_holdout",
                "comparison",
                "recent",
            ],
            "task": [
                "engagement",
                "engagement",
                "engagement",
            ],
            "model": [
                "svd",
                "svd",
                "svd",
            ],
            "k": [10, 10, 10],
            "ndcg_at_k": [
                0.010,
                0.008,
                0.005,
            ],
            "recall_at_k": [
                0.020,
                0.016,
                0.010,
            ],
        }
    )

    result = prepare_performance(performance)

    comparison = result[
        result["period"] == "comparison"
    ].iloc[0]

    recent = result[
        result["period"] == "recent"
    ].iloc[0]

    assert np.isclose(
        comparison["ndcg_relative_change_pct"],
        -20.0,
    )

    assert np.isclose(
        comparison["recall_relative_change_pct"],
        -20.0,
    )

    assert np.isclose(
        recent["ndcg_relative_change_pct"],
        -50.0,
    )

    assert np.isclose(
        recent["recall_relative_change_pct"],
        -50.0,
    )


def test_prepare_cold_start_calculates_rates():
    cold_start = pd.DataFrame(
        {
            "period": ["comparison"],
            "task": ["engagement"],
            "total_test_users": [100],
            "warm_users": [10],
            "cold_users": [90],
            "total_test_items": [200],
            "known_catalog_items": [150],
            "new_items": [50],
        }
    )

    result = prepare_cold_start(
        cold_start
    )

    row = result.iloc[0]

    assert np.isclose(
        row["cold_user_rate_pct"],
        90.0,
    )

    assert np.isclose(
        row["new_item_rate_pct"],
        25.0,
    )


def test_build_relationship_table_merges_all_sources():
    drift = pd.DataFrame(
        {
            "period": [
                "comparison",
                "recent",
            ],
            "event_jsd": [
                0.01,
                0.02,
            ],
            "product_jsd": [
                0.30,
                0.50,
            ],
            "user_jsd": [
                0.70,
                0.80,
            ],
        }
    )

    performance = pd.DataFrame(
        {
            "period": ["comparison"],
            "task": ["engagement"],
            "model": ["svd"],
            "baseline_ndcg_at_10": [0.01],
            "current_ndcg_at_10": [0.008],
            "ndcg_change_vs_baseline": [-0.002],
            "ndcg_relative_change_pct": [-20.0],
            "baseline_recall_at_10": [0.02],
            "current_recall_at_10": [0.016],
            "recall_change_vs_baseline": [-0.004],
            "recall_relative_change_pct": [-20.0],
        }
    )

    cold_start = pd.DataFrame(
        {
            "period": ["comparison"],
            "task": ["engagement"],
            "total_test_users": [100],
            "warm_users": [10],
            "cold_users": [90],
            "cold_user_rate_pct": [90.0],
            "total_test_items": [200],
            "known_catalog_items": [150],
            "new_items": [50],
            "new_item_rate_pct": [25.0],
        }
    )

    result = build_relationship_table(
        drift,
        performance,
        cold_start,
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert row["product_jsd"] == 0.30
    assert row["user_jsd"] == 0.70
    assert row["cold_user_rate_pct"] == 90.0
    assert row["new_item_rate_pct"] == 25.0

    assert (
        "not evidence of causation"
        in row["association_note"]
    )


def test_add_model_health_flags_degradation():
    data = pd.DataFrame(
        {
            "ndcg_relative_change_pct": [
                -10.0,
                -20.0,
                -50.0,
                15.0,
            ]
        }
    )

    result = add_model_health(
        data,
        threshold=20.0,
    )

    assert result["performance_status"].tolist() == [
        "stable",
        "degraded",
        "degraded",
        "stable",
    ]


def test_add_model_health_custom_threshold():
    data = pd.DataFrame(
        {
            "ndcg_relative_change_pct": [
                -15.0,
                -25.0,
            ]
        }
    )

    result = add_model_health(
        data,
        threshold=30.0,
    )

    assert result["performance_status"].tolist() == [
        "stable",
        "stable",
    ]