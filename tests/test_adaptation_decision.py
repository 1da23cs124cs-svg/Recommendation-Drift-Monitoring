"""
Tests for Phase 5A — Automated Adaptation Decision
"""

import pandas as pd
import pytest

from src.adaptation_decision import (
    determine_action,
    determine_validation_decision,
    build_decision_table,
)


# ---------------------------------------------------------------------
# Health status -> action
# ---------------------------------------------------------------------

def test_healthy_status_results_in_monitor():
    assert (
        determine_action("healthy")
        == "MONITOR"
    )


def test_warning_status_results_in_monitor():
    assert (
        determine_action("warning")
        == "MONITOR"
    )


def test_degraded_status_results_in_retrain():
    assert (
        determine_action("degraded")
        == "RETRAIN"
    )


def test_invalid_health_status_raises_error():
    with pytest.raises(ValueError):
        determine_action("unknown")


# ---------------------------------------------------------------------
# Adaptation status -> validation decision
# ---------------------------------------------------------------------

def test_recovered_results_in_accept():
    assert (
        determine_validation_decision(
            "RECOVERED"
        )
        == "ACCEPT"
    )


def test_improved_from_zero_results_in_accept():
    assert (
        determine_validation_decision(
            "IMPROVED_FROM_ZERO"
        )
        == "ACCEPT"
    )


def test_partial_recovery_results_in_review():
    assert (
        determine_validation_decision(
            "PARTIAL_RECOVERY"
        )
        == "REVIEW"
    )


def test_not_recovered_results_in_reject():
    assert (
        determine_validation_decision(
            "NOT_RECOVERED"
        )
        == "REJECT"
    )


def test_invalid_adaptation_status_raises_error():
    with pytest.raises(ValueError):
        determine_validation_decision(
            "UNKNOWN_STATUS"
        )


# ---------------------------------------------------------------------
# Final lifecycle decisions
# ---------------------------------------------------------------------

def make_health_data(
    status,
    model="svd",
    task="engagement",
):
    return pd.DataFrame(
        [
            {
                "period": "recent",
                "task": task,
                "model": model,
                "health_status": status,
                "health_threshold_pct": 20.0,
                "ndcg_relative_change_pct": -30.0
                if status == "degraded"
                else 5.0,
                "recall_relative_change_pct": -25.0
                if status == "warning"
                else 5.0,
                "association_note": (
                    "Temporal association only; "
                    "not evidence  of causation."
                ),
            }
        ]
    )


def make_adaptation_data(
    status,
    model="svd",
    task="engagement",
):
    return pd.DataFrame(
        [
            {
                "task": task,
                "model": model,
                "evaluation_period": "recent",
                "evaluation_users": 100,
                "target_items": 500,
                "before_ndcg_at_10": 0.002,
                "after_ndcg_at_10": 0.003,
                "before_recall_at_10": 0.002,
                "after_recall_at_10": 0.003,
                "ndcg_relative_change_pct": 50.0,
                "recall_relative_change_pct": 50.0,
                "adaptation_status": status,
            }
        ]
    )


def test_healthy_model_keeps_current_model():

    health = make_health_data(
        "healthy"
    )

    adaptation = pd.DataFrame(
        columns=[
            "task",
            "model",
            "evaluation_period",
            "evaluation_users",
            "target_items",
            "before_ndcg_at_10",
            "after_ndcg_at_10",
            "before_recall_at_10",
            "after_recall_at_10",
            "ndcg_relative_change_pct",
            "recall_relative_change_pct",
            "adaptation_status",
        ]
    )

    result = build_decision_table(
        health,
        adaptation,
    )

    assert (
        result.iloc[0]["recommended_action"]
        == "MONITOR"
    )

    assert (
        result.iloc[0]["lifecycle_decision"]
        == "KEEP_CURRENT_MODEL"
    )


def test_warning_model_keeps_current_model():

    health = make_health_data(
        "warning"
    )

    adaptation = pd.DataFrame(
        columns=[
            "task",
            "model",
            "evaluation_period",
            "evaluation_users",
            "target_items",
            "before_ndcg_at_10",
            "after_ndcg_at_10",
            "before_recall_at_10",
            "after_recall_at_10",
            "ndcg_relative_change_pct",
            "recall_relative_change_pct",
            "adaptation_status",
        ]
    )

    result = build_decision_table(
        health,
        adaptation,
    )

    assert (
        result.iloc[0]["recommended_action"]
        == "MONITOR"
    )

    assert (
        result.iloc[0]["lifecycle_decision"]
        == "KEEP_CURRENT_MODEL"
    )


def test_degraded_model_without_adaptation_is_pending():

    health = make_health_data(
        "degraded"
    )

    adaptation = pd.DataFrame(
        columns=[
            "task",
            "model",
            "evaluation_period",
            "evaluation_users",
            "target_items",
            "before_ndcg_at_10",
            "after_ndcg_at_10",
            "before_recall_at_10",
            "after_recall_at_10",
            "ndcg_relative_change_pct",
            "recall_relative_change_pct",
            "adaptation_status",
        ]
    )

    result = build_decision_table(
        health,
        adaptation,
    )

    assert (
        result.iloc[0]["recommended_action"]
        == "RETRAIN"
    )

    assert not result.iloc[0]["adaptation_performed"]

    assert (
        result.iloc[0]["lifecycle_decision"]
        == "RETRAIN_PENDING"
    )


def test_degraded_recovered_model_is_accepted():

    health = make_health_data(
        "degraded"
    )

    adaptation = make_adaptation_data(
        "RECOVERED"
    )

    result = build_decision_table(
        health,
        adaptation,
    )

    assert (
        result.iloc[0]["recommended_action"]
        == "RETRAIN"
    )

    assert result.iloc[0]["adaptation_performed"]

    assert (
        result.iloc[0]["validation_decision"]
        == "ACCEPT"
    )

    assert (
        result.iloc[0]["lifecycle_decision"]
        == "ACCEPT"
    )


def test_degraded_partial_recovery_requires_review():

    health = make_health_data(
        "degraded"
    )

    adaptation = make_adaptation_data(
        "PARTIAL_RECOVERY"
    )

    result = build_decision_table(
        health,
        adaptation,
    )

    assert (
        result.iloc[0]["recommended_action"]
        == "RETRAIN"
    )

    assert (
        result.iloc[0]["validation_decision"]
        == "REVIEW"
    )

    assert (
        result.iloc[0]["lifecycle_decision"]
        == "REVIEW"
    )


def test_degraded_not_recovered_model_is_rejected():

    health = make_health_data(
        "degraded"
    )

    adaptation = make_adaptation_data(
        "NOT_RECOVERED"
    )

    result = build_decision_table(
        health,
        adaptation,
    )

    assert (
        result.iloc[0]["recommended_action"]
        == "RETRAIN"
    )

    assert (
        result.iloc[0]["validation_decision"]
        == "REJECT"
    )

    assert (
        result.iloc[0]["lifecycle_decision"]
        == "REJECT"
    )