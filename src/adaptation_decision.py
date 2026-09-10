"""
Phase 5A — Automated Adaptation Decision

Purpose
-------
Convert model-health monitoring results into an explicit adaptation
decision and interpret the outcome of a historical retraining experiment.

Decision flow
-------------
HEALTHY
    -> MONITOR

WARNING
    -> MONITOR

DEGRADED
    -> RETRAIN

After retraining:

RECOVERED
    -> ACCEPT

PARTIAL_RECOVERY
    -> REVIEW

IMPROVED_FROM_ZERO
    -> ACCEPT

NOT_RECOVERED
    -> REJECT

Important
---------
This module is a decision/monitoring layer.

It does not claim that drift caused model degradation.
It does not automatically deploy a model.
It does not replace the Phase 4 adaptation experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_HEALTH_INPUT = Path(
    "data/processed/model_health_status.csv"
)

DEFAULT_ADAPTATION_INPUT = Path(
    "data/processed/adaptation_results.csv"
)

DEFAULT_OUTPUT = Path(
    "data/processed/adaptation_decision.csv"
)

DEFAULT_THRESHOLD = 20.0


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_health_data(
    path: Path,
) -> pd.DataFrame:
    """Load Phase 3B model-health results."""

    if not path.exists():
        raise FileNotFoundError(
            f"Model health results not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "period",
        "task",
        "model",
        "health_status",
        "health_threshold_pct",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Model health results are missing columns: "
            f"{sorted(missing)}"
        )

    return df


def load_adaptation_data(
    path: Path,
) -> pd.DataFrame:
    """Load Phase 4 adaptation results."""

    if not path.exists():
        raise FileNotFoundError(
            f"Adaptation results not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "task",
        "model",
        "evaluation_period",
        "before_ndcg_at_10",
        "after_ndcg_at_10",
        "before_recall_at_10",
        "after_recall_at_10",
        "ndcg_relative_change_pct",
        "recall_relative_change_pct",
        "adaptation_status",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Adaptation results are missing columns: "
            f"{sorted(missing)}"
        )

    return df


# ---------------------------------------------------------------------
# Initial adaptation decision
# ---------------------------------------------------------------------

def determine_action(
    health_status: str,
) -> str:
    """
    Determine the monitoring action from model health.

    Rules
    -----
    healthy  -> MONITOR
    warning  -> MONITOR
    degraded -> RETRAIN
    """

    status = str(health_status).strip().lower()

    if status == "degraded":
        return "RETRAIN"

    if status in {
        "healthy",
        "warning",
    }:
        return "MONITOR"

    raise ValueError(
        f"Unknown health status: {health_status}"
    )


# ---------------------------------------------------------------------
# Adaptation validation decision
# ---------------------------------------------------------------------

def determine_validation_decision(
    adaptation_status: str,
) -> str:
    """
    Determine what to do after adaptation validation.

    Rules
    -----
    RECOVERED
        -> ACCEPT

    IMPROVED_FROM_ZERO
        -> ACCEPT

    PARTIAL_RECOVERY
        -> REVIEW

    NOT_RECOVERED
        -> REJECT
    """

    status = (
        str(adaptation_status)
        .strip()
        .upper()
    )

    if status in {
        "RECOVERED",
        "IMPROVED_FROM_ZERO",
    }:
        return "ACCEPT"

    if status == "PARTIAL_RECOVERY":
        return "REVIEW"

    if status == "NOT_RECOVERED":
        return "REJECT"

    raise ValueError(
        "Unknown adaptation status: "
        f"{adaptation_status}"
    )


# ---------------------------------------------------------------------
# Combine monitoring and adaptation results
# ---------------------------------------------------------------------

def build_decision_table(
    health_df: pd.DataFrame,
    adaptation_df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Build the final adaptation-decision table.

    Phase 3B health results determine whether retraining
    should be considered.

    Phase 4 adaptation results determine whether the adapted
    model should be accepted, reviewed, or rejected.

    Only degraded configurations that have corresponding
    adaptation results receive a validation decision.
    """

    if threshold <= 0:
        raise ValueError(
            "Decision threshold must be greater than zero."
        )

    health = health_df.copy()
    adaptation = adaptation_df.copy()

    # ---------------------------------------------------------
    # Determine action from health
    # ---------------------------------------------------------

    health["recommended_action"] = (
        health["health_status"]
        .apply(determine_action)
    )

    # ---------------------------------------------------------
    # Normalize model/task fields
    # ---------------------------------------------------------

    health["model"] = (
        health["model"]
        .astype(str)
        .str.lower()
    )

    health["task"] = (
        health["task"]
        .astype(str)
        .str.lower()
    )

    adaptation["model"] = (
        adaptation["model"]
        .astype(str)
        .str.lower()
    )

    adaptation["task"] = (
        adaptation["task"]
        .astype(str)
        .str.lower()
    )

    # ---------------------------------------------------------
    # Adaptation results are associated with the recent
    # evaluation period.
    #
    # The health result that triggered adaptation may be from
    # an earlier period, so we join on model/task rather than
    # requiring the health period to equal "recent".
    # ---------------------------------------------------------

    adaptation_subset = adaptation[
        [
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
    ].copy()

    adaptation_subset[
        "validation_decision"
    ] = adaptation_subset[
        "adaptation_status"
    ].apply(
        determine_validation_decision
    )

    # ---------------------------------------------------------
    # Keep only the most relevant health row for each
    # model/task pair when an adaptation result exists.
    #
    # We select degraded health states first, because these
    # are the states that trigger retraining.
    # ---------------------------------------------------------

    health["health_priority"] = (
        health["health_status"]
        .map(
            {
                "degraded": 0,
                "warning": 1,
                "healthy": 2,
            }
        )
        .fillna(99)
    )

    health = health.sort_values(
        [
            "model",
            "task",
            "health_priority",
        ]
    )

    health = health.drop_duplicates(
        subset=["model", "task"],
        keep="first",
    )

    # ---------------------------------------------------------
    # Merge
    # ---------------------------------------------------------

    result = health.merge(
        adaptation_subset,
        on=["model", "task"],
        how="left",
        suffixes=("", "_adaptation"),
    )

    # ---------------------------------------------------------
    # Adaptation status
    # ---------------------------------------------------------

    result["adaptation_performed"] = (
        result["adaptation_status"]
        .notna()
    )

    result["validation_required"] = (
        result["recommended_action"]
        == "RETRAIN"
    )

    # ---------------------------------------------------------
    # Final lifecycle decision
    # ---------------------------------------------------------

    def final_decision(row):

        action = row["recommended_action"]

        if action == "MONITOR":
            return "KEEP_CURRENT_MODEL"

        if action == "RETRAIN":

            if not row["adaptation_performed"]:
                return "RETRAIN_PENDING"

            return row["validation_decision"]

        return "REVIEW"

    result["lifecycle_decision"] = (
        result.apply(
            final_decision,
            axis=1,
        )
    )

    # ---------------------------------------------------------
    # Add explicit threshold
    # ---------------------------------------------------------

    result["decision_threshold_pct"] = threshold

    # ---------------------------------------------------------
    # Human-readable explanation
    # ---------------------------------------------------------

    def explain(row):

        action = row["recommended_action"]

        if action == "MONITOR":
            return (
                "Model health is not sufficiently degraded "
                "to trigger automatic adaptation."
            )

        if (
            action == "RETRAIN"
            and not row["adaptation_performed"]
        ):
            return (
                "Model health is degraded; retraining "
                "should be performed and validated."
            )

        status = str(
            row["adaptation_status"]
        ).upper()

        if status in {
            "RECOVERED",
            "IMPROVED_FROM_ZERO",
        }:
            return (
                "Adaptation improved both monitored "
                "performance dimensions sufficiently "
                "to accept the adapted model."
            )

        if status == "PARTIAL_RECOVERY":
            return (
                "Adaptation improved one monitored metric "
                "but not the other; manual review is required."
            )

        if status == "NOT_RECOVERED":
            return (
                "Adaptation did not improve the monitored "
                "performance metrics; retain the current model."
            )

        return (
            "Further validation is required."
        )

    result["decision_explanation"] = (
        result.apply(
            explain,
            axis=1,
        )
    )

    # ---------------------------------------------------------
    # Remove helper column
    # ---------------------------------------------------------

    result = result.drop(
        columns=["health_priority"],
        errors="ignore",
    )

    return result


# ---------------------------------------------------------------------
# Output preparation
# ---------------------------------------------------------------------

def prepare_output(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Select columns required by the lifecycle layer."""

    columns = [
        "period",
        "task",
        "model",

        "health_status",
        "health_threshold_pct",

        "ndcg_relative_change_pct",
        "recall_relative_change_pct",

        "recommended_action",

        "adaptation_performed",
        "evaluation_period",
        "evaluation_users",
        "target_items",

        "before_ndcg_at_10",
        "after_ndcg_at_10",
        "before_recall_at_10",
        "after_recall_at_10",

        "adaptation_status",

        "validation_decision",
        "validation_required",
        "lifecycle_decision",

        "decision_threshold_pct",
        "decision_explanation",

        "association_note",
        "causal_note",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[available].copy()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_decision_summary(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print(
        "PHASE 5A - ADAPTATION DECISION ENGINE"
    )
    print("=" * 80)

    print("\nModel lifecycle decisions:")

    summary_columns = [
        "period",
        "task",
        "model",
        "health_status",
        "recommended_action",
        "adaptation_status",
        "validation_decision",
        "lifecycle_decision",
    ]

    available = [
        column
        for column in summary_columns
        if column in df.columns
    ]

    print(
        df[available].to_string(
            index=False
        )
    )

    print("\nAction summary:")

    action_counts = (
        df["recommended_action"]
        .value_counts()
    )

    for action in [
        "MONITOR",
        "RETRAIN",
    ]:
        print(
            f"{action:<10}: "
            f"{action_counts.get(action, 0)}"
        )

    print("\nLifecycle summary:")

    lifecycle_counts = (
        df["lifecycle_decision"]
        .value_counts()
    )

    for decision in [
        "KEEP_CURRENT_MODEL",
        "ACCEPT",
        "REVIEW",
        "REJECT",
        "RETRAIN_PENDING",
    ]:
        print(
            f"{decision:<22}: "
            f"{lifecycle_counts.get(decision, 0)}"
        )

    print("\nInterpretation:")

    print(
        "- HEALTHY and WARNING states remain under "
        "monitoring without triggering retraining."
    )

    print(
        "- DEGRADED states trigger a RETRAIN recommendation."
    )

    print(
        "- A RECOVERED adaptation is marked ACCEPT."
    )

    print(
        "- A PARTIAL_RECOVERY adaptation is marked REVIEW "
        "because only one monitored metric improved."
    )

    print(
        "- A NOT_RECOVERED adaptation is marked REJECT."
    )

    print(
        "- The decision engine does not automatically deploy "
        "a retrained model."
    )

    print(
        "- Monitoring and adaptation results do not establish "
        "that data drift caused performance degradation."
    )

    print("=" * 80)


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def run_decision_engine(
    health_input: Path = DEFAULT_HEALTH_INPUT,
    adaptation_input: Path = DEFAULT_ADAPTATION_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Run the complete Phase 5A decision pipeline."""

    if threshold <= 0:
        raise ValueError(
            "Decision threshold must be greater than zero."
        )

    health = load_health_data(
        health_input
    )

    adaptation = load_adaptation_data(
        adaptation_input
    )

    decision = build_decision_table(
        health,
        adaptation,
        threshold=threshold,
    )

    output = prepare_output(
        decision
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print_decision_summary(
        output
    )

    print(
        f"\nSaved adaptation decisions to: "
        f"{output_path}"
    )

    return output


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate adaptation and model-lifecycle "
            "decisions from monitoring results."
        )
    )

    parser.add_argument(
        "--health-input",
        type=Path,
        default=DEFAULT_HEALTH_INPUT,
        help=(
            "Path to the Phase 3B model-health CSV."
        ),
    )

    parser.add_argument(
        "--adaptation-input",
        type=Path,
        default=DEFAULT_ADAPTATION_INPUT,
        help=(
            "Path to the Phase 4 adaptation-results CSV."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Path for the Phase 5A decision CSV."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Decision threshold in percent. "
            "Default: 20."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    run_decision_engine(
        health_input=args.health_input,
        adaptation_input=args.adaptation_input,
        output_path=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()