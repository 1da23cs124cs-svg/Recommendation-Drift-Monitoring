"""
Phase 3B - Model Health Monitoring

Combines:
    - data drift
    - cold-start pressure
    - recommendation performance degradation

into a transparent model-health status.

Health policy
-------------
DEGRADED:
    Recommendation NDCG degradation reaches the configured threshold.

WARNING:
    No NDCG degradation, but Recall degradation reaches the configured
    threshold.

HEALTHY:
    Neither NDCG nor Recall reaches the degradation threshold.

Important:
    This is a monitoring rule, not a causal model and not a statistical
    significance test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path(
    "data/processed/drift_performance_relationship.csv"
)

DEFAULT_OUTPUT = Path(
    "data/processed/model_health_status.csv"
)

DEFAULT_FIGURES = Path(
    "reports/figures"
)

DEFAULT_THRESHOLD = 20.0


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_relationship_data(
    path: Path,
) -> pd.DataFrame:
    """Load the Phase 3A relationship-analysis output."""

    if not path.exists():
        raise FileNotFoundError(
            f"Relationship analysis not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "period",
        "task",
        "model",
        "event_jsd",
        "product_jsd",
        "user_jsd",
        "cold_user_rate_pct",
        "new_item_rate_pct",
        "ndcg_relative_change_pct",
        "recall_relative_change_pct",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Relationship analysis is missing columns: "
            f"{sorted(missing)}"
        )

    return df


# ---------------------------------------------------------------------
# Health classification
# ---------------------------------------------------------------------

def classify_health(
    ndcg_change: float,
    recall_change: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """
    Classify model health using transparent monitoring rules.

    Priority:
        1. NDCG degradation -> DEGRADED
        2. Recall degradation -> WARNING
        3. Otherwise -> HEALTHY
    """

    if pd.notna(ndcg_change):
        if ndcg_change <= -threshold:
            return "degraded"

    if pd.notna(recall_change):
        if recall_change <= -threshold:
            return "warning"

    return "healthy"


def add_health_status(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Add model-health status and monitoring flags."""

    if threshold <= 0:
        raise ValueError(
            "Health threshold must be greater than zero."
        )

    result = df.copy()

    result["ndcg_degradation_flag"] = (
        result["ndcg_relative_change_pct"]
        <= -threshold
    )

    result["recall_degradation_flag"] = (
        result["recall_relative_change_pct"]
        <= -threshold
    )

    result["health_status"] = result.apply(
        lambda row: classify_health(
            row["ndcg_relative_change_pct"],
            row["recall_relative_change_pct"],
            threshold=threshold,
        ),
        axis=1,
    )

    result["health_threshold_pct"] = threshold

    return result


# ---------------------------------------------------------------------
# Monitoring indicators
# ---------------------------------------------------------------------

def add_monitoring_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add interpretable drift and cold-start indicators.

    These indicators describe the environment and are not themselves
    used to claim causation.
    """

    result = df.copy()

    result["product_drift_level"] = result[
        "product_jsd"
    ].apply(
        lambda value:
            "higher_recent"
            if value >= 0.5
            else "moderate"
    )

    result["user_drift_level"] = result[
        "user_jsd"
    ].apply(
        lambda value:
            "high"
            if value >= 0.5
            else "moderate"
    )

    result["cold_start_pressure"] = result[
        "cold_user_rate_pct"
    ].apply(
        lambda value:
            "very_high"
            if value >= 90
            else "high"
            if value >= 70
            else "moderate"
            if value >= 40
            else "low"
    )

    result["new_item_pressure"] = result[
        "new_item_rate_pct"
    ].apply(
        lambda value:
            "very_high"
            if value >= 70
            else "high"
            if value >= 40
            else "moderate"
            if value >= 20
            else "low"
    )

    return result


# ---------------------------------------------------------------------
# Output preparation
# ---------------------------------------------------------------------

def prepare_health_output(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Select columns required by the monitoring layer."""

    columns = [
        "period",
        "task",
        "model",

        "event_jsd",
        "product_jsd",
        "user_jsd",

        "cold_user_rate_pct",
        "new_item_rate_pct",

        "ndcg_relative_change_pct",
        "recall_relative_change_pct",

        "ndcg_degradation_flag",
        "recall_degradation_flag",

        "product_drift_level",
        "user_drift_level",
        "cold_start_pressure",
        "new_item_pressure",

        "health_threshold_pct",
        "health_status",

        "association_note",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[available].copy()


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def create_health_chart(
    df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """
    Create a model-health overview chart.

    The chart displays relative NDCG change for each model/task/period.
    """

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        figures_dir
        / "model_health_overview.png"
    )

    plot_data = df.copy()

    plot_data["label"] = (
        plot_data["model"]
        + " - "
        + plot_data["task"]
        + " - "
        + plot_data["period"]
    )

    plot_data = plot_data.sort_values(
        "ndcg_relative_change_pct"
    )

    plt.figure(figsize=(12, 7))

    plt.barh(
        plot_data["label"],
        plot_data["ndcg_relative_change_pct"],
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    threshold = float(
        plot_data["health_threshold_pct"].iloc[0]
    )

    plt.axvline(
        -threshold,
        linestyle=":",
        linewidth=1,
    )

    plt.xlabel(
        "Relative NDCG@10 Change (%)"
    )

    plt.ylabel(
        "Model / Task / Period"
    )

    plt.title(
        "Recommendation Model Health Overview"
    )

    plt.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=150,
    )

    plt.close()

    return figure_path


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_health_summary(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print(
        "PHASE 3B - MODEL HEALTH MONITORING"
    )
    print("=" * 80)

    print("\nModel health status:")

    summary = df[
        [
            "period",
            "task",
            "model",
            "ndcg_relative_change_pct",
            "recall_relative_change_pct",
            "health_status",
        ]
    ].copy()

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nMonitoring environment:")

    environment = df[
        [
            "period",
            "task",
            "product_jsd",
            "user_jsd",
            "cold_user_rate_pct",
            "new_item_rate_pct",
            "cold_start_pressure",
            "new_item_pressure",
        ]
    ].copy()

    print(
        environment.to_string(
            index=False,
            col_space=12,
        )
    )

    counts = (
        df["health_status"]
        .value_counts()
    )

    print("\nHealth summary:")

    print(
        f"Healthy   : "
        f"{counts.get('healthy', 0)}"
    )

    print(
        f"Warning   : "
        f"{counts.get('warning', 0)}"
    )

    print(
        f"Degraded  : "
        f"{counts.get('degraded', 0)}"
    )

    print("\nInterpretation:")

    degraded = df[
        df["health_status"]
        == "degraded"
    ]

    warning = df[
        df["health_status"]
        == "warning"
    ]

    print(
        f"- {len(degraded)} combinations are "
        "currently classified as degraded "
        "because NDCG declined beyond the threshold."
    )

    print(
        f"- {len(warning)} combinations are "
        "classified as warning because Recall "
        "declined beyond the threshold without "
        "NDCG crossing the degradation threshold."
    )

    print(
        "- Drift and cold-start indicators provide "
        "environmental context for model monitoring."
    )

    print(
    "- Health status indicates monitoring conditions "
    "and does not establish that drift caused degradation."
)

    print("=" * 80)


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def run_monitoring(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    figures_dir: Path = DEFAULT_FIGURES,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Run the complete Phase 3B monitoring pipeline."""

    if threshold <= 0:
        raise ValueError(
            "Health threshold must be greater than zero."
        )

    relationship = load_relationship_data(
        input_path
    )

    health = add_health_status(
        relationship,
        threshold=threshold,
    )

    health = add_monitoring_indicators(
        health
    )

    output = prepare_health_output(
        health
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    create_health_chart(
        output,
        figures_dir,
    )

    print_health_summary(
        output
    )

    print(
        f"\nSaved model health results to: "
        f"{output_path}"
    )

    print(
        f"Saved figure to: {figures_dir}"
    )

    return output


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Monitor recommendation model health "
            "using drift and performance signals."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Path to the Phase 3A "
            "relationship-analysis CSV."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Path for the model-health CSV."
        ),
    )

    parser.add_argument(
        "--figures",
        type=Path,
        default=DEFAULT_FIGURES,
        help=(
            "Directory for monitoring figures."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Relative performance threshold "
            "in percent. Default: 20."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    run_monitoring(
        input_path=args.input,
        output_path=args.output,
        figures_dir=args.figures,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()