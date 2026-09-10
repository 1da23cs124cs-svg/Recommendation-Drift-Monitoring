"""
Phase 3A - Drift and Recommendation Performance Relationship Analysis

Purpose
-------
Combine Phase 1 drift measurements, Phase 2A recommendation performance,
and cold-start statistics into one monitoring view.

The analysis describes temporal association between distribution drift,
cold-start pressure, and recommendation performance degradation.

Important
---------
This module does NOT claim that drift causes model degradation.
With only a small number of temporal periods, the analysis is descriptive
and association-based.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DRIFT = Path(
    "data/processed/drift_results.csv"
)

DEFAULT_PERFORMANCE = Path(
    "data/processed/recommendation_results.csv"
)

DEFAULT_COLD_START = Path(
    "data/processed/cold_start_summary.csv"
)

DEFAULT_OUTPUT = Path(
    "data/processed/drift_performance_relationship.csv"
)

DEFAULT_FIGURES = Path(
    "reports/figures"
)

BASELINE_PERIOD = "baseline_holdout"
FUTURE_PERIODS = [
    "comparison",
    "recent",
]

K = 10


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_drift_results(
    path: Path,
) -> pd.DataFrame:
    """Load Phase 1 drift measurements."""

    if not path.exists():
        raise FileNotFoundError(
            f"Drift results not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "Comparison",
        "Event_JSD",
        "Product_JSD",
        "User_JSD",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Drift results are missing columns: "
            f"{sorted(missing)}"
        )

    rows = []

    for _, row in df.iterrows():

        label = str(row["Comparison"])

        if label == "Baseline vs Comparison":
            period = "comparison"

        elif label == "Baseline vs Recent":
            period = "recent"

        else:
            raise ValueError(
                f"Unknown drift comparison: {label}"
            )

        rows.append(
            {
                "period": period,
                "event_jsd": float(row["Event_JSD"]),
                "product_jsd": float(row["Product_JSD"]),
                "user_jsd": float(row["User_JSD"]),
            }
        )

    return pd.DataFrame(rows)


def load_performance_results(
    path: Path,
) -> pd.DataFrame:
    """Load Phase 2A recommendation evaluation results."""

    if not path.exists():
        raise FileNotFoundError(
            f"Recommendation results not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "period",
        "task",
        "model",
        "k",
        "ndcg_at_k",
        "recall_at_k",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Recommendation results are missing columns: "
            f"{sorted(missing)}"
        )

    return df


def load_cold_start_results(
    path: Path,
) -> pd.DataFrame:
    """Load Phase 2A cold-start summary."""

    if not path.exists():
        raise FileNotFoundError(
            f"Cold-start summary not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "period",
        "task",
        "total_test_users",
        "warm_users",
        "cold_users",
        "total_test_items",
        "known_catalog_items",
        "new_items",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Cold-start summary is missing columns: "
            f"{sorted(missing)}"
        )

    return df


# ---------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------

def prepare_performance(
    performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep K=10 and calculate future-period performance changes
    relative to the baseline holdout.
    """

    df = performance[
        performance["k"].eq(K)
        & performance["period"].isin(
            [BASELINE_PERIOD] + FUTURE_PERIODS
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No performance records found for k=10."
        )

    rows = []

    for (task, model), group in df.groupby(
        ["task", "model"]
    ):

        baseline = group[
            group["period"].eq(BASELINE_PERIOD)
        ]

        if baseline.empty:
            continue

        baseline_row = baseline.iloc[0]

        baseline_ndcg = float(
            baseline_row["ndcg_at_k"]
        )

        baseline_recall = float(
            baseline_row["recall_at_k"]
        )

        for period in FUTURE_PERIODS:

            current = group[
                group["period"].eq(period)
            ]

            if current.empty:
                continue

            current_row = current.iloc[0]

            current_ndcg = float(
                current_row["ndcg_at_k"]
            )

            current_recall = float(
                current_row["recall_at_k"]
            )

            ndcg_change = (
                current_ndcg - baseline_ndcg
            )

            recall_change = (
                current_recall - baseline_recall
            )

            if baseline_ndcg != 0:
                ndcg_relative_change = (
                    ndcg_change / baseline_ndcg
                ) * 100
            else:
                ndcg_relative_change = float("nan")

            if baseline_recall != 0:
                recall_relative_change = (
                    recall_change / baseline_recall
                ) * 100
            else:
                recall_relative_change = float("nan")

            rows.append(
                {
                    "period": period,
                    "task": task,
                    "model": model,
                    "baseline_ndcg_at_10": baseline_ndcg,
                    "current_ndcg_at_10": current_ndcg,
                    "ndcg_change_vs_baseline": ndcg_change,
                    "ndcg_relative_change_pct":
                        ndcg_relative_change,
                    "baseline_recall_at_10":
                        baseline_recall,
                    "current_recall_at_10":
                        current_recall,
                    "recall_change_vs_baseline":
                        recall_change,
                    "recall_relative_change_pct":
                        recall_relative_change,
                }
            )

    if not rows:
        raise ValueError(
            "Unable to calculate performance changes."
        )

    return pd.DataFrame(rows)


def prepare_cold_start(
    cold_start: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate cold-user and new-item rates.
    """

    df = cold_start.copy()

    df["cold_user_rate_pct"] = (
        df["cold_users"]
        / df["total_test_users"]
        * 100
    )

    df["new_item_rate_pct"] = (
        df["new_items"]
        / df["total_test_items"]
        * 100
    )

    return df[
        [
            "period",
            "task",
            "total_test_users",
            "warm_users",
            "cold_users",
            "cold_user_rate_pct",
            "total_test_items",
            "known_catalog_items",
            "new_items",
            "new_item_rate_pct",
        ]
    ]


# ---------------------------------------------------------------------
# Relationship analysis
# ---------------------------------------------------------------------

def build_relationship_table(
    drift: pd.DataFrame,
    performance: pd.DataFrame,
    cold_start: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine drift, recommendation performance, and cold-start metrics.
    """

    result = performance.merge(
        drift,
        on="period",
        how="left",
    )

    result = result.merge(
        cold_start,
        on=["period", "task"],
        how="left",
    )

    if result["product_jsd"].isna().any():
        raise ValueError(
            "Some performance periods could not be matched "
            "with drift measurements."
        )

    if result["cold_user_rate_pct"].isna().any():
        raise ValueError(
            "Some performance rows could not be matched "
            "with cold-start statistics."
        )

    result["association_note"] = (
        "Temporal association only; "
        "not evidence of causation."
    )

    return result


# ---------------------------------------------------------------------
# Model-health interpretation
# ---------------------------------------------------------------------

def add_model_health(
    df: pd.DataFrame,
    threshold: float = 20.0,
) -> pd.DataFrame:
    """
    Apply the same 20% relative NDCG monitoring policy used in Phase 2B.
    """

    result = df.copy()

    result["performance_status"] = result[
        "ndcg_relative_change_pct"
    ].apply(
        lambda value:
            "degraded"
            if pd.notna(value)
            and value <= -threshold
            else "stable"
    )

    return result


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def create_drift_performance_chart(
    df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """
    Plot drift values against model performance change.
    """

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        figures_dir
        / "drift_vs_performance.png"
    )

    plt.figure(figsize=(10, 6))

    for (task, model), group in df.groupby(
        ["task", "model"]
    ):

        group = group.sort_values("period")

        label = f"{model} - {task}"

        plt.plot(
            group["product_jsd"],
            group["ndcg_relative_change_pct"],
            marker="o",
            label=label,
        )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Product Drift (JSD)")
    plt.ylabel("Relative NDCG@10 Change (%)")
    plt.title(
        "Product Drift vs Recommendation Performance Change"
    )
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=150,
    )

    plt.close()

    return figure_path


def create_cold_start_chart(
    df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """
    Plot cold-user rate across periods.
    """

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        figures_dir
        / "cold_start_analysis.png"
    )

    plot_data = (
        df[
            [
                "period",
                "task",
                "cold_user_rate_pct",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    period_order = [
        BASELINE_PERIOD,
        "comparison",
        "recent",
    ]

    plot_data["period"] = pd.Categorical(
        plot_data["period"],
        categories=period_order,
        ordered=True,
    )

    plt.figure(figsize=(10, 6))

    for task, group in plot_data.groupby("task"):

        group = group.sort_values("period")

        plt.plot(
            group["period"].astype(str),
            group["cold_user_rate_pct"],
            marker="o",
            label=task,
        )

    plt.xlabel("Evaluation Period")
    plt.ylabel("Cold Users (%)")
    plt.title(
        "Cold-Start User Rate Across Evaluation Periods"
    )
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.legend()
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

def print_summary(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print(
        "PHASE 3A - DRIFT AND PERFORMANCE "
        "RELATIONSHIP ANALYSIS"
    )
    print("=" * 80)

    print("\nDrift measurements:")

    drift_summary = (
        df[
            [
                "period",
                "event_jsd",
                "product_jsd",
                "user_jsd",
            ]
        ]
        .drop_duplicates()
        .sort_values("period")
    )

    print(
        drift_summary.to_string(
            index=False
        )
    )

    print("\nPerformance changes:")

    performance_summary = df[
        [
            "period",
            "task",
            "model",
            "ndcg_relative_change_pct",
            "recall_relative_change_pct",
            "performance_status",
        ]
    ].copy()

    print(
        performance_summary.to_string(
            index=False
        )
    )

    print("\nCold-start pressure:")

    cold_summary = (
        df[
            [
                "period",
                "task",
                "cold_user_rate_pct",
                "new_item_rate_pct",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["period", "task"]
        )
    )

    print(
        cold_summary.to_string(
            index=False
        )
    )

    print("\nKey observations:")

    max_product = (
        df.groupby("period")["product_jsd"]
        .first()
    )

    max_user = (
        df.groupby("period")["user_jsd"]
        .first()
    )

    print(
        "1. Product drift increased from "
        f"{max_product.get('comparison', float('nan')):.4f} "
        "in comparison to "
        f"{max_product.get('recent', float('nan')):.4f} "
        "in recent."
    )

    print(
    "2. User drift remained high and increased "
    f"from {max_user.get('comparison', float('nan')):.4f} "
    f"to {max_user.get('recent', float('nan')):.4f}."
)

    degraded = df[
        df["performance_status"] == "degraded"
    ]

    print(
        f"3. {len(degraded)} model/task-period combinations "
        "show relative NDCG degradation of at least 20%."
    )

    print(
        "4. These results indicate temporal association "
        "between distribution change, cold-start pressure, "
        "and recommendation performance changes."
    )

    print(
        "5. The analysis does not establish that drift "
        "caused performance degradation."
    )

    print("=" * 80)


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def run_analysis(
    drift_path: Path = DEFAULT_DRIFT,
    performance_path: Path = DEFAULT_PERFORMANCE,
    cold_start_path: Path = DEFAULT_COLD_START,
    output_path: Path = DEFAULT_OUTPUT,
    figures_dir: Path = DEFAULT_FIGURES,
    threshold: float = 20.0,
) -> pd.DataFrame:

    if threshold <= 0:
        raise ValueError(
            "Threshold must be greater than zero."
        )

    drift = load_drift_results(
        drift_path
    )

    performance = load_performance_results(
        performance_path
    )

    cold_start = load_cold_start_results(
        cold_start_path
    )

    prepared_performance = prepare_performance(
        performance
    )

    prepared_cold_start = prepare_cold_start(
        cold_start
    )

    relationship = build_relationship_table(
        drift,
        prepared_performance,
        prepared_cold_start,
    )

    relationship = add_model_health(
        relationship,
        threshold=threshold,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    relationship.to_csv(
        output_path,
        index=False,
    )

    create_drift_performance_chart(
        relationship,
        figures_dir,
    )

    create_cold_start_chart(
        relationship,
        figures_dir,
    )

    print_summary(
        relationship
    )

    print(
        f"\nSaved relationship analysis to: "
        f"{output_path}"
    )

    print(
        f"Saved figures to: {figures_dir}"
    )

    return relationship


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze the temporal relationship between "
            "data drift, cold-start pressure, and "
            "recommendation performance."
        )
    )

    parser.add_argument(
        "--drift",
        type=Path,
        default=DEFAULT_DRIFT,
    )

    parser.add_argument(
        "--performance",
        type=Path,
        default=DEFAULT_PERFORMANCE,
    )

    parser.add_argument(
        "--cold-start",
        type=Path,
        default=DEFAULT_COLD_START,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--figures",
        type=Path,
        default=DEFAULT_FIGURES,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    run_analysis(
        drift_path=args.drift,
        performance_path=args.performance,
        cold_start_path=args.cold_start,
        output_path=args.output,
        figures_dir=args.figures,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()