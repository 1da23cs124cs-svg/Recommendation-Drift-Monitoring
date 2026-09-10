"""
Phase 2B - Recommendation Performance Degradation Monitoring

This module monitors recommendation-model performance over time using
the results produced by Phase 2A.

It compares future-period performance against the baseline holdout
and flags degradation when the relative NDCG@10 decline reaches
the configured threshold.

Default degradation threshold:
    20%

Important:
    The threshold is a monitoring policy, not a statistical significance
    test and does not establish that data drift caused degradation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESULTS = Path("data/processed/recommendation_results.csv")
DEFAULT_OUTPUT = Path("data/processed/performance_degradation.csv")
DEFAULT_FIGURES = Path("reports/figures")

BASELINE_PERIOD = "baseline_holdout"
FUTURE_PERIODS = ["comparison", "recent"]
DEFAULT_K = 10
DEFAULT_THRESHOLD = 20.0


def load_results(results_path: Path) -> pd.DataFrame:
    """Load Phase 2A recommendation evaluation results."""

    if not results_path.exists():
        raise FileNotFoundError(
            f"Recommendation results file not found: {results_path}"
        )

    results = pd.read_csv(results_path)

    required_columns = {
        "period",
        "task",
        "model",
        "k",
        "ndcg_at_k",
        "recall_at_k",
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            "Recommendation results are missing required columns: "
            f"{sorted(missing)}"
        )

    return results


def filter_monitoring_results(
    results: pd.DataFrame,
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    """
    Keep only the periods and ranking cutoff required for monitoring.
    """

    filtered = results[
        results["k"].eq(k)
        & results["period"].isin(
            [BASELINE_PERIOD] + FUTURE_PERIODS
        )
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No monitoring results found for k={k}."
        )

    return filtered


def calculate_degradation(
    results: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Compare future-period performance with baseline holdout.

    Degradation is based primarily on relative NDCG decline.

    A result is classified as degraded when:

        ndcg_relative_change_pct <= -threshold

    Example:
        threshold = 20
        baseline NDCG = 0.010
        recent NDCG = 0.007

        relative change = -30%

        status = degraded
    """

    rows = []

    group_columns = ["task", "model"]

    for (task, model), group in results.groupby(group_columns):
        baseline = group[
            group["period"].eq(BASELINE_PERIOD)
        ]

        if baseline.empty:
            continue

        baseline_row = baseline.iloc[0]

        baseline_ndcg = float(baseline_row["ndcg_at_k"])
        baseline_recall = float(baseline_row["recall_at_k"])

        for period in FUTURE_PERIODS:
            current = group[
                group["period"].eq(period)
            ]

            if current.empty:
                continue

            current_row = current.iloc[0]

            current_ndcg = float(current_row["ndcg_at_k"])
            current_recall = float(current_row["recall_at_k"])

            ndcg_change = current_ndcg - baseline_ndcg
            recall_change = current_recall - baseline_recall

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

            if pd.notna(ndcg_relative_change):
                status = (
                    "degraded"
                    if ndcg_relative_change <= -threshold
                    else "stable"
                )
            else:
                status = "unknown"

            rows.append(
                {
                    "period": period,
                    "task": task,
                    "model": model,
                    "k": int(current_row["k"]),
                    "baseline_ndcg_at_k": baseline_ndcg,
                    "current_ndcg_at_k": current_ndcg,
                    "ndcg_change_vs_baseline": ndcg_change,
                    "ndcg_relative_change_pct": ndcg_relative_change,
                    "baseline_recall_at_k": baseline_recall,
                    "current_recall_at_k": current_recall,
                    "recall_change_vs_baseline": recall_change,
                    "recall_relative_change_pct": recall_relative_change,
                    "degradation_threshold_pct": threshold,
                    "status": status,
                }
            )

    if not rows:
        raise ValueError(
            "Could not calculate degradation. "
            "Check that baseline_holdout and future periods exist."
        )

    return pd.DataFrame(rows)


def save_results(
    degradation: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save degradation results as CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    degradation.to_csv(
        output_path,
        index=False,
    )


def create_ndcg_chart(
    results: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """Create NDCG@10 performance chart."""

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = results[
        results["k"].eq(DEFAULT_K)
    ].copy()

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

    figure_path = (
        figures_dir / "recommendation_ndcg_performance.png"
    )

    plt.figure(figsize=(10, 6))

    for (task, model), group in plot_data.groupby(
        ["task", "model"]
    ):
        group = group.sort_values("period")

        label = f"{model} - {task}"

        plt.plot(
            group["period"].astype(str),
            group["ndcg_at_k"],
            marker="o",
            label=label,
        )

    plt.title("Recommendation NDCG@10 Performance")
    plt.xlabel("Evaluation Period")
    plt.ylabel("NDCG@10")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=150,
    )

    plt.close()

    return figure_path


def create_recall_chart(
    results: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """Create Recall@10 performance chart."""

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = results[
        results["k"].eq(DEFAULT_K)
    ].copy()

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

    figure_path = (
        figures_dir / "recommendation_recall_performance.png"
    )

    plt.figure(figsize=(10, 6))

    for (task, model), group in plot_data.groupby(
        ["task", "model"]
    ):
        group = group.sort_values("period")

        label = f"{model} - {task}"

        plt.plot(
            group["period"].astype(str),
            group["recall_at_k"],
            marker="o",
            label=label,
        )

    plt.title("Recommendation Recall@10 Performance")
    plt.xlabel("Evaluation Period")
    plt.ylabel("Recall@10")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=150,
    )

    plt.close()

    return figure_path


def print_monitoring_summary(
    degradation: pd.DataFrame,
) -> None:
    """Print a concise monitoring summary."""

    print("\n" + "=" * 80)
    print("PHASE 2B - PERFORMANCE DEGRADATION MONITORING")
    print("=" * 80)

    for _, row in degradation.iterrows():

        ndcg_change = row["ndcg_relative_change_pct"]

        if pd.isna(ndcg_change):
            ndcg_text = "N/A"
        else:
            ndcg_text = f"{ndcg_change:+.2f}%"

        recall_change = row["recall_relative_change_pct"]

        if pd.isna(recall_change):
            recall_text = "N/A"
        else:
            recall_text = f"{recall_change:+.2f}%"

        print(
            f"\nPeriod : {row['period']}"
            f"\nTask   : {row['task']}"
            f"\nModel  : {row['model']}"
            f"\nNDCG@10 change : {ndcg_text}"
            f"\nRecall@10 change : {recall_text}"
            f"\nStatus : {row['status'].upper()}"
        )

    degraded_count = (
        degradation["status"]
        .eq("degraded")
        .sum()
    )

    stable_count = (
        degradation["status"]
        .eq("stable")
        .sum()
    )

    print("\n" + "-" * 80)
    print(f"Degraded combinations : {degraded_count}")
    print(f"Stable combinations   : {stable_count}")
    print("-" * 80)


def run_monitoring(
    results_path: Path = DEFAULT_RESULTS,
    output_path: Path = DEFAULT_OUTPUT,
    figures_dir: Path = DEFAULT_FIGURES,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Run the complete Phase 2B monitoring pipeline."""

    if threshold <= 0:
        raise ValueError(
            "Degradation threshold must be greater than 0."
        )

    results = load_results(results_path)

    filtered_results = filter_monitoring_results(
        results,
        k=k,
    )

    degradation = calculate_degradation(
        filtered_results,
        threshold=threshold,
    )

    save_results(
        degradation,
        output_path,
    )

    create_ndcg_chart(
        filtered_results,
        figures_dir,
    )

    create_recall_chart(
        filtered_results,
        figures_dir,
    )

    print_monitoring_summary(
        degradation,
    )

    print(
        f"\nSaved monitoring results to: {output_path}"
    )

    print(
        f"Saved figures to: {figures_dir}"
    )

    return degradation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Monitor recommendation performance degradation "
            "using Phase 2A evaluation results."
        )
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path to recommendation_results.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path",
    )

    parser.add_argument(
        "--figures",
        type=Path,
        default=DEFAULT_FIGURES,
        help="Directory for generated figures",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Recommendation cutoff to monitor",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Relative NDCG degradation threshold in percent. "
            "Default: 20"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_args()

    run_monitoring(
        results_path=args.results,
        output_path=args.output,
        figures_dir=args.figures,
        k=args.k,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()