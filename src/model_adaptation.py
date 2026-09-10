"""
Phase 4 — Model Adaptation & Retraining

Historical adaptation simulation:

1. Train the original SVD model on the first 80% of baseline.
2. Simulate a monitoring trigger after the comparison period.
3. Retrain using baseline training data + full comparison period.
4. Evaluate BOTH models on the SAME unseen recent-period test set.
5. Compare NDCG@10 and Recall@10.

Important:
- Recent data is NEVER used for training.
- Before/after models use identical evaluation users and targets.
- The experiment measures temporal adaptation performance.
- It does not establish that drift caused degradation.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from src.recommendation_pipeline import (
        build_training_matrix,
        make_ground_truth,
        recommend_batch,
        ranking_metrics,
    )
except ModuleNotFoundError:
    from recommendation_pipeline import (
        build_training_matrix,
        make_ground_truth,
        recommend_batch,
        ranking_metrics,
    )


DATA_PATH = Path("data/processed/events_cleaned.csv")
MODEL_DIR = Path("models")
RESULTS_PATH = Path("data/processed/adaptation_results.csv")

N_COMPONENTS = 16
K = 10


def evaluate_model(model, ground_truth):
    """
    Evaluate a model on a fixed ground-truth dictionary.

    Returns:
        precision, recall, ndcg, hit_rate
    """

    if not ground_truth:
        return 0.0, 0.0, 0.0, 0.0

    users = list(ground_truth.keys())

    recommendations = recommend_batch(
        model,
        users,
        method="svd",
        max_k=K,
    )

    precisions = []
    recalls = []
    ndcgs = []
    hit_rates = []

    for user_id, relevant_items in ground_truth.items():

        if not relevant_items:
            continue

        recommended_items = recommendations.get(
            user_id,
            [],
        )

        precision, recall, ndcg, hit_rate = ranking_metrics(
            recommended_items,
            relevant_items,
            K,
        )

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        hit_rates.append(hit_rate)

    if not ndcgs:
        return 0.0, 0.0, 0.0, 0.0

    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(ndcgs)),
        float(np.mean(hit_rates)),
    )


def calculate_change(before, after):
    """
    Calculate relative percentage change.

    Returns NaN when the baseline metric is zero.
    """

    if before == 0:
        return np.nan

    return ((after - before) / before) * 100.0


def classify_adaptation(
    before_ndcg,
    after_ndcg,
    before_recall,
    after_recall,
):
    """
    Classify adaptation outcome.

    Rules:
    - IMPROVED_FROM_ZERO:
        both baseline metrics are zero and both become positive.
    - RECOVERED:
        both NDCG and Recall improve.
    - PARTIAL_RECOVERY:
        exactly one metric improves.
    - NOT_RECOVERED:
        neither metric improves.
    """

    ndcg_improved = after_ndcg > before_ndcg
    recall_improved = after_recall > before_recall

    if (
        before_ndcg == 0
        and before_recall == 0
        and after_ndcg > 0
        and after_recall > 0
    ):
        return "IMPROVED_FROM_ZERO"

    if ndcg_improved and recall_improved:
        return "RECOVERED"

    if ndcg_improved or recall_improved:
        return "PARTIAL_RECOVERY"

    return "NOT_RECOVERED"


def build_fixed_ground_truth(
    df,
    original_model,
    adapted_model,
    task,
):
    """
    Build ONE fixed recent-period evaluation set.

    The same users and target items are used for both
    original and adapted models.

    We start from the recent-period ground truth and
    retain only users that exist in BOTH model catalogs.

    We do NOT remove targets merely because the adapted
    model has seen them during training. Otherwise the
    definition of the test target would depend on the
    model being evaluated.
    """

    original_users = set(original_model["user_ids"])
    adapted_users = set(adapted_model["user_ids"])

    common_users = (
        original_users
        & adapted_users
    )

    recent_ground_truth = make_ground_truth(
        df,
        "recent",
        original_model,
        task,
    )

    fixed_ground_truth = {
        user_id: set(items)
        for user_id, items in recent_ground_truth.items()
        if user_id in common_users
    }

    fixed_ground_truth = {
        user_id: items
        for user_id, items in fixed_ground_truth.items()
        if items
    }

    return fixed_ground_truth


def main():

    print("=" * 72)
    print("PHASE 4 — MODEL ADAPTATION & RETRAINING")
    print("=" * 72)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["visitorid"] = df["visitorid"].astype(
        "int64"
    )

    df["itemid"] = df["itemid"].astype(
        "int64"
    )

    # --------------------------------------------------------
    # Temporal periods
    # --------------------------------------------------------

    baseline = (
        df[df["period"] == "baseline"]
        .sort_values("timestamp")
        .copy()
    )

    comparison = (
        df[df["period"] == "comparison"]
        .sort_values("timestamp")
        .copy()
    )

    recent = (
        df[df["period"] == "recent"]
        .sort_values("timestamp")
        .copy()
    )

    # --------------------------------------------------------
    # Original training = first 80% of baseline
    # --------------------------------------------------------

    split_time = baseline["timestamp"].quantile(
        0.80
    )

    original_train = baseline[
        baseline["timestamp"] <= split_time
    ].copy()

    # --------------------------------------------------------
    # Adapted training = original + comparison
    # --------------------------------------------------------

    adapted_train = pd.concat(
        [
            original_train,
            comparison,
        ],
        ignore_index=True,
    )

    print(
        "\nAdaptation trigger: "
        "comparison-period SVD degradation detected."
    )

    print(
        "Triggered configuration: "
        "SVD recommendation model"
    )

    print(
        "\nOriginal training rows:",
        len(original_train),
    )

    print(
        "Adapted training rows "
        "(baseline + comparison):",
        len(adapted_train),
    )

    print(
        "Recent evaluation rows "
        "(completely unseen during adaptation):",
        len(recent),
    )

    # --------------------------------------------------------
    # Train original model
    # --------------------------------------------------------

    print("\nTraining original SVD model...")

    original_model, _ = build_training_matrix(
        original_train,
        n_components=N_COMPONENTS,
    )

    print("Original SVD training complete.")

    # --------------------------------------------------------
    # Train adapted model
    # --------------------------------------------------------

    print("Training adapted SVD model...")

    adapted_model, _ = build_training_matrix(
        adapted_train,
        n_components=N_COMPONENTS,
    )

    print("Adapted SVD training complete.")

    # --------------------------------------------------------
    # Save adapted model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapted_model_path = (
        MODEL_DIR / "adapted_svd_model.joblib"
    )

    joblib.dump(
        adapted_model,
        adapted_model_path,
    )

    print(
        "\nAdapted model saved to:",
        adapted_model_path,
    )

    print(
        "\nOriginal model matrix shape:",
        original_model["matrix"].shape,
    )

    print(
        "Adapted model matrix shape:",
        adapted_model["matrix"].shape,
    )

    # --------------------------------------------------------
    # Evaluate engagement and purchase
    # --------------------------------------------------------

    results = []

    for task in [
        "engagement",
        "purchase",
    ]:

        print("\n" + "-" * 72)
        print(
            f"{task.upper()} — FIXED RECENT TEST SET"
        )
        print("-" * 72)

        fixed_ground_truth = build_fixed_ground_truth(
            df,
            original_model,
            adapted_model,
            task,
        )

        print(
            "Common evaluation users:",
            len(fixed_ground_truth),
        )

        print(
            "Total target items:",
            sum(
                len(items)
                for items in fixed_ground_truth.values()
            ),
        )

        # ----------------------------------------------------
        # BEFORE
        # ----------------------------------------------------

        (
            before_precision,
            before_recall,
            before_ndcg,
            before_hit_rate,
        ) = evaluate_model(
            original_model,
            fixed_ground_truth,
        )

        # ----------------------------------------------------
        # AFTER
        # ----------------------------------------------------

        (
            after_precision,
            after_recall,
            after_ndcg,
            after_hit_rate,
        ) = evaluate_model(
            adapted_model,
            fixed_ground_truth,
        )

        # ----------------------------------------------------
        # Changes
        # ----------------------------------------------------

        ndcg_change = calculate_change(
            before_ndcg,
            after_ndcg,
        )

        recall_change = calculate_change(
            before_recall,
            after_recall,
        )

        status = classify_adaptation(
            before_ndcg,
            after_ndcg,
            before_recall,
            after_recall,
        )

        print(
            "\nBefore NDCG@10:",
            f"{before_ndcg:.6f}",
        )

        print(
            "After NDCG@10:",
            f"{after_ndcg:.6f}",
        )

        if pd.notna(ndcg_change):
            print(
                "NDCG change:",
                f"{ndcg_change:+.2f}%",
            )
        else:
            print(
                "NDCG change:",
                "N/A (baseline was zero)",
            )

        print(
            "\nBefore Recall@10:",
            f"{before_recall:.6f}",
        )

        print(
            "After Recall@10:",
            f"{after_recall:.6f}",
        )

        if pd.notna(recall_change):
            print(
                "Recall change:",
                f"{recall_change:+.2f}%",
            )
        else:
            print(
                "Recall change:",
                "N/A (baseline was zero)",
            )

        print(
            "\nAdaptation result:",
            status,
        )

        results.append(
            {
                "task": task,
                "model": "svd",
                "evaluation_period": "recent",
                "evaluation_users": len(
                    fixed_ground_truth
                ),
                "target_items": sum(
                    len(items)
                    for items in fixed_ground_truth.values()
                ),
                "before_precision_at_10":
                    before_precision,
                "after_precision_at_10":
                    after_precision,
                "before_recall_at_10":
                    before_recall,
                "after_recall_at_10":
                    after_recall,
                "before_ndcg_at_10":
                    before_ndcg,
                "after_ndcg_at_10":
                    after_ndcg,
                "before_hit_rate_at_10":
                    before_hit_rate,
                "after_hit_rate_at_10":
                    after_hit_rate,
                "ndcg_relative_change_pct":
                    ndcg_change,
                "recall_relative_change_pct":
                    recall_change,
                "adaptation_status":
                    status,
                "adaptation_training":
                    "baseline first 80% + full comparison",
                "evaluation_data":
                    "recent",
                "causal_note":
                    "Temporal adaptation experiment; "
                    "does not establish that drift caused "
                    "performance degradation.",
            }
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        "\nAdaptation results saved to:",
        RESULTS_PATH,
    )

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    print(
        "The adapted model was trained using "
        "baseline training data plus the full "
        "comparison period."
    )

    print(
        "Both original and adapted models were "
        "evaluated on the same unseen recent-period "
        "test population."
    )

    print(
        "The recent period was not used during "
        "adaptation training."
    )

    print(
        "Percentage improvement is reported as "
        "N/A when the original metric is zero."
    )

    print(
        "The experiment measures temporal adaptation "
        "performance and does not establish that "
        "drift caused the original degradation."
    )


if __name__ == "__main__":
    main()