"""
Phase 2A: Temporal Recommendation Baseline

Goal
----
Given a user's historical interactions, recommend products that the user
will interact with later.

Models
------
1. Popularity baseline
2. SVD latent-factor recommender

Evaluation
----------
- Temporal split:
    * First 80% of baseline -> training
    * Last 20% of baseline -> baseline_holdout
    * comparison -> future evaluation
    * recent -> future evaluation
- Tasks:
    * engagement: all future interactions
    * purchase: transactions only
- Metrics:
    * Precision@K
    * Recall@K
    * NDCG@K
    * Hit Rate@K
    * Catalog Coverage@K
- Warm-start evaluation:
    * user must exist in training
    * item must exist in training catalog
    * training-seen items are excluded from recommendations

This is a research baseline for studying recommendation degradation
under temporal data drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}

TASKS = (
    "engagement",
    "purchase",
)

METHODS = (
    "popularity",
    "svd",
)


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def build_weighted_interactions(
    train_df: pd.DataFrame,
    min_user_interactions: int = 3,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """
    Convert event types to implicit-feedback weights and filter
    users/items with insufficient history.
    """

    required = {
        "visitorid",
        "itemid",
        "event",
    }

    missing = required.difference(train_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = train_df[
        ["visitorid", "itemid", "event"]
    ].copy()

    data["weight"] = data["event"].map(EVENT_WEIGHTS)

    data = data.dropna(subset=["weight"])

    # Minimum number of event rows per user.
    user_counts = data.groupby("visitorid").size()

    valid_users = set(
        user_counts[
            user_counts >= min_user_interactions
        ].index.astype(int)
    )

    # Minimum number of event rows per item.
    item_counts = data.groupby("itemid").size()

    valid_items = set(
        item_counts[
            item_counts >= min_item_interactions
        ].index.astype(int)
    )

    data = data[
        data["visitorid"].isin(valid_users)
        & data["itemid"].isin(valid_items)
    ].copy()

    if data.empty:
        raise ValueError(
            "Training data became empty after filtering."
        )

    # Multiple events for the same user-item pair are combined.
    interactions = (
        data
        .groupby(
            ["visitorid", "itemid"],
            as_index=False
        )["weight"]
        .sum()
    )

    return interactions


# ---------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------

def build_training_matrix(
    train_df: pd.DataFrame,
    min_user_interactions: int = 3,
    min_item_interactions: int = 5,
    n_components: int = 16,
    random_state: int = 42,
) -> Tuple[dict, csr_matrix]:
    """
    Build the weighted user-item matrix and recommendation models.
    """

    interactions = build_weighted_interactions(
        train_df,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )

    user_ids = pd.Index(
        interactions["visitorid"].unique(),
        name="visitorid",
    )

    item_ids = pd.Index(
        interactions["itemid"].unique(),
        name="itemid",
    )

    user_to_idx = {
        int(user_id): index
        for index, user_id in enumerate(user_ids)
    }

    item_to_idx = {
        int(item_id): index
        for index, item_id in enumerate(item_ids)
    }

    rows = (
        interactions["visitorid"]
        .map(user_to_idx)
        .to_numpy(dtype=np.int32)
    )

    cols = (
        interactions["itemid"]
        .map(item_to_idx)
        .to_numpy(dtype=np.int32)
    )

    values = (
        interactions["weight"]
        .astype("float32")
        .to_numpy()
    )

    matrix = csr_matrix(
        (
            values,
            (rows, cols),
        ),
        shape=(
            len(user_ids),
            len(item_ids),
        ),
        dtype=np.float32,
    )

    if matrix.shape[1] != len(item_ids):
        raise RuntimeError(
            "Internal item dimension mismatch."
        )

    # -------------------------------------------------------------
    # Popularity model
    # -------------------------------------------------------------

    popularity = (
        interactions
        .groupby("itemid")["weight"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    popularity_ids = [
        int(item_id)
        for item_id in popularity.index
    ]

    # -------------------------------------------------------------
    # SVD model
    # -------------------------------------------------------------

    max_components = max(
        1,
        min(matrix.shape) - 1,
    )

    n_components = min(
        int(n_components),
        max_components,
    )

    svd = TruncatedSVD(
        n_components=n_components,
        n_iter=10,
        random_state=random_state,
    )

    user_factors = (
        svd.fit_transform(matrix)
        .astype("float32")
    )

    # -------------------------------------------------------------
    # Seen-item dictionary
    # -------------------------------------------------------------

    seen_items: Dict[int, Set[int]] = {}

    for row in interactions.itertuples(
        index=False
    ):
        user_id = int(row.visitorid)
        item_id = int(row.itemid)

        seen_items.setdefault(
            user_id,
            set(),
        ).add(item_id)

    model = {
        "svd": svd,

        "user_factors": user_factors,

        "item_factors": (
            svd.components_
            .astype("float32")
        ),

        "user_ids": (
            user_ids
            .to_numpy(dtype=np.int64)
        ),

        "item_ids": (
            item_ids
            .to_numpy(dtype=np.int64)
        ),

        "user_to_idx": user_to_idx,

        "item_to_idx": item_to_idx,

        "seen_items": seen_items,

        "popularity_ids": popularity_ids,

        "matrix": matrix,

        "config": {
            "min_user_interactions":
                int(min_user_interactions),

            "min_item_interactions":
                int(min_item_interactions),

            "n_components":
                int(n_components),

            "event_weights":
                EVENT_WEIGHTS,
        },
    }

    return model, matrix


# ---------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------

def make_ground_truth(
    df: pd.DataFrame,
    period: str,
    model: dict,
    task: str = "engagement",
) -> Dict[int, Set[int]]:
    """
    Construct future relevant items.

    Only warm users and items from the training catalog are included.
    """

    if task not in TASKS:
        raise ValueError(
            f"Unknown task: {task}"
        )

    test = df.loc[
        df["period"] == period,
        [
            "visitorid",
            "itemid",
            "event",
        ],
    ].copy()

    if task == "purchase":
        test = test[
            test["event"] == "transaction"
        ]

    catalog = set(
        int(item_id)
        for item_id in model["item_ids"]
    )

    known_users = model["user_to_idx"]

    seen_items = model["seen_items"]

    ground_truth: Dict[
        int,
        Set[int]
    ] = {}

    for row in test.itertuples(
        index=False
    ):

        user_id = int(row.visitorid)
        item_id = int(row.itemid)

        # Warm user.
        if user_id not in known_users:
            continue

        # Known training catalog item.
        if item_id not in catalog:
            continue

        # Exclude items already seen during training.
        if item_id in seen_items.get(
            user_id,
            set(),
        ):
            continue

        ground_truth.setdefault(
            user_id,
            set(),
        ).add(item_id)

    return ground_truth


# ---------------------------------------------------------------------
# Recommendation functions
# ---------------------------------------------------------------------

def topk_popularity(
    model: dict,
    user_id: int,
    k: int,
) -> List[int]:

    seen = model[
        "seen_items"
    ].get(
        user_id,
        set(),
    )

    recommendations = []

    for item_id in model[
        "popularity_ids"
    ]:

        if item_id not in seen:
            recommendations.append(
                item_id
            )

        if len(recommendations) >= k:
            break

    return recommendations


def _top_indices(
    scores: np.ndarray,
    item_ids: np.ndarray,
    k: int,
) -> List[int]:

    if k <= 0:
        return []

    finite = np.isfinite(scores)

    candidates = np.flatnonzero(
        finite
    )

    if len(candidates) == 0:
        return []

    k_eff = min(
        k,
        len(candidates),
    )

    candidate_scores = scores[
        candidates
    ]

    selected = np.argpartition(
        candidate_scores,
        -k_eff,
    )[-k_eff:]

    selected = selected[
        np.argsort(
            candidate_scores[selected]
        )[::-1]
    ]

    return [
        int(item_ids[candidates[index]])
        for index in selected
    ]


def topk_svd(
    model: dict,
    user_id: int,
    k: int,
) -> List[int]:

    user_index = model[
        "user_to_idx"
    ][user_id]

    scores = (
        model["user_factors"][user_index]
        @ model["item_factors"]
    )

    seen = model[
        "seen_items"
    ].get(
        user_id,
        set(),
    )

    for item_id in seen:

        item_index = model[
            "item_to_idx"
        ].get(item_id)

        if item_index is not None:
            scores[item_index] = -np.inf

    return _top_indices(
        scores,
        model["item_ids"],
        k,
    )


# ---------------------------------------------------------------------
# Batch recommendation
# ---------------------------------------------------------------------

def recommend_batch(
    model: dict,
    user_ids: List[int],
    method: str,
    max_k: int,
) -> Dict[int, List[int]]:

    if method == "popularity":

        return {
            user_id:
                topk_popularity(
                    model,
                    user_id,
                    max_k,
                )
            for user_id in user_ids
        }

    if method != "svd":
        raise ValueError(
            f"Unknown recommendation method: {method}"
        )

    user_indices = np.asarray(
        [
            model["user_to_idx"][user_id]
            for user_id in user_ids
        ],
        dtype=np.int32,
    )

    recommendations = {}

    batch_size = 256

    item_ids = model["item_ids"]

    for start in range(
        0,
        len(user_ids),
        batch_size,
    ):

        batch_indices = (
            user_indices[
                start:
                start + batch_size
            ]
        )

        scores = (
            model["user_factors"][
                batch_indices
            ]
            @ model["item_factors"]
        )

        batch_users = user_ids[
            start:
            start + batch_size
        ]

        for local_index, user_id in enumerate(
            batch_users
        ):

            row_scores = (
                scores[local_index]
                .copy()
            )

            seen = model[
                "seen_items"
            ].get(
                user_id,
                set(),
            )

            for item_id in seen:

                item_index = model[
                    "item_to_idx"
                ].get(item_id)

                if item_index is not None:
                    row_scores[
                        item_index
                    ] = -np.inf

            recommendations[user_id] = (
                _top_indices(
                    row_scores,
                    item_ids,
                    max_k,
                )
            )

    return recommendations


# ---------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------

def ranking_metrics(
    recommended: Iterable[int],
    relevant: Set[int],
    k: int,
) -> Tuple[
    float,
    float,
    float,
    float,
]:

    recommendations = list(
        recommended
    )[:k]

    hits = np.asarray(
        [
            1
            if item in relevant
            else 0
            for item in recommendations
        ],
        dtype=float,
    )

    hit_count = float(
        hits.sum()
    )

    precision = (
        hit_count / k
        if k > 0
        else 0.0
    )

    recall = (
        hit_count / len(relevant)
        if relevant
        else 0.0
    )

    dcg = float(
        sum(
            hit /
            np.log2(rank + 2)
            for rank, hit
            in enumerate(hits)
        )
    )

    ideal_length = min(
        len(relevant),
        k,
    )

    idcg = float(
        sum(
            1 /
            np.log2(rank + 2)
            for rank
            in range(ideal_length)
        )
    )

    ndcg = (
        dcg / idcg
        if idcg > 0
        else 0.0
    )

    hit_rate = float(
        hit_count > 0
    )

    return (
        precision,
        recall,
        ndcg,
        hit_rate,
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_model(
    model: dict,
    df: pd.DataFrame,
    period: str,
    method: str,
    task: str,
    k_values: Tuple[int, ...],
    max_eval_users: int | None = None,
    random_state: int = 42,
) -> List[dict]:

    if method not in METHODS:
        raise ValueError(
            f"Unknown method: {method}"
        )

    ground_truth = make_ground_truth(
        df,
        period,
        model,
        task,
    )

    users = sorted(
        ground_truth
    )

    if (
        max_eval_users is not None
        and len(users) > max_eval_users
    ):

        rng = np.random.default_rng(
            random_state
        )

        users = sorted(
            rng.choice(
                np.asarray(users),
                size=max_eval_users,
                replace=False,
            ).tolist()
        )

        ground_truth = {
            user_id:
                ground_truth[user_id]
            for user_id in users
        }

    if not users:

        return [
            {
                "period": period,
                "task": task,
                "model": method,
                "k": int(k),
                "n_eval_users": 0,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "ndcg_at_k": 0.0,
                "hit_rate_at_k": 0.0,
                "catalog_coverage_at_k": 0.0,
            }
            for k in k_values
        ]

    max_k = max(k_values)

    recommendations = recommend_batch(
        model,
        users,
        method,
        max_k,
    )

    results = []

    for k in k_values:

        metric_rows = []

        recommended_catalog = set()

        for user_id in users:

            recs = recommendations[
                user_id
            ][:k]

            recommended_catalog.update(
                recs
            )

            metric_rows.append(
                ranking_metrics(
                    recs,
                    ground_truth[user_id],
                    k,
                )
            )

        metric_array = np.asarray(
            metric_rows,
            dtype=float,
        )

        results.append(
            {
                "period": period,
                "task": task,
                "model": method,
                "k": int(k),
                "n_eval_users": len(users),

                "precision_at_k":
                    float(
                        metric_array[:, 0].mean()
                    ),

                "recall_at_k":
                    float(
                        metric_array[:, 1].mean()
                    ),

                "ndcg_at_k":
                    float(
                        metric_array[:, 2].mean()
                    ),

                "hit_rate_at_k":
                    float(
                        metric_array[:, 3].mean()
                    ),

                "catalog_coverage_at_k":
                    float(
                        len(recommended_catalog)
                        /
                        len(model["item_ids"])
                    ),
            }
        )

    return results


# ---------------------------------------------------------------------
# Cold-start analysis
# ---------------------------------------------------------------------

def cold_start_summary(
    df: pd.DataFrame,
    period: str,
    model: dict,
    task: str,
) -> dict:

    test = df.loc[
        df["period"] == period,
        [
            "visitorid",
            "itemid",
            "event",
        ],
    ]

    if task == "purchase":

        test = test[
            test["event"]
            == "transaction"
        ]

    known_users = set(
        model["user_to_idx"]
    )

    known_items = set(
        int(item_id)
        for item_id in model["item_ids"]
    )

    test_users = set(
        test["visitorid"]
        .astype(int)
    )

    test_items = set(
        test["itemid"]
        .astype(int)
    )

    return {
        "period": period,
        "task": task,

        "total_test_users":
            len(test_users),

        "warm_users":
            len(
                test_users
                & known_users
            ),

        "cold_users":
            len(
                test_users
                - known_users
            ),

        "total_test_items":
            len(test_items),

        "known_catalog_items":
            len(
                test_items
                & known_items
            ),

        "new_items":
            len(
                test_items
                - known_items
            ),
    }


# ---------------------------------------------------------------------
# Degradation analysis
# ---------------------------------------------------------------------

def add_degradation_metrics(
    results: pd.DataFrame,
) -> pd.DataFrame:

    output = results.copy()

    output[
        "ndcg_change_vs_baseline"
    ] = np.nan

    output[
        "ndcg_relative_change_pct"
    ] = np.nan

    output[
        "recall_change_vs_baseline"
    ] = np.nan

    output[
        "recall_relative_change_pct"
    ] = np.nan

    baseline = (
        output[
            output["period"]
            == "baseline_holdout"
        ]
        .set_index(
            [
                "task",
                "model",
                "k",
            ]
        )
    )

    for index, row in output.iterrows():

        key = (
            row["task"],
            row["model"],
            row["k"],
        )

        if (
            key not in baseline.index
            or row["period"]
            == "baseline_holdout"
        ):
            continue

        baseline_ndcg = float(
            baseline.loc[
                key,
                "ndcg_at_k",
            ]
        )

        baseline_recall = float(
            baseline.loc[
                key,
                "recall_at_k",
            ]
        )

        output.at[
            index,
            "ndcg_change_vs_baseline",
        ] = (
            row["ndcg_at_k"]
            - baseline_ndcg
        )

        if baseline_ndcg > 0:

            output.at[
                index,
                "ndcg_relative_change_pct",
            ] = (
                (
                    row["ndcg_at_k"]
                    /
                    baseline_ndcg
                )
                - 1
            ) * 100

        output.at[
            index,
            "recall_change_vs_baseline",
        ] = (
            row["recall_at_k"]
            - baseline_recall
        )

        if baseline_recall > 0:

            output.at[
                index,
                "recall_relative_change_pct",
            ] = (
                (
                    row["recall_at_k"]
                    /
                    baseline_recall
                )
                - 1
            ) * 100

    return output


# ---------------------------------------------------------------------
# Complete experiment
# ---------------------------------------------------------------------

def run_experiment(
    data_path: str,
    model_dir: str,
    results_path: str,
    cold_start_path: str,
    components: int = 16,
    k_values: Tuple[int, ...] = (
        5,
        10,
    ),
    max_eval_users: int | None = 5000,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    df = pd.read_csv(
        data_path,
        usecols=[
            "timestamp",
            "visitorid",
            "itemid",
            "event",
            "period",
        ],
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "visitorid",
            "itemid",
            "event",
            "period",
        ]
    )

    # -------------------------------------------------------------
    # Temporal baseline split
    # -------------------------------------------------------------

    baseline = (
        df[
            df["period"]
            == "baseline"
        ]
        .sort_values(
            "timestamp"
        )
    )

    if baseline.empty:
        raise ValueError(
            "No baseline rows found."
        )

    split_timestamp = float(
        baseline[
            "timestamp"
        ].quantile(0.80)
    )

    train_df = baseline[
        baseline["timestamp"]
        <= split_timestamp
    ].copy()

    holdout_df = baseline[
        baseline["timestamp"]
        > split_timestamp
    ].copy()

    holdout_df["period"] = (
        "baseline_holdout"
    )

    # -------------------------------------------------------------
    # Build recommendation model
    # -------------------------------------------------------------

    model, matrix = (
        build_training_matrix(
            train_df,
            n_components=components,
        )
    )

    # -------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------

    model_path = Path(
        model_dir
    )

    model_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path
        / "recommendation_models.joblib",
        compress=3,
    )

    # -------------------------------------------------------------
    # Evaluation data
    # -------------------------------------------------------------

    evaluation_df = pd.concat(
        [
            holdout_df,
            df[
                df["period"].isin(
                    [
                        "comparison",
                        "recent",
                    ]
                )
            ],
        ],
        ignore_index=True,
    )

    periods = [
        "baseline_holdout",
        "comparison",
        "recent",
    ]

    results = []

    cold_rows = []

    # -------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------

    for period in periods:

        for task in TASKS:

            cold_rows.append(
                cold_start_summary(
                    evaluation_df,
                    period,
                    model,
                    task,
                )
            )

            for method in METHODS:

                results.extend(
                    evaluate_model(
                        model,
                        evaluation_df,
                        period,
                        method,
                        task,
                        k_values,
                        max_eval_users=max_eval_users,
                    )
                )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        add_degradation_metrics(
            results_df
        )
    )

    cold_df = pd.DataFrame(
        cold_rows
    )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    Path(
        results_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    Path(
        cold_start_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cold_df.to_csv(
        cold_start_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Save experiment metadata
    # -------------------------------------------------------------

    metadata = {

        "training_period":
            "first_80_percent_of_baseline",

        "baseline_train_cut_timestamp":
            int(split_timestamp),

        "baseline_train_end":
            str(
                pd.to_datetime(
                    split_timestamp,
                    unit="ms",
                )
            ),

        "baseline_train_interactions":
            int(len(train_df)),

        "baseline_holdout_interactions":
            int(len(holdout_df)),

        "matrix_shape":
            list(matrix.shape),

        "nonzero_interactions":
            int(matrix.nnz),

        "explained_variance_ratio":
            float(
                model[
                    "svd"
                ]
                .explained_variance_ratio_
                .sum()
            ),

        "model_config":
            model["config"],

        "evaluation_tasks":
            list(TASKS),

        "evaluation_methods":
            list(METHODS),

        "k_values":
            list(k_values),

        "max_eval_users_per_period_task":
            max_eval_users,

        "warm_start_definition":
            (
                "User and item must occur "
                "in training; training-seen "
                "items are excluded."
            ),

        "results_file":
            results_path,

        "cold_start_file":
            cold_start_path,
    }

    (
        model_path
        / "training_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        results_df,
        cold_df,
    )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 2A temporal "
            "recommendation evaluation."
        )
    )

    parser.add_argument(
        "--data",
        default=(
            "data/processed/"
            "events_cleaned.csv"
        ),
    )

    parser.add_argument(
        "--model-dir",
        default="models",
    )

    parser.add_argument(
        "--results",
        default=(
            "data/processed/"
            "recommendation_results.csv"
        ),
    )

    parser.add_argument(
        "--cold-start",
        default=(
            "data/processed/"
            "cold_start_summary.csv"
        ),
    )

    parser.add_argument(
        "--components",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[
            5,
            10,
        ],
    )

    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=5000,
        help=(
            "Maximum users sampled per "
            "period/task. Use 0 for all."
        ),
    )

    args = parser.parse_args()

    results, cold = run_experiment(
        data_path=args.data,
        model_dir=args.model_dir,
        results_path=args.results,
        cold_start_path=args.cold_start,
        components=args.components,
        k_values=tuple(
            sorted(
                set(args.k)
            )
        ),
        max_eval_users=(
            None
            if args.max_eval_users == 0
            else args.max_eval_users
        ),
    )

    print(
        "\nRecommendation results:"
    )

    print(
        results.to_string(
            index=False
        )
    )

    print(
        "\nCold-start summary:"
    )

    print(
        cold.to_string(
            index=False
        )
    )

    print(
        "\nSaved model:"
        f" {Path(args.model_dir) / 'recommendation_models.joblib'}"
    )

    print(
        "\nSaved results:"
        f" {args.results}"
    )

    print(
        "\nSaved cold-start summary:"
        f" {args.cold_start}"
    )


if __name__ == "__main__":
    main()