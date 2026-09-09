import numpy as np
import pandas as pd

from src.recommendation_pipeline import (
    ranking_metrics,
    build_training_matrix,
    make_ground_truth,
    add_degradation_metrics,
)


def test_ranking_metrics():
    recommendations = [10, 20, 30, 40, 50]
    relevant = {20, 50}

    precision, recall, ndcg, hit_rate = ranking_metrics(
        recommendations,
        relevant,
        k=5,
    )

    assert precision > 0
    assert recall > 0
    assert ndcg > 0
    assert hit_rate == 1.0


def test_build_training_matrix():
    train_df = pd.DataFrame(
        {
            "visitorid": [1, 1, 2, 2, 3, 3, 3],
            "itemid": [10, 20, 10, 30, 20, 30, 40],
            "event": [
                "view",
                "transaction",
                "view",
                "addtocart",
                "view",
                "transaction",
                "view",
            ],
        }
    )

    model, matrix = build_training_matrix(
        train_df,
        min_user_interactions=1,
        min_item_interactions=1,
        n_components=2,
    )

    assert model is not None

    assert matrix.shape[0] == len(model["user_ids"])
    assert matrix.shape[1] == len(model["item_ids"])

    assert "svd" in model
    assert "user_factors" in model
    assert "item_factors" in model
    assert "popularity_ids" in model
    assert "seen_items" in model


def test_make_ground_truth_excludes_seen_and_unknown():
    train_df = pd.DataFrame(
        {
            "visitorid": [1, 1, 2],
            "itemid": [10, 20, 10],
            "event": [
                "view",
                "view",
                "view",
            ],
        }
    )

    model, _ = build_training_matrix(
        train_df,
        min_user_interactions=1,
        min_item_interactions=1,
        n_components=1,
    )

    future_df = pd.DataFrame(
        {
            "visitorid": [1, 1, 1, 2],
            "itemid": [10, 20, 10, 20],
            "event": [
                "view",
                "view",
                "view",
                "view",
            ],
            "period": [
                "comparison",
                "comparison",
                "comparison",
                "comparison",
            ],
        }
    )

    ground_truth = make_ground_truth(
        future_df,
        "comparison",
        model,
        task="engagement",
    )

    # User 1 has already interacted with both 10 and 20.
    assert 10 not in ground_truth.get(1, set())
    assert 20 not in ground_truth.get(1, set())

    # User 2 has not interacted with item 20.
    assert 20 in ground_truth.get(2, set())


def test_purchase_ground_truth():
    # Item 20 is deliberately included in the training catalog,
    # but user 1 has not interacted with it.
    train_df = pd.DataFrame(
        {
            "visitorid": [1, 2],
            "itemid": [10, 20],
            "event": [
                "view",
                "view",
            ],
        }
    )

    model, _ = build_training_matrix(
        train_df,
        min_user_interactions=1,
        min_item_interactions=1,
        n_components=1,
    )

    future_df = pd.DataFrame(
        {
            "visitorid": [1, 1, 1],
            "itemid": [10, 20, 20],
            "event": [
                "view",
                "transaction",
                "addtocart",
            ],
            "period": [
                "comparison",
                "comparison",
                "comparison",
            ],
        }
    )

    ground_truth = make_ground_truth(
        future_df,
        "comparison",
        model,
        task="purchase",
    )

    # Only the transaction should count.
    # Item 10 is already seen by user 1, so it is excluded.
    assert ground_truth.get(1, set()) == {20}


def test_degradation_metrics():
    results = pd.DataFrame(
        {
            "period": [
                "baseline_holdout",
                "comparison",
            ],
            "task": [
                "engagement",
                "engagement",
            ],
            "model": [
                "svd",
                "svd",
            ],
            "k": [
                10,
                10,
            ],
            "ndcg_at_k": [
                0.010,
                0.008,
            ],
            "recall_at_k": [
                0.020,
                0.016,
            ],
        }
    )

    result = add_degradation_metrics(results)

    comparison = result[
        result["period"] == "comparison"
    ].iloc[0]

    assert np.isclose(
        comparison["ndcg_change_vs_baseline"],
        -0.002,
    )

    assert np.isclose(
        comparison["recall_change_vs_baseline"],
        -0.004,
    )

    assert np.isclose(
        comparison["ndcg_relative_change_pct"],
        -20.0,
    )

    assert np.isclose(
        comparison["recall_relative_change_pct"],
        -20.0,
    )