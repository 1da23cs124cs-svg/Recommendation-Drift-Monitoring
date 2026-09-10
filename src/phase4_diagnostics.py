import pandas as pd

from recommendation_pipeline import (
    build_training_matrix,
    make_ground_truth,
    recommend_batch,
    ranking_metrics,
)


DATA_PATH = "data/processed/events_cleaned.csv"


def evaluate(recommendations, ground_truth, k=10):
    precisions = []
    recalls = []
    ndcgs = []
    hits = []

    for user_id, relevant_items in ground_truth.items():
        recs = recommendations.get(user_id, [])

        if not relevant_items:
            continue

        p, r, n, h = ranking_metrics(
            recs,
            relevant_items,
            k,
        )

        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)
        hits.append(h)

    if not precisions:
        return 0.0, 0.0, 0.0, 0.0

    return (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(ndcgs) / len(ndcgs),
        sum(hits) / len(hits),
    )


print("=" * 72)
print("PHASE 4 — DIAGNOSTIC CHECK")
print("=" * 72)

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["visitorid"] = df["visitorid"].astype("int64")
df["itemid"] = df["itemid"].astype("int64")

baseline = df[df["period"] == "baseline"].sort_values("timestamp")
comparison = df[df["period"] == "comparison"].sort_values("timestamp")
recent = df[df["period"] == "recent"].sort_values("timestamp")

# Same temporal split used by Phase 2A / Phase 4
split_time = baseline["timestamp"].quantile(0.80)

original_train = baseline[
    baseline["timestamp"] <= split_time
].copy()

adapted_train = pd.concat(
    [original_train, comparison],
    ignore_index=True,
)

print("\nTraining data")
print("-" * 72)
print("Original training rows:", len(original_train))
print("Adapted training rows:", len(adapted_train))
print("Recent rows:", len(recent))

print("\nTraining original SVD...")
original_model, _ = build_training_matrix(
    original_train,
    n_components=16,
)

print("Training adapted SVD...")
adapted_model, _ = build_training_matrix(
    adapted_train,
    n_components=16,
)

# ============================================================
# DIAGNOSTIC 1 — REPRODUCE PHASE 2A EVALUATION
# ============================================================

print("\n" + "=" * 72)
print("1. ORIGINAL MODEL — PHASE 2A STYLE GROUND TRUTH")
print("=" * 72)

for task in ["engagement", "purchase"]:

    gt = make_ground_truth(
        df,
        "recent",
        original_model,
        task,
    )

    users_with_targets = len(gt)
    total_targets = sum(len(v) for v in gt.values())

    recommendations = recommend_batch(
        original_model,
        list(gt.keys()),
        method="svd",
        max_k=10,
    )

    metrics = evaluate(
        recommendations,
        gt,
        k=10,
    )

    print(f"\n{task.upper()}")
    print("Users with targets:", users_with_targets)
    print("Total target interactions/items:", total_targets)
    print("NDCG@10:", round(metrics[2], 6))
    print("Recall@10:", round(metrics[1], 6))


# ============================================================
# DIAGNOSTIC 2 — CURRENT PHASE 4 COMMON FILTER
# ============================================================

print("\n" + "=" * 72)
print("2. CURRENT PHASE 4 COMMON-GROUND-TRUTH FILTER")
print("=" * 72)

original_users = set(original_model["user_ids"])
adapted_users = set(adapted_model["user_ids"])

original_items = set(original_model["item_ids"])
adapted_items = set(adapted_model["item_ids"])

recent_users = set(recent["visitorid"].unique())
recent_items = set(recent["itemid"].unique())

common_users = (
    recent_users
    & original_users
    & adapted_users
)

common_items = (
    recent_items
    & original_items
    & adapted_items
)

print("\nRecent unique users:", len(recent_users))
print("Original model users:", len(original_users))
print("Adapted model users:", len(adapted_users))
print("Common users:", len(common_users))

print("\nRecent unique items:", len(recent_items))
print("Original model items:", len(original_items))
print("Adapted model items:", len(adapted_items))
print("Common items:", len(common_items))


# ============================================================
# DIAGNOSTIC 3 — TARGET REDUCTION
# ============================================================

print("\n" + "=" * 72)
print("3. TARGET FILTERING EFFECT")
print("=" * 72)

for task in ["engagement", "purchase"]:

    print(f"\n--- {task.upper()} ---")

    phase2a_gt = make_ground_truth(
        df,
        "recent",
        original_model,
        task,
    )

    # Keep only common users
    common_user_gt = {
        user: items
        for user, items in phase2a_gt.items()
        if user in common_users
    }

    # Keep only items existing in both model catalogs
    common_catalog_gt = {
        user: {
            item
            for item in items
            if item in common_items
        }
        for user, items in common_user_gt.items()
    }

    # Remove items seen by adapted model too
    adapted_seen = adapted_model["seen_items"]

    strict_gt = {
        user: {
            item
            for item in items
            if item not in adapted_seen.get(user, set())
        }
        for user, items in common_catalog_gt.items()
    }

    # Remove empty target sets
    strict_gt = {
        user: items
        for user, items in strict_gt.items()
        if items
    }

    print(
        "Original Phase 2A target users:",
        len(phase2a_gt),
    )

    print(
        "After common-user filter:",
        len(common_user_gt),
    )

    print(
        "After common-catalog filter:",
        sum(bool(v) for v in common_catalog_gt.values()),
    )

    print(
        "After adapted-seen-item filter:",
        len(strict_gt),
    )

    print(
        "Phase 2A target items:",
        sum(len(v) for v in phase2a_gt.values()),
    )

    print(
        "Common-user target items:",
        sum(len(v) for v in common_user_gt.values()),
    )

    print(
        "Common-catalog target items:",
        sum(len(v) for v in common_catalog_gt.values()),
    )

    print(
        "Strict Phase 4 target items:",
        sum(len(v) for v in strict_gt.values()),
    )


# ============================================================
# DIAGNOSTIC 4 — COMPARE RECOMMENDATION HITS
# ============================================================

print("\n" + "=" * 72)
print("4. ORIGINAL VS ADAPTED — STRICT COMMON TARGETS")
print("=" * 72)

for task in ["engagement", "purchase"]:

    phase2a_gt = make_ground_truth(
        df,
        "recent",
        original_model,
        task,
    )

    strict_gt = {}

    for user, items in phase2a_gt.items():

        if user not in common_users:
            continue

        valid_items = {
            item
            for item in items
            if item in common_items
            and item not in adapted_model["seen_items"].get(
                user,
                set(),
            )
        }

        if valid_items:
            strict_gt[user] = valid_items

    original_recs = recommend_batch(
        original_model,
        list(strict_gt.keys()),
        method="svd",
        max_k=10,
    )

    adapted_recs = recommend_batch(
        adapted_model,
        list(strict_gt.keys()),
        method="svd",
        max_k=10,
    )

    original_hits = 0
    adapted_hits = 0

    for user, targets in strict_gt.items():

        original_hits += len(
            set(original_recs.get(user, []))
            & targets
        )

        adapted_hits += len(
            set(adapted_recs.get(user, []))
            & targets
        )

    original_metrics = evaluate(
        original_recs,
        strict_gt,
        10,
    )

    adapted_metrics = evaluate(
        adapted_recs,
        strict_gt,
        10,
    )

    print(f"\n{task.upper()}")

    print(
        "Users evaluated:",
        len(strict_gt),
    )

    print(
        "Original recommendation hits:",
        original_hits,
    )

    print(
        "Adapted recommendation hits:",
        adapted_hits,
    )

    print(
        "Original NDCG@10:",
        round(original_metrics[2], 6),
    )

    print(
        "Adapted NDCG@10:",
        round(adapted_metrics[2], 6),
    )

    print(
        "Original Recall@10:",
        round(original_metrics[1], 6),
    )

    print(
        "Adapted Recall@10:",
        round(adapted_metrics[1], 6),
    )


print("\n" + "=" * 72)
print("DIAGNOSTIC COMPLETE")
print("=" * 72)
