from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# E-Commerce Recommendation Health Monitor
# A plain-language dashboard for non-technical users
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(
    page_title="Recommendation System Health",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PERIOD_ORDER = ["baseline_holdout", "comparison", "recent"]
PERIOD_LABELS = {
    "baseline_holdout": "Earlier period",
    "comparison": "Middle period",
    "recent": "Latest period",
}
HEALTH_LABELS = {
    "healthy": "Healthy",
    "warning": "Needs attention",
    "degraded": "Performance has dropped",
}
ACTION_LABELS = {
    "MONITOR": "Keep watching",
    "RETRAIN": "Recommend retraining",
}
LIFECYCLE_LABELS = {
    "KEEP_CURRENT_MODEL": "Keep current model",
    "ACCEPT": "Adapted model passed checks",
    "REVIEW": "Needs human review",
    "REJECT": "Adapted model did not pass",
    "RETRAIN_PENDING": "Retraining pending",
}


# ------------------------------ Styling ------------------------------

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem;}
    .main-title {font-size: 2.1rem; font-weight: 750; margin-bottom: .15rem;}
    .subtitle {font-size: 1rem; color: #667085; margin-bottom: 1.25rem;}
    .plain-card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px; padding: 1rem 1.1rem;
        background: rgba(128,128,128,.045); min-height: 112px;
    }
    .card-label {font-size: .88rem; color: #667085; margin-bottom: .4rem;}
    .card-value {font-size: 1.45rem; font-weight: 750;}
    .helper {color: #667085; font-size: .88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------ Data loading ------------------------------

@st.cache_data
def load_data():
    required = {
        "health": DATA_DIR / "model_health_status.csv",
        "relationship": DATA_DIR / "drift_performance_relationship.csv",
        "recommendation": DATA_DIR / "recommendation_results.csv",
        "cold_start": DATA_DIR / "cold_start_summary.csv",
        "adaptation": DATA_DIR / "adaptation_decision.csv",
    }

    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))

    frames = {name: pd.read_csv(path) for name, path in required.items()}

    for name, frame in frames.items():
        for col in ("period", "task", "model", "health_status",
                    "recommended_action", "lifecycle_decision",
                    "adaptation_status", "validation_decision"):
            if col in frame.columns:
                frame[col] = frame[col].astype("string").str.strip()

    return frames


try:
    data = load_data()
except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
    st.error("Some project result files are missing or empty.")
    st.write("Run your monitoring and adaptation scripts first, then refresh this page.")
    st.code(str(exc))
    st.stop()

health_df = data["health"]
relationship_df = data["relationship"]
recommendation_df = data["recommendation"]
cold_start_df = data["cold_start"]
adaptation_df = data["adaptation"]


# ------------------------------ Header ------------------------------

st.markdown('<div class="main-title">🛍️ Recommendation System Health</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A simple overview of whether the shop’s recommendations are '
    'working well, changing over time, or need attention.</div>',
    unsafe_allow_html=True,
)

with st.expander("What does this dashboard do?", expanded=False):
    st.write(
        "An online shop recommends products to visitors. This dashboard checks whether those "
        "recommendations remain useful over time, whether the data is changing, and whether a "
        "model that was retrained performs better on a later test period."
    )
    st.caption(
        "Important: a change happening at the same time as a performance drop does not, by itself, "
        "prove that the data change caused the drop."
    )


# ------------------------------ Sidebar filters ------------------------------

st.sidebar.title("Choose what to view")
st.sidebar.caption("These filters update the numbers, charts, alerts, and tables.")

models = sorted(health_df["model"].dropna().astype(str).unique()) if "model" in health_df else []
tasks = sorted(health_df["task"].dropna().astype(str).unique()) if "task" in health_df else []
periods_available = [p for p in PERIOD_ORDER if p in set(health_df.get("period", pd.Series(dtype=str)).dropna())]

selected_model = st.sidebar.selectbox("Recommendation method", ["All methods"] + models)
selected_task = st.sidebar.selectbox("What is being measured?", ["All tasks"] + tasks)
selected_period_label = st.sidebar.selectbox(
    "Time period",
    ["All periods"] + [PERIOD_LABELS.get(p, p.title()) for p in periods_available],
)
selected_period = next(
    (p for p, label in PERIOD_LABELS.items() if label == selected_period_label),
    None,
) if selected_period_label != "All periods" else None

def apply_filters(frame, period_column=True):
    result = frame.copy()
    if selected_model != "All methods" and "model" in result.columns:
        result = result[result["model"].astype(str) == selected_model]
    if selected_task != "All tasks" and "task" in result.columns:
        result = result[result["task"].astype(str) == selected_task]
    if selected_period and period_column and "period" in result.columns:
        result = result[result["period"].astype(str) == selected_period]
    return result

filtered_health = apply_filters(health_df)
filtered_adaptation = apply_filters(adaptation_df)
filtered_performance = recommendation_df.copy()
if "k" in filtered_performance.columns:
    filtered_performance = filtered_performance[filtered_performance["k"] == 10]
filtered_performance = apply_filters(filtered_performance)
if "period" in filtered_performance.columns:
    filtered_performance = filtered_performance[
        filtered_performance["period"].isin(PERIOD_ORDER)
    ]


# ------------------------------ Overview ------------------------------

st.header("1. At a glance")

def count_health(label):
    if "health_status" not in filtered_health:
        return 0
    return int(filtered_health["health_status"].astype(str).str.lower().eq(label).sum())

degraded_count = count_health("degraded")
warning_count = count_health("warning")
healthy_count = count_health("healthy")

if filtered_health.empty:
    overall = "No matching data"
    overall_icon = "⚪"
elif degraded_count:
    overall, overall_icon = "Needs attention", "🔴"
elif warning_count:
    overall, overall_icon = "Keep an eye on it", "🟡"
else:
    overall, overall_icon = "Healthy", "🟢"

c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "Overall status", f"{overall_icon} {overall}"),
    (c2, "Healthy checks", f"🟢 {healthy_count}"),
    (c3, "Checks to watch", f"🟡 {warning_count}"),
    (c4, "Checks with a drop", f"🔴 {degraded_count}"),
]:
    with col:
        st.markdown(
            f'<div class="plain-card"><div class="card-label">{label}</div>'
            f'<div class="card-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.caption(
    "These counts refer to the rows visible under your chosen filters—not necessarily the number "
    "of unique models in production."
)


# ------------------------------ Performance ------------------------------

st.header("2. Are recommendations still useful?")
st.write(
    "**NDCG@10** checks whether useful products appear near the top of the list. "
    "**Recall@10** checks how many of the relevant products appear in the top 10."
)

if filtered_performance.empty:
    st.info("No recommendation results match these filters. Try selecting “All methods”, “All tasks”, or “All periods”.")
else:
    perf = filtered_performance.copy()
    if "period" not in perf.columns or not {"ndcg_at_k", "recall_at_k"}.issubset(perf.columns):
        st.warning("The recommendation results file does not contain the expected period/NDCG/Recall columns.")
    else:
        if selected_period:
            chosen_period = selected_period
        else:
            existing_periods = [p for p in PERIOD_ORDER if p in set(perf["period"].astype(str))]
            chosen_period = existing_periods[-1] if existing_periods else None

        current = perf[perf["period"].astype(str) == chosen_period] if chosen_period else perf.iloc[0:0]

        # When “All” is selected, avoid implying a single row is the system-wide score.
        if selected_model == "All methods" or selected_task == "All tasks":
            st.info(
                "You have selected more than one configuration. The score below is the average of "
                "the matching rows. Choose one method and one task in the sidebar for a direct comparison."
            )

        if not current.empty:
            ndcg_value = pd.to_numeric(current["ndcg_at_k"], errors="coerce").mean()
            recall_value = pd.to_numeric(current["recall_at_k"], errors="coerce").mean()
            chosen_label = PERIOD_LABELS.get(chosen_period, str(chosen_period).title())

            m1, m2, m3 = st.columns(3)
            m1.metric(f"NDCG@10 · {chosen_label}", f"{ndcg_value:.4f}" if pd.notna(ndcg_value) else "—")
            m2.metric(f"Recall@10 · {chosen_label}", f"{recall_value:.4f}" if pd.notna(recall_value) else "—")
            m3.metric("Performance alert threshold", "20% drop")
            st.caption(
                "The scores are shown on a 0–1 scale. Higher is generally better. "
                "The 20% threshold is the project’s monitoring rule, not a universal industry standard."
            )
        else:
            st.info("There are no results for this period and filter combination.")

        # Table of values used in the chart, with clear labels.
        chart_data = perf.copy()
        chart_data["period"] = chart_data["period"].astype(str)
        chart_data["period_order"] = chart_data["period"].map({p: i for i, p in enumerate(PERIOD_ORDER)})
        chart_data = chart_data.sort_values("period_order")
        chart_data["configuration"] = (
            chart_data.get("model", pd.Series("Model", index=chart_data.index)).astype(str)
            + " · "
            + chart_data.get("task", pd.Series("Task", index=chart_data.index)).astype(str)
        )
        chart_data["time_label"] = chart_data["period"].map(PERIOD_LABELS).fillna(chart_data["period"])
        chart_data["ndcg_at_k"] = pd.to_numeric(chart_data["ndcg_at_k"], errors="coerce")
        chart_data["recall_at_k"] = pd.to_numeric(chart_data["recall_at_k"], errors="coerce")

        if selected_model != "All methods" and selected_task != "All tasks":
            chart_data = chart_data[chart_data["configuration"] == f"{selected_model} · {selected_task}"]

        if selected_period:
            chart_data = chart_data[chart_data["period"] == selected_period]

        left, right = st.columns(2)
        with left:
            st.subheader("Recommendation placement")
            st.caption("NDCG@10 · higher is better")
            ndcg_plot = chart_data.pivot_table(
                index="time_label", columns="configuration", values="ndcg_at_k", aggfunc="mean"
            )
            order_labels = [PERIOD_LABELS[p] for p in PERIOD_ORDER if p in set(chart_data["period"])]
            ndcg_plot = ndcg_plot.reindex(order_labels)
            if not ndcg_plot.empty:
                st.line_chart(ndcg_plot)
            else:
                st.caption("Not enough data to draw this chart.")

        with right:
            st.subheader("Relevant products found")
            st.caption("Recall@10 · higher is better")
            recall_plot = chart_data.pivot_table(
                index="time_label", columns="configuration", values="recall_at_k", aggfunc="mean"
            )
            order_labels = [PERIOD_LABELS[p] for p in PERIOD_ORDER if p in set(chart_data["period"])]
            recall_plot = recall_plot.reindex(order_labels)
            if not recall_plot.empty:
                st.line_chart(recall_plot)
            else:
                st.caption("Not enough data to draw this chart.")

        with st.expander("See the exact performance values"):
            display = chart_data.copy()
            display["period"] = display["period"].map(PERIOD_LABELS).fillna(display["period"])
            columns = [c for c in ["period", "model", "task", "k", "ndcg_at_k", "recall_at_k"] if c in display]
            st.dataframe(display[columns].rename(columns={
                "period": "Time period", "model": "Method", "task": "Task",
                "k": "Top products checked", "ndcg_at_k": "NDCG@10", "recall_at_k": "Recall@10",
            }), use_container_width=True, hide_index=True)


# ------------------------------ Alerts ------------------------------

st.header("3. Things that may need attention")

if filtered_health.empty:
    st.info("No health-monitoring rows match the current filters.")
else:
    alert_rows = filtered_health.copy()
    alert_rows["health_status"] = alert_rows["health_status"].astype(str).str.lower()
    bad = alert_rows[alert_rows["health_status"] == "degraded"]
    warning = alert_rows[alert_rows["health_status"] == "warning"]

    if bad.empty and warning.empty:
        st.success("No warning or degraded checks appear under the current filters.")
    else:
        if not bad.empty:
            st.error("A performance drop was detected in these checks:")
            st.dataframe(
                bad[[c for c in ["period", "task", "model", "ndcg_relative_change_pct",
                                  "recall_relative_change_pct", "health_status"] if c in bad.columns]]
                .rename(columns={
                    "period": "Time period", "task": "Task", "model": "Method",
                    "ndcg_relative_change_pct": "NDCG change (%)",
                    "recall_relative_change_pct": "Recall change (%)",
                    "health_status": "Status",
                }),
                use_container_width=True, hide_index=True,
            )
        if not warning.empty:
            st.warning("Some checks are approaching or meeting the warning condition.")
            st.dataframe(
                warning[[c for c in ["period", "task", "model", "ndcg_relative_change_pct",
                                      "recall_relative_change_pct", "health_status"] if c in warning.columns]]
                .rename(columns={
                    "period": "Time period", "task": "Task", "model": "Method",
                    "ndcg_relative_change_pct": "NDCG change (%)",
                    "recall_relative_change_pct": "Recall change (%)",
                    "health_status": "Status",
                }),
                use_container_width=True, hide_index=True,
            )

st.caption(
    "A negative percentage means the score fell compared with the reference period. "
    "A positive percentage means it rose. These are associations over time, not proof of cause."
)


# ------------------------------ Adaptation / lifecycle ------------------------------

st.header("4. What happened after retraining?")
st.write(
    "When a model’s performance is marked as degraded, the system can recommend retraining. "
    "The retrained model is then checked against the original model using the same recent test targets."
)

if filtered_adaptation.empty:
    st.info("No adaptation decisions match the current filters. Check that Phase 5A has been run and the CSV is up to date.")
else:
    lifecycle = filtered_adaptation.copy()

    def readable(value, mapping=None):
        if pd.isna(value) or str(value).strip() in ("", "<NA>", "nan", "None"):
            return "Not available"
        text_value = str(value).strip()
        return mapping.get(text_value, text_value.replace("_", " ").title()) if mapping else text_value.replace("_", " ").title()

    for _, row in lifecycle.iterrows():
        model = readable(row.get("model")).upper()
        task = readable(row.get("task"))
        period = readable(row.get("period"), PERIOD_LABELS)
        decision_raw = str(row.get("lifecycle_decision", ""))
        decision = readable(row.get("lifecycle_decision"), LIFECYCLE_LABELS)
        health = readable(row.get("health_status"), HEALTH_LABELS)
        action = readable(row.get("recommended_action"), ACTION_LABELS)
        adaptation_status = readable(row.get("adaptation_status"))
        validation = readable(row.get("validation_decision"))
        ndcg_change = pd.to_numeric(pd.Series([row.get("ndcg_relative_change_pct")]), errors="coerce").iloc[0]
        recall_change = pd.to_numeric(pd.Series([row.get("recall_relative_change_pct")]), errors="coerce").iloc[0]

        icon = {
            "ACCEPT": "🟢", "KEEP_CURRENT_MODEL": "🟢",
            "REVIEW": "🟡", "REJECT": "🔴", "RETRAIN_PENDING": "🟠",
        }.get(decision_raw, "⚪")

        with st.container(border=True):
            st.subheader(f"{model} · {task} · {period}")
            st.markdown(f"### {icon} {decision}")
            a, b, c = st.columns(3)
            a.metric("NDCG change after adaptation", f"{ndcg_change:+.2f}%" if pd.notna(ndcg_change) else "—")
            b.metric("Recall change after adaptation", f"{recall_change:+.2f}%" if pd.notna(recall_change) else "—")
            c.metric("Model health before decision", health)
            st.write(f"**Suggested next step:** {action}")
            st.write(f"**Adaptation result:** {adaptation_status}")
            st.write(f"**Validation:** {validation}")
            if decision_raw == "ACCEPT":
                st.success("The adapted model passed the project’s current validation rule. This is not automatic deployment.")
            elif decision_raw == "REVIEW":
                st.warning("The measures disagree or show a mixed result, so a person should review it.")
            elif decision_raw == "REJECT":
                st.error("The adapted model did not meet the project’s recovery rule.")
            elif decision_raw == "KEEP_CURRENT_MODEL":
                st.info("The monitoring rule did not call for retraining for this row.")

    st.caption(
        "ACCEPT means the configured checks passed; it does not mean the model has been deployed. "
        "REVIEW means a human should inspect the results before deciding what to do."
    )

    with st.expander("Show all lifecycle decision details"):
        columns = [
            "period", "task", "model", "health_status", "recommended_action",
            "adaptation_status", "validation_decision", "lifecycle_decision",
            "decision_explanation",
        ]
        columns = [c for c in columns if c in lifecycle.columns]
        st.dataframe(lifecycle[columns].rename(columns={
            "period": "Time period", "task": "Task", "model": "Method",
            "health_status": "Health", "recommended_action": "Suggested action",
            "adaptation_status": "Adaptation result", "validation_decision": "Validation",
            "lifecycle_decision": "Lifecycle decision", "decision_explanation": "Why this decision?",
        }), use_container_width=True, hide_index=True)


# ------------------------------ Cold start / changing data ------------------------------

st.header("5. Are there many new users or products?")
st.write(
    "A recommendation system has less history to learn from when a visitor is new or a product has "
    "not appeared in its training data. This is commonly called the **cold-start problem**."
)

recent_relationship = relationship_df.copy()
if "period" in recent_relationship.columns:
    recent_relationship = recent_relationship[recent_relationship["period"].astype(str) == "recent"]

if recent_relationship.empty:
    st.info("No latest-period cold-start summary is available.")
else:
    row = recent_relationship.iloc[0]
    cold_users = pd.to_numeric(pd.Series([row.get("cold_user_rate_pct")]), errors="coerce").iloc[0]
    new_items = pd.to_numeric(pd.Series([row.get("new_item_rate_pct")]), errors="coerce").iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Visitors with little/no history", f"{cold_users:.2f}%" if pd.notna(cold_users) else "—")
        st.caption("A higher percentage means the system has less past activity to personalize recommendations for.")
    with c2:
        st.metric("Products not seen in training", f"{new_items:.2f}%" if pd.notna(new_items) else "—")
        st.caption("A higher percentage means more products have little or no training history.")


# ------------------------------ Data transparency ------------------------------

st.header("6. Monitoring details")
with st.expander("View the health-monitoring data"):
    detail_cols = [
        c for c in ["period", "task", "model", "ndcg_relative_change_pct",
                    "recall_relative_change_pct", "health_status"]
        if c in filtered_health.columns
    ]
    details = filtered_health[detail_cols].copy()
    details = details.rename(columns={
        "period": "Time period", "task": "Task", "model": "Method",
        "ndcg_relative_change_pct": "NDCG change (%)",
        "recall_relative_change_pct": "Recall change (%)",
        "health_status": "Health status",
    })
    st.dataframe(details, use_container_width=True, hide_index=True)

with st.expander("How to read the terms"):
    st.markdown(
        """
        - **NDCG@10:** gives more credit when relevant products appear near the top of the first 10 recommendations.
        - **Recall@10:** the share of relevant products found in the first 10 recommendations.
        - **Data drift:** the data distribution changes over time.
        - **Cold start:** the system has too little history about a visitor or product.
        - **Retraining/adaptation:** teaching the model again with newer historical data.
        - **ACCEPT / REVIEW / REJECT:** the result of the project's validation rule, not a guarantee of business success.
        """
    )

st.divider()
st.caption(
    "Recommendation Drift Monitoring System · Retailrocket evaluation dataset · "
    "Project monitoring threshold: 20% relative performance drop"
)
