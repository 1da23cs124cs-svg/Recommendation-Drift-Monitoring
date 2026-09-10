from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(
    page_title="Recommendation Health Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.25rem;
        margin-bottom: 0.8rem;
    }

    .status-card {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.20);
        background: rgba(128, 128, 128, 0.05);
        min-height: 112px;
    }

    .status-label {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 0.3rem;
    }

    .status-value {
        font-size: 1.5rem;
        font-weight: 650;
    }

    .finding-card {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        background: rgba(128, 128, 128, 0.04);
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .finding-title {
        font-size: 0.95rem;
        font-weight: 650;
        margin-bottom: 0.35rem;
    }

    .finding-text {
        font-size: 0.9rem;
        color: #4b5563;
        line-height: 1.5;
    }

    .small-text {
        color: #6b7280;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    health = pd.read_csv(
        DATA_DIR / "model_health_status.csv"
    )

    relationship = pd.read_csv(
        DATA_DIR / "drift_performance_relationship.csv"
    )

    recommendation = pd.read_csv(
        DATA_DIR / "recommendation_results.csv"
    )

    cold_start = pd.read_csv(
        DATA_DIR / "cold_start_summary.csv"
    )

    return (
        health,
        relationship,
        recommendation,
        cold_start,
    )


try:

    (
        health_df,
        relationship_df,
        recommendation_df,
        cold_start_df,
    ) = load_data()

except FileNotFoundError as error:

    st.error("Required project output is missing.")

    st.code(str(error))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'E-Commerce Recommendation Health Monitor'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Monitor data drift, recommendation performance, '
    'cold-start pressure, and model health'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Monitoring Controls")

st.sidebar.caption(
    "Use these filters to investigate model behaviour."
)


available_models = sorted(
    health_df["model"].unique()
)

available_tasks = sorted(
    health_df["task"].unique()
)

available_periods = sorted(
    health_df["period"].unique()
)


selected_model = st.sidebar.selectbox(
    "Model",
    ["All"] + available_models,
)

selected_task = st.sidebar.selectbox(
    "Task",
    ["All"] + available_tasks,
)

selected_period = st.sidebar.selectbox(
    "Evaluation Period",
    ["All"] + available_periods,
)


# ============================================================
# FILTER HEALTH DATA
# ============================================================

filtered_health = health_df.copy()


if selected_model != "All":

    filtered_health = filtered_health[
        filtered_health["model"] == selected_model
    ]


if selected_task != "All":

    filtered_health = filtered_health[
        filtered_health["task"] == selected_task
    ]


if selected_period != "All":

    filtered_health = filtered_health[
        filtered_health["period"] == selected_period
    ]


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">System Overview</div>',
    unsafe_allow_html=True,
)


degraded_count = int(
    filtered_health["health_status"]
    .eq("degraded")
    .sum()
)

warning_count = int(
    filtered_health["health_status"]
    .eq("warning")
    .sum()
)

healthy_count = int(
    filtered_health["health_status"]
    .eq("healthy")
    .sum()
)


if degraded_count > 0:

    overall_status = "DEGRADED"
    status_icon = "🔴"

elif warning_count > 0:

    overall_status = "WARNING"
    status_icon = "🟡"

else:

    overall_status = "HEALTHY"
    status_icon = "🟢"


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Overall Model Health</div>
            <div class="status-value">
                {status_icon} {overall_status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Healthy Models</div>
            <div class="status-value">
                🟢 {healthy_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Warnings</div>
            <div class="status-value">
                🟡 {warning_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with col4:

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Degraded</div>
            <div class="status-value">
                🔴 {degraded_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PERFORMANCE DATA
# ============================================================

performance = recommendation_df[
    recommendation_df["k"] == 10
].copy()


if selected_model != "All":

    performance = performance[
        performance["model"] == selected_model
    ]


if selected_task != "All":

    performance = performance[
        performance["task"] == selected_task
    ]


performance = performance[
    performance["period"].isin(
        [
            "baseline_holdout",
            "comparison",
            "recent",
        ]
    )
]


if selected_period != "All":

    performance = performance[
        performance["period"] == selected_period
    ]


# ============================================================
# CURRENT PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">Current Performance</div>',
    unsafe_allow_html=True,
)


if not performance.empty:

    latest_period_order = {
        "baseline_holdout": 0,
        "comparison": 1,
        "recent": 2,
    }

    latest_period = max(
        performance["period"],
        key=lambda value: latest_period_order.get(value, -1),
    )

    latest_data = performance[
        performance["period"] == latest_period
    ].copy()

    latest_ndcg = float(
        latest_data["ndcg_at_k"].mean()
    )

    latest_recall = float(
        latest_data["recall_at_k"].mean()
    )

    period_label = {
        "baseline_holdout": "Baseline",
        "comparison": "Comparison",
        "recent": "Recent",
    }.get(
        latest_period,
        latest_period.title(),
    )

else:

    latest_period = None
    latest_ndcg = None
    latest_recall = None
    period_label = None


perf1, perf2, perf3 = st.columns(3)


with perf1:

    if latest_ndcg is not None:

        st.metric(
            f"NDCG@10 • {period_label}",
            f"{latest_ndcg:.4f}",
            help=(
                "Mean NDCG@10 across the configurations "
                "selected by the current filters."
            ),
        )

    else:

        st.metric(
            "NDCG@10",
            "—",
        )


with perf2:

    if latest_recall is not None:

        st.metric(
            f"Recall@10 • {period_label}",
            f"{latest_recall:.4f}",
            help=(
                "Mean Recall@10 across the configurations "
                "selected by the current filters."
            ),
        )

    else:

        st.metric(
            "Recall@10",
            "—",
        )


with perf3:

    st.metric(
        "Monitoring Threshold",
        "20%",
        help=(
            "Relative NDCG or Recall degradation threshold "
            "used by the monitoring policy."
        ),
    )

# ============================================================
# KEY FINDING
# ============================================================

# Use the current filters to determine whether a specific
# degradation finding should be highlighted.

finding_data = health_df.copy()


if selected_model != "All":

    finding_data = finding_data[
        finding_data["model"] == selected_model
    ]


if selected_task != "All":

    finding_data = finding_data[
        finding_data["task"] == selected_task
    ]


if selected_period != "All":

    finding_data = finding_data[
        finding_data["period"] == selected_period
    ]


# Prefer degraded configurations when generating the finding.
degraded_findings = finding_data[
    finding_data["health_status"] == "degraded"
].copy()


if not degraded_findings.empty:

    # Highlight the largest NDCG degradation.
    degraded_findings = degraded_findings.sort_values(
        "ndcg_relative_change_pct"
    )

    key_row = degraded_findings.iloc[0]

    key_model = str(
        key_row["model"]
    ).upper()

    key_task = str(
        key_row["task"]
    ).title()

    key_period = str(
        key_row["period"]
    ).title()

    key_ndcg = float(
        key_row["ndcg_relative_change_pct"]
    )

    key_recall = float(
        key_row["recall_relative_change_pct"]
    )

    st.markdown(
        f"""
        <div class="finding-card">
            <div class="finding-title">
                🔎 Key Monitoring Finding
            </div>
            <div class="finding-text">
                <strong>{key_model} / {key_task} / {key_period}</strong>
                shows the strongest monitored degradation under the
                current filters: NDCG@10 changed by
                <strong>{key_ndcg:+.2f}%</strong> and Recall@10 changed by
                <strong>{key_recall:+.2f}%</strong> relative to the
                baseline holdout.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif not finding_data.empty:

    # If there is no degradation, show a neutral finding.
    st.markdown(
        """
        <div class="finding-card">
            <div class="finding-title">
                🔎 Key Monitoring Finding
            </div>
            <div class="finding-text">
                No configuration under the current filters crosses
                the 20% NDCG degradation threshold.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# PERFORMANCE OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Recommendation Performance'
    '</div>',
    unsafe_allow_html=True,
)


if not performance.empty:

    performance["period_order"] = (
        performance["period"]
        .map(
            {
                "baseline_holdout": 0,
                "comparison": 1,
                "recent": 2,
            }
        )
    )

    performance = performance.sort_values(
        [
            "model",
            "task",
            "period_order",
        ]
    )


    performance["series"] = (
        performance["model"]
        .astype(str)
        .str.title()
        + " - "
        + performance["task"]
        .astype(str)
        .str.title()
    )


    chart1, chart2 = st.columns(2)


    # ========================================================
    # NDCG@10
    # ========================================================

    with chart1:

        st.caption("NDCG@10 — Temporal Performance")

        ndcg_chart = (
            performance.pivot(
                index="period",
                columns="series",
                values="ndcg_at_k",
            )
            .reindex(
                [
                    "baseline_holdout",
                    "comparison",
                    "recent",
                ]
            )
        )

        ndcg_chart.index = [
            "Baseline",
            "Comparison",
            "Recent",
        ]

        ndcg_chart.columns = [
            str(column)
            for column in ndcg_chart.columns
        ]

        st.line_chart(
            ndcg_chart,
            width="stretch",
        )


    # ========================================================
    # RECALL@10
    # ========================================================

    with chart2:

        st.caption("Recall@10 — Temporal Performance")

        recall_chart = (
            performance.pivot(
                index="period",
                columns="series",
                values="recall_at_k",
            )
            .reindex(
                [
                    "baseline_holdout",
                    "comparison",
                    "recent",
                ]
            )
        )

        recall_chart.index = [
            "Baseline",
            "Comparison",
            "Recent",
        ]

        recall_chart.columns = [
            str(column)
            for column in recall_chart.columns
        ]

        st.line_chart(
            recall_chart,
            width="stretch",
        )


else:

    st.info(
        "No recommendation performance data matches "
        "the current filters."
    )


# ============================================================
# DEGRADATION ALERTS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Model Degradation Alerts'
    '</div>',
    unsafe_allow_html=True,
)


degraded_rows = filtered_health[
    filtered_health["health_status"] == "degraded"
].copy()


warning_rows = filtered_health[
    filtered_health["health_status"] == "warning"
].copy()


if degraded_rows.empty and warning_rows.empty:

    st.success(
        "No model degradation alerts under the current filters."
    )

else:

    for _, row in degraded_rows.iterrows():

        ndcg = float(
            row["ndcg_relative_change_pct"]
        )

        st.error(
            f"🔴 {row['model'].upper()} / "
            f"{row['task'].title()} / "
            f"{row['period'].title()} — "
            f"NDCG@10 changed by {ndcg:+.2f}%"
        )


    for _, row in warning_rows.iterrows():

        recall = float(
            row["recall_relative_change_pct"]
        )

        st.warning(
            f"🟡 {row['model'].upper()} / "
            f"{row['task'].title()} / "
            f"{row['period'].title()} — "
            f"Recall@10 changed by {recall:+.2f}%"
        )


# ============================================================
# COLD-START MONITORING
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Cold-Start Monitoring'
    '</div>',
    unsafe_allow_html=True,
)


recent_relationship = relationship_df[
    relationship_df["period"] == "recent"
]


if not recent_relationship.empty:

    recent = recent_relationship.iloc[0]

    cold_users = float(
        recent["cold_user_rate_pct"]
    )

    new_items = float(
        recent["new_item_rate_pct"]
    )


    cold1, cold2 = st.columns(2)


    with cold1:

        st.metric(
            "Cold Users",
            f"{cold_users:.2f}%",
        )

        if cold_users >= 90:

            st.caption(
                "🔴 Very high cold-user pressure"
            )

        elif cold_users >= 70:

            st.caption(
                "🟡 High cold-user pressure"
            )

        else:

            st.caption(
                "🟢 Lower cold-user pressure"
            )

        st.caption(
            "Users without sufficient historical interaction "
            "history for the monitored period."
        )


    with cold2:

        st.metric(
            "New Items",
            f"{new_items:.2f}%",
        )

        if new_items >= 70:

            st.caption(
                "🔴 Very high new-item pressure"
            )

        elif new_items >= 40:

            st.caption(
                "🟡 High new-item pressure"
            )

        elif new_items >= 20:

            st.caption(
                "🟡 Moderate new-item pressure"
            )

        else:

            st.caption(
                "🟢 Lower new-item pressure"
            )

        st.caption(
            "Items not represented in the relevant "
            "training history."
        )


# ============================================================
# HEALTH DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Monitoring Details'
    '</div>',
    unsafe_allow_html=True,
)


details_columns = [
    "period",
    "task",
    "model",
    "ndcg_relative_change_pct",
    "recall_relative_change_pct",
    "health_status",
]


details = filtered_health[
    details_columns
].copy()


details = details.rename(
    columns={
        "period": "Period",
        "task": "Task",
        "model": "Model",
        "ndcg_relative_change_pct":
            "NDCG Change (%)",
        "recall_relative_change_pct":
            "Recall Change (%)",
        "health_status":
            "Health",
    }
)


details["NDCG Change (%)"] = (
    details["NDCG Change (%)"]
    .round(2)
)


details["Recall Change (%)"] = (
    details["Recall Change (%)"]
    .round(2)
)


st.dataframe(
    details,
    width="stretch",
    hide_index=True,
)


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Monitoring Interpretation'
    '</div>',
    unsafe_allow_html=True,
)


st.info(
    "The monitoring results show temporal changes in data "
    "distribution, cold-start pressure, and recommendation "
    "performance. These measurements identify conditions "
    "requiring investigation; they do not establish that "
    "data drift caused model degradation."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Recommendation Drift Monitoring System • "
    "Retailrocket evaluation dataset • "
    "20% relative performance degradation threshold"
)