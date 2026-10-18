from pathlib import Path

import pandas as pd
import streamlit as st

from src.ecommerce.recommendation_service import RecommendationService
from src.ecommerce.event_logger import EventLogger


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
# PAGE SELECTION
# ============================================================

st.sidebar.title("Application")

selected_page = st.sidebar.radio(
    "Go to",
    ["E-commerce Demo", "Admin Monitoring"],
)

st.sidebar.divider()

# ============================================================
# E-COMMERCE DEMO
# ============================================================

def render_ecommerce_demo():

    st.markdown(
        '<div class="main-title">'
        '?? E-Commerce Recommendation Demo'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Interact with the recommendation system as an online shopper'
        '</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This page simulates an online shopping experience. "
        "Choose a visitor, generate recommendations, and record "
        "shopper actions. These actions are logged for later monitoring."
    )

    try:
        recommendation_service = RecommendationService()
        event_logger = EventLogger()
    except Exception as error:
        st.error("The recommendation service could not be loaded.")
        st.code(str(error))
        return

    st.markdown(
        '<div class="section-title">1. Choose a visitor</div>',
        unsafe_allow_html=True,
    )

    model = recommendation_service.models

    user_ids = list(model.get("user_ids", []))

    if not user_ids:
        st.warning("No known visitors are available in the recommendation model.")
        return

    visitor_options = user_ids[:1000]

    selected_visitor = st.selectbox(
        "Visitor ID",
        visitor_options,
        help="Choose a visitor from the recommendation model."
    )

    recommendation_count = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
    )

    if st.button(
        "Get Recommendations",
        type="primary",
        use_container_width=True,
    ):
        try:
            recommendations = recommendation_service.recommend(
                visitor_id=int(selected_visitor),
                method="svd",
                k=int(recommendation_count),
            )

            st.session_state["demo_recommendations"] = recommendations
            st.session_state["demo_visitor"] = int(selected_visitor)

        except Exception as error:
            st.error("Could not generate recommendations.")
            st.code(str(error))

    recommendations = st.session_state.get(
        "demo_recommendations",
        [],
    )

    demo_visitor = st.session_state.get(
        "demo_visitor",
        int(selected_visitor),
    )

    if recommendations:

        st.markdown(
            '<div class="section-title">2. Recommended products</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            f"Recommendations generated for visitor {demo_visitor}. "
            "The current model ranks products using historical interaction data."
        )

        for index, item_id in enumerate(recommendations, start=1):

            col1, col2, col3, col4, col5 = st.columns(
                [1, 2, 1, 1, 1]
            )

            with col1:
                st.write(f"**#{index}**")

            with col2:
                st.write(f"Product `{item_id}`")

            with col3:
                if st.button(
                    "View",
                    key=f"view_{item_id}_{index}",
                ):
                    event_logger.log_event(
                        visitor_id=demo_visitor,
                        item_id=int(item_id),
                        event="view",
                    )
                    st.toast("View recorded")

            with col4:
                if st.button(
                    "Click",
                    key=f"click_{item_id}_{index}",
                ):
                    event_logger.log_event(
                        visitor_id=demo_visitor,
                        item_id=int(item_id),
                        event="click",
                    )
                    st.toast("Click recorded")

            with col5:
                if st.button(
                    "Cart",
                    key=f"cart_{item_id}_{index}",
                ):
                    event_logger.log_event(
                        visitor_id=demo_visitor,
                        item_id=int(item_id),
                        event="addtocart",
                    )
                    st.toast("Add-to-cart recorded")

        st.markdown(
            '<div class="section-title">3. Purchase simulation</div>',
            unsafe_allow_html=True,
        )

        purchase_item = st.selectbox(
            "Select a recommended product to purchase",
            recommendations,
            key="purchase_item",
        )

        if st.button(
            "Complete Transaction",
            use_container_width=True,
        ):
            event_logger.log_event(
                visitor_id=demo_visitor,
                item_id=int(purchase_item),
                event="transaction",
            )

            st.success(
                f"Transaction recorded for product {purchase_item}."
            )

        st.markdown(
            '<div class="section-title">4. Recent shopper events</div>',
            unsafe_allow_html=True,
        )

        try:
            recent_events = event_logger.read_events()

            if not recent_events.empty:
                st.dataframe(
                    recent_events.tail(20).iloc[::-1],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("No shopper events have been recorded yet.")

        except Exception as error:
            st.warning("Could not read recent shopper events.")
            st.code(str(error))

    else:
        st.markdown(
            '<div class="section-title">2. Recommended products</div>',
            unsafe_allow_html=True,
        )

        st.info(
            "Click **Get Recommendations** to generate products for this visitor."
        )

    st.divider()

    st.caption(
        "Demo note: shopper actions are logged immediately, while "
        "drift detection, performance evaluation, and model adaptation "
        "remain part of the batch monitoring pipeline."
    )


# ============================================================
# PAGE ROUTING
# ============================================================

if selected_page == "E-commerce Demo":
    render_ecommerce_demo()
    st.stop()

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

    adaptation = pd.read_csv(
        DATA_DIR / "adaptation_decision.csv"
    )

    return (
        health,
        relationship,
        recommendation,
        cold_start,
        adaptation,
    )


try:

    (
        health_df,
        relationship_df,
        recommendation_df,
        cold_start_df,
        adaptation_df,
    ) = load_data()

except FileNotFoundError as error:

    st.error(
        "Required project output is missing. "
        "Run the monitoring and adaptation pipeline first."
    )

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
# FILTER ADAPTATION DATA
# ============================================================

filtered_adaptation = adaptation_df.copy()


if selected_model != "All":

    filtered_adaptation = filtered_adaptation[
        filtered_adaptation["model"] == selected_model
    ]


if selected_task != "All":

    filtered_adaptation = filtered_adaptation[
        filtered_adaptation["task"] == selected_task
    ]


if selected_period != "All":

    filtered_adaptation = filtered_adaptation[
        filtered_adaptation["period"] == selected_period
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
        key=lambda value: latest_period_order.get(
            value,
            -1,
        ),
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


degraded_findings = finding_data[
    finding_data["health_status"] == "degraded"
].copy()


if not degraded_findings.empty:

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

        st.caption(
            "NDCG@10 — Temporal Performance"
        )

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

        st.caption(
            "Recall@10 — Temporal Performance"
        )

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
# ADAPTATION & MODEL LIFECYCLE
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Adaptation & Model Lifecycle'
    '</div>',
    unsafe_allow_html=True,
)


if filtered_adaptation.empty:

    st.info(
        "No adaptation lifecycle information matches "
        "the current filters."
    )

else:

    st.caption(
        "Shows the degradation condition that triggered adaptation "
        "and the validation results obtained on the unseen recent period."
    )

    lifecycle_rows = filtered_adaptation.copy()


    for start in range(
        0,
        len(lifecycle_rows),
        2,
    ):

        card_rows = lifecycle_rows.iloc[
            start:start + 2
        ]

        card_columns = st.columns(
            len(card_rows)
        )


        for card_column, (_, row) in zip(
            card_columns,
            card_rows.iterrows(),
        ):

            with card_column:

                model_label = str(
                    row["model"]
                ).upper()

                task_label = str(
                    row["task"]
                ).title()

                trigger_period = str(
                    row["period"]
                ).title()


                # ------------------------------------------------
                # Health status
                # ------------------------------------------------

                health_status = str(
                    row["health_status"]
                ).upper()

                health_icon = {
                    "HEALTHY": "🟢",
                    "WARNING": "🟡",
                    "DEGRADED": "🔴",
                }.get(
                    health_status,
                    "⚪",
                )


                # ------------------------------------------------
                # Lifecycle decision
                # ------------------------------------------------

                lifecycle_decision = str(
                    row["lifecycle_decision"]
                )

                lifecycle_icon = {
                    "KEEP_CURRENT_MODEL": "🟢",
                    "ACCEPT": "🟢",
                    "REVIEW": "🟡",
                    "REJECT": "🔴",
                    "RETRAIN_PENDING": "🟠",
                }.get(
                    lifecycle_decision,
                    "⚪",
                )


                # ------------------------------------------------
                # Recommended action
                # ------------------------------------------------

                recommended_action = str(
                    row["recommended_action"]
                )


                # ------------------------------------------------
                # Adaptation status
                # ------------------------------------------------

                adaptation_status = row[
                    "adaptation_status"
                ]

                if pd.notna(adaptation_status):

                    adaptation_label = (
                        str(adaptation_status)
                        .replace("_", " ")
                        .title()
                    )

                else:

                    adaptation_label = (
                        "Not performed"
                    )


                # ------------------------------------------------
                # Validation decision
                # ------------------------------------------------

                validation_decision = row[
                    "validation_decision"
                ]

                if pd.notna(validation_decision):

                    validation_label = str(
                        validation_decision
                    )

                else:

                    validation_label = (
                        "Not required"
                    )


                # ------------------------------------------------
                # Validation period
                # ------------------------------------------------

                evaluation_period = row.get(
                    "evaluation_period",
                    None,
                )

                if pd.notna(evaluation_period):

                    validation_period_label = {
                        "baseline_holdout": "Baseline",
                        "comparison": "Comparison",
                        "recent": "Recent",
                    }.get(
                        str(evaluation_period),
                        str(evaluation_period).title(),
                    )

                else:

                    validation_period_label = (
                        "Not applicable"
                    )


                # ------------------------------------------------
                # BEFORE / AFTER PERFORMANCE
                # IMPORTANT:
                # These values come directly from Phase 4.
                # ------------------------------------------------

                before_ndcg = row.get(
                    "before_ndcg_at_10",
                    float("nan"),
                )

                after_ndcg = row.get(
                    "after_ndcg_at_10",
                    float("nan"),
                )

                before_recall = row.get(
                    "before_recall_at_10",
                    float("nan"),
                )

                after_recall = row.get(
                    "after_recall_at_10",
                    float("nan"),
                )


                ndcg_change = row.get(
                    "ndcg_relative_change_pct",
                    float("nan"),
                )

                recall_change = row.get(
                    "recall_relative_change_pct",
                    float("nan"),
                )


                # ------------------------------------------------
                # Format NDCG
                # ------------------------------------------------

                if (
                    pd.notna(before_ndcg)
                    and pd.notna(after_ndcg)
                ):

                    ndcg_before_text = (
                        f"{float(before_ndcg):.6f}"
                    )

                    ndcg_after_text = (
                        f"{float(after_ndcg):.6f}"
                    )

                else:

                    ndcg_before_text = "—"
                    ndcg_after_text = "—"


                if pd.notna(ndcg_change):

                    ndcg_change_text = (
                        f"{float(ndcg_change):+.2f}%"
                    )

                else:

                    ndcg_change_text = "—"


                # ------------------------------------------------
                # Format Recall
                # ------------------------------------------------

                if (
                    pd.notna(before_recall)
                    and pd.notna(after_recall)
                ):

                    recall_before_text = (
                        f"{float(before_recall):.6f}"
                    )

                    recall_after_text = (
                        f"{float(after_recall):.6f}"
                    )

                else:

                    recall_before_text = "—"
                    recall_after_text = "—"


                if pd.notna(recall_change):

                    recall_change_text = (
                        f"{float(recall_change):+.2f}%"
                    )

                else:

                    recall_change_text = "—"


                # ------------------------------------------------
                # Lifecycle card
                # ------------------------------------------------

                with st.container(
                    border=True
                ):

                    st.subheader(
                        f"{model_label} / "
                        f"{task_label}"
                    )

                    st.markdown(
                        f"### {lifecycle_icon} "
                        f"{lifecycle_decision}"
                    )

                    st.markdown(
                        f"**Health period:** "
                        f"{trigger_period}"
                    )

                    st.markdown(
                        f"**Model Health:** "
                        f"{health_icon} {health_status}"
                    )

                    st.markdown(
                        f"**Recommended Action:** "
                        f"{recommended_action}"
                    )

                    st.markdown(
                        f"**Adaptation:** "
                        f"{adaptation_label}"
                    )

                    st.markdown(
                        f"**Validation:** "
                        f"{validation_label}"
                    )

                    st.markdown(
                        f"**Validation Period:** "
                        f"{validation_period_label}"
                    )

                    st.markdown(
                        "#### Post-Adaptation Validation"
                    )

                    metric1, metric2 = st.columns(2)


                    with metric1:

                        st.metric(
                            "NDCG@10 Before",
                            ndcg_before_text,
                        )

                        st.metric(
                            "NDCG@10 After",
                            ndcg_after_text,
                        )

                        st.caption(
                            f"Change: {ndcg_change_text}"
                        )


                    with metric2:

                        st.metric(
                            "Recall@10 Before",
                            recall_before_text,
                        )

                        st.metric(
                            "Recall@10 After",
                            recall_after_text,
                        )

                        st.caption(
                            f"Change: {recall_change_text}"
                        )


                    # ------------------------------------------------
                    # Validation interpretation
                    # ------------------------------------------------

                    if lifecycle_decision == "ACCEPT":

                        st.success(
                            "Adaptation improved the monitored "
                            "performance dimensions sufficiently "
                            "to pass validation."
                        )

                    elif lifecycle_decision == "REVIEW":

                        st.warning(
                            "Adaptation produced mixed validation "
                            "results. Manual review is required."
                        )

                    elif lifecycle_decision == "REJECT":

                        st.error(
                            "Adaptation did not recover the "
                            "monitored performance."
                        )

                    elif lifecycle_decision == "KEEP_CURRENT_MODEL":

                        st.info(
                            "Current model remains under monitoring; "
                            "adaptation was not required."
                        )

                    else:

                        st.info(
                            "Adaptation is pending validation."
                        )


    # ========================================================
    # DECISION EXPLANATIONS
    # ========================================================

    st.markdown(
        "#### Decision Explanation"
    )

    explanation_columns = [
        "period",
        "task",
        "model",
        "recommended_action",
        "adaptation_status",
        "validation_decision",
        "lifecycle_decision",
        "decision_explanation",
    ]

    available_explanation_columns = [
        column
        for column in explanation_columns
        if column in filtered_adaptation.columns
    ]

    explanation_df = filtered_adaptation[
        available_explanation_columns
    ].copy()

    explanation_df = explanation_df.rename(
        columns={
            "period": "Health Period",
            "task": "Task",
            "model": "Model",
            "recommended_action": "Action",
            "adaptation_status": "Adaptation",
            "validation_decision": "Validation",
            "lifecycle_decision": "Lifecycle Decision",
            "decision_explanation": "Explanation",
        }
    )

    st.dataframe(
        explanation_df,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "ACCEPT indicates that the adaptation passed the "
        "defined validation rule. REVIEW indicates that the "
        "adaptation produced mixed results and requires "
        "manual assessment. REJECT indicates that the "
        "adaptation did not recover the monitored performance."
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


# ------------------------------------------------------------
# Determine which period to display
# ------------------------------------------------------------

if selected_period == "All":

    cold_period = "recent"

else:

    cold_period = selected_period


cold_relationship = relationship_df.copy()


# ------------------------------------------------------------
# Filter by selected period
# ------------------------------------------------------------

if "period" in cold_relationship.columns:

    cold_relationship = cold_relationship[
        cold_relationship["period"] == cold_period
    ]


# ------------------------------------------------------------
# Filter by selected task
# ------------------------------------------------------------

if (
    selected_task != "All"
    and "task" in cold_relationship.columns
):

    cold_relationship = cold_relationship[
        cold_relationship["task"] == selected_task
    ]


# ------------------------------------------------------------
# Filter by selected model
# ------------------------------------------------------------

if (
    selected_model != "All"
    and "model" in cold_relationship.columns
):

    cold_relationship = cold_relationship[
        cold_relationship["model"] == selected_model
    ]


if not cold_relationship.empty:

    cold_values = cold_relationship.iloc[0]

    cold_users = float(
        cold_values["cold_user_rate_pct"]
    )

    new_items = float(
        cold_values["new_item_rate_pct"]
    )

    displayed_cold_period = {
        "baseline_holdout": "Baseline",
        "comparison": "Comparison",
        "recent": "Recent",
    }.get(
        cold_period,
        cold_period.title(),
    )


    st.caption(
        f"Cold-start indicators for the "
        f"{displayed_cold_period} evaluation period "
        f"under the current filters."
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

        elif cold_users >= 40:

            st.caption(
                "🟡 Moderate cold-user pressure"
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


else:

    st.info(
        "No cold-start information matches "
        "the current filters."
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



