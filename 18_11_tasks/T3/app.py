# Create interactive charts, such as distribution plots and decision boundary visualizations, using Streamlit components to provide stakeholders with a clear understanding of model behavior.

import os

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.set_page_config(
    page_title="ModelVision — Interactive Model Behavior Studio",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 8%, #1a153a 0%, #080b16 52%, #04060d 100%);
        color: #e6f1ff;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1024 0%, #060812 100%);
        border-right: 1px solid rgba(167, 139, 250, 0.25);
    }
    .hero-box {
        background: linear-gradient(135deg, rgba(22, 18, 54, 0.9), rgba(14, 26, 52, 0.7));
        border: 1px solid rgba(167, 139, 250, 0.38);
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 18px;
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.14);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.97rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(167, 139, 250, 0.28);
        border-radius: 15px;
        padding: 16px 18px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    }
    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #a78bfa;
        margin-top: 4px;
    }
    .insight-box {
        background: rgba(15, 23, 42, 0.72);
        border-left: 4px solid #a78bfa;
        border-radius: 12px;
        padding: 15px 18px;
        margin-top: 12px;
        color: #cbd5e1;
    }
    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 8px;
        background: rgba(167, 139, 250, 0.14);
        border: 1px solid rgba(167, 139, 250, 0.38);
        color: #ddd6fe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

NUMERIC_FEATURES = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
]
ALL_FEATURES = NUMERIC_FEATURES + ["Research"]


@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Adm.csv")
    return pd.read_csv(csv_path)


df = load_data()

st.markdown(
    f"""
    <div class="hero-box">
        <p class="hero-title">ModelVision // Interactive Charts & Decision Boundary Studio</p>
        <p class="hero-sub">
            Stakeholder analytics cockpit for exploring score distributions, model decision surfaces, and feature behavior across applicants.
        </p>
        <div style="margin-top: 12px;">
            <span class="pill">Dataset: Adm.csv ({len(df)} Records)</span>
            <span class="pill">5 Interactive Classifiers</span>
            <span class="pill">Live Probability Contours</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## 🎛️ Stakeholder Filters")
all_regions = sorted(df["region"].unique().tolist())
selected_regions = st.sidebar.multiselect("Regions", all_regions, default=all_regions)
research_filter = st.sidebar.radio("Research Filter", ["All", "With Research", "Without Research"], horizontal=False)
rating_range = st.sidebar.slider("University Rating Range", 1, 5, (1, 5))
threshold = st.sidebar.slider("Admission Threshold (Chance of Admit)", 0.50, 0.90, 0.75, 0.05)

filtered = df[
    (df["region"].isin(selected_regions))
    & (df["University Rating"] >= rating_range[0])
    & (df["University Rating"] <= rating_range[1])
].copy()

if research_filter == "With Research":
    filtered = filtered[filtered["Research"] == 1]
elif research_filter == "Without Research":
    filtered = filtered[filtered["Research"] == 0]

filtered["Admitted"] = (filtered["Chance of Admit"] >= threshold).astype(int)
filtered["Status"] = filtered["Admitted"].map({1: "Admitted", 0: "Not Admitted"})

if len(filtered) < 15:
    st.warning("Not enough records match the current filters. Please broaden your sidebar filters.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Filtered Applicants</div>'
    f'<div class="kpi-value">{len(filtered)}</div><div class="kpi-sub">Active Cohort</div></div>',
    unsafe_allow_html=True,
)
k2.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Admitted</div>'
    f'<div class="kpi-value" style="color:#34d399;">{int(filtered["Admitted"].sum())}</div>'
    f'<div class="kpi-sub">Cutoff >= {threshold:.2f}</div></div>',
    unsafe_allow_html=True,
)
k3.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Admission Rate</div>'
    f'<div class="kpi-value" style="color:#38bdf8;">{filtered["Admitted"].mean() * 100:.1f}%</div>'
    f'<div class="kpi-sub">Selected Regions</div></div>',
    unsafe_allow_html=True,
)
k4.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Mean Chance</div>'
    f'<div class="kpi-value">{filtered["Chance of Admit"].mean() * 100:.1f}%</div>'
    f'<div class="kpi-sub">Cohort Average</div></div>',
    unsafe_allow_html=True,
)
k5.markdown(
    f'<div class="kpi-card"><div class="kpi-label">Mean CGPA</div>'
    f'<div class="kpi-value" style="color:#a78bfa;">{filtered["CGPA"].mean():.2f}</div>'
    f'<div class="kpi-sub">Mean GRE: {filtered["GRE Score"].mean():.0f}</div></div>',
    unsafe_allow_html=True,
)

st.write("")
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Distribution Plots",
    "🎯 Decision Boundary Studio",
    "🔍 Correlations & Scatter",
    "🧠 Feature Importance & Insights",
])

with tab1:
    c_opt1, c_opt2, c_opt3 = st.columns([1.2, 1.2, 1.0])
    dist_feature = c_opt1.selectbox("Feature to Visualize", NUMERIC_FEATURES + ["Chance of Admit"], index=5)
    chart_style = c_opt2.selectbox("Distribution Style", ["Histogram + Boxplot", "Violin Plot", "Overlaid Histogram"])
    nbins = c_opt3.slider("Histogram Bins", 10, 45, 22)

    color_seq = {"Admitted": "#34d399", "Not Admitted": "#fb7185"}

    if chart_style == "Histogram + Boxplot":
        fig_dist = px.histogram(
            filtered,
            x=dist_feature,
            color="Status",
            nbins=nbins,
            barmode="overlay",
            marginal="box",
            opacity=0.68,
            color_discrete_map=color_seq,
            title=f"{dist_feature} Distribution by Admission Status",
        )
    elif chart_style == "Violin Plot":
        fig_dist = px.violin(
            filtered,
            x="Status",
            y=dist_feature,
            color="Status",
            box=True,
            points="all",
            color_discrete_map=color_seq,
            title=f"{dist_feature} Spread & Density by Status",
        )
    else:
        fig_dist = px.histogram(
            filtered,
            x=dist_feature,
            color="Status",
            nbins=nbins,
            barmode="overlay",
            histnorm="probability density",
            opacity=0.65,
            color_discrete_map=color_seq,
            title=f"{dist_feature} Normalized Density Comparison",
        )

    fig_dist.update_layout(
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.48)",
        font={"color": "#e2e8f0"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    reg_stats = (
        filtered.groupby("region", as_index=False)
        .agg(Avg_Chance=("Chance of Admit", "mean"), Admit_Rate=("Admitted", "mean"), Count=("Serial No.", "count"))
    )
    reg_stats["Avg_Chance"] = reg_stats["Avg_Chance"] * 100
    reg_stats["Admit_Rate"] = reg_stats["Admit_Rate"] * 100

    fig_reg = go.Figure()
    fig_reg.add_trace(
        go.Bar(
            x=reg_stats["region"],
            y=reg_stats["Avg_Chance"],
            name="Avg Chance of Admit (%)",
            marker_color="#38bdf8",
            text=[f"{v:.1f}%" for v in reg_stats["Avg_Chance"]],
            textposition="outside",
        )
    )
    fig_reg.add_trace(
        go.Bar(
            x=reg_stats["region"],
            y=reg_stats["Admit_Rate"],
            name="Admission Rate (%)",
            marker_color="#a78bfa",
            text=[f"{v:.1f}%" for v in reg_stats["Admit_Rate"]],
            textposition="outside",
        )
    )
    fig_reg.update_layout(
        title="Regional Performance Breakdown",
        barmode="group",
        yaxis=dict(title="Percentage (%)", range=[0, 112], gridcolor="rgba(148,163,184,0.14)"),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.48)",
        font={"color": "#e2e8f0"},
    )
    st.plotly_chart(fig_reg, use_container_width=True)

with tab2:
    b1, b2, b3, b4 = st.columns([1, 1, 1.1, 1])
    x_feat = b1.selectbox("X-Axis Feature", NUMERIC_FEATURES, index=0)
    y_feat = b2.selectbox("Y-Axis Feature", NUMERIC_FEATURES, index=5)
    model_choice = b3.selectbox(
        "Classifier Model",
        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "SVM (RBF)", "Random Forest"],
    )
    if model_choice == "Decision Tree":
        param_val = b4.slider("Max Depth", 1, 10, 4)
    elif model_choice == "K-Nearest Neighbors":
        param_val = b4.slider("Neighbors (K)", 3, 25, 9, 2)
    elif model_choice == "Random Forest":
        param_val = b4.slider("Trees / Depth", 2, 10, 5)
    else:
        param_val = b4.slider("Regularization (C)", 0.1, 5.0, 1.0, 0.1)

    if x_feat == y_feat:
        st.warning("Please choose two different features for the X and Y axes.")
    elif filtered["Admitted"].nunique() < 2:
        st.warning("Current threshold produces only one class. Adjust the Admission Threshold in the sidebar.")
    else:
        X_pair = filtered[[x_feat, y_feat]]
        y_pair = filtered["Admitted"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_pair, y_pair, test_size=0.2, random_state=42, stratify=y_pair
        )
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if model_choice == "Logistic Regression":
            clf = LogisticRegression(C=float(param_val), random_state=42)
        elif model_choice == "Decision Tree":
            clf = DecisionTreeClassifier(max_depth=int(param_val), random_state=42)
        elif model_choice == "K-Nearest Neighbors":
            clf = KNeighborsClassifier(n_neighbors=int(param_val))
        elif model_choice == "SVM (RBF)":
            clf = SVC(C=float(param_val), probability=True, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=100, max_depth=int(param_val), random_state=42)

        clf.fit(X_tr_s, y_tr)
        y_hat = clf.predict(X_te_s)

        acc = accuracy_score(y_te, y_hat)
        prec = precision_score(y_te, y_hat, zero_division=0)
        rec = recall_score(y_te, y_hat, zero_division=0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Test Accuracy", f"{acc * 100:.1f}%")
        m2.metric("Precision (Admitted)", f"{prec * 100:.1f}%")
        m3.metric("Recall (Admitted)", f"{rec * 100:.1f}%")

        x_pad = (X_pair[x_feat].max() - X_pair[x_feat].min()) * 0.06 + 0.1
        y_pad = (X_pair[y_feat].max() - X_pair[y_feat].min()) * 0.06 + 0.1
        x_grid = np.linspace(X_pair[x_feat].min() - x_pad, X_pair[x_feat].max() + x_pad, 110)
        y_grid = np.linspace(X_pair[y_feat].min() - y_pad, X_pair[y_feat].max() + y_pad, 110)
        xx, yy = np.meshgrid(x_grid, y_grid)

        grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[x_feat, y_feat])
        probs_grid = clf.predict_proba(scaler.transform(grid_df))[:, 1].reshape(xx.shape)

        fig_bound = go.Figure()
        fig_bound.add_trace(
            go.Contour(
                x=x_grid,
                y=y_grid,
                z=probs_grid,
                colorscale=[
                    [0.0, "rgba(251, 113, 133, 0.65)"],
                    [0.5, "rgba(15, 23, 42, 0.72)"],
                    [1.0, "rgba(52, 211, 153, 0.68)"],
                ],
                zmin=0,
                zmax=1,
                contours=dict(showlines=False),
                colorbar=dict(title="Admit Prob"),
                hovertemplate=f"{x_feat}: %{{x:.2f}}<br>{y_feat}: %{{y:.2f}}<br>Admit Prob: %{{z:.2f}}<extra></extra>",
            )
        )
        fig_bound.add_trace(
            go.Contour(
                x=x_grid,
                y=y_grid,
                z=probs_grid,
                contours=dict(start=0.5, end=0.5, size=0.1, coloring="none"),
                line=dict(color="#f8fafc", width=3, dash="dash"),
                showscale=False,
                hoverinfo="skip",
                name="Decision Boundary (p=0.5)",
            )
        )

        for label, color, symbol in [("Not Admitted", "#fb7185", "circle"), ("Admitted", "#34d399", "diamond")]:
            sub = filtered[filtered["Status"] == label]
            fig_bound.add_trace(
                go.Scatter(
                    x=sub[x_feat],
                    y=sub[y_feat],
                    mode="markers",
                    name=label,
                    marker=dict(size=7, color=color, symbol=symbol, line=dict(color="#070b14", width=1)),
                    text=sub["University_Name"] + " | Chance: " + (sub["Chance of Admit"] * 100).round(1).astype(str) + "%",
                    hovertemplate=f"<b>{label}</b><br>%{{text}}<br>{x_feat}: %{{x}}<br>{y_feat}: %{{y}}<extra></extra>",
                )
            )

        fig_bound.update_layout(
            title=f"{model_choice} Probability Surface & Decision Boundary ({x_feat} vs {y_feat})",
            xaxis=dict(title=x_feat, gridcolor="rgba(148,163,184,0.12)"),
            yaxis=dict(title=y_feat, gridcolor="rgba(148,163,184,0.12)"),
            height=540,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font={"color": "#e2e8f0"},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bound, use_container_width=True)

        st.markdown(
            f"""
            <div class="insight-box">
                <b>Stakeholder Takeaway:</b> The dashed white line marks the 50% decision threshold for <b>{model_choice}</b>
                using <b>{x_feat}</b> and <b>{y_feat}</b>. Applicants in the emerald region have high predicted admission probability,
                while those in the coral region fall below the admission cutoff ({threshold:.2f}).
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab3:
    sc1, sc2, sc3 = st.columns(3)
    sx = sc1.selectbox("Scatter X", NUMERIC_FEATURES, index=0, key="sx")
    sy = sc2.selectbox("Scatter Y", NUMERIC_FEATURES + ["Chance of Admit"], index=6, key="sy")
    scolor = sc3.selectbox("Color By", ["Status", "region", "University Rating"])

    fig_sc = px.scatter(
        filtered,
        x=sx,
        y=sy,
        color=scolor,
        size="CGPA",
        hover_data=["University_Name", "GRE Score", "TOEFL Score", "CGPA", "Chance of Admit"],
        color_discrete_map={"Admitted": "#34d399", "Not Admitted": "#fb7185"} if scolor == "Status" else None,
        title=f"Interactive Feature Relationship: {sx} vs {sy}",
    )
    fig_sc.update_layout(
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.48)",
        font={"color": "#e2e8f0"},
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    corr_cols = ALL_FEATURES + ["Chance of Admit"]
    corr_mat = filtered[corr_cols].corr().round(2)
    fig_corr = px.imshow(
        corr_mat,
        text_auto=True,
        color_continuous_scale="Tealgrn",
        aspect="auto",
        title="Feature Correlation Matrix",
    )
    fig_corr.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    if filtered["Admitted"].nunique() >= 2:
        rf_imp = RandomForestClassifier(n_estimators=120, random_state=42)
        rf_imp.fit(filtered[ALL_FEATURES], filtered["Admitted"])
        imp_df = pd.DataFrame({
            "Feature": ALL_FEATURES,
            "Importance": rf_imp.feature_importances_ * 100,
        }).sort_values("Importance", ascending=True)

        fig_imp = go.Figure(
            go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation="h",
                text=[f"{v:.1f}%" for v in imp_df["Importance"]],
                textposition="outside",
                marker=dict(
                    color=imp_df["Importance"],
                    colorscale="Viridis",
                    line=dict(color="rgba(255,255,255,0.2)", width=1),
                ),
            )
        )
        fig_imp.update_layout(
            title="Global Feature Importance (Random Forest Classifier)",
            xaxis=dict(title="Relative Importance (%)", gridcolor="rgba(148,163,184,0.14)"),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.48)",
            font={"color": "#e2e8f0"},
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        top_feat = imp_df.iloc[-1]["Feature"]
        second_feat = imp_df.iloc[-2]["Feature"]
        st.markdown(
            f"""
            <div class="insight-box">
                <b>Key Model Behavior Summary:</b> Across the {len(filtered)} selected applicants, <b>{top_feat}</b> and
                <b>{second_feat}</b> are the strongest predictors of admission decisions. Use the Decision Boundary Studio tab
                to compare linear vs. non-linear boundaries for these top drivers.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Adjust the admission threshold so both classes exist to compute classifier feature importance.")

    st.markdown("### 📋 Filtered Cohort Preview")
    st.dataframe(
        filtered[["Serial No.", "University_Name", "region", "GRE Score", "TOEFL Score", "CGPA", "Research", "Chance of Admit", "Status"]].head(30),
        use_container_width=True,
        hide_index=True,
    )
