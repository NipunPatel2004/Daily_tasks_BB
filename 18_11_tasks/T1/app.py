# Design and develop an intuitive web-based UI using Streamlit for inputting applicant data, visualizing prediction probabilities, and providing explanations for individual admission decisions.

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(
    page_title="AdmitAI — Futuristic Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 12% 10%, #111b33 0%, #070b14 55%, #05070e 100%);
        color: #e6f1ff;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1224 0%, #070b14 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.2);
    }
    .hero-box {
        background: linear-gradient(135deg, rgba(14, 24, 48, 0.9), rgba(20, 32, 62, 0.65));
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 0 28px rgba(56, 189, 248, 0.12);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 18px 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #38bdf8;
        margin-top: 4px;
    }
    .decision-admit {
        color: #34d399;
        text-shadow: 0 0 14px rgba(52, 211, 153, 0.45);
    }
    .decision-reject {
        color: #fb7185;
        text-shadow: 0 0 14px rgba(251, 113, 133, 0.45);
    }
    .explain-card {
        background: rgba(15, 23, 42, 0.68);
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .explain-pos {
        border-left-color: #34d399;
    }
    .explain-neg {
        border-left-color: #fb7185;
    }
    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 8px;
        background: rgba(56, 189, 248, 0.14);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #7dd3fc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]


@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Adm.csv")
    return pd.read_csv(csv_path)


@st.cache_resource
def train_models(df):
    X = df[FEATURES]
    y_reg = df["Chance of Admit"]
    y_clf = (df["Chance of Admit"] >= 0.75).astype(int)

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_clf_train)

    reg = RandomForestRegressor(n_estimators=120, random_state=42)
    reg.fit(X_train, y_reg_train)

    acc = accuracy_score(y_clf_test, clf.predict(X_test_scaled))
    r2 = r2_score(y_reg_test, reg.predict(X_test))

    return clf, reg, scaler, acc, r2


df = load_data()
clf, reg, scaler, acc, r2 = train_models(df)

st.markdown(
    f"""
    <div class="hero-box">
        <p class="hero-title">AdmitAI // Neural Admission Decision Studio</p>
        <p class="hero-sub">
            Interactive applicant evaluation with calibrated admission probabilities, radar benchmarking, and transparent per-applicant explanations.
        </p>
        <div style="margin-top: 12px;">
            <span class="pill">Dataset: Adm.csv ({len(df)} Applicants)</span>
            <span class="pill">Classifier Accuracy: {acc * 100:.1f}%</span>
            <span class="pill">Regressor R²: {r2:.3f}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

PRESETS = {
    "Custom Input": None,
    "Top Research Scholar": {
        "gre": 334, "toefl": 117, "rating": 5, "sop": 4.5, "lor": 4.5, "cgpa": 9.55, "research": 1
    },
    "Strong Balanced Applicant": {
        "gre": 322, "toefl": 110, "rating": 4, "sop": 4.0, "lor": 3.5, "cgpa": 8.80, "research": 1
    },
    "Borderline Candidate": {
        "gre": 313, "toefl": 104, "rating": 3, "sop": 3.0, "lor": 3.0, "cgpa": 8.35, "research": 0
    },
    "Needs Improvement": {
        "gre": 298, "toefl": 97, "rating": 2, "sop": 2.0, "lor": 2.5, "cgpa": 7.55, "research": 0
    },
}

st.sidebar.markdown("## 🎛️ Applicant Profile")
preset_choice = st.sidebar.selectbox("Quick Profile Preset", list(PRESETS.keys()))
defaults = PRESETS[preset_choice] or {
    "gre": 322, "toefl": 109, "rating": 3, "sop": 3.5, "lor": 3.5, "cgpa": 8.75, "research": 1
}

gre = st.sidebar.slider("GRE Score", 290, 340, defaults["gre"])
toefl = st.sidebar.slider("TOEFL Score", 92, 120, defaults["toefl"])
rating = st.sidebar.select_slider("University Rating", options=[1, 2, 3, 4, 5], value=defaults["rating"])
sop = st.sidebar.slider("SOP Strength", 1.0, 5.0, float(defaults["sop"]), 0.5)
lor = st.sidebar.slider("LOR Strength", 1.0, 5.0, float(defaults["lor"]), 0.5)
cgpa = st.sidebar.slider("CGPA (out of 10)", 6.80, 10.00, float(defaults["cgpa"]), 0.05)
research = st.sidebar.radio(
    "Research Experience",
    options=[1, 0],
    index=0 if defaults["research"] == 1 else 1,
    format_func=lambda v: "Yes (Published / Lab)" if v == 1 else "No Research",
    horizontal=True,
)

input_df = pd.DataFrame([{
    "GRE Score": gre,
    "TOEFL Score": toefl,
    "University Rating": rating,
    "SOP": sop,
    "LOR": lor,
    "CGPA": cgpa,
    "Research": research,
}])

input_scaled = scaler.transform(input_df)
probs = clf.predict_proba(input_scaled)[0]
admit_prob = float(probs[1])
reject_prob = float(probs[0])
chance_pred = float(np.clip(reg.predict(input_df)[0], 0.0, 1.0))
is_admitted = admit_prob >= 0.5
percentile = float((df["Chance of Admit"] < chance_pred).mean() * 100)

contributions = input_scaled[0] * clf.coef_[0]
exp_df = pd.DataFrame({
    "Feature": FEATURES,
    "Applicant Value": input_df.iloc[0].values,
    "Dataset Avg": [round(df[f].mean(), 2) for f in FEATURES],
    "Impact": contributions,
}).sort_values("Impact", ascending=True)

k1, k2, k3, k4 = st.columns(4)
decision_class = "decision-admit" if is_admitted else "decision-reject"
decision_text = "ADMITTED" if is_admitted else "NOT ADMITTED"

k1.markdown(
    f"""
    <div class="kpi-card">
        <div class="kpi-label">Predicted Decision</div>
        <div class="kpi-value {decision_class}">{decision_text}</div>
        <div class="kpi-sub">Cutoff: 75% Chance of Admit</div>
    </div>
    """,
    unsafe_allow_html=True,
)
k2.markdown(
    f"""
    <div class="kpi-card">
        <div class="kpi-label">Admission Probability</div>
        <div class="kpi-value">{admit_prob * 100:.1f}%</div>
        <div class="kpi-sub">Reject Prob: {reject_prob * 100:.1f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)
k3.markdown(
    f"""
    <div class="kpi-card">
        <div class="kpi-label">Predicted Chance Score</div>
        <div class="kpi-value">{chance_pred * 100:.1f}%</div>
        <div class="kpi-sub">Random Forest Estimate</div>
    </div>
    """,
    unsafe_allow_html=True,
)
k4.markdown(
    f"""
    <div class="kpi-card">
        <div class="kpi-label">Applicant Percentile</div>
        <div class="kpi-value">Top {max(1.0, 100 - percentile):.0f}%</div>
        <div class="kpi-sub">Higher than {percentile:.1f}% of peers</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
col_left, col_right = st.columns([1.05, 1.15])

with col_left:
    gauge_color = "#34d399" if admit_prob >= 0.5 else "#fb7185"
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(admit_prob * 100, 1),
            number={"suffix": "%", "font": {"size": 44, "color": "#f8fafc"}},
            delta={"reference": 50, "increasing": {"color": "#34d399"}, "decreasing": {"color": "#fb7185"}},
            title={"text": "Admission Probability Meter", "font": {"size": 16, "color": "#cbd5e1"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "rgba(15,23,42,0.7)",
                "borderwidth": 1,
                "bordercolor": "rgba(56,189,248,0.3)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(251,113,133,0.18)"},
                    {"range": [40, 70], "color": "rgba(250,204,21,0.16)"},
                    {"range": [70, 100], "color": "rgba(52,211,153,0.18)"},
                ],
                "threshold": {"line": {"color": "#38bdf8", "width": 3}, "thickness": 0.8, "value": 50},
            },
        )
    )
    fig_gauge.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    fig_prob = go.Figure(
        go.Bar(
            x=["Not Admitted", "Admitted"],
            y=[reject_prob * 100, admit_prob * 100],
            text=[f"{reject_prob * 100:.1f}%", f"{admit_prob * 100:.1f}%"],
            textposition="outside",
            marker=dict(
                color=["#fb7185", "#34d399"],
                line=dict(color=["#fda4af", "#6ee7b7"], width=1.5),
            ),
            width=0.45,
        )
    )
    fig_prob.update_layout(
        title="Class Probability Breakdown",
        yaxis=dict(title="Probability (%)", range=[0, 115], gridcolor="rgba(148,163,184,0.15)"),
        xaxis=dict(title=""),
        height=270,
        margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font={"color": "#e2e8f0"},
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with col_right:
    colors = ["#34d399" if v >= 0 else "#fb7185" for v in exp_df["Impact"]]
    fig_exp = go.Figure(
        go.Bar(
            x=exp_df["Impact"],
            y=exp_df["Feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=[f"{v:+.2f}" for v in exp_df["Impact"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Impact: %{x:+.3f}<extra></extra>",
        )
    )
    fig_exp.add_vline(x=0, line_color="#38bdf8", line_width=2)
    fig_exp.update_layout(
        title="Individual Decision Explanation (Feature Contributions)",
        xaxis=dict(title="Log-Odds Impact (+ Supports Admit | - Lowers Chance)", gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(title=""),
        height=320,
        margin=dict(l=20, r=40, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font={"color": "#e2e8f0"},
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    admitted_avg = df[df["Chance of Admit"] >= 0.75][FEATURES].mean()
    radar_cols = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA"]
    app_norm = [
        (input_df[c].iloc[0] - df[c].min()) / (df[c].max() - df[c].min()) * 100 for c in radar_cols
    ]
    adm_norm = [
        (admitted_avg[c] - df[c].min()) / (df[c].max() - df[c].min()) * 100 for c in radar_cols
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=adm_norm + [adm_norm[0]],
            theta=radar_cols + [radar_cols[0]],
            fill="toself",
            name="Avg Admitted Peer",
            line=dict(color="#818cf8", width=2),
            fillcolor="rgba(129,140,248,0.2)",
        )
    )
    fig_radar.add_trace(
        go.Scatterpolar(
            r=app_norm + [app_norm[0]],
            theta=radar_cols + [radar_cols[0]],
            fill="toself",
            name="This Applicant",
            line=dict(color="#38bdf8", width=2.5),
            fillcolor="rgba(56,189,248,0.28)",
        )
    )
    fig_radar.update_layout(
        title="Profile Radar vs. Average Admitted Applicant",
        polar=dict(
            bgcolor="rgba(15,23,42,0.5)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.2)"),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
        ),
        height=310,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("### 🧠 Plain-Language Decision Explanation")
pos_df = exp_df[exp_df["Impact"] > 0].sort_values("Impact", ascending=False)
neg_df = exp_df[exp_df["Impact"] < 0].sort_values("Impact", ascending=True)

e1, e2 = st.columns(2)
with e1:
    pos_items = "".join(
        f"<li><b>{r['Feature']}</b> ({r['Applicant Value']}) vs avg {r['Dataset Avg']} → <span style='color:#34d399;'>+{r['Impact']:.2f}</span></li>"
        for _, r in pos_df.head(4).iterrows()
    ) or "<li>No strong positive factors above baseline.</li>"
    st.markdown(
        f"""
        <div class="explain-card explain-pos">
            <h4 style="margin:0 0 8px 0; color:#34d399;">✅ Key Strengths Driving Admission</h4>
            <ul style="margin:0; padding-left:18px; color:#cbd5e1;">{pos_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with e2:
    neg_items = "".join(
        f"<li><b>{r['Feature']}</b> ({r['Applicant Value']}) vs avg {r['Dataset Avg']} → <span style='color:#fb7185;'>{r['Impact']:.2f}</span></li>"
        for _, r in neg_df.head(4).iterrows()
    ) or "<li>No major negative factors dragging down this profile!</li>"
    st.markdown(
        f"""
        <div class="explain-card explain-neg">
            <h4 style="margin:0 0 8px 0; color:#fb7185;">⚠️ Areas Lowering Admission Probability</h4>
            <ul style="margin:0; padding-left:18px; color:#cbd5e1;">{neg_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("📋 View Full Feature Attribution Table"):
    display_exp = exp_df.sort_values("Impact", ascending=False).copy()
    display_exp["Impact"] = display_exp["Impact"].round(4)
    st.dataframe(display_exp, use_container_width=True, hide_index=True)
