# Integrate real-time data streams into the Streamlit application, enabling live updates of applicant status and model predictions as new information becomes available.

import os
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="AdmitStream — Live Admission Telemetry",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 85% 10%, #0e2238 0%, #070b14 55%, #04060c 100%);
        color: #e6f1ff;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1325 0%, #060a12 100%);
        border-right: 1px solid rgba(52, 211, 153, 0.22);
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.75); }
        70% { box-shadow: 0 0 0 10px rgba(52, 211, 153, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }
    .live-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #34d399;
        margin-right: 8px;
        animation: pulse 1.6s infinite;
    }
    .hero-box {
        background: linear-gradient(135deg, rgba(10, 24, 44, 0.92), rgba(14, 34, 58, 0.68));
        border: 1px solid rgba(52, 211, 153, 0.35);
        border-radius: 18px;
        padding: 20px 26px;
        margin-bottom: 18px;
        box-shadow: 0 0 26px rgba(52, 211, 153, 0.12);
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        background: linear-gradient(90deg, #34d399, #38bdf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.96rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .ticker-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-left: 5px solid #38bdf8;
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 16px;
        font-family: monospace;
        font-size: 0.92rem;
        color: #e2e8f0;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(56, 189, 248, 0.25);
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
    .kpi-meta {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 4px;
    }
    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 8px;
        background: rgba(52, 211, 153, 0.14);
        border: 1px solid rgba(52, 211, 153, 0.38);
        color: #6ee7b7;
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
def load_stream_datasets():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Adm.csv")
    df = pd.read_csv(csv_path)
    df["Application_DateTime"] = pd.to_datetime(df["Application_DateTime"])
    df = df.sort_values("Application_DateTime").reset_index(drop=True)
    train_df = df.iloc[:800].copy()
    stream_pool = df.iloc[800:].copy().reset_index(drop=True)
    return df, train_df, stream_pool


@st.cache_resource
def train_stream_model(train_df, stream_pool):
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(train_df[FEATURES], train_df["Chance of Admit"])
    preds = model.predict(stream_pool[FEATURES])
    r2 = r2_score(stream_pool["Chance of Admit"], preds)
    mae = mean_absolute_error(stream_pool["Chance of Admit"], preds)
    return model, r2, mae


def classify_status(prob, admit_cutoff, waitlist_cutoff):
    if prob >= admit_cutoff:
        return "Admitted"
    if prob >= waitlist_cutoff:
        return "Waitlisted"
    return "Rejected"


def build_event(row_dict, app_id, uni, region, model, admit_cutoff, waitlist_cutoff, source="Live Feed"):
    X_in = pd.DataFrame([{f: row_dict[f] for f in FEATURES}])
    prob = float(np.clip(model.predict(X_in)[0], 0.0, 1.0))
    status = classify_status(prob, admit_cutoff, waitlist_cutoff)
    return {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Applicant_ID": int(app_id),
        "University": str(uni),
        "Region": str(region),
        "GRE Score": int(row_dict["GRE Score"]),
        "TOEFL Score": int(row_dict["TOEFL Score"]),
        "CGPA": round(float(row_dict["CGPA"]), 2),
        "Research": int(row_dict["Research"]),
        "Predicted_Chance": round(prob, 4),
        "Status": status,
        "Source": source,
    }


df_all, train_df, stream_pool = load_stream_datasets()
model, model_r2, model_mae = train_stream_model(train_df, stream_pool)

st.markdown(
    f"""
    <div class="hero-box">
        <p class="hero-title"><span class="live-dot"></span>AdmitStream // Real-Time Prediction Telemetry</p>
        <p class="hero-sub">
            Live chronological applicant ingestion with instant status classification, real-time score updates, and streaming analytics.
        </p>
        <div style="margin-top: 12px;">
            <span class="pill">LIVE FEED READY</span>
            <span class="pill">Stream Pool: {len(stream_pool)} Incoming Records</span>
            <span class="pill">Model R²: {model_r2:.3f} | MAE: {model_mae:.3f}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## 📡 Stream Controls")
admit_cutoff = st.sidebar.slider("Admit Cutoff", 0.65, 0.90, 0.75, 0.01)
waitlist_cutoff = st.sidebar.slider("Waitlist Cutoff", 0.45, 0.74, 0.60, 0.01)
batch_size = st.sidebar.slider("Events per Stream Burst", 5, 40, 12)
delay_sec = st.sidebar.slider("Stream Speed (sec/event)", 0.05, 1.50, 0.30, 0.05)

b1, b2 = st.sidebar.columns(2)
start_stream = b1.button("▶ Start Stream", use_container_width=True)
step_one = b2.button("⏭ Next Event", use_container_width=True)
reset_stream = st.sidebar.button("↺ Reset Feed", use_container_width=True)

if "live_events" not in st.session_state:
    initial_rows = []
    for i in range(6):
        r = stream_pool.iloc[i]
        initial_rows.append(
            build_event(r.to_dict(), r["Serial No."], r["University_Name"], r["region"], model, admit_cutoff, waitlist_cutoff)
        )
    st.session_state.live_events = pd.DataFrame(initial_rows)
    st.session_state.stream_cursor = 6
    st.session_state.last_banner = "Stream initialized with 6 live applicants from chronological feed."

if reset_stream:
    st.session_state.live_events = st.session_state.live_events.iloc[0:0]
    st.session_state.stream_cursor = 0
    st.session_state.last_banner = "Stream cleared. Click 'Start Stream' or push a manual update."

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚡ Push Live Applicant Update")
with st.sidebar.form("manual_live_form"):
    app_id = st.number_input("Applicant ID (new or existing)", 1, 9999, 801)
    uni_name = st.selectbox("University", sorted(df_all["University_Name"].unique().tolist()))
    region = st.selectbox("Region", sorted(df_all["region"].unique().tolist()))
    gre = st.slider("GRE Score", 290, 340, 326)
    toefl = st.slider("TOEFL Score", 92, 120, 112)
    rating = st.select_slider("University Rating", options=[1, 2, 3, 4, 5], value=4)
    sop = st.slider("SOP", 1.0, 5.0, 4.0, 0.5)
    lor = st.slider("LOR", 1.0, 5.0, 4.0, 0.5)
    cgpa = st.slider("CGPA", 6.80, 10.00, 9.05, 0.05)
    research = st.radio("Research", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
    push_btn = st.form_submit_button("🚀 Push Live Update", use_container_width=True)

if push_btn:
    payload = {
        "GRE Score": gre,
        "TOEFL Score": toefl,
        "University Rating": rating,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": research,
    }
    existing = st.session_state.live_events
    prev_match = existing[existing["Applicant_ID"] == int(app_id)]
    old_status = prev_match["Status"].iloc[-1] if len(prev_match) > 0 else None

    new_ev = build_event(payload, app_id, uni_name, region, model, admit_cutoff, waitlist_cutoff, source="Manual Update")
    cleaned = existing[existing["Applicant_ID"] != int(app_id)]
    st.session_state.live_events = pd.concat([cleaned, pd.DataFrame([new_ev])], ignore_index=True)

    if old_status:
        st.session_state.last_banner = (
            f"LIVE UPDATE | Applicant #{int(app_id)} ({uni_name}) status updated: {old_status} -> {new_ev['Status']} "
            f"({new_ev['Predicted_Chance'] * 100:.1f}%)"
        )
    else:
        st.session_state.last_banner = (
            f"NEW EVENT | Applicant #{int(app_id)} ({uni_name}) ingested -> {new_ev['Status']} "
            f"({new_ev['Predicted_Chance'] * 100:.1f}%)"
        )


def ingest_next_from_pool():
    idx = st.session_state.stream_cursor
    if idx >= len(stream_pool):
        idx = 0
    row = stream_pool.iloc[idx]
    ev = build_event(row.to_dict(), row["Serial No."], row["University_Name"], row["region"], model, admit_cutoff, waitlist_cutoff)
    st.session_state.live_events = pd.concat([st.session_state.live_events, pd.DataFrame([ev])], ignore_index=True)
    st.session_state.stream_cursor = idx + 1
    st.session_state.last_banner = (
        f"INCOMING STREAM | #{ev['Applicant_ID']} | {ev['University']} ({ev['Region']}) | "
        f"GRE {ev['GRE Score']}, CGPA {ev['CGPA']} -> {ev['Predicted_Chance'] * 100:.1f}% [{ev['Status'].upper()}]"
    )


if step_one:
    ingest_next_from_pool()

live_placeholder = st.empty()


def render_telemetry(events_df, banner_text):
    if len(events_df) > 0:
        events_df = events_df.copy()
        events_df["Status"] = events_df["Predicted_Chance"].apply(
            lambda p: classify_status(p, admit_cutoff, waitlist_cutoff)
        )

    with live_placeholder.container():
        st.markdown(f'<div class="ticker-box">⚡ {banner_text}</div>', unsafe_allow_html=True)

        total = len(events_df)
        n_adm = int((events_df["Status"] == "Admitted").sum()) if total else 0
        n_wait = int((events_df["Status"] == "Waitlisted").sum()) if total else 0
        n_rej = int((events_df["Status"] == "Rejected").sum()) if total else 0
        avg_p = float(events_df["Predicted_Chance"].mean() * 100) if total else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Events Processed</div>'
            f'<div class="kpi-value">{total}</div><div class="kpi-meta">Live Queue Active</div></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Admitted</div>'
            f'<div class="kpi-value" style="color:#34d399;">{n_adm}</div>'
            f'<div class="kpi-meta">{(n_adm / total * 100) if total else 0:.1f}% Rate</div></div>',
            unsafe_allow_html=True,
        )
        c3.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Waitlisted</div>'
            f'<div class="kpi-value" style="color:#fbbf24;">{n_wait}</div>'
            f'<div class="kpi-meta">{(n_wait / total * 100) if total else 0:.1f}% Rate</div></div>',
            unsafe_allow_html=True,
        )
        c4.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Rejected</div>'
            f'<div class="kpi-value" style="color:#fb7185;">{n_rej}</div>'
            f'<div class="kpi-meta">{(n_rej / total * 100) if total else 0:.1f}% Rate</div></div>',
            unsafe_allow_html=True,
        )
        c5.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Rolling Avg Chance</div>'
            f'<div class="kpi-value" style="color:#38bdf8;">{avg_p:.1f}%</div>'
            f'<div class="kpi-meta">Live Mean Score</div></div>',
            unsafe_allow_html=True,
        )

        st.write("")
        if total == 0:
            st.info("No live events yet. Click **Start Stream** or **Push Live Update** in the sidebar.")
            return

        left, right = st.columns([1.45, 0.95])

        with left:
            color_map = {"Admitted": "#34d399", "Waitlisted": "#fbbf24", "Rejected": "#fb7185"}
            marker_colors = [color_map[s] for s in events_df["Status"]]
            x_seq = list(range(1, total + 1))

            fig_trend = go.Figure()
            fig_trend.add_trace(
                go.Scatter(
                    x=x_seq,
                    y=events_df["Predicted_Chance"] * 100,
                    mode="lines+markers",
                    line=dict(color="#38bdf8", width=2.8),
                    fill="tozeroy",
                    fillcolor="rgba(56, 189, 248, 0.14)",
                    marker=dict(size=9, color=marker_colors, line=dict(color="#0f172a", width=1.5)),
                    text=events_df["University"] + " (" + events_df["Status"] + ")",
                    hovertemplate="<b>Event #%{x}</b><br>%{text}<br>Chance: %{y:.1f}%<extra></extra>",
                    name="Predicted Chance",
                )
            )
            fig_trend.add_hline(y=admit_cutoff * 100, line_dash="dash", line_color="#34d399", annotation_text="Admit Cutoff")
            fig_trend.add_hline(y=waitlist_cutoff * 100, line_dash="dot", line_color="#fbbf24", annotation_text="Waitlist Cutoff")
            fig_trend.update_layout(
                title="Live Prediction Stream Signal",
                xaxis=dict(title="Incoming Event Sequence", gridcolor="rgba(148,163,184,0.12)"),
                yaxis=dict(title="Predicted Chance (%)", range=[30, 102], gridcolor="rgba(148,163,184,0.12)"),
                height=340,
                margin=dict(l=20, r=20, t=45, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.48)",
                font={"color": "#e2e8f0"},
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with right:
            status_order = ["Admitted", "Waitlisted", "Rejected"]
            counts = events_df["Status"].value_counts().reindex(status_order, fill_value=0)
            fig_donut = go.Figure(
                go.Pie(
                    labels=status_order,
                    values=counts.values,
                    hole=0.58,
                    marker=dict(colors=["#34d399", "#fbbf24", "#fb7185"], line=dict(color="#070b14", width=2)),
                    textinfo="label+percent",
                )
            )
            fig_donut.update_layout(
                title="Live Status Mix",
                height=340,
                margin=dict(l=15, r=15, t=45, b=15),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"},
                showlegend=False,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        reg_avg = events_df.groupby("Region")["Predicted_Chance"].mean().sort_values(ascending=False) * 100
        fig_reg = go.Figure(
            go.Bar(
                x=reg_avg.index,
                y=reg_avg.values,
                text=[f"{v:.1f}%" for v in reg_avg.values],
                textposition="outside",
                marker=dict(color="#818cf8", line=dict(color="#c4b5fd", width=1)),
            )
        )
        fig_reg.update_layout(
            title="Live Regional Average Admission Chance",
            yaxis=dict(title="Avg Chance (%)", range=[0, 110], gridcolor="rgba(148,163,184,0.12)"),
            height=260,
            margin=dict(l=20, r=20, t=42, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.48)",
            font={"color": "#e2e8f0"},
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        st.markdown("### 📋 Live Incoming Applicant Feed (Newest First)")
        st.dataframe(events_df.iloc[::-1].head(20), use_container_width=True, hide_index=True)


if start_stream:
    for _ in range(batch_size):
        ingest_next_from_pool()
        render_telemetry(st.session_state.live_events, st.session_state.last_banner)
        time.sleep(delay_sec)
else:
    render_telemetry(st.session_state.live_events, st.session_state.last_banner)

if len(st.session_state.live_events) > 0:
    csv_bytes = st.session_state.live_events.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Live Stream Log (CSV)",
        data=csv_bytes,
        file_name="live_admission_stream.csv",
        mime="text/csv",
    )
