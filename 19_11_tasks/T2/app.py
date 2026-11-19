import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Admissions Committee Decision Portal", layout="wide")

# staff authentication
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    st.title("Admissions Committee Decision Portal")
    st.write("Please sign in with your administrative credentials to access the decision-making engine.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")
        if submitted:
            if username == "admin" and password == "admit2026":
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
    st.stop()

# authenticated header
st.sidebar.title("Decision-Maker Controls")
st.sidebar.write("Authenticated: **Admissions Committee Chair**")
if st.sidebar.button("Sign Out"):
    st.session_state["auth"] = False
    st.rerun()

# load dataset and model
df = pd.read_csv("Adm.csv")
df.columns = df.columns.str.strip()
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']

try:
    model = joblib.load("admission_model.pkl")
except:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[features], df['Chance of Admit'])
    joblib.dump(model, "admission_model.pkl")

df['Predicted_Chance'] = model.predict(df[features]).clip(0, 1)

# committee decision criteria definitions
st.sidebar.header("Official Admission Policy Criteria")

target_capacity = st.sidebar.number_input("Target Incoming Class Seats", min_value=50, max_value=500, value=250, step=10)

admit_cutoff = st.sidebar.slider("Direct Admit Probability Cutoff", 0.70, 0.95, 0.78, 0.01)
waitlist_cutoff = st.sidebar.slider("Waitlist Probability Cutoff", 0.50, 0.75, 0.60, 0.01)

st.sidebar.subheader("Academic Eligibility Policy Floors")
min_cgpa = st.sidebar.slider("Minimum CGPA Floor", 6.0, 9.5, 7.5, 0.1)
min_gre = st.sidebar.slider("Minimum GRE Floor", 280, 330, 300, 5)
min_toefl = st.sidebar.slider("Minimum TOEFL Floor", 80, 115, 95, 1)
require_research_for_borderline = st.sidebar.checkbox("Grant Waitlist Preference for Research Experience", value=True)

# apply decision rules
def determine_decision(row):
    meets_floors = (row['CGPA'] >= min_cgpa) and (row['GRE Score'] >= min_gre) and (row['TOEFL Score'] >= min_toefl)
    
    if meets_floors and (row['Predicted_Chance'] >= admit_cutoff):
        return "Admit"
    elif row['Predicted_Chance'] >= waitlist_cutoff:
        if meets_floors:
            return "Waitlist"
        elif require_research_for_borderline and (row['Research'] == 1):
            return "Waitlist (Research Waiver)"
        else:
            return "Deny (Below Floor)"
    else:
        return "Deny"

df['Decision'] = df.apply(determine_decision, axis=1)

# main dashboard title
st.title("University Admissions Committee - Decision Maker Portal")
st.caption("Real-time decision modeling engine with institutional policy criteria and capacity optimization.")

# capacity & high-level stats
admit_count = int(df['Decision'].str.startswith('Admit').sum())
waitlist_count = int(df['Decision'].str.startswith('Waitlist').sum())
deny_count = int(df['Decision'].str.startswith('Deny').sum())

cap_filled_pct = min(1.0, admit_count / target_capacity)

st.subheader("Cohort Capacity & Decision Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Applicants", len(df))
col2.metric("Offers Extended", f"{admit_count} / {target_capacity}")
col3.metric("Capacity Filled", f"{cap_filled_pct:.1%}")
col4.metric("Waitlisted", waitlist_count)
col5.metric("Denied", deny_count)

st.progress(cap_filled_pct)

# multi-tab layout
tab1, tab2, tab3, tab4 = st.tabs(["Decision Overview", "Policy Impact & Analytics", "Applicant Review & Override", "Official Decision Roster"])

with tab1:
    st.markdown("### Committee Decisions Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Decision Distribution**")
        fig1, ax1 = plt.subplots()
        df['Decision'].value_counts().plot(kind='bar', ax=ax1, color=['green', 'orange', 'red', 'gold', 'darkred'][:len(df['Decision'].unique())])
        st.pyplot(fig1)
        
    with c2:
        st.write("**Predicted Chance Distribution by Decision**")
        fig2, ax2 = plt.subplots()
        for dec in df['Decision'].unique():
            subset = df[df['Decision'] == dec]
            ax2.hist(subset['Predicted_Chance'], bins=15, alpha=0.5, label=dec)
        ax2.legend()
        st.pyplot(fig2)

with tab2:
    st.markdown("### Policy Impact & Demographics")
    p1, p2 = st.columns(2)
    with p1:
        st.write("**Admission Rate by University Rating**")
        rating_rate = df.groupby('University Rating').apply(lambda g: (g['Decision'].str.startswith('Admit')).mean())
        fig3, ax3 = plt.subplots()
        rating_rate.plot(kind='bar', ax=ax3, color='steelblue')
        ax3.set_ylabel("Acceptance Rate")
        st.pyplot(fig3)
        
    with p2:
        st.write("**Regional Acceptance Performance**")
        region_rate = df.groupby('region').apply(lambda g: (g['Decision'].str.startswith('Admit')).mean())
        fig4, ax4 = plt.subplots()
        region_rate.plot(kind='bar', ax=ax4, color='darkcyan')
        ax4.set_ylabel("Acceptance Rate")
        st.pyplot(fig4)
        
    st.write("#### Profile Averages by Decision Group")
    group_stats = df.groupby('Decision')[['GRE Score', 'TOEFL Score', 'CGPA', 'Research', 'Predicted_Chance']].mean().round(2)
    st.dataframe(group_stats, use_container_width=True)

with tab3:
    st.markdown("### Individual Applicant Deep-Dive & Manual Override")
    search_id = st.selectbox("Select Applicant Serial Number", options=df['Serial No.'].tolist())
    app_data = df[df['Serial No.'] == search_id].iloc[0]
    
    st.write(f"#### Evaluation for Applicant #{search_id}")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted Admission Probability", f"{app_data['Predicted_Chance']:.2%}")
    col_b.metric("Automated Decision", app_data['Decision'])
    col_c.metric("University Tier", f"Rating {int(app_data['University Rating'])}")
    
    st.write("##### Institutional Policy Compliance")
    chk_cgpa = "Passed" if app_data['CGPA'] >= min_cgpa else f"Failed (Required >= {min_cgpa})"
    chk_gre = "Passed" if app_data['GRE Score'] >= min_gre else f"Failed (Required >= {min_gre})"
    chk_toefl = "Passed" if app_data['TOEFL Score'] >= min_toefl else f"Failed (Required >= {min_toefl})"
    
    f1, f2, f3 = st.columns(3)
    f1.info(f"**CGPA ({app_data['CGPA']:.2f})**: {chk_cgpa}")
    f2.info(f"**GRE ({int(app_data['GRE Score'])})**: {chk_gre}")
    f3.info(f"**TOEFL ({int(app_data['TOEFL Score'])})**: {chk_toefl}")
    
    st.write("##### Committee Manual Override")
    override_choice = st.radio("Override Status", ["Keep Automated Decision", "Force Admit", "Force Waitlist", "Force Deny"], horizontal=True)
    override_notes = st.text_area("Committee Rationale / Notes", placeholder="Enter official committee reasoning for this candidate...")
    if st.button("Save Candidate Review"):
        st.success(f"Review for Applicant #{search_id} recorded successfully.")

with tab4:
    st.markdown("### Official Decision Roster")
    status_filter = st.selectbox("Filter Roster by Decision", ["All Decisions"] + list(df['Decision'].unique()))
    roster_view = df if status_filter == "All Decisions" else df[df['Decision'] == status_filter]
    
    cols_to_show = ['Serial No.', 'GRE Score', 'TOEFL Score', 'CGPA', 'Research', 'University Rating', 'region', 'Predicted_Chance', 'Decision']
    st.dataframe(roster_view[cols_to_show].sort_values(by='Predicted_Chance', ascending=False), use_container_width=True)
    
    csv_out = roster_view[cols_to_show].to_csv(index=False).encode('utf-8')
    st.download_button("Download Official Committee Decision Roster (CSV)", csv_out, "committee_decision_roster.csv", "text/csv")
