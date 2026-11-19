import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Admission Prediction & Report Portal", layout="wide")

st.title("University Admission - Dynamic Prediction & Reporting System")
st.caption("Comprehensive evaluation portal for admission officers to analyze applicant datasets dynamically.")

# sidebar controls for dataset and thresholds
st.sidebar.header("Dataset & Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Applicant Batch (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Uploaded custom batch: {len(df)} records")
else:
    df = pd.read_csv("Adm.csv")
    st.sidebar.info("Using standard dataset: Adm.csv")

df.columns = df.columns.str.strip()
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']

# load or train model
try:
    model = joblib.load("admission_model.pkl")
except:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[features], df['Chance of Admit'])
    joblib.dump(model, "admission_model.pkl")

# model predictions and dynamic scoring
df['Predicted_Chance'] = model.predict(df[features]).clip(0, 1)
df['Percentile'] = (df['Predicted_Chance'].rank(pct=True) * 100).round(1)

# classification criteria
tier_cutoff_high = st.sidebar.slider("High Likelihood Cutoff (Tier 1)", 0.70, 0.95, 0.80, 0.05)
tier_cutoff_med = st.sidebar.slider("Competitive Cutoff (Tier 2)", 0.50, 0.75, 0.60, 0.05)

df['Admission_Tier'] = np.where(df['Predicted_Chance'] >= tier_cutoff_high, 'Tier 1 - Strong Admit',
                       np.where(df['Predicted_Chance'] >= tier_cutoff_med, 'Tier 2 - Competitive Review', 'Tier 3 - Low Probability'))

# sidebar filters
st.sidebar.header("Filter Applicants")
selected_tier = st.sidebar.multiselect("Select Tiers", options=df['Admission_Tier'].unique(), default=df['Admission_Tier'].unique())
selected_research = st.sidebar.radio("Research Experience", ["All", "Only with Research", "Without Research"])

filtered_df = df[df['Admission_Tier'].isin(selected_tier)]
if selected_research == "Only with Research":
    filtered_df = filtered_df[filtered_df['Research'] == 1]
elif selected_research == "Without Research":
    filtered_df = filtered_df[filtered_df['Research'] == 0]

# executive summary metrics
t1_count = int((df['Admission_Tier'] == 'Tier 1 - Strong Admit').sum())
t2_count = int((df['Admission_Tier'] == 'Tier 2 - Competitive Review').sum())
t3_count = int((df['Admission_Tier'] == 'Tier 3 - Low Probability').sum())
avg_chance = df['Predicted_Chance'].mean()
top_cgpa = df[df['Admission_Tier'] == 'Tier 1 - Strong Admit']['CGPA'].mean() if t1_count > 0 else 0

st.subheader("Batch Executive Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Applicants", len(df))
col2.metric("Tier 1 (Strong)", f"{t1_count} ({t1_count/len(df)*100:.1f}%)")
col3.metric("Tier 2 (Competitive)", f"{t2_count} ({t2_count/len(df)*100:.1f}%)")
col4.metric("Tier 3 (Low)", f"{t3_count} ({t3_count/len(df)*100:.1f}%)")
col5.metric("Avg Predicted Chance", f"{avg_chance:.2%}")

# tabs for detailed report views
tab1, tab2, tab3 = st.tabs(["Prediction Report & Table", "Cohort Visual Analytics", "Individual Applicant Scorecard"])

with tab1:
    st.markdown("### Applicant Predictions Table")
    st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** applicants based on active filters.")
    
    display_cols = ['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'CGPA', 'Research', 'Predicted_Chance', 'Percentile', 'Admission_Tier']
    st.dataframe(filtered_df[display_cols].sort_values(by='Predicted_Chance', ascending=False), use_container_width=True)
    
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Predictions CSV", csv_bytes, "applicant_predictions.csv", "text/csv")

with tab2:
    st.markdown("### Cohort Visual Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Predicted Admission Chance Distribution**")
        fig1, ax1 = plt.subplots()
        df['Predicted_Chance'].plot(kind='hist', bins=15, ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_xlabel("Predicted Chance")
        st.pyplot(fig1)
        
    with c2:
        st.write("**Applicant Distribution by Admission Tier**")
        fig2, ax2 = plt.subplots()
        df['Admission_Tier'].value_counts().plot(kind='bar', ax=ax2, color=['mediumseagreen', 'orange', 'salmon'])
        st.pyplot(fig2)
        
    c3, c4 = st.columns(2)
    with c3:
        st.write("**CGPA vs Predicted Admission Chance**")
        fig3, ax3 = plt.subplots()
        ax3.scatter(df['CGPA'], df['Predicted_Chance'], alpha=0.6, color='steelblue')
        ax3.set_xlabel("CGPA")
        ax3.set_ylabel("Predicted Chance")
        st.pyplot(fig3)
        
    with c4:
        st.write("**Average Admission Chance by University Rating**")
        fig4, ax4 = plt.subplots()
        df.groupby('University Rating')['Predicted_Chance'].mean().plot(kind='bar', ax=ax4, color='teal')
        ax4.set_ylabel("Mean Chance")
        st.pyplot(fig4)

with tab3:
    st.markdown("### Individual Applicant Deep-Dive")
    applicant_ids = df['Serial No.'].tolist()
    chosen_id = st.selectbox("Select Applicant ID", options=applicant_ids)
    
    app_row = df[df['Serial No.'] == chosen_id].iloc[0]
    
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Predicted Chance", f"{app_row['Predicted_Chance']:.2%}")
    sc2.metric("Cohort Percentile", f"Top {100 - app_row['Percentile']:.1f}%")
    sc3.metric("Assigned Status", app_row['Admission_Tier'])
    
    st.write("#### Applicant Profile Breakdown")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("GRE Score", f"{int(app_row['GRE Score'])} / 340")
    p2.metric("TOEFL Score", f"{int(app_row['TOEFL Score'])} / 120")
    p3.metric("Undergrad CGPA", f"{app_row['CGPA']:.2f} / 10.0")
    p4.metric("Research Experience", "Yes" if app_row['Research'] == 1 else "No")
    
    p5, p6, p7 = st.columns(3)
    p5.metric("University Rating", f"{int(app_row['University Rating'])} / 5")
    p6.metric("Statement of Purpose (SOP)", f"{app_row['SOP']:.1f} / 5.0")
    p7.metric("Letter of Rec (LOR)", f"{app_row['LOR']:.1f} / 5.0")
    
    # automated feedback summary
    st.write("#### Admission Recommendation Summary")
    if app_row['Predicted_Chance'] >= tier_cutoff_high:
        st.success("Recommendation: Expedited Direct Admission. Strong academic metrics exceeding cohort benchmarks.")
    elif app_row['Predicted_Chance'] >= tier_cutoff_med:
        st.warning("Recommendation: Committee Holistic Review. Competitive candidate with strengths suitable for departmental evaluation.")
    else:
        st.error("Recommendation: Deny Admission or Transfer Pathway. Candidate metrics below competitive threshold.")
