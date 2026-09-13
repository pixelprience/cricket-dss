import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. LOAD THE SAVED AI MODELS ---
rf_model = joblib.load('models/cricket_rf_model.joblib')
model_features = joblib.load('models/model_features.joblib')

# --- 2. BUILD THE USER INTERFACE ---
st.set_page_config(page_title="T20 Predictor", layout="wide")
st.title("🏏 T20 Cricket Win Predictor")
st.markdown("### Live Decision Support System (DSS)")

st.sidebar.header("Match Context")
batting_team = st.sidebar.selectbox("Batting Team", ['Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'])
bowling_team = st.sidebar.selectbox("Bowling Team", ['Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'], index=1)
venue = st.sidebar.selectbox("Venue", ['Wankhede Stadium', 'Eden Gardens', 'M Chinnaswamy Stadium', 'Arun Jaitley Stadium', 'MA Chidambaram Stadium'])

st.sidebar.header("Current Match Situation")
target_score = st.sidebar.number_input("Target Score", min_value=1, value=180)
current_score = st.sidebar.number_input("Current Score", min_value=0, value=90)
overs_completed = st.sidebar.number_input("Overs Completed", min_value=0, max_value=19, value=10)
balls_in_current_over = st.sidebar.number_input("Balls in Current Over", min_value=0, max_value=5, value=0)
wickets_lost = st.sidebar.number_input("Wickets Lost", min_value=0, max_value=10, value=2)
runs_last_5 = st.sidebar.number_input("Runs in Last 5 Overs", min_value=0, value=45)
wickets_last_5 = st.sidebar.number_input("Wickets in Last 5 Overs", min_value=0, max_value=10, value=1)

# --- 3. THE PREDICTION ENGINE ---
if st.button("Calculate Live Win Probability", type="primary"):
    
    # Calculate the dynamic momentum features
    balls_bowled = (overs_completed * 6) + balls_in_current_over
    balls_remaining = 120 - balls_bowled
    runs_required = target_score - current_score
    wickets_in_hand = 10 - wickets_lost
    
    crr = current_score / (balls_bowled / 6) if balls_bowled > 0 else 0
    rrr = (runs_required / balls_remaining) * 6 if balls_remaining > 0 else 0

    # Package the user inputs
    input_data = {
        'target_score': target_score,
        'current_score': current_score,
        'runs_required': runs_required,
        'wickets_in_hand': wickets_in_hand,
        'balls_remaining': balls_remaining,
        'crr': crr,
        'rrr': rrr,
        'runs_last_5_overs': runs_last_5,
        'wickets_last_5_overs': wickets_last_5
    }

    input_df = pd.DataFrame([input_data])
    categorical_data = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'venue': [venue]})
    categorical_encoded = pd.get_dummies(categorical_data)
    final_input = pd.concat([input_df, categorical_encoded], axis=1)
    final_input = final_input.reindex(columns=model_features, fill_value=0)

    # Ask the Random Forest model for its live probability read
    probability = rf_model.predict_proba(final_input)[0]
    loss_prob = probability[0] * 100
    win_prob = probability[1] * 100

    # --- 4. DISPLAY THE RESULTS & VISUALS ---
    st.write("---")
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Live Match Momentum")
        st.metric(label=f"{batting_team} Win Probability", value=f"{win_prob:.1f}%")
        st.metric(label=f"{bowling_team} Win Probability", value=f"{loss_prob:.1f}%")
        
        st.info(f"**Strategic Insight:** At {crr:.1f} runs per over, the chasing team is currently maintaining pressure. However, they must monitor their required run rate of {rrr:.1f} while preserving their {wickets_in_hand} remaining wickets.")

    with col2:
        st.subheader("What is Driving this Prediction?")
        # Generate the Feature Importance Chart
        importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': model_features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(5)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
        ax.set_title("Top 5 Momentum Drivers", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        st.pyplot(fig)