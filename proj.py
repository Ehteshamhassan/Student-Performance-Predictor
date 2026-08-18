import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. TRAIN THE AI (Cached for speed)
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000
    
    study = np.random.uniform(2, 40, n)
    prev_grade = np.random.uniform(50, 100, n)
    attendance = np.random.uniform(50, 100, n)
    sleep = np.random.uniform(4, 10, n)
    
    extracurricular = np.random.choice([0, 1], n)
    part_time_job = np.random.choice([0, 1], n)
    course_code = np.random.choice([0, 1, 2, 3], n) 
    
    score = (0.3 * prev_grade) + (0.25 * attendance) + (0.4 * study) + (1.2 * sleep)
    score += (extracurricular * 2.0) - (part_time_job * 3.5) - (course_code * 2.5) 
    score += np.random.normal(0, 3, n) 
    score = np.clip(score, 0, 100)
    
    X = pd.DataFrame({
        'Course_Code': course_code,
        'Study_Hours': study,
        'Previous_Grade': prev_grade,
        'Attendance': attendance,
        'Sleep_Hours': sleep,
        'Extracurricular': extracurricular,
        'Part_Time_Job': part_time_job
    })
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, score)
    return model, X.columns

ai_model, feature_names = train_model()

# ---------------------------------------------------------
# 2. STATE MANAGEMENT & CUSTOM UI WIDGET
# ---------------------------------------------------------
# Default values for our inputs
defaults = {"prev": 75.0, "study": 15.0, "att": 85.0, "sleep": 7.0}

def reset_inputs():
    st.session_state.course = "Computer Science"
    st.session_state.extra = "No"
    st.session_state.job = "No"
    # Reset all synchronized inputs
    for key, val in defaults.items():
        st.session_state[f"{key}_slider"] = val
        st.session_state[f"{key}_text"] = val

# Initialize defaults on first load
if 'course' not in st.session_state:
    reset_inputs()

# Custom UI Component: A synchronized slider and text box
def synced_input(label, min_val, max_val, step, key):
    st.markdown(f"**{label}**")
    
    # Define callbacks to keep both widgets in sync
    def update_from_slider():
        st.session_state[f"{key}_text"] = st.session_state[f"{key}_slider"]
        
    def update_from_text():
        st.session_state[f"{key}_slider"] = st.session_state[f"{key}_text"]
        
    # Layout: Slider gets 70% width, Text Input gets 30% width
    col_slider, col_text = st.columns([3, 1.5])
    
    with col_slider:
        st.slider(label, min_value=float(min_val), max_value=float(max_val), step=float(step), 
                  key=f"{key}_slider", on_change=update_from_slider, label_visibility="collapsed")
    with col_text:
        st.number_input(label, min_value=float(min_val), max_value=float(max_val), step=float(step), 
                        key=f"{key}_text", on_change=update_from_text, label_visibility="collapsed")

# ---------------------------------------------------------
# 3. BUILD THE INTERACTIVE WEB UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Student Predictor", page_icon="🎓", layout="wide")
st.title("AI Student Performance Predictor")
st.write("Adjust the metrics using the sliders or type exact values to predict the final score.")

course_map = {"Arts": 0, "Commerce": 1, "Mechanical Engineering": 2, "Computer Science": 3}
binary_map = {"No": 0, "Yes": 1}

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Academic Profile")
    st.selectbox("**Select Course/Major**", list(course_map.keys()), key="course")
    st.write("") # Spacer
    synced_input("Previous Semester Grade (%)", 0.0, 100.0, 1.0, "prev")

with col2:
    st.subheader("Study Habits")
    synced_input("Weekly Study Hours", 0.0, 40.0, 1.0, "study")
    synced_input("Attendance (%)", 0.0, 100.0, 1.0, "att")

with col3:
    st.subheader("Lifestyle Factors")
    synced_input("Average Sleep (Hours/Night)", 2.0, 12.0, 0.5, "sleep")
    st.radio("**Extracurricular Activities?**", ["No", "Yes"], key="extra", horizontal=True)
    st.radio("**Part-Time Job?**", ["No", "Yes"], key="job", horizontal=True)

st.markdown("---")

# Buttons Row
btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    predict_clicked = st.button("Predict Final Score", type="primary", use_container_width=True)
with btn_col2:
    st.button("Reset Inputs", on_click=reset_inputs, use_container_width=True)

# ---------------------------------------------------------
# 4. RUN PREDICTION & RENDER CHARTS
# ---------------------------------------------------------
if predict_clicked:
    
    # Pull the values from the slider session state
    user_data = pd.DataFrame({
        'Course_Code': [course_map[st.session_state.course]],
        'Study_Hours': [st.session_state.study_slider],
        'Previous_Grade': [st.session_state.prev_slider],
        'Attendance': [st.session_state.att_slider],
        'Sleep_Hours': [st.session_state.sleep_slider],
        'Extracurricular': [binary_map[st.session_state.extra]],
        'Part_Time_Job': [binary_map[st.session_state.job]]
    })
    
    prediction = ai_model.predict(user_data)[0]
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown("### Prediction Result")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Final Score", 'font': {'size': 24}},
            number = {'suffix': "%"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightcoral"},
                    {'range': [50, 75], 'color': "khaki"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        if prediction >= 85:
            st.balloons()
            
    with res_col2:
        st.markdown("### How the AI Made This Decision")
        st.write("This chart shows which factors carry the most weight in the AI's calculation.")
        
        importances = ai_model.feature_importances_
        clean_names = [name.replace('_', ' ') for name in feature_names]
        
        chart_data = pd.DataFrame({"Importance": importances}, index=clean_names)
        chart_data = chart_data.sort_values(by="Importance", ascending=True)
        
        st.bar_chart(chart_data, horizontal=True)
