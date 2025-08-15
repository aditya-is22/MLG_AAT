import streamlit as st
import pickle
import math
import numpy as np

@st.cache_resource
def load_ipl_models():
    with open('ipl_models.pkl', 'rb') as f:
        components = pickle.load(f)
    return (
        components['models'],
        components['pca'],
        components['scaler'],
        components['results']
    )

# Load models and components once and reuse
models, pca, scaler, results = load_ipl_models()
# Page config
st.set_page_config(page_title='IPL Score Predictor', layout="centered")

# Title and styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #FF4B4B;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 30px;
    }
    </style>
    <div class="main-header">IPL Score Predictor 2024</div>
""", unsafe_allow_html=True)

# Background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://4.bp.blogspot.com/-F6aZF5PMwBQ/Wrj5h204qxI/AAAAAAAABao/4QLn48RP3x0P8Ry0CcktxilJqRfv1IfcACLcBGAs/s1600/GURU%2BEDITZ%2Bbackground.jpg");
        background-attachment: fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Model selection
model_name = st.selectbox(
    'Select Prediction Model',
    ['random_forest', 'xgboost', 'stacking', 'pca_rf'],
    format_func=lambda x: {
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'stacking': 'Stacking Ensemble',
        'pca_rf': 'PCA Random Forest'
    }[x]
)

# Show model performance
st.sidebar.header("Model Performance")
metrics = results[model_name]
st.sidebar.metric("R² Score", f"{metrics['r2']:.3f}")
st.sidebar.metric("RMSE", f"{metrics['rmse']:.2f}")
st.sidebar.metric("MAE", f"{metrics['mae']:.2f}")

# Team selection
teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
         'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
         'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Batting Team', teams)
with col2:
    bowling_team = st.selectbox('Bowling Team', teams, index=1)

if batting_team == bowling_team:
    st.error('Batting and Bowling teams must be different!')

# Match statistics input
col3, col4 = st.columns(2)
with col3:
    overs = st.number_input('Current Over', min_value=5.1, max_value=19.5, value=5.1, step=0.1)
    if overs - math.floor(overs) > 0.5:
        st.error('Please enter valid over (6 balls per over)')
with col4:
    runs = st.number_input('Current Runs', min_value=0, max_value=354, value=0, step=1)

wickets = st.slider('Wickets Fallen', 0, 9, 0)

col5, col6 = st.columns(2)
with col5:
    runs_last_5 = st.number_input('Runs (Last 5 Overs)', min_value=0, max_value=runs, value=0, step=1)
with col6:
    wickets_last_5 = st.number_input('Wickets (Last 5 Overs)', min_value=0, max_value=wickets, value=0, step=1)

# Create prediction array
def create_prediction_array(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5):
    prediction_array = []
    
    # Batting Team
    for team in teams:
        prediction_array.append(1 if team == batting_team else 0)
    
    # Bowling Team
    for team in teams:
        prediction_array.append(1 if team == bowling_team else 0)
    
    # Add other features
    prediction_array.extend([runs, wickets, overs, runs_last_5, wickets_last_5])
    
    return np.array([prediction_array])

# Make prediction
def predict_score(prediction_array, model_name):
    if model_name == 'pca_rf':
        prediction_array_transformed = scaler.transform(prediction_array)
        prediction_array_transformed = pca.transform(prediction_array_transformed)
        prediction = models['pca_rf'].predict(prediction_array_transformed)
    elif model_name == 'stacking':
        rf_pred = models['stacking']['rf'].predict(prediction_array)
        xgb_pred = models['stacking']['xgb'].predict(prediction_array)
        stack_features = np.column_stack((rf_pred, xgb_pred))
        prediction = models['stacking']['meta'].predict(stack_features)
    else:
        prediction = models[model_name].predict(prediction_array)
    
    return int(round(prediction[0]))

# Prediction button
if st.button('Predict Score', key='predict'):
    if batting_team != bowling_team:
        # Create prediction array
        prediction_array = create_prediction_array(
            batting_team, bowling_team, runs, wickets, overs,
            runs_last_5, wickets_last_5
        )
        
        # Get prediction
        prediction = predict_score(prediction_array, model_name)
        
        # Display prediction
        st.success(f'Predicted Final Score: {prediction-5} to {prediction+5} runs')
        
        # Additional insights
        run_rate = runs/overs if overs > 0 else 0
        required_run_rate = (prediction-runs)/(20-overs) if overs < 20 else 0
        
        # Display insights
        col7, col8 = st.columns(2)
        with col7:
            st.info(f'Current Run Rate: {run_rate:.2f}')
        with col8:
            st.info(f'Required Run Rate: {required_run_rate:.2f}')