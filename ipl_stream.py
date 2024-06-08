import streamlit as st
import pickle
import pandas as pd

# List of teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

# Load the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Streamlit app title
st.title('IPL Win Predictor')

# Function to style input fields
def colorful_input(label, options, key):
    return st.markdown(f'<div style="background-color:#f0f0f5;padding:10px;border-radius:10px;"><label style="color:#4B0082;font-size:16px;">{label}</label><br><select style="border-radius:5px;border:2px solid #800080;padding:7px;" key="{key}">' +
                       ''.join([f'<option value="{option}">{option}</option>' for option in options]) +
                       '</select></div>', unsafe_allow_html=True)

# Layout for input fields
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))
with col2:
    selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=0, max_value=300, step=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, max_value=300, step=1)
with col4:
    overs = st.number_input('Overs completed', step=1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Prediction button
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    # Create input DataFrame for the model
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict probabilities
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Display results
    st.header(batting_team + " - " + str(round(win * 100, 2)) + "%")
    st.header(bowling_team + " - " + str(round(loss * 100, 2)) + "%")
