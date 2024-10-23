import streamlit as st
import joblib

model = joblib.load('regression.joblib')

with st.form(key='prediction_form'):
    size = st.number_input('Size (sq ft)', min_value=0)
    bedrooms = st.number_input('Number of Bedrooms', min_value=0)
    garden = st.number_input('Has Garden (1 for Yes, 0 for No)', min_value=0, max_value=1)
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    prediction = model.predict([[size, bedrooms, garden]])
    st.write(f'The predicted price is: ${prediction[0]:,.2f}')
