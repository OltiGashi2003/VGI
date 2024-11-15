import streamlit as st
import pandas as pd
import joblib
from prophet import Prophet
from datetime import datetime
from streamlit_navigation_bar import st_navbar
import streamlit_home 
import streamlit_products
import streamlit_about


# Function to load all pre-trained models
def load_all_models(pickup_ids):
    models = {}
    for pickup_id in pickup_ids: 
        model_filename = f"prophet_model_pickup_{pickup_id}.pkl"
        try:
            models[pickup_id] = joblib.load(model_filename)
        except FileNotFoundError:
            st.write(f"Model for Pickup ID {pickup_id} not found.")
    return models

# Function to predict demand using a selected model
def predict_demand(model, forecast_time):
    future = pd.DataFrame({'ds': [forecast_time]})
    future['Weekday'] = future['ds'].dt.dayofweek
    forecast = model.predict(future)
    predicted_demand = forecast['yhat'].values[0]
    return int(round(predicted_demand))


# Create the top navigation bar
page = st.radio("", ["Home", "Products", "About Us"], horizontal=True, key="page_navigation")
st.write(page)

if page == "Home":
    streamlit_home.run()
if page == "Products":
    streamlit_products.run()
if page == "About Us":
    streamlit_about.run()