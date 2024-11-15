import streamlit as st 
from app import load_all_models, predict_demand
from datetime import datetime
import pandas as pd
import plotly.express as px
import altair as alt
import folium
from streamlit_folium import folium_static
import requests
from sklearn.cluster import KMeans
from geopy.distance import geodesic
from prophet import Prophet
import joblib


def run():
    st.title("VGI FLEXI - App")
    # st.subheader("Products - Demand Forecasting & Visualization")?

    # Define available products
    products = {
        "Forecasting Demand": "forecast_demand",
        "Actual Demand": "actual_demand"
    }

    # Select the product
    selected_product = st.selectbox("Select a Product", list(products.keys()))

    if selected_product == "Forecasting Demand":
        # List of Pickup IDs (replace with actual Pickup IDs)
        pickup_ids = [0, 8, 19, 30, 31]

        # Load all models
        with st.spinner('Loading models...'):
            models = load_all_models(pickup_ids)
        st.success("Models loaded.")

        # User input: Select Pickup ID
        pickup_id = st.selectbox("Select Pickup ID", pickup_ids)

        # User input: Date and Time
        forecast_date = st.date_input("Select Date", value=datetime.today())
        forecast_hour = st.slider("Select Hour", min_value=0, max_value=23, step=1)
        forecast_datetime = datetime.combine(forecast_date, datetime.min.time()) + pd.Timedelta(hours=forecast_hour)

        st.write(f"Forecasting for Pickup ID {pickup_id} on {forecast_datetime.strftime('%Y-%m-%d %H:%M')}")

        # Make prediction
        if st.button("Get Prediction"):
            if pickup_id in models:
                model = models[pickup_id]
                demand = predict_demand(model, forecast_datetime)
                st.success(f"Predicted demand for Pickup ID {pickup_id} at {forecast_datetime.strftime('%Y-%m-%d %H:%M')} is {demand} bookings.")
            else:
                st.error(f"Model for Pickup ID {pickup_id} is not available.")

    # elif selected_product == "Actual Demand":
    #     # Add functionality to visualize actual demand
    #     # Here you can add code to load the actual demand data and visualize it.
    #     # Example: Load actual demand data from a CSV or database

    #     st.subheader("Visualizing Actual Demand")
    
    elif selected_product == "Actual Demand":
        data = pd.read_csv('datasets/cleaned_dataset.csv', usecols=['Actual Pickup Time', 'Actual Dropoff Time', 'Passenger status', "district", "Pickup Latitude", "Pickup Longitude"])

        # Convert date columns to datetime format for easier manipulation
        data['Actual Pickup Time'] = pd.to_datetime(data['Actual Pickup Time'])
        data['Actual Dropoff Time'] = pd.to_datetime(data['Actual Dropoff Time'])
        data['week'] = data['Actual Pickup Time'].dt.isocalendar().week
        data['weekday'] = data['Actual Pickup Time'].dt.day_name()  # Extract weekday
        data['hour'] = data['Actual Pickup Time'].dt.hour           # Extract hour

        # Sidebar filters
        st.sidebar.title("🚍VGI-Flexi Dashboard")
        view_option = st.sidebar.radio("View Data", ["Weekly Trends", "General Trends"])
        selected_week = st.sidebar.selectbox("Select Week", sorted(data['week'].unique()), disabled=(view_option == "General Trends"))
        selected_status = st.sidebar.multiselect("Select Trip Status", options=data['Passenger status'].unique(), default=data['Passenger status'].unique())
        selected_weekdays = st.sidebar.multiselect("Select Weekday(s)", options=data['weekday'].unique(), default=data['weekday'].unique())  # Weekday filter
        selected_hours = st.sidebar.slider("Select Hour of Day", 0, 23, (0, 23))  # Hour of the day filter

        # Filter data based on selections
        if view_option == "Weekly Trends":
            filtered_data = data[
                (data['week'] == selected_week) &
                (data['Passenger status'].isin(selected_status)) &
                (data['weekday'].isin(selected_weekdays)) &
                (data['hour'].between(selected_hours[0], selected_hours[1]))
            ]
        else:
            filtered_data = data[
                (data['Passenger status'].isin(selected_status)) &
                (data['weekday'].isin(selected_weekdays)) &
                (data['hour'].between(selected_hours[0], selected_hours[1]))
            ]

        # Layout setup
        st.title("VGI-Flexi Statistics Dashboard")

        # Summary Metrics with wider columns
        st.subheader("Summary Metrics")
        col1, col2, col3 = st.columns([2.0, 2.0, 2.0])
        total_trips = filtered_data.shape[0]
        completed_trips = filtered_data[filtered_data['Passenger status'] == 'Trip completed'].shape[0]
        canceled_trips = filtered_data[filtered_data['Passenger status'] == 'Cancelled'].shape[0]
        col1.metric("Total Trips", total_trips)
        col2.metric("Completed Trips", completed_trips)
        col3.metric("Canceled Trips", canceled_trips)

        # Top Districts with Most Canceled and Completed Trips with wider columns
        st.subheader("Top Districts by Trip Status")
        col1, col2 = st.columns([1.5, 2.5])
        with col1:
            if "Cancelled" in selected_status:
                st.markdown("**Top Districts with Most Canceled Trips**")
                top_canceled_districts = filtered_data[filtered_data['Passenger status'] == 'Cancelled']['district'].value_counts().head(5)
                st.write(top_canceled_districts)

            if "Trip completed" in selected_status:
                st.markdown("**Top Districts with Most Completed Trips**")
                top_completed_districts = filtered_data[filtered_data['Passenger status'] == 'Trip completed']['district'].value_counts().head(5)
                st.write(top_completed_districts)

        # Pickup Count Heatmap
        with col2:
            if not filtered_data.empty:
                st.subheader("Pickup Location Heatmap")
                pickup_count_data = filtered_data.groupby(['Pickup Latitude', 'Pickup Longitude']).size().reset_index(name='Pickup Count')
                
                fig = px.density_mapbox(
                    pickup_count_data,
                    lat='Pickup Latitude',
                    lon='Pickup Longitude',
                    z='Pickup Count',
                    radius=10,
                    mapbox_style="open-street-map",
                    color_continuous_scale="turbo",
                    title="Heatmap of Pickup Locations (Count of Pickups)"
                )
                
                fig.update_layout(width=600, height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No data available for the selected filters.")

        # Weekly or General Trends Line Chart
        st.subheader("Trends of Trips")
        if view_option == "Weekly Trends":
            trends_data = data.groupby(['week', 'Passenger status']).size().unstack(fill_value=0).reset_index()
            fig = px.line(trends_data, x='week', y=['Trip completed', 'Cancelled'], labels={'value': 'Trip Count', 'week': 'Week Number'})
        else:
            trends_data = data.groupby(data['Actual Pickup Time'].dt.date)['Passenger status'].value_counts().unstack(fill_value=0).reset_index()
            fig = px.line(trends_data, x='Actual Pickup Time', y=['Trip completed', 'Cancelled'], labels={'value': 'Trip Count', 'Actual Pickup Time': 'Date'})

        st.plotly_chart(fig, use_container_width=True)
