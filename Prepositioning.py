import streamlit as st
import pandas as pd
import joblib
from prophet import Prophet
from datetime import datetime
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static
import requests

# Coordinates mapping for each pickup ID
pickup_coordinates = {
    0: {"lat": 48.992168, "lon": 11.377365},
    8: {"lat": 49.017505, "lon": 11.404733},
    19: {"lat": 49.033832, "lon": 11.471982},
    30: {"lat": 49.036378, "lon": 11.470632},
    31: {"lat": 49.035227, "lon": 11.467885}
}

# Map each pickup location to its nearest city ID in ParkAPI
city_id_map = {  
    0: "city1",  # Replace "city1" with the nearest city ID in ParkAPI
    8: "city2",  # Replace accordingly based on supported cities near coordinates
    19: "city3",
    30: "city4",
    31: "city5"
}

# Load models function
def load_all_models(pickup_ids):
    models = {}
    for pickup_id in pickup_ids:
        model_filename = f"prophet_model_pickup_{pickup_id}.pkl"
        try:
            models[pickup_id] = joblib.load(model_filename)
        except FileNotFoundError:
            st.write(f"Model for Pickup ID {pickup_id} not found.")
    return models

# Prediction function
def predict_demand(model, forecast_time):
    future = pd.DataFrame({'ds': [forecast_time]})
    future['Weekday'] = future['ds'].dt.dayofweek
    forecast = model.predict(future)
    predicted_demand = forecast['yhat'].values[0]
    return int(round(predicted_demand))

# Function to find parking spots around coordinates using ParkAPI
def find_parking_spots_around_coordinates(coordinates_dict, city_id_map):
    url_base = "https://api.parkendd.de/{city}/"  # Corrected placeholder to match the format function
    all_parking_spots = []
    
    for pickup_id, coords in coordinates_dict.items():
        city = city_id_map.get(pickup_id)  # Fetch the city name based on the pickup ID
        if city:
            url = url_base.format(city=city)  # Corrected to use 'city' instead of 'city_id'
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for lot in data.get("lots", []):
                    name = lot.get("name", "Unnamed Parking Spot")
                    lat = lot["coords"]["lat"]
                    lon = lot["coords"]["lng"]
                    all_parking_spots.append((name, lat, lon, pickup_id))
            else:
                st.write("")
        else:
            st.write(f"No city mapping found for Pickup ID: {pickup_id}")
    return all_parking_spots

# Main application code
st.title("Optimal Shuttle Parking with Coordinates Display")

# User input for hour
forecast_date = st.date_input("Select Date", value=datetime.today())
forecast_hour = st.slider("Select Hour", min_value=0, max_value=23, step=1)
forecast_datetime = datetime.combine(forecast_date, datetime.min.time()) + pd.Timedelta(hours=forecast_hour)

# List of Pickup IDs
pickup_ids = [0, 8, 19, 30, 31]

# Load models
with st.spinner('Loading models...'):
    models = load_all_models(pickup_ids)
st.success("Models loaded.")

# Create a DataFrame to store predictions and coordinates for clustering
predictions = []

# Get demand prediction for each pickup ID
for pickup_id, model in models.items():
    if model is not None:
        demand = predict_demand(model, forecast_datetime)
        predictions.append({
            'Pickup ID': pickup_id,
            'Predicted Demand': demand,
            'Latitude': pickup_coordinates[pickup_id]["lat"],
            'Longitude': pickup_coordinates[pickup_id]["lon"]
        })

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions)

if not predictions_df.empty:
    st.subheader(f"Predicted Demand and Optimal Shuttle Parking for {forecast_datetime.strftime('%Y-%m-%d %H:%M')}")

    # Prepare data for clustering (latitude, longitude, and demand as weight)
    X = predictions_df[['Latitude', 'Longitude']].values
    weights = predictions_df['Predicted Demand'].values

    # Apply weighted KMeans clustering to find two optimal shuttle bus locations
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X, sample_weight=weights)
    predictions_df['Cluster'] = kmeans.labels_

    # Calculate weighted centroids (optimal bus locations) for each cluster
    optimal_positions = pd.DataFrame(kmeans.cluster_centers_, columns=['Latitude', 'Longitude'])

    # Display the coordinates of the optimal shuttle bus positions
    st.write("Optimal Shuttle Bus Positions:")
    for i, row in optimal_positions.iterrows():
        st.write(f"Shuttle {i + 1}: Coordinates ({row['Latitude']}, {row['Longitude']})")

    # Find parking spots around each pickup coordinate
    all_parking_spots = find_parking_spots_around_coordinates(pickup_coordinates, city_id_map)

    # Ensure there are parking spots available before assigning to shuttles
    if all_parking_spots:
        # Determine the closest parking spot to each shuttle position
        nearest_parking = []
        for i, shuttle_pos in optimal_positions.iterrows():
            shuttle_location = (shuttle_pos['Latitude'], shuttle_pos['Longitude'])
            # Find the nearest parking spot for each shuttle position
            closest_spot = min(all_parking_spots, key=lambda x: geodesic(shuttle_location, (x[1], x[2])).meters)
            nearest_parking.append(closest_spot)

        # Display the coordinates of the nearest parking spots
        st.write("Assigned Parking Spot Coordinates for Each Shuttle:")
        for i, spot in enumerate(nearest_parking):
            st.write(f"Shuttle {i + 1} Assigned Parking Spot: {spot[0]}, Coordinates ({spot[1]}, {spot[2]})")
    else:
        st.write("No parking spots found for the specified coordinates.")
        nearest_parking = []

    # Initialize Folium map
    map_center = [predictions_df['Latitude'].mean(), predictions_df['Longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=13)

    # Add demand hotspots to the map as blue circles
    for _, row in predictions_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=row['Predicted Demand'] / 2 + 5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.7,
            tooltip=f"Pickup ID: {row['Pickup ID']}<br>Predicted Demand: {row['Predicted Demand']}"
        ).add_to(m)

    # Add optimal shuttle bus positions to the map
    for i, row in optimal_positions.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            icon=folium.Icon(color="red", icon="bus", prefix="fa"),
            tooltip=f"Optimal Shuttle Position {i+1}"
        ).add_to(m)

    # Mark nearest parking spots on the map, if available
    if nearest_parking:
        for i, spot in enumerate(nearest_parking):
            folium.Marker(
                location=[spot[1], spot[2]],
                icon=folium.Icon(color="green", icon="parking", prefix="fa"),
                tooltip=f"Assigned Parking for Shuttle {i+1}: {spot[0]}"
            ).add_to(m)

    # Display the map in Streamlit
folium_static(m)




