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
import folium
from folium.plugins import MarkerCluster
import datetime
import random
from streamlit_folium import st_folium
from folium import CustomIcon




def run():
    st.title("VGI FLEXI - App")
    # st.subheader("Products - Demand Forecasting & Visualization")?

    # Define available products
    products = {
        "Pre Position": "Pre Position",
        "DaRP": "DaRP"
    }

    
    selected_product = st.selectbox("Select a Product", list(products.keys()))
    if selected_product == "Pre Position":
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
            0: "city1",  
            8: "city2",  
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
                    pass  
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
            url_base = "https://api.parkendd.de/{city}/"
            all_parking_spots = []
            
            for pickup_id, coords in coordinates_dict.items():
                city = city_id_map.get(pickup_id)
                if city:
                    url = url_base.format(city=city)
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        for lot in data.get("lots", []):
                            name = lot.get("name", "Unnamed Parking Spot")
                            lat = lot["coords"]["lat"]
                            lon = lot["coords"]["lng"]
                            all_parking_spots.append((name, lat, lon, pickup_id))
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

            try:
                # Apply weighted KMeans clustering to find two optimal shuttle bus locations
                kmeans = KMeans(n_clusters=2, random_state=42)
                kmeans.fit(X, sample_weight=weights)
                predictions_df['Cluster'] = kmeans.labels_

                # Calculate weighted centroids (optimal bus locations) for each cluster
                optimal_positions = pd.DataFrame(kmeans.cluster_centers_, columns=['Latitude', 'Longitude'])

                # Display each shuttle position in a new row with full decimal precision
                st.write("Optimal Shuttle Bus Positions:")
                for i, row in optimal_positions.iterrows():
                    lat = row['Latitude']
                    lon = row['Longitude']
                    st.metric(label=f"Shuttle {i + 1} Position", value=f"{lat}° N, {lon}° E")

                # Find parking spots around each pickup coordinate
                all_parking_spots = find_parking_spots_around_coordinates(pickup_coordinates, city_id_map)

                # Determine the closest parking spot to each shuttle position
                nearest_parking = []
                if all_parking_spots:  # Check if there are any parking spots available
                    for i, shuttle_pos in optimal_positions.iterrows():
                        shuttle_location = (shuttle_pos['Latitude'], shuttle_pos['Longitude'])
                        closest_spot = min(all_parking_spots, key=lambda x: geodesic(shuttle_location, (x[1], x[2])).meters)
                        nearest_parking.append(closest_spot)

                    # Display the coordinates of the nearest parking spots with full precision
                    st.write("Assigned Parking Spot Coordinates for Each Shuttle:")
                    for i, spot in enumerate(nearest_parking):
                        lat = spot[1]
                        lon = spot[2]
                        st.metric(label=f"Parking Spot for Shuttle {i + 1}", value=f"{lat}° N, {lon}° E")

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

                # Mark nearest parking spots on the map
                if nearest_parking:
                    for i, spot in enumerate(nearest_parking):
                        folium.Marker(
                            location=[spot[1], spot[2]],
                            icon=folium.Icon(color="green", icon="parking", prefix="fa"),
                            tooltip=f"Assigned Parking for Shuttle {i+1}: {spot[0]}"
                        ).add_to(m)

                # Display the map in Streamlit
                        folium_static(m)
            except ValueError:
                        st.warning("Not enough predictions to triangulate a position.")
            else:
                    st.warning("Not enough predictions to triangulate a position.")
    
#     elif selected_product == "DaRP":


# # OpenRouteService API Key
# #ORS_API_KEY = "5b3ce3597851110001cf62484030fcec9fa24003bf779a0485100d73"
#         MAPBOX_API_KEY = "pk.eyJ1Ijoic2FuaWEzMyIsImEiOiJjbTNnNmNxaWgwMTdjMnFzZGdrMGgzaHR5In0.97xI8z8QUeioHazpvtFjCQ"

#         # Constants
#         MAX_PASSENGERS = 8
#         CO2_EMISSION_PER_KM = 0.14  # kg per km
#         SERVICE_CENTER = {"lat": 48.992168, "lon": 11.377365}
#         shuttle_ids = {f"shuttle_{i+1}": f"Shuttle ID {i+1}" for i in range(2)}  # Rename shuttles as Shuttle ID 1, Shuttle ID 2

#         # Fixed pickup and drop-off locations (stations)
#         STATIONS = {
#             0: {"lat": 48.992168, "lon": 11.377365},
#             8: {"lat": 49.017505, "lon": 11.404733},
#             19: {"lat": 49.033832, "lon": 11.471982},
#             30: {"lat": 49.036378, "lon": 11.470632},
#             31: {"lat": 49.035227, "lon": 11.467885}
#         }

#         # Cost function weights
#         DISTANCE_WEIGHT = 0.5
#         TIME_WEIGHT = 0.3
#         CO2_WEIGHT = 0.2

#         # Initialize session state variables
#         if "shuttle_routes" not in st.session_state:
#             st.session_state.shuttle_routes = {"shuttle_1": [], "shuttle_2": []}
#         if "map_data" not in st.session_state:
#             st.session_state.map_data = None
#         if "status_messages" not in st.session_state:
#             st.session_state.status_messages = []
#         if "initialized_30_days" not in st.session_state:
#             st.session_state.initialized_30_days = False
#         if "shuttle_positions" not in st.session_state:
#             st.session_state.shuttle_positions = {
#                 "shuttle_1": SERVICE_CENTER,  # Starting position at the service center
#                 "shuttle_2": SERVICE_CENTER   # Starting position at the service center
#             }

#         # Create a booking request based on user input for pickup/drop-off locations and time
#         def create_request(pickup_station_index, dropoff_station_index, pickup_time, dropoff_time, passengers):
#             pickup_station = STATIONS[pickup_station_index]
#             dropoff_station = STATIONS[dropoff_station_index]
#             return {
#                 "pickup": (pickup_station["lat"], pickup_station["lon"]),
#                 "dropoff": (dropoff_station["lat"], dropoff_station["lon"]),
#                 "requested_pickup_time": pickup_time,
#                 "dropoff_time": dropoff_time,
#                 "passengers": passengers
#             }

#         # Initialize recent bookings with 5 bookings spaced in 15-minute intervals
#         def initialize_recent_bookings():
#             if not st.session_state.initialized_30_days:
#                 current_date = datetime.datetime.now().date()  # Use today's date as the base date
#                 base_time = datetime.datetime.combine(current_date, datetime.time(0, 0))  # Start at midnight

#                 # Define specific pickup and drop-off locations
#                 pickup_dropoff_pairs = [
#                     (0, 8),   # From Station 0 to Station 8
#                     (8, 19),  # From Station 8 to Station 19
#                     (19, 30), # From Station 19 to Station 30
#                     (30, 31), # From Station 30 to Station 31
#                     (31, 0)   # From Station 31 back to Station 0
#                 ]

#                 # Generate bookings for each hour in a 24-hour period
#                 for hour in range(24):
#                     pickup_time = base_time + datetime.timedelta(hours=hour)
#                     dropoff_time = pickup_time + datetime.timedelta(minutes=30)
#                     num_passengers = random.randint(1, MAX_PASSENGERS)
#                     pickup_station, dropoff_station = random.choice(pickup_dropoff_pairs)  # Random pickup/drop-off stations

#                     # Create the booking request
#                     request = create_request(pickup_station, dropoff_station, pickup_time, dropoff_time, num_passengers)
                    
#                     # Alternate between shuttles
#                     shuttle_key = "shuttle_1" if hour % 2 == 0 else "shuttle_2"
#                     st.session_state.shuttle_routes[shuttle_key].append(request)

#                 st.session_state.initialized_30_days = True

#         import time



#         def get_ors_route(pickup, dropoff, retries=3, retry_delay=1):
#             if not isinstance(pickup, tuple) or not isinstance(dropoff, tuple) or len(pickup) != 2 or len(dropoff) != 2:
#                 print("Invalid coordinates provided.")
#                 return [], 0, 0, 0

#             url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{pickup[1]},{pickup[0]};{dropoff[1]},{dropoff[0]}"
#             params = {
#                 "access_token": MAPBOX_API_KEY,
#                 "geometries": "geojson",
#                 "overview": "simplified",
#             }

#             for attempt in range(retries):
#                 try:
#                     response = requests.get(url, params=params)
#                     if response.status_code == 200:
#                         data = response.json()
#                         if "routes" in data and data["routes"]:
#                             route = data["routes"][0]
#                             distance_km = route["distance"] / 1000  # Convert meters to kilometers
#                             duration_min = route["duration"] / 60    # Convert seconds to minutes
#                             co2_emission = distance_km * CO2_EMISSION_PER_KM

#                             # Skip entries with zero distance as invalid responses
#                             if distance_km > 0:
#                                 print(f"Route Distance: {distance_km:.2f} km")
#                                 print(f"Route Duration: {duration_min:.2f} minutes")
#                                 print(f"CO₂ Emission: {co2_emission:.2f} kg")
#                                 return route["geometry"]["coordinates"], distance_km, duration_min, co2_emission
#                             else:
#                                 print("Skipped zero distance route.")

#                     else:
#                         print(f"Attempt {attempt + 1} failed with status {response.status_code}. Retrying...")

#                 except Exception as e:
#                     print(f"Error fetching route data: {e}. Retrying...")

#                 time.sleep(retry_delay)

#             print("Failed to fetch route data after retries.")
#             return [], 0, 0, 0


#         def evaluate_shuttle(shuttle, new_request):
#             # Start from the shuttle's current position
#             current_position = st.session_state.shuttle_positions.get(shuttle, (SERVICE_CENTER["lat"], SERVICE_CENTER["lon"]))
#             temp_route = list(st.session_state.shuttle_routes[shuttle])  # Copy current route for evaluation
#             temp_route.append(new_request)  # Add new request for evaluation

#             # Reset cumulative metrics
#             total_distance = 0
#             total_duration = 0
#             total_co2 = 0

#             print(f"\nEvaluating shuttle: {shuttle}")
#             print(f"Initial Position: {current_position}")
            
#             # Calculate metrics for each leg from current position to each stop on the route
#             last_position = current_position
#             for idx, request in enumerate(temp_route):
#                 print(f"\nLeg {idx + 1}:")
                
#                 # Track each leg's distance, duration, and CO₂ separately
#                 pickup_coords, pickup_distance, pickup_duration, pickup_co2 = get_ors_route(last_position, request["pickup"])
                
#                 # Check and accumulate pickup leg only if valid
#                 if pickup_distance > 0:
#                     print(f"Pickup Distance: {pickup_distance} km, Duration: {pickup_duration} mins, CO2: {pickup_co2} kg")
#                     total_distance += pickup_distance
#                     total_duration += pickup_duration
#                     total_co2 += pickup_co2

#                 # Update last position to pickup location
#                 last_position = request["pickup"]

#                 # Calculate from pickup to drop-off
#                 dropoff_coords, dropoff_distance, dropoff_duration, dropoff_co2 = get_ors_route(request["pickup"], request["dropoff"])
                
#                 # Check and accumulate drop-off leg only if valid
#                 if dropoff_distance > 0:
#                     print(f"Dropoff Distance: {dropoff_distance} km, Duration: {dropoff_duration} mins, CO2: {dropoff_co2} kg")
#                     total_distance += dropoff_distance
#                     total_duration += dropoff_duration
#                     total_co2 += dropoff_co2

#                 # Update last position to dropoff
#                 last_position = request["dropoff"]

#             # Display cumulative results for this route evaluation
#             print(f"\nCumulative Results for {shuttle}:")
#             print(f"Total Distance: {total_distance:.2f} km")
#             print(f"Total Duration: {total_duration:.2f} mins")
#             print(f"Total CO2: {total_co2:.2f} kg\n")

#             # Calculate cost based on weights and accumulated values
#             cost = DISTANCE_WEIGHT * total_distance + TIME_WEIGHT * total_duration + CO2_WEIGHT * total_co2
#             earliest_arrival_time = datetime.datetime.now() + datetime.timedelta(minutes=total_duration)
#             recommended_pickup_time = max(earliest_arrival_time, new_request["requested_pickup_time"])
#             on_time = earliest_arrival_time <= new_request["requested_pickup_time"]

#             # Prepare the scheduling information for the new request
#             new_request["scheduled_pickup_time"] = recommended_pickup_time
#             new_request["on_time"] = on_time

#             return {
#                 "cost": cost,
#                 "distance": total_distance,
#                 "duration": total_duration,
#                 "co2": total_co2,
#                 "is_feasible": new_request["passengers"] <= MAX_PASSENGERS,
#                 "earliest_arrival_time": earliest_arrival_time,
#                 "recommended_pickup_time": recommended_pickup_time,
#                 "route": temp_route,
#                 "on_time": on_time
#             }




#         def find_next_available_time():
#             next_available_time = datetime.datetime.now() + datetime.timedelta(minutes=15)
            
#             # Iterate through each shuttle's routes
#             for shuttle_routes in st.session_state.shuttle_routes.values():
#                 if shuttle_routes:  # Only process if there are existing routes for the shuttle
#                     latest_request_time = max(req["requested_pickup_time"] for req in shuttle_routes)
#                     # Calculate the time 15 minutes after the latest request time for the shuttle
#                     shuttle_next_available_time = latest_request_time + datetime.timedelta(minutes=15)
#                     # Update next_available_time to be the maximum of current and calculated times
#                     next_available_time = max(next_available_time, shuttle_next_available_time)
            
#             print(f"Recommended Next Available Pickup Time: {next_available_time.strftime('%H:%M')}")
#             return next_available_time


#         # Updated Map Plotting Function
#         def plot_map(shuttle_info, selected_shuttle=None):
#             m = folium.Map(location=[SERVICE_CENTER["lat"], SERVICE_CENTER["lon"]], zoom_start=12)
#             colors = {"shuttle_1": "blue", "shuttle_2": "green"}
#             bus_icon_url = "https://img.icons8.com/color/48/bus.png"

#             for shuttle, info in shuttle_info.items():
#                 color = colors.get(shuttle, "blue")
#                 route = info["route"]
#                 shuttle_display_name = shuttle_ids.get(shuttle, shuttle)  # Use display names in marker text only

#                 # Display markers for each shuttle's pickup and drop-off locations
#                 for i, request in enumerate(route):
#                     pickup_time = request.get("scheduled_pickup_time", request["requested_pickup_time"])
#                     pickup_marker_text = (
#                         f"{shuttle_display_name} Pickup at {pickup_time.strftime('%H:%M')}, "
#                         f"Passengers: {request['passengers']}, "
#                         f"{'On Time' if request.get('on_time', False) else 'Delayed'}"
#                     )
#                     dropoff_marker_text = f"{shuttle_display_name} Drop-off at {request['dropoff_time'].strftime('%H:%M')}"

#                     # Use custom bus icon for pickups
#                     pickup_icon = folium.CustomIcon(bus_icon_url, icon_size=(40, 40))
#                     folium.Marker(
#                         location=request["pickup"],
#                         popup=pickup_marker_text,
#                         icon=pickup_icon
#                     ).add_to(m)

#                     # Add dropoff marker with a color-coded icon
#                     folium.Marker(
#                         location=request["dropoff"],
#                         popup=dropoff_marker_text,
#                         icon=folium.Icon(color=color, icon="stop", prefix="fa")
#                     ).add_to(m)

#                     # Draw route line if coordinates are valid
#                     if i < len(route) - 1:
#                         route_coords, _, _, _ = get_ors_route(request["pickup"], request["dropoff"])
#                         if route_coords:  # Check if coordinates are not empty
#                             folium.PolyLine(route_coords, color=color, weight=5, opacity=0.7).add_to(m)

#             return m

#         # UI
#         # UI
#         st.title("Dial-a-Route Solution with Time-Sensitive Pickup Scheduling")

#         if st.button("Initialize Recent Bookings"):
#             initialize_recent_bookings()
#             st.write("Recent bookings initialized.")

#         pickup_station_index = st.selectbox("Select Pickup Station", options=STATIONS.keys(), format_func=lambda x: f"Station {x}")
#         dropoff_station_index = st.selectbox("Select Dropoff Station", options=STATIONS.keys(), format_func=lambda x: f"Station {x}")
#         pickup_time = st.time_input("Choose your desired pickup time", datetime.time(13, 0))
#         pickup_datetime = datetime.datetime.combine(datetime.datetime.today(), pickup_time)
#         dropoff_time = pickup_datetime + datetime.timedelta(minutes=30)
#         passengers = st.slider("Number of Passengers", 1, MAX_PASSENGERS)

#         if st.button("Create Booking"):
#             new_request = create_request(pickup_station_index, dropoff_station_index, pickup_datetime, dropoff_time, passengers)
#             st.session_state.status_messages.clear()
#             shuttle_evaluations = {}
#             selected_shuttle = None
#             min_cost = float("inf")

#             for shuttle in st.session_state.shuttle_routes.keys():
#                 eval_info = evaluate_shuttle(shuttle, new_request)
#                 shuttle_evaluations[shuttle] = eval_info

#                 if eval_info["is_feasible"]:
#                     status = f"""
#                     #### 🚐 {shuttle.capitalize()}
#                     - **Earliest Arrival**: {eval_info['earliest_arrival_time'].strftime('%H:%M')}
#                     - **Recommended Pickup Time**: {eval_info['recommended_pickup_time'].strftime('%H:%M')}
#                     - **Cost**: {eval_info['cost']:.2f}
#                     - **Distance**: {eval_info['distance']:.2f} km
#                     - **CO₂ Emission**: {eval_info['co2']:.2f} kg
#                     - **Status**: Can accommodate the request.
#                     """
#                     st.session_state.status_messages.append(status)
                    
#                     if eval_info["cost"] < min_cost:
#                         min_cost = eval_info["cost"]
#                         selected_shuttle = shuttle
#                 else:
#                     status = f"#### 🚐 {shuttle.capitalize()} - **Status**: Full and cannot take new passengers."
#                     st.session_state.status_messages.append(status)

#             if selected_shuttle:
#                 selected_status = f"""
#                 ### ✅ Selected Shuttle: {selected_shuttle.capitalize()}
#                 - **Total Cost**: {min_cost:.2f}
#                 - **Total Distance**: {shuttle_evaluations[selected_shuttle]['distance']:.2f} km
#                 - **Total CO₂ Emission**: {shuttle_evaluations[selected_shuttle]['co2']:.2f} kg
#                 """
#                 st.session_state.status_messages.append(selected_status)
#                 st.session_state.shuttle_routes[selected_shuttle].append(new_request)
#                 st.session_state.map_data = plot_map(shuttle_evaluations, selected_shuttle=selected_shuttle)
#             else:
#                 recommended_time = find_next_available_time()
#                 no_shuttle_status = f"""
#                 ### 🚫 No Shuttles Available
#                 - **Recommended Next Available Pickup Time**: {recommended_time.strftime('%H:%M')}
#                 """
#                 st.session_state.status_messages.append(no_shuttle_status)

#         # Display all messages in a clean format
#         if st.session_state.status_messages:
#             st.markdown("### 🚍 Shuttle Constraints and Cost Evaluation")
#             for msg in st.session_state.status_messages:
#                 st.markdown(msg)

#         if st.session_state.map_data:
#             st.write("### Shuttle Route Map with Pickup Scheduling")
#             st_folium(st.session_state.map_data, width=700, height=500)


