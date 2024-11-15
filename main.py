import pandas as pd

# Load the datasets
trip_data = pd.read_excel('C:/Users/ASUS/OneDrive/Desktop/VGI/FLEXI_trip_data.xlsx')  # Replace with the actual path
bus_stops = pd.read_excel('C:/Users/ASUS/OneDrive/Desktop/VGI/FLEXI_bus_stops.xlsx')  # Replace with the actual path

# Step 1: Filter the trip data to include only rows with Pickup ID and Dropoff ID <= 69, and only completed trips
merged_data = trip_data.merge(bus_stops, left_on='Pickup ID', right_on='index', suffixes=('', '_pickup'))
merged_data = merged_data.merge(bus_stops, left_on='Dropoff ID', right_on='index', suffixes=('', '_dropoff'))

# Drop extra index columns
merged_data = merged_data.drop(columns=['index', 'index_dropoff'])

merged_data['Actual Pickup Time'] = pd.to_datetime(merged_data['Actual Pickup Time'])
merged_data['Actual Dropoff Time'] = pd.to_datetime(merged_data['Actual Dropoff Time'])

# Calculate trip duration in minutes
merged_data['Trip Duration (minutes)'] = (merged_data['Actual Dropoff Time'] - merged_data['Actual Pickup Time']).dt.total_seconds() / 60

# Display the updated dataset with the trip duration
#print(merged_data[['Booking ID', 'Pickup ID', 'Dropoff ID', 'Actual Pickup Time', 'Actual Dropoff Time', 'Trip Duration (minutes)']].head())

# Select the required columns for heatmap visualization
heatmap_data = merged_data[['Booking ID', 'Pickup ID', 'Dropoff ID',
                            'Actual Pickup Time', 'Actual Dropoff Time',
                            'latitude', 'longitude', 'latitude_dropoff', 'longitude_dropoff', "name", "district", "Passenger status"]]

# Rename columns for clarity (e.g., labeling pickup and dropoff coordinates)
heatmap_data = heatmap_data.rename(columns={
    'latitude': 'Pickup Latitude',
    'longitude': 'Pickup Longitude',
    'latitude_dropoff': 'Dropoff Latitude',
    'longitude_dropoff': 'Dropoff Longitude'
})

heatmap_data = heatmap_data[(heatmap_data['Pickup ID'] <= 69) &
                              (heatmap_data['Dropoff ID'] <= 69)]
# Display the structured data for heatmap
print(heatmap_data.head())