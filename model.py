import pandas as pd
from prophet import Prophet
import joblib  # For saving the model

class DemandForecast:
    def __init__(self, data):
        # Initialize with data and preprocess
        self.data = data.copy()
        self.data['Actual Pickup Time'] = pd.to_datetime(self.data['Actual Pickup Time'])
        self.data['Rounded Pickup Time'] = self.data['Actual Pickup Time'].dt.floor('H')
        self.data['Weekday'] = self.data['Rounded Pickup Time'].dt.dayofweek
        
        # Aggregate demand by time and location
        self.pickup_demand = self.data.groupby(['Rounded Pickup Time', 'Pickup ID']).size().reset_index(name='demand')

    def train_model(self, pickup_id, forecast_days=7, save_model=True):
        # Filter data for the specified pickup location
        location_data = self.pickup_demand[self.pickup_demand['Pickup ID'] == pickup_id][['Rounded Pickup Time', 'demand']]
        location_data.rename(columns={'Rounded Pickup Time': 'ds', 'demand': 'y'}, inplace=True)
        location_data['Weekday'] = location_data['ds'].dt.dayofweek

        # Initialize Prophet model and add 'Weekday' as a regressor
        model = Prophet()
        model.add_regressor('Weekday')
        
        # Train the model
        model.fit(location_data)
        self.model = model  # Save model for future use

        # Prepare future dataframe
        future = model.make_future_dataframe(periods=24 * forecast_days, freq='H')
        future['Weekday'] = future['ds'].dt.dayofweek
        
        # Make predictions
        forecast = model.predict(future)

        # Optionally, save the model as a file
        if save_model:
            model_filename = f"prophet_model_pickup_{pickup_id}.pkl"
            joblib.dump(model, model_filename)
            print(f"Model for Pickup ID {pickup_id} saved as {model_filename}")
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def forecast_for_multiple_locations(self, pickup_ids, forecast_days=7):
        forecasts = {}
        for pickup_id in pickup_ids:
            print(f"Forecasting for Pickup ID {pickup_id}")
            forecast = self.train_model(pickup_id, forecast_days=forecast_days)
            forecasts[pickup_id] = forecast
        return forecasts

# Load data
heatmap_data = pd.read_csv("heatmap_data.csv")

# Initialize forecasting class with data
demand_forecast = DemandForecast(heatmap_data)

# Train model and forecast for Pickup ID 0 (example)
forecast_for_pickup_0 = demand_forecast.train_model(0)
print(forecast_for_pickup_0.head())

# Forecast for multiple pickup IDs (e.g., top 5 most frequent)
top_pickup_ids = [0, 8, 19, 30, 31]  # Replace with most frequent IDs
multiple_forecasts = demand_forecast.forecast_for_multiple_locations(top_pickup_ids)
print(multiple_forecasts[0].head())  # Forecast for Pickup ID 0