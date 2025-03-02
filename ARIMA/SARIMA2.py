import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the yearly dataset
yearly_data = pd.read_csv('ARIMA/yearly_trend.csv')
yearly_data['Year'] = pd.to_datetime(yearly_data['Year'], format='%Y')  # Convert 'Year' to datetime
yearly_data.set_index('Year', inplace=True)  # Set 'Year' as the index

# Load the monthly dataset
monthly_data = pd.read_csv('ARIMA/monthly_trend.csv', header=None, names=['Month', 'Value'])
monthly_data['Month'] = monthly_data['Month'].astype(str).str.zfill(2)  # Ensure month is two digits (e.g., '01' for January)

# Create a date column for the monthly data (assume all data is for the most recent year in the yearly dataset)
most_recent_year = yearly_data.index.max().year  # Extract the year as an integer
monthly_data['Date'] = pd.to_datetime(
    str(most_recent_year) + '-' + monthly_data['Month'] + '-01', format='%Y-%m-%d'  # Correct format
)
monthly_data.set_index('Date', inplace=True)

# Identify the peak year in the yearly trend
peak_year = yearly_data['value'].idxmax()

# Filter monthly data for the peak year
peak_monthly_data = monthly_data[monthly_data.index.year == peak_year.year]

# Combine yearly and monthly data for modeling
# Resample monthly data to yearly frequency by taking the mean
monthly_resampled = peak_monthly_data.resample('Y').mean()

# Combine with yearly data
combined_data = pd.concat([yearly_data['value'], monthly_resampled['Value']])

# Drop NaN values (if any)
combined_data.dropna(inplace=True)

# Fit an ARIMA model
# Adjust the order (p, d, q) as needed based on your data
model = ARIMA(combined_data, order=(5, 1, 0))  # Example order, tune as needed
model_fit = model.fit()

# Forecast the next few years
forecast_steps = 5  # Number of years to predict
forecast = model_fit.forecast(steps=forecast_steps)

# Create a DataFrame for the forecasted values
forecast_index = pd.date_range(start=combined_data.index[-1] + pd.DateOffset(years=1), periods=forecast_steps, freq='Y')
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(combined_data.index, combined_data, label='Historical Data')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
plt.title('ARIMA Model: Historical Data and Forecast')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()

# Save the forecast to a CSV file
forecast_df.to_csv('ARIMA/forecasted_values.csv')
print("Forecast saved to 'forecasted_values.csv'")