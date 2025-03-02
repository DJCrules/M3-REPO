import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the datasets
yearly_data = pd.read_csv('ARIMA/yearly_trend.csv')
monthly_data = pd.read_csv('ARIMA/monthly_trend.csv')
# Identify peak values in the yearly trend
peak_years = yearly_data[yearly_data['Value'] == yearly_data['Value'].max()]

# Filter monthly data for the peak years
peak_monthly_data = monthly_data[monthly_data.index.year.isin(peak_years.index.year)]

# Combine yearly and monthly data for modeling
combined_data = pd.concat([yearly_data, peak_monthly_data.resample('Y').mean()])

# Drop NaN values (if any)
combined_data.dropna(inplace=True)

# Fit an ARIMA model
# Adjust the order (p, d, q) as needed based on your data
model = ARIMA(combined_data['Value'], order=(5, 1, 0))
model_fit = model.fit()

# Forecast the next few years
forecast_steps = 5  # Number of years to predict
forecast = model_fit.forecast(steps=forecast_steps)

# Create a DataFrame for the forecasted values
forecast_index = pd.date_range(start=combined_data.index[-1] + pd.DateOffset(years=1), periods=forecast_steps, freq='Y')
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(combined_data.index, combined_data['Value'], label='Historical Data')
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