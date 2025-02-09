import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# Step 1: Generate Synthetic Data
np.random.seed(42)  # For reproducibility

# Simulate a multivariate time series (e.g., 3 variables: GDP, Interest Rate, Inflation)
n_obs = 100  # Number of observations (time steps)
n_vars = 3  # Number of variables (e.g., GDP, Interest Rate, Inflation Rate)

# Create synthetic time series data
time = np.arange(n_obs)
gdp = np.cumsum(np.random.randn(n_obs)) + 10000  # Random walk for GDP (sum of random steps)
interest_rate = np.cumsum(np.random.randn(n_obs)) + 5  # Random walk for Interest Rate
inflation_rate = np.cumsum(np.random.randn(n_obs)) + 2  # Random walk for Inflation Rate

# Create a DataFrame
data = pd.DataFrame({
    'GDP': gdp,
    'Interest Rate': interest_rate,
    'Inflation Rate': inflation_rate
}, index=time)

# Step 2: Visualize the Synthetic Data
data.plot(figsize=(10, 6))
plt.title('Synthetic Time Series Data: GDP, Interest Rate, and Inflation Rate')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='best')
plt.show()

# Step 3: Fit the VAR Model
model = VAR(data)  # Fit the VAR model to the data

# Choose the optimal lag length using the AIC criterion
lag_order = model.select_order(maxlags=10).aic
print(f'Optimal Lag Order: {lag_order}')

# Fit the model using the optimal lag order
var_result = model.fit(lag_order)

# Step 4: Make Forecasts (for the next 10 time steps)
forecast_steps = 10  # Forecast for the next 10 time steps
forecast = var_result.forecast(data.values[-lag_order:], steps=forecast_steps)

# Step 5: Visualize the Forecast
forecast_df = pd.DataFrame(forecast, columns=data.columns, index=np.arange(n_obs, n_obs + forecast_steps))

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['GDP'], label='Actual GDP', color='blue')
plt.plot(forecast_df.index, forecast_df['GDP'], label='Forecasted GDP', color='red', linestyle='--')
plt.title('Synthetic GDP Forecast with VAR Model')
plt.xlabel('Time')
plt.ylabel('GDP Value')
plt.legend(loc='best')
plt.show()

# Print the forecasted values
print("Forecasted Values for the next 10 time steps:")
print(forecast_df)
