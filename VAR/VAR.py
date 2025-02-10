import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

# Step 1: Generate Synthetic Data
np.random.seed(42)

years = np.arange(2000, 2024)  # 24 years of data (2000-2023)
n_obs = len(years)

# Simulate revenue streams (in million dollars)
broadcasting_revenue = pd.read_csv(r'D:\M3 Repository\M3-REPO\VAR\Liverpool_revenue', converters={
    'Broadcasting': lambda x: int(x * 1000000)})
commercial_revenue = pd.read_csv(r'D:\M3 Repository\M3-REPO\VAR\Liverpool_revenue', converters={
    'Commercial': lambda x: int(x * 1000000)})
matchday_revenue = pd.read_csv(r'D:\M3 Repository\M3-REPO\VAR\Liverpool_revenue', converters={
    'Matchday': lambda x: int(x * 1000000)})

# Create DataFrame
data = pd.DataFrame({
    'Year': years,
    'Broadcasting Revenue': broadcasting_revenue,
    'Commercial Revenue': commercial_revenue,
    'Matchday Revenue': matchday_revenue,
})

# Compute total revenue
data['Total Revenue'] = data['Broadcasting Revenue'] + data['Commercial Revenue'] + data['Matchday Revenue']

# Step 2: Prepare Data for VAR
data.set_index('Year', inplace=True)  # Use Year as the index
data_diff = data.diff().dropna()  # Differencing to ensure stationarity

# Step 3: Fit the VAR Model
max_lags_possible = min(10, len(data_diff) // 2)  # Ensure enough observations
model = VAR(data_diff)
lag_order = model.select_order(maxlags=max_lags_possible).aic  # Auto-adjust maxlags
var_model = model.fit(lag_order)

# Step 4: Forecast Future Revenue
forecast_steps = 10  # Predict next 10 years (2024-2033)
forecast_input = data_diff.values[-lag_order:]
forecast_diff = var_model.forecast(forecast_input, steps=forecast_steps)

# Convert differenced forecast back to original values
forecast_df = pd.DataFrame(forecast_diff, columns=data.columns, index=np.arange(2024, 2024+forecast_steps))
forecast_df = forecast_df.cumsum() + data.iloc[-1]  # Reverse differencing

# Step 5: Plot Forecasts
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Total Revenue'], label='Historical Total Revenue', color='blue')
plt.plot(forecast_df.index, forecast_df['Total Revenue'], label='Forecasted Total Revenue', color='red', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Total Revenue (in million $)')
plt.title('Football Club Revenue Forecast with VAR Model')
plt.legend()
plt.show()

# Print Forecasted Values
print("Forecasted Revenue for Next 10 Years:")
print(forecast_df)
