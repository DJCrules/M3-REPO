import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler

# Step 1: Load and Prepare Data
broadcasting_revenue = pd.read_csv(r'D:\M3 Repository\M3-REPO\VAR\Liverpool_revenue.csv', converters={
    'Broadcasting': lambda x: int(x)})
commercial_revenue = pd.read_csv(r'D:\M3 Repository\M3-REPO\VAR\Liverpool_revenue.csv', converters={
    'Commercial': lambda x: int(x)})
matchday_revenue = pd.read_csv(r'D:\M3 Repository\M3-REPO\VAR\Liverpool_revenue.csv', converters={
    'Matchday': lambda x: int(x)})

# Combine revenues into a single DataFrame
years = np.arange(2000, 2024)  # 24 years of data (2000-2023)
data = pd.DataFrame({
    'Year': years,
    'Broadcasting Revenue': broadcasting_revenue['Broadcasting'],
    'Commercial Revenue': commercial_revenue['Commercial'],
    'Matchday Revenue': matchday_revenue['Matchday'],
})

# Step 2: Check for Multicollinearity (Correlation)
print(data.corr())  # Check correlation matrix

# Optionally, remove or combine variables if correlation is too high
# For example, if Broadcasting and Matchday are highly correlated, we could drop one or combine

# Step 3: Standardize/Scale the Data
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data.drop(columns='Year')), columns=data.columns[1:])

# Step 4: Differencing for Stationarity
data_diff = data_scaled.diff().dropna()  # Differencing to make the series stationary

# Step 5: Fit the VAR Model with reduced lag order
max_lags_possible = min(2, len(data_diff) // 2)  # Use 2 lags, you can increase or decrease this
model = VAR(data_diff)
lag_order = model.select_order(maxlags=max_lags_possible).aic  # AIC selection of lags
var_model = model.fit(lag_order)

# Step 6: Forecast Future Revenue
forecast_steps = 10  # Predict next 10 years (2024-2033)
forecast_input = data_diff.values[-lag_order:]  # Use the most recent data for forecasting
forecast_diff = var_model.forecast(forecast_input, steps=forecast_steps)

# Step 7: Reverse Differencing and Rescale Back to Original Data
forecast_df = pd.DataFrame(forecast_diff, columns=data.columns[1:], index=np.arange(2024, 2024 + forecast_steps))
forecast_df = forecast_df.cumsum() + data.iloc[-1, 1:]  # Reverse differencing

# Rescale the data back to the original scale
forecast_df = scaler.inverse_transform(forecast_df)

# Step 8: Plot the Results
plt.figure(figsize=(10, 6))
plt.plot(data['Year'], data['Broadcasting Revenue'], label='Broadcasting Revenue', color='blue')
plt.plot(forecast_df.index, forecast_df[:, 0], label='Forecasted Broadcasting Revenue', color='blue', linestyle='--')
plt.plot(forecast_df.index, forecast_df[:, 1], label='Forecasted Commercial Revenue', color='green', linestyle='--')
plt.plot(forecast_df.index, forecast_df[:, 2], label='Forecasted Matchday Revenue', color='red', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Revenue (in million $)')
plt.title('Football Club Revenue Forecast with VAR Model')
plt.legend()
plt.show()

# Print Forecasted Values
print("Forecasted Revenue for Next 10 Years:")
print(forecast_df)
