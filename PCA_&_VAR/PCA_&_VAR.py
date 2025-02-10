import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler

# Step 1: Load the CSV data
file_path = 'D:/M3 Repository/M3-REPO/PCA_&_VAR/Liverpool_revenue.csv'
df = pd.read_csv(file_path)

# Step 2: Clean the 'Year' column
# Remove commas and convert to integer
df['Year'] = df['Year'].astype(str).str.replace(',', '').astype(int)

# Convert 'Year' to datetime format
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Set 'Year' as the index
df.set_index('Year', inplace=True)

# Add the 'Total' column for historical data
df['Total'] = df['Broadcasting'] + df['Commercial'] + df['Matchday']

# Step 3: Standardize the data
scaler = StandardScaler()
revenue_data = df[['Broadcasting', 'Commercial', 'Matchday']]
scaled_data = scaler.fit_transform(revenue_data)

# Step 4: Apply PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2', 'PC3'], index=df.index)

# Step 5: Fit the VAR model
pca_diff = pca_df.diff().dropna()
max_lags_possible = min(2, len(pca_diff) // 2)
model = VAR(pca_diff)
lag_order = model.select_order(maxlags=max_lags_possible).aic
var_model = model.fit(lag_order)

# Step 6: Forecast Future PCA Components starting from 2023
forecast_steps = 10
forecast_input = pca_diff.values[-lag_order:]  # Use the last 'lag_order' number of observations
forecast_diff = var_model.forecast(forecast_input, steps=forecast_steps)

# Create a DataFrame for forecasted PCA components starting from 2023
forecast_pca_df = pd.DataFrame(forecast_diff, columns=['PC1', 'PC2', 'PC3'], 
                                index=pd.date_range(start='2023', periods=forecast_steps, freq='Y'))

# Reverse the differencing and add the cumulative sum of forecasted components
forecast_pca_df = forecast_pca_df.cumsum() + pca_df.iloc[-1]

# Step 7: Inverse PCA to get back the original revenue values
forecast_scaled_data = pca.inverse_transform(forecast_pca_df)
forecast_revenue_df = scaler.inverse_transform(forecast_scaled_data)
forecast_revenue_df = pd.DataFrame(forecast_revenue_df, columns=['Broadcasting', 'Commercial', 'Matchday'], index=forecast_pca_df.index)

# Step 8: Calculate total revenue (sum of Broadcasting, Commercial, and Matchday)
forecast_revenue_df['Total'] = forecast_revenue_df['Broadcasting'] + forecast_revenue_df['Commercial'] + forecast_revenue_df['Matchday']

# Step 9: Plot Forecasts (including total revenue)
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Broadcasting'], label='Historical Broadcasting Revenue', color='blue')
plt.plot(forecast_revenue_df.index, forecast_revenue_df['Broadcasting'], label='Forecasted Broadcasting Revenue', color='blue', linestyle='--')
plt.plot(df.index, df['Commercial'], label='Historical Commercial Revenue', color='green')
plt.plot(forecast_revenue_df.index, forecast_revenue_df['Commercial'], label='Forecasted Commercial Revenue', color='green', linestyle='--')
plt.plot(df.index, df['Matchday'], label='Historical Matchday Revenue', color='purple')
plt.plot(forecast_revenue_df.index, forecast_revenue_df['Matchday'], label='Forecasted Matchday Revenue', color='purple', linestyle='--')

# Plot Historical Total Revenue
plt.plot(df.index, df['Total'], label='Historical Total Revenue', color='orange')

# Plot Forecasted Total Revenue
plt.plot(forecast_revenue_df.index, forecast_revenue_df['Total'], label='Forecasted Total Revenue', color='red', linestyle='--')

# Format x-axis as years (2 digits for year)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y'))  # Display only last two digits of the year
plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.xlabel('Year')
plt.ylabel('Revenue (in million $)')
plt.title('Football Club Revenue Forecast with PCA and VAR Model')
plt.legend()
plt.show()

# Step 10: Print Forecasted Total Revenue for Next 10 Years
print("Forecasted Total Revenue for Next 10 Years:")
print(forecast_revenue_df[['Total']])
