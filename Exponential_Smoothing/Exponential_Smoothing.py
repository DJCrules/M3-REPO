import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#Loads revenue data from CSV
csv_file = "D:/M3 Repository/M3-REPO/Exponential_Smoothing/Liverpool_revenue.csv"  
df = pd.read_csv(csv_file)

#Converts "Year" to datetime and set as index
df["Year"] = pd.to_datetime(df["Year"], format="%Y")
df.set_index("Year", inplace=True)

#Applies Exponential Smoothing to a revenue stream
def forecast_revenue(series, steps=10):
    model = ExponentialSmoothing(series, trend="add", seasonal=None, damped_trend=True)
    fitted_model = model.fit()
    return fitted_model.forecast(steps=steps), fitted_model.fittedvalues, fitted_model.predict(start=series.index[0], end=series.index[-1] + pd.DateOffset(years=steps))

#Forecasts each revenue stream separately
future_years = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), periods=10, freq="Y")

forecast_broadcasting, fitted_broadcasting, full_fit_broadcasting = forecast_revenue(df["Broadcasting"])
forecast_commercial, fitted_commercial, full_fit_commercial = forecast_revenue(df["Commercial"])
forecast_matchday, fitted_matchday, full_fit_matchday = forecast_revenue(df["Matchday"])

#Creates a DataFrame for the future predictions
forecast_df = pd.DataFrame({
    "Year": future_years,
    "Broadcasting": forecast_broadcasting,
    "Commercial": forecast_commercial,
    "Matchday": forecast_matchday
})

#Calculates total revenue
forecast_df["Total_Revenue"] = (
    forecast_df["Broadcasting"] + forecast_df["Commercial"] + forecast_df["Matchday"]
)
forecast_df.set_index("Year", inplace=True)

#Creates a continuous total revenue series (historical + forecast)
total_revenue_historical = df["Broadcasting"] + df["Commercial"] + df["Matchday"]
total_revenue_fitted = full_fit_broadcasting + full_fit_commercial + full_fit_matchday
total_revenue_forecast = forecast_df["Total_Revenue"]

#Combine for a continuous plot
combined_years = list(df.index) + list(forecast_df.index)
combined_revenue = list(total_revenue_historical) + list(total_revenue_forecast)

#Gets the last forecasted year only
last_forecast_year = forecast_df.index[-1]
last_forecast_value = forecast_df.loc[last_forecast_year, "Total_Revenue"]

#Plots total revenue forecast
plt.figure(figsize=(12, 6))

#Plots the exponential smoothing curve (historical + forecast)
plt.plot(combined_years, total_revenue_fitted, color="red", label="Exponential Smoothing Curve", linestyle="-")

#Plots historical total revenue data points (without connecting lines)
plt.scatter(df.index, total_revenue_historical, color="blue", label="Historical Total Revenue", zorder=5)

#Adds labels, legend, and title
plt.xlabel("Year")
plt.ylabel("Total Revenue")
plt.legend()
plt.title("Football Team Total Revenue Forecast")
plt.grid(True)
plt.show()