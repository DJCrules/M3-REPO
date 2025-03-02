import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import warnings

warnings.filterwarnings("ignore")  # Stops lots of warning notices

def interpolate_monthly(yearly_data, years, reference_monthly_pattern):
    """Distributes yearly data across 12 months while preserving seasonality from reference data."""
    monthly_data = []
    monthly_dates = []
    normalized_pattern = reference_monthly_pattern / reference_monthly_pattern.sum()  # Normalize seasonal pattern
    
    for year, year_total in zip(years, yearly_data):
        monthly_values = year_total * normalized_pattern
        for month, value in enumerate(monthly_values, start=1):
            monthly_dates.append(pd.Timestamp(f"{year}-{month:02d}-01"))
            monthly_data.append(value)
    return monthly_dates, monthly_data

def compute_trend(yearly_x, yearly_y, window):
    """Computes the overall trend using a moving average window."""
    if len(yearly_x) < window:
        trend = np.mean(yearly_y)  # If not enough data points, use the overall mean
    else:
        trend = np.mean(yearly_y[-window:])  # Use the last 'window' years to estimate the trend
    return trend

def grid_search_sarima(x, y, param_grid, seasonal_period=12, forecast_steps=12):
    x = pd.to_datetime(x)
    ts = pd.Series(y, index=x).sort_index()
    
    # Ensure there are no duplicate dates
    ts = ts[~ts.index.duplicated(keep='first')]
    
    # Apply smoothing using a moving average to capture the overall trend
    smoothed_ts = ts.rolling(window=window * 12, min_periods=1).mean()
    trend_slope = (smoothed_ts.iloc[-1] - smoothed_ts.iloc[-window * 12]) / (window * 12)  # Compute slope over past 'window' years
    
    p_values, d_values, q_values = param_grid['p'], param_grid['d'], param_grid['q']
    P_values, D_values, Q_values = param_grid['P'], param_grid['D'], param_grid['Q']
    
    # Variables tracking for the best model
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None
    
    # Iterates over all combinations
    for order in product(p_values, d_values, q_values):
        for seasonal_order in product(P_values, D_values, Q_values):
            try:
                seasonal_order_full = (seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_period)
                model = SARIMAX(
                    smoothed_ts, order=order, seasonal_order=seasonal_order_full,
                    enforce_stationarity=False, enforce_invertibility=False
                )
                model_fit = model.fit(disp=False)
                
                # Update best model if AIC is better
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
                    best_seasonal_order = seasonal_order_full
                    best_model = model_fit
                    print(f"New best AIC: {best_aic} | Order: {order} | Seasonal Order: {seasonal_order_full}")
            except Exception:
                continue
    
    # Generate the forecast with the model that has best AIC
    forecast = best_model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean + np.arange(1, forecast_steps + 1) * trend_slope
    
    # Generates the dates for the prediction
    future_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
    forecast_mean.index = future_dates
    
    # Debugging: Print forecasted values
    print("\nForecasted Values:")
    print(forecast_mean)
    
    # Apply monthly seasonality pattern to forecasted values correctly
    seasonal_pattern = ts[-seasonal_period:].values / np.mean(ts[-seasonal_period:])
    seasonal_pattern = np.tile(seasonal_pattern, int(np.ceil(len(forecast_mean) / seasonal_period)))[:len(forecast_mean)]
    forecast_monthly = forecast_mean * seasonal_pattern
    
    # Debugging: Print seasonally adjusted forecast
    print("\nSeasonally Adjusted Forecast Values:")
    print(forecast_monthly)
    
    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Main forecast plot
    axs[0].plot(ts.index, ts, label='Observed', color='blue')
    axs[0].plot(smoothed_ts.index, smoothed_ts, label='Smoothed Trend', color='green', linestyle='dashed')
    axs[0].plot([ts.index[-1], forecast_mean.index[0]], [ts.iloc[-1], forecast_mean.iloc[0]], color='blue', linestyle='dashed')
    axs[0].plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
    axs[0].set_title(f'SARIMA {best_order}x{best_seasonal_order} Forecast (AIC={best_aic:.2f})')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Value')
    axs[0].legend()
    
    # Subplot applying monthly seasonality to forecast
    axs[1].plot(forecast_mean.index, forecast_monthly, label='Seasonally Adjusted Forecast', color='purple')
    axs[1].set_title('Monthly Seasonal Pattern Applied to Forecast')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Value')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

    return best_model, best_order, best_seasonal_order, forecast_mean

if __name__ == "__main__":
    # Seed for the sample data -- can be removed for actual data
    np.random.seed(42)
    
    # Input one year's worth of monthly data
    monthly_x = pd.date_range(start="2019-01-01", periods=12, freq='MS')
    monthly_y = [385849808.2, 335376020.6, 358609741.9, 320008018.3, 312154067.4, 291720378, 302475976.3, 297524892.2, 298460444.7, 343314758, 368403830.9, 371802296.2]
    population_2019 = 1145500
    monthly_y = np.array(monthly_y) / population_2019  # Normalize monthly consumption by population
    reference_seasonality = np.array(monthly_y) / np.sum(monthly_y)  # Extract seasonal pattern
    
    # Input yearly totals and population data
    yearly_x = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    yearly_y = [3526358380, 3559905506, 3540478638, 3587412567, 3614993602, 3744760554, 3723931951, 3659963650, 3414885059, 3418629809, 3320164729]
    population = [1085400, 1092300, 1101400, 1111300, 1120200, 1128100, 1136900, 1145500, 1153300, 1144900, 1153300]
    
    yearly_y = np.array(yearly_y) / np.array(population)
    window = 4  # Define the window explicitly
    trend_value = compute_trend(yearly_x, yearly_y, window)
    
    yearly_dates, yearly_monthly_y = interpolate_monthly(yearly_y, yearly_x, reference_seasonality)
    x = yearly_dates + list(monthly_x)
    y = yearly_monthly_y + list(monthly_y)
    
    param_grid = {'p': [0, 1, 2], 'd': [0, 1], 'q': [0, 1, 2], 'P': [0, 1], 'D': [0, 1], 'Q': [0, 1]}
    
    best_model, best_order, best_seasonal_order, forecast_mean = grid_search_sarima(x, y, param_grid, 12, 240)
