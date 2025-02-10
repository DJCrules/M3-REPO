import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import warnings

warnings.filterwarnings("ignore") # Stops lots of warning notices

def grid_search_sarima(x, y, param_grid, seasonal_period=12, forecast_steps=1):

    x = pd.to_datetime(x)
    ts = pd.Series(y, index=x).sort_index()
    
    # Makes sure all of the time intervals are regular
    freq = pd.infer_freq(ts.index)
    if freq is None:
        raise ValueError("Dates are irregular")
    
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
                    ts, order=order, seasonal_order=seasonal_order_full,
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
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    offset = pd.tseries.frequencies.to_offset(freq)
    
    # Generates the dates for the prediction
    future_dates = pd.date_range(start=ts.index[-1] + offset, periods=forecast_steps, freq=freq)
    forecast_mean.index = future_dates
    forecast_conf_int.index = future_dates
    
    # Plots result
    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts, label='Observed', color='blue')
    plt.plot([ts.index[-1], forecast_mean.index[0]], [ts.iloc[-1], forecast_mean.iloc[0]], color='blue', linestyle='dashed')      # Connects the last known point to the first forecasted point
    plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
    plt.fill_between(
        forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1],
        color='pink', alpha=0.3
    )
    plt.title(f'SARIMA {best_order}x{best_seasonal_order} Forecast (AIC={best_aic:.2f})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    return best_model, best_order, best_seasonal_order, (forecast_mean, forecast_conf_int)

if __name__ == "__main__":
    # Seed for the sample data -- can be removed for actual data
    np.random.seed(42)
    
    x = pd.date_range(start="2020-01-01", end="2022-12-31", freq="MS")  # Replace with appropriate time frame and interval e.g 'MS' --> month start, 'M' --> month end, 'Q' --> quarterly end
    trend = 10 * (x.year - 2020)  # VARIABLE JUST FOR TEST DATA
    noise = np.random.normal(0, 8, len(x))  #  VARIABLE JUST FOR TEST DATA
    y = np.round(trend + noise).astype(int)  # Replace with the appropriate data
    
    param_grid = {
        'p': [0, 1, 2],  # Non-seasonal Auto-Regression order
        'd': [0, 1],     # Non-seasonal differencing
        'q': [0, 1, 2],  # Non-seasonal Moving-Average order
        'P': [0, 1],     # Seasonal Auto-Regression order
        'D': [0, 1],     # Seasonal differencing
        'Q': [0, 1]      # Seasonal Moving-Average order
    }
    
    # This parameter defines the length in units of the season, e.g 12 --> monthly, 4 --> quarterly
    seasonal_period = 12  # This value can be changed depending on how the data cycles (quarterly, monthly, etc.)
    
    # Number of time intervals that will be predicted
    forecast_steps = 12  # This can be changed for the model to predict a different amount of time
    
    # Runs a grid search to optimise parameters
    best_model, best_order, best_seasonal_order, forecasts = grid_search_sarima(
        x, y, param_grid, seasonal_period, forecast_steps
    )
    
    # Prints results
    print("\nBest Parameters:")
    print(f"Order (p, d, q): {best_order}")
    print(f"Seasonal Order (P, D, Q, m): {best_seasonal_order}")
    print(f"Best AIC: {best_model.aic:.2f}")
    
    forecast_mean, forecast_conf_int = forecasts
    print("\nForecasted Values:\n", forecast_mean)
    print("\nConfidence Intervals:\n", forecast_conf_int)
