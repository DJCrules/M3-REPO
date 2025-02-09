import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def logistic_growth(t, P0, r):
    # Carrying capacity for Seattle based off sources
    K = 532000
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))

def add_noise(y, sigma=0.05):
    # Adds a random 5% noise for jittering to test model sensitivity
    return y * (1 + np.random.normal(0, sigma, size=y.shape))

# Load the dataset
seattle_housing = df = pd.read_csv('D:\M3 Repository\M3-REPO\Logarithmic\seattle_housing_units.csv', converters={
    'Total housing units': lambda x: int(x.replace(',', ''))
})

# Clean column names
seattle_housing.columns = seattle_housing.columns.str.strip()

# Uncomment to not include Covid data
# seattle_housing = seattle_housing[seattle_housing['Year'] < 2020]

# Prepare the data
X = seattle_housing['Year'] - seattle_housing['Year'].min()
y = seattle_housing['Total housing units']

# Fit logistic model
params, _ = curve_fit(logistic_growth, X, y, p0=[y.iloc[0], 0.1])

# Predict for future years
start_year = seattle_housing['Year'].min()
end_year = 2070
future_years = np.arange(start_year, end_year + 1)
future_times = future_years - start_year
future_predictions = logistic_growth(future_times, *params)

# Selected future years for predictions
selected_future_years = np.array([2030, 2040, 2070])
selected_predictions = logistic_growth(selected_future_years - start_year, *params)

# Sensitivity analysis averaged over 1000 trials
num_trials = 1000
percent_differences = []

for _ in range(num_trials):
    # Add noise to the data
    y_jittered = add_noise(y)
    params_jittered, _ = curve_fit(logistic_growth, X, y_jittered, p0=[y.iloc[0], 0.1])
    
    # Get predictions for selected future years with jittered data
    selected_predictions_jittered = logistic_growth(selected_future_years - start_year, *params_jittered)
    
    # Calculate percentage differences
    percent_diff = 100 * np.abs(selected_predictions - selected_predictions_jittered) / selected_predictions
    percent_differences.append(percent_diff)

# Calculate average percent differences
average_percent_diff = np.mean(percent_differences, axis=0)

print("Average Percent Differences for Selected Future Predictions:")
for year, diff in zip(selected_future_years, average_percent_diff):
    print(f'Year: {year}, Average Percent Difference: {diff:.3f}%')

print("Estimated Number of Houses for Selected Future Years:")
for year, prediction in zip(selected_future_years, selected_predictions):
    print(f'Year: {year}, Estimated Total Housing Units: {prediction:.0f}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(seattle_housing['Year'], y, color='blue', label='Actual Data')
plt.plot(future_years, logistic_growth(future_times, *params), color='red', linewidth=2, label='Logistic Growth Model')
plt.scatter(selected_future_years, selected_predictions, color='green', label='Selected Future Predictions')
plt.xlabel('Year')
plt.ylabel('Total Housing Units')
plt.title('Seattle Housing Units Prediction with Logistic Growth')
plt.legend()
plt.show()
