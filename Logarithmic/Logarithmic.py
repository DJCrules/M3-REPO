import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def logistic_growth(t, P0, r):
    #Carrying capacity for population
    K = 1000000000 #Guess
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))

def add_noise(y, sigma=0.05):
    #Adds a random 5% noise for jittering to test model sensitivity
    return y * (1 + np.random.normal(0, sigma, size=y.shape))

#Loads the dataset
United_Kingdom_population = df = pd.read_csv(r"D:\M3 Repository\M3-REPO\Logarithmic\united_kingdom_population.csv", converters={
    "Population": lambda x: int(x.replace(",", ""))})

# Clean column names
United_Kingdom_population.columns = United_Kingdom_population.columns.str.strip()

#Prepares the data
X = United_Kingdom_population["Year"] - United_Kingdom_population["Year"].min()
y = United_Kingdom_population["Population"]

#Fits logistic model
params, _ = curve_fit(logistic_growth, X, y, p0=[y.iloc[0], 0.1])


start_year = United_Kingdom_population["Year"].min()
end_year = 2200
future_years = np.arange(start_year, end_year + 1)
future_times = future_years - start_year
future_predictions = logistic_growth(future_times, *params)

#Selects future years for predictions
selected_future_years = np.array([2100, 2150, 2200])
selected_predictions = logistic_growth(selected_future_years - start_year, *params)

#Sensitivity analysis averaged over 10 trials
num_trials = 10
percent_differences = []

for _ in range(num_trials):
    #Adds noise to the data
    y_jittered = add_noise(y)
    params_jittered, _ = curve_fit(logistic_growth, X, y_jittered, p0=[y.iloc[0], 0.1])
    
    #Gets predictions for selected future years with jittered data
    selected_predictions_jittered = logistic_growth(selected_future_years - start_year, *params_jittered)
    
    #Calculates percentage differences
    percent_diff = 100 * np.abs(selected_predictions - selected_predictions_jittered) / selected_predictions
    percent_differences.append(percent_diff)

#Calculates average percent differences
average_percent_diff = np.mean(percent_differences, axis=0)

P0_est, r_est = params
print(f"Estimated Growth Rate (r): {r_est:.4f}")

print("Average Percent Differences for Selected Future Predictions:")
for year, diff in zip(selected_future_years, average_percent_diff):
    print(f"Year: {year}, Average Percent Difference: {diff:.3f}%")

print("Estimated Population for Selected Future Years:")
for year, prediction in zip(selected_future_years, selected_predictions):
    print(f"Year: {year}, Estimated Population: {prediction:.0f}")

#Plots the results
plt.figure(figsize=(10, 6))
plt.scatter(United_Kingdom_population["Year"], y, color="blue", label="Actual Data")
plt.plot(future_years, logistic_growth(future_times, *params), color="red", linewidth=2, label="Logistic Growth Model")
plt.scatter(selected_future_years, selected_predictions, color="green", label="Selected Future Predictions")
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("United Kingdom population prediction with Logistic Growth")
plt.legend()
plt.show()
