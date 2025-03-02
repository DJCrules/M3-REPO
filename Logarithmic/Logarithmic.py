import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def logistic_growth(t, P0, r):
    # Carrying capacity for population
    K = 1524920  # calculated from size of birmingham in km^2 and average population density of London
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))

# Loads the dataset
United_Kingdom_population = pd.read_csv(r"C:\Users\joshk\OneDrive - Hills Road Sixth Form College\Documents\GitHub\M3-REPO\Logarithmic\Manchester_housing.csv", converters={
    "Population": lambda x: int(x.replace(",", ""))})

# Clean column names
United_Kingdom_population.columns = United_Kingdom_population.columns.str.strip()

# Prepares the data
X = United_Kingdom_population["Year"] - United_Kingdom_population["Year"].min()
y = United_Kingdom_population["Population"]

# Fits logistic model
params, _ = curve_fit(logistic_growth, X, y)

start_year = United_Kingdom_population["Year"].min()
end_year = 2045
future_years = np.arange(start_year, end_year + 1)
future_times = future_years - start_year
future_predictions = logistic_growth(future_times, * params)

# Print the estimated population for each year from 2012 to 2045
print("Estimated Population for Each Year from 2012 to 2045:")
for i in range(len(future_years)):
    year = future_years[i]
    prediction = future_predictions[i]
    print(f"Year: {year}, Estimated Population: {prediction:.0f}")

# Plots the results
plt.figure(figsize=(10, 6))
plt.scatter(United_Kingdom_population["Year"], y, color="blue", label="Actual Data")
plt.plot(future_years, logistic_growth(future_times, *params), color="red", linewidth=2, label="Logistic Growth Model")
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("United Kingdom Population Prediction with Logistic Growth")
plt.legend()
plt.show()
