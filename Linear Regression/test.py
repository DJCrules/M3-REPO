from matplotlib import pyplot as pp
import numpy as np

data = [
    [1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 
    2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    [187070, 187440, 186560, 188100, 189190, 189520, 190860, 192570, 193810, 194920, 197000, 
    199150, 201190, 203240, 208300, 213940, 216160, 217870, 218270, 219280, 221340, 221850, 
    222730, 224570, 226310, 228300, 230900, 234290, 238800],
    [1780, 1840, 1950, 1990, 2070, 2140, 2250, 2310, 2380, 2420, 2470, 2500, 2570, 2620, 2620, 
    2670, 2680, 2690, 2710, 2760, 2770, 2780, 2830, 2830, 2840, 2840, 2840, 2850, 2850],
    [39620, 40270, 42600, 44750, 47210, 48770, 50210, 51860, 54060, 56450, 59080, 61170, 63280, 
    65170, 69390, 74130, 76120, 77790, 78420, 78940, 80210, 80300, 80600, 81500, 82470, 83760, 
    85420, 88120, 92150],
    [59010, 60770, 62650, 64670, 66660, 69360, 71750, 74100, 76170, 77070, 77500, 77840, 77710, 
    77680, 78140, 78600, 78670, 78550, 78180, 78390, 78650, 78740, 78930, 79340, 79640, 79860, 
    80050, 80260, 80430],
    [39060, 40350, 41790, 43140, 44590, 46310, 47930, 49360, 50820, 51500, 51810, 52020, 52260, 
    52340, 52550, 52790, 52890, 53010, 53090, 53280, 53590, 53850, 54100, 54540, 54880, 55150, 
    55630, 55990, 56280],
    [2710, 2860, 3070, 3290, 3490, 3730, 3970, 4210, 4410, 4540, 4610, 4740, 4880, 4990, 5170, 
    5320, 5370, 5440, 5480, 5530, 5640, 5730, 5820, 5910, 5990, 6080, 6180, 6240, 6290]
]

# Normalize the x_data (years)
x_data = np.array(data[0])  # Years
x_min, x_max = min(x_data), max(x_data)
x_data_normalized = (x_data - x_min) / (x_max - x_min)

# Linear regression function
def linear_regression(training_array, target_array, learning_rate=0.01, iterations=10000):
    m, n = training_array.shape
    training_array = np.c_[np.ones(m), training_array]  # Add bias
    theta = np.random.randn(n + 1)  # Use random initialization
    cost_history = []

    for i in range(iterations):
        predictions = np.dot(training_array, theta)
        errors = predictions - target_array
        gradient = (1 / m) * np.dot(training_array.T, errors)
        theta -= learning_rate * gradient

        # Calculate and record cost
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        cost_history.append(cost)

        if i % 1000 == 0:  # Print every 1000 iterations for feedback
            print(f"Iteration {i}, Cost: {cost:.4f}")

    return theta, cost_history

# Apply linear regression to normalized x_data
theta_y, cost_history_y = linear_regression(x_data_normalized.reshape(-1, 1), np.array(data[1]))
theta_z, cost_history_z = linear_regression(x_data_normalized.reshape(-1, 1), np.array(data[2]))
theta_w, cost_history_w = linear_regression(x_data_normalized.reshape(-1, 1), np.array(data[3]))

# Plot the cost history
pp.plot(range(len(cost_history_y)), cost_history_y, label="Cost for Y")
pp.plot(range(len(cost_history_z)), cost_history_z, label="Cost for Z")
pp.plot(range(len(cost_history_w)), cost_history_w, label="Cost for W")
pp.xlabel("Iterations")
pp.ylabel("Cost")
pp.legend()
pp.show()

# Predictions for future years (2030, 2040, 2050) - denormalize the result
def predict(theta, year):
    year_normalized = (year - x_min) / (x_max - x_min)  # Normalize the year
    pred_normalized = theta[0] + year_normalized * theta[1]
    return pred_normalized * (x_max - x_min) + x_min  # Denormalize the prediction

# Predict for future years (2030, 2040, 2050)
future_years = [2030, 2040, 2050]
predictions_y = [predict(theta_y, year) for year in future_years]
predictions_z = [predict(theta_z, year) for year in future_years]
predictions_w = [predict(theta_w, year) for year in future_years]

print("Predictions for the year 2030, 2040, 2050:")
for i, year in enumerate(future_years):
    print(f"Year {year}: Y: {predictions_y[i]:.2f}, Z: {predictions_z[i]:.2f}, W: {predictions_w[i]:.2f}")
