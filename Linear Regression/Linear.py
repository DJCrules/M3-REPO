from matplotlib import pyplot as pp
import numpy as np

x = np.array([2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012])
y = np.array([1424117394, 1525486540, 1609229857, 1535046668, 1551441077, 1584877006, 1602218919, 1632646065, 1650323202, 1632305132, 1663571950])

# Reshaping x and y to ensure proper 2D arrays for matrix operations
x = x.reshape(-1, 1)  # Ensure x is a column vector
y = y.reshape(-1, 1)  # Ensure y is a column vector

# Linear regression function
def linear_regression(x, y, learning_rate=0.0001, iterations=5000):
    m = len(x)  # Number of data points
    x = np.c_[np.ones(m), x]  # Add bias term (ones column)
    theta = np.zeros(x.shape[1])  # Initialize parameters (theta)

    for i in range(iterations):
        predictions = x @ theta  # Matrix multiplication for predictions
        errors = predictions - y
        gradient = (1/m) * (x.T @ errors)  # Gradient should have shape (2, 1)
        
        # Ensure the gradient has the same shape as theta before updating
        gradient = gradient.flatten()  # Make sure gradient is a 1D array
        
        theta -= learning_rate * gradient  # Update theta
    
    return theta



theta = linear_regression(x, y)  # Perform linear regression
print("Learned parameters:", theta)

# Scatter plot
pp.scatter(x, y)

# Plot the regression line
x_line = np.linspace(min(x), max(x), 100)  # Create a range for x values to plot the line
x_line = x_line.reshape(-1, 1)  # Reshape for matrix multiplication
x_line = np.c_[np.ones(x_line.shape[0]), x_line]  # Add bias term (ones column)
y_line = x_line @ theta  # Predictions for the regression line
pp.plot(x_line[:, 1], y_line, color='red')  # Plot the line

# Labels and grid
pp.xlabel("Year")
pp.ylabel("Value")
pp.grid(True)
pp.show()
