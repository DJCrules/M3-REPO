from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# # Create random set of data
# data = np.array([[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]])

# dimensions = 2

# # Initialising data with noise for random
# for i in range(dimensions):
#     slope = np.random.randint(-100, 100)
#     intercept = np.random.randint(0, 100)
#     noise = np.random.normal(0, 50, size=len(data[0]))

#     new_row = np.array([slope * data[0] + intercept + noise])  
#     data = np.concatenate([data, new_row], axis=0)

data = np.trunc(data)
print(data)

def linear_regression(training_array, target_array, learning_rate=1e-6, iterations=100000):
    m, n = training_array.shape

    # Add the biases
    training_array = np.c_[np.ones(m), training_array] 

    # Initialize theta
    theta = np.zeros(n + 1)

    # Gradient descent
    for i in range(iterations):
        predictions = np.dot(training_array, theta)
        errors = predictions - target_array
        gradient = (1 / m) * (np.dot(training_array.T, errors))
        theta -= learning_rate * gradient

    return theta

# Split data 
x_data = data[0]
y_data = data[1]
z_data = data[2]

theta_y = linear_regression(x_data.reshape(-1, 1), y_data)
theta_z = linear_regression(x_data.reshape(-1, 1), z_data)

print("Learned parameters for y:", theta_y)
print("Learned parameters for z:", theta_z)

# Create 3D plot
fig = pp.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the training data
ax.scatter(x_data, y_data, z_data)

# Predicted y and z based on x
x_range = np.linspace(min(x_data), max(x_data), 100)
y_pred = theta_y[0] + theta_y[1] * x_range
z_pred = theta_z[0] + theta_z[1] * x_range

# Plot the regression lines
ax.plot(x_range, y_pred, z_pred, color='red')

pp.show()
