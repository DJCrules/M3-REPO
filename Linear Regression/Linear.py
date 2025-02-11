from matplotlib import pyplot as pp
import numpy as np

x = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])

# Define a linear relationship with random noise
slope = np.random.randint(-100, 100)
intercept = np.random.randint(0,100)
noise = np.random.normal(0, 50, size=len(x))

# Generate y values
y = slope * x + intercept + noise

# Convert to list for desired format
x = list(x)
y = list(y.astype(int))

def linear_regression(x, y, learning_rate=0.0001, iterations=5000):  # Reduced learning rate
    m, n = x.shape
    x = np.c_[np.ones(m), x]  # Add bias term (ones column)
    theta = np.zeros(n + 1)  # Initialize parameters
    
    for i in range(iterations):
        predictions = x @ theta
        errors = predictions - y
        gradient = (1/m) * (x.T @ errors)
        theta -= learning_rate * gradient
    
    return theta

x = np.array(x)
x = x.reshape(-1, 1)
y = np.array(y)
theta = linear_regression(x, y)
print("Learned parameters:", theta)

pp.scatter(x, y)
x = np.linspace(0, 50, 100) 
y = x * theta[1] + theta[0]
pp.plot(x, y)

pp.xlabel("x")
pp.ylabel("y")
pp.grid(True)
pp.show()
