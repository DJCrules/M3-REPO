import numpy as np
from matplotlib import pyplot as plt

def linear_regression(x, y, learning_rate=0.002, iterations=100000):
    m, n = x.shape
    x = np.c_[np.ones(m), x]
    theta = np.zeros(n + 1)

    for i in range(iterations):
        predictions = x @ theta
        errors = predictions - y
        gradient = (1/m) * (x.T @ errors)
        theta -= learning_rate * gradient

    return theta

x = np.array([21, 19, 18, 17, 16, 15, 14, 13, 12]).reshape(-1, 1)
y = np.array([
    15.25486540,
    15.35046668,
    15.51441077,
    15.84877006,
    16.02218919,
    16.32646065,
    16.50323202,
    16.32305132,
    16.63571950
])

theta = linear_regression(x, y)

plt.scatter(x, y, label="Actual Data", color="blue")

x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
x_range_with_bias = np.c_[np.ones(x_range.shape[0]), x_range]
y_range = x_range_with_bias @ theta

plt.plot(x_range, y_range, label="Linear Regression Fit", color="red")

plt.xlabel("Year (20--)")
plt.ylabel("Energy (100 GWh)")
plt.title("Linear Regression Fit for Domestic power usage in Birmingham")
plt.grid(True)
plt.legend()

print(theta)
plt.show()
