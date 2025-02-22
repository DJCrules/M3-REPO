from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

def linear_regression(X, y, lr=0.01, epochs=10000, dec = 0.999):
    m, n = X.shape
    X = np.c_[np.ones(m), X]
    theta = np.random.randn(n + 1)

    for epoch in range(epochs):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * X.T @ errors
        theta -= lr * gradient
        lr *= dec
    
    return theta

# Load data
raw_data = pd.read_csv("/Users/oscarhorton/Documents/Programming/M3/MainGit/M3-REPO/Linear Regression/WeatherDataM.csv")
X = raw_data[["Te", "WS"]]
y = raw_data.Hu

# Train model
theta = linear_regression(X, y, lr=0.01, epochs=10000)
print("Theta:", theta)

# Plot
x1 = np.linspace(X["Te"].min(), X["Te"].max(), 100)
x2_fixed = np.mean(X["WS"])

y_pred = theta[0] + theta[1] * x1 + theta[2] * x2_fixed

fig = pp.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(x1, np.full_like(x1, x2_fixed), y_pred, color='blue')

ax.scatter(X["Te"], X["WS"], y, color='red')

pp.show()