from matplotlib import pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

# Plot
def plotresults(theta):
    x1 = np.linspace(X["Te"].min(), X["Te"].max(), 100)
    x2_fixed = np.mean(X["WS"])
    
    y_pred = theta[0] + theta[1] * x1 + theta[2] * x2_fixed
    
    fig = pp.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot(x1, np.full_like(x1, x2_fixed), y_pred, color='blue')
    
    ax.scatter(X["Te"], X["WS"], y, color='red')
    def update(angle):
        ax.view_init(elev=20, azim=angle)

    # Animate rotation
    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)

    pp.show()

def linear_regression(X, y, lr=0.01, epochs=10000, dec=0.9999, catch=1e-10, min_lr=1e-10):
    np.random.seed(60) 
    m, n = X.shape
    X = np.c_[np.ones(m), X] 
    theta = np.random.randn(n + 1)
    mse = float("inf")

    for epoch in range(epochs):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * X.T @ errors
        theta -= lr * gradient
        lr = max(lr * dec, min_lr)

        new_mse = np.mean(errors ** 2)
        if epoch % 10000 == 0:
            print(f"Epoch {epoch}: MSE = {new_mse}")

        if abs(mse - new_mse) < catch:
            break 
        mse = new_mse
    
    return theta

 
# Load data
raw_data = pd.read_csv("./Linear Regression/WeatherDataM.csv")
X = raw_data[["Te", "WS"]]
y = raw_data.Hu
 
# Train model
theta = linear_regression(X, y, lr=0.0001, epochs=500000, dec=1)
print("Theta:", theta)
 
plotresults(theta)
