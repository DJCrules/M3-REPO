import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class polynomial_trend:
    def __init__(self, filename, order=0, new=False):
        self.csv_file = filename
        if order != 0 and new:
            self.generate_poly_data(order, noise_level=100)
            self.plot_data()
        elif order == 0:
            self.plot_data()
        df = pd.read_csv(self.csv_file)
        self.xs = df.iloc[:, 0]
        self.ys = df.iloc[:, 1]
        self.numpy_xs = np.array(self.xs)
        self.numpy_ys = np.array(self.ys)

    def generate_poly_data(self, order, num_points=100, noise_level=2):
        coefficients=np.random.normal(0, 1, order + 1)
        x = np.linspace(-10, 10, num_points)
        y = np.sum([c * (x**i) for i, c in enumerate(coefficients)], axis=0)
        noise = np.random.normal(0, noise_level, num_points)
        y += noise
        df = pd.DataFrame({'x': x, 'y': y})
        df.to_csv(self.csv_file, index=False)
    
    def plot_data(self, coefficients=None):
        plt.scatter(self.xs, self.ys)
        if coefficients is not None:
            poly_func = np.poly1d(coefficients[::-1])
            x_range = np.linspace(min(self.xs), max(self.xs), 100)
            y_fit = poly_func(x_range)
            plt.plot(x_range, y_fit, color='red', label="Polynomial Fit")
        plt.show()
    
    def regress(self, order, rate=0.01, tolerance=1e-6, max_iterations=10000):
        coefficients = np.random.normal(0, 0.1, order + 1)
        print(coefficients)
        
        last_mse = self.MSE(coefficients)
        print(last_mse)
        iteration = 0

        while iteration < max_iterations:
            predicted_ys = np.polyval(coefficients[::-1], self.numpy_xs)
            
            errors = predicted_ys - self.numpy_ys
            
            gradient = np.zeros(order + 1)
            for i in range(order + 1):
                gradient[i] = np.mean(errors * (self.numpy_xs ** i))
            
            coefficients -= rate * gradient
            
            current_mse = self.MSE(coefficients)
            
            if abs(last_mse - current_mse) < tolerance:
                print(f"Converged after {iteration} iterations.")
                break
            
            last_mse = current_mse
            iteration += 1
            
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}: MSE = {current_mse}")
        
        return coefficients
    
    def MSE(self, coefficients):
        predictions = np.polyval(coefficients[::-1], self.xs)
        errors = (predictions - self.ys) ** 2
        return np.mean(errors)


newtrend = polynomial_trend('.\Polynomial\sample_data.csv', 3, False)
cos = newtrend.regress(3)
newtrend.plot_data(cos)