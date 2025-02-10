import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class polynomial_trend:
    def __init__(self, filename):
        self.csv_file = filename
    def generate_cubic_data(self, num_points=100, noise_level=2, coefficients=[0.05, 0.005, 0.005, 0]):
        """
        wipes csvv and writes noisy cubic
        example: generate_cubic_data(".\Polynomial\sample_data.csv", num_points=200, noise_level=20, coefficients=[1, -2, 3, -5])
        """
        x = np.linspace(-5, 4.9, num_points)

        a, b, c, d = coefficients[0], coefficients[1], coefficients[2], coefficients[3]
        y = a * x**3 + b * x**2 + c * x + d

        noise = np.random.normal(0, noise_level, num_points)
        y_noisy = y + noise

        df = pd.DataFrame({'x': x, 'y': y_noisy})

        print(self.csv_file)
        df.to_csv(self.csv_file, index=False)
    
    def generate_quadratic_data(self, num_points=100, noise_level=10, coefficients=[1, -2, 3]):
        """
        wipes csvv and writes noisy cubic
        example: generate_cubic_data(".\Polynomial\sample_data.csv", num_points=200, noise_level=20, coefficients=[1, -2, 3])
        """
        x = np.linspace(-10, 10, num_points)

        a, b, c = coefficients[0], coefficients[1], coefficients[2]
        y = a * x**2 + b * x + c

        noise = np.random.normal(0, noise_level, num_points)
        y_noisy = y + noise

        df = pd.DataFrame({'x': x, 'y': y_noisy})

        print(self.csv_file)
        df.to_csv(self.csv_file, index=False)

    def plot_data(self):
        """
        Reads a CSV file and plots a scatter graph.
        Assumes the CSV has two columns: x-values and y-values (no headers).
        """
        try:
            # Read CSV file
            df = pd.read_csv(self.csv_file)
            
            # Ensure there are at least two columns
            if df.shape[1] < 2:
                raise ValueError("CSV file must have at least two columns for x and y values.")
            
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            
            # Create scatter plot
            plt.scatter(x, y)
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.title("Scatter")
            
            # Show plot
            plt.show()
        except Exception as e:
            print(f"Error: {e}")

newtrend = polynomial_trend('.\Polynomial\sample_data.csv')

newtrend.generate_cubic_data()
newtrend.plot_data()

   
