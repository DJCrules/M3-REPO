import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class polynomial_trend:
    def __init__(self, filename, order=0, new=False):
        """
        use order to set generated data trend, 2 for quadratic, 3 for cubic etc
        filename is where to save to or where to read from
        new is whether to make a new one, set to false auto if order=0
        """
        self.csv_file = filename
        if order != 0 and new:
            self.generate_poly_data(order)
        df = pd.read_csv(self.csv_file)
        self.xs = df.iloc[:, 0]
        self.ys = df.iloc[:, 1]
        self.numpy_xs = np.array(self.xs)
        self.numpy_ys = np.array(self.ys)
        self.plot_data()

    def generate_poly_data(self, order, num_points=4):
        """
        the added noise is order ** 2
        """
        coefficients=np.random.normal(0, 1.5, order + 1)
        x = np.linspace(-3, 3, num_points)
        y = np.sum([c * (x**i) for i, c in enumerate(coefficients)], axis=0)
        noise = np.random.normal(0, order**2, num_points)
        y += noise
        df = pd.DataFrame({'x': x, 'y': y})
        df.to_csv(self.csv_file, index=False)
    
    def plot_data(self, coefficients=None, extra=0, mse=0):
        """
        extra is short for extrapolation
        mse is on title
        if coefficients is none, no line of best fit
        """
        plt.scatter(self.xs, self.ys)
        if coefficients is not None:
            poly_func = np.poly1d(coefficients[::-1])
            x_range = np.linspace(min(self.xs), max(self.xs) + extra, 100)
            y_fit = poly_func(x_range)
            plt.plot(x_range, y_fit, color='red')
            plt.title(f"Relative MSE: {np.round((mse /(max(self.xs) - min(self.xs))), 2)}\ny = " + " + ".join(f"{np.round(c, 2)} * x^{i}" for i, c in enumerate(coefficients)))
        plt.show()
    
    def regress(self, order, rate=0.0000005, tolerance=0.0001, max_iterations=25000):
        """
        honestly not that great, really good at cubic but quite bad at anything < 3
        """
        coefficients = np.random.normal(0, pow(max(self.xs) - min(self.xs), (1/order)), order + 1)
        #randomise coefficients
        last_mse = self.MSE(coefficients)
        iteration = 0
        while iteration < max_iterations:
            poly_func = np.poly1d(coefficients[::-1])
            predicted_ys = poly_func(self.numpy_xs)
            errors = predicted_ys - self.numpy_ys
            gradient = np.zeros(order + 1)
            for i in range(order + 1):
                gradient[i] = np.mean(2 * errors * np.power(self.numpy_xs, i))
            coefficients -= rate * gradient
            current_mse = self.MSE(coefficients)
            if abs(last_mse - current_mse) < tolerance:
                print(f"MSE: {current_mse}")
                break
            last_mse = current_mse
            iteration += 1
        return coefficients, current_mse
    
    def MSE(self, coefficients):
        poly_func = np.poly1d(coefficients[::-1])
        predicted_ys = poly_func(self.numpy_xs)
        errors = (predicted_ys - self.numpy_ys) ** 2
        return np.mean(errors)

    def general_regress(self):
        best_coefficients, best_mse = self.regress(1)
        bestBIC = self.BIC(best_coefficients, best_mse)
        datapoints = len(self.xs)
        for i in range(1, int(np.round(np.sqrt(datapoints), 0)) + 1):
            coeffs, mse = self.regress(i)
            newtrend.plot_data(coeffs, mse=mse)
            current_BIC = self.BIC(coeffs, mse)
            if current_BIC < bestBIC:
                best_coefficients, best_mse = coeffs, mse
                bestBIC = current_BIC
        return best_coefficients, best_mse
    
    def BIC(self, coeffs, mse):
        n = len(self.xs)
        k = len(coeffs) 
        return (n * np.log(mse)) + (k * np.log(n))

newtrend = polynomial_trend(r'.\Polynomial\sampledata.csv', np.random.randint(2, 4), True)
cos, mse = newtrend.general_regress()
