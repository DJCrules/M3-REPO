import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import random
import csv
warnings.simplefilter("error", RuntimeWarning)

class polynomial_trend:
    def __init__(self, filename, order=0, show=False, random=False):
        """
        use order to set generated data trend, 2 for quadratic, 3 for cubic etc
        filename is where to save to or where to read from
        show is whether to show scatter of data
        """
        self.csv_file = filename
        if order != 0:
            self.generate_poly_data(order)
            #^create csv file
        if random:
            self.generate_poly_data(np.random.randint(1, 5))
        df = pd.read_csv(self.csv_file)
        #^read csv file
        self.coeffs = None
        self.mse = None
        self.xs = df.iloc[:, 0]
        self.ys = df.iloc[:, 1]
        self.numpy_xs = np.array(self.xs)
        self.numpy_ys = np.array(self.ys)
        if show:
            self.plot_data()

    def generate_poly_data(self, order, num_points=35, shift=False):
        """
        the added noise = order
        """
        coefficients=np.random.normal(0, 1, order + 1)
        for i in range(1, len(coefficients)):
            coefficients[i] *= (i ** -1)
        #^init these coeffs randomly

        x = np.linspace(-5, 5, num_points)
        y = np.sum([c * (x**i) for i, c in enumerate(coefficients)], axis=0)
        #^find polynomial value for each x value

        noise = np.random.normal(0, order, num_points)
        y += noise
        #add noise

        df = pd.DataFrame({'x': x, 'y': y})
        df.to_csv(self.csv_file, index=False)
    
    def plot_data(self, coefficients=None, extra=0, mse=0, pred=0, avg=False):
        """
        extra is short for extrapolation
        """
        plt.scatter(self.xs, self.ys)
        #^plot data as scatter

        if coefficients is not None:
            poly_func = np.poly1d(coefficients[::-1])
            x_range = np.linspace(min(self.xs), max(self.xs), 100)
            y_fit = poly_func(x_range)
            #^numpy fitting polynomial to axis

            plt.plot(x_range, y_fit, color='red')
            x_xtra = np.linspace(max(self.xs), max(self.xs) + extra)
            y_xtra = poly_func(x_xtra)
            plt.title(f"Relative MSE: {np.round((mse /(max(self.xs) - min(self.xs))), 2)}\ny = " + " + ".join(f"{np.round(c, 2)} * x^{i}" for i, c in enumerate(coefficients)))
            #^title includes mse and all the coeffs nicely organised
            
            if pred != 0:
                poly_derivative = np.polyder(poly_func)
                last_x = max(self.xs)
                last_y = poly_func(last_x)
                last_slope = poly_derivative(last_x) 
                
                x_extrapolate = np.linspace(last_x, last_x + pred, 50)
                y_extrapolate =  last_slope * (x_extrapolate - last_x) + last_y
                # y = mx + c
                if avg:
                    y_avg = np.divide((y_extrapolate + y_xtra), 2)
                    plt.plot(x_extrapolate, y_avg, color='blue', linestyle="dashed")
                plt.plot(x_extrapolate, y_extrapolate, color='red', linestyle="dashed")
            plt.plot(x_xtra, y_xtra, color='red', linestyle="dashed")
        plt.show()
    
    def regress(self, order, rate=0.00001, tolerance=0.0001, max_iterations=20000):
        """
        really good at cubic but quite bad at anything < 3
        """
        try:
            coefficients = np.ones(order + 1)
            scale_factor = (max(self.xs) - min(self.xs)) ** (1 / order)
            for i in range(1, len(coefficients)):
                coefficients[i] = scale_factor / (i ** 2)
            # ^initialise the coeffs like this to avoid overflow, from them being too far 
            # from the datapoints initially. dont use random coeffs because this makes 
            # repeats inconsistent

            last_mse = self.MSE(coefficients)
            iteration = 0
            while iteration < max_iterations:
                poly_func = np.poly1d(coefficients[::-1])
                predicted_ys = poly_func(self.numpy_xs)
                errors = predicted_ys - self.numpy_ys
                # ^the difference in predicted polynomial from datapoints

                gradient = np.zeros(order + 1)
                for i in range(order + 1):
                    gradient[i] = np.mean(2 * errors * np.power(self.numpy_xs, i))
                coefficients -= rate * gradient
                #^gradient descent for coeffs

                current_mse = self.MSE(coefficients)
                if abs(last_mse - current_mse) < tolerance:
                    break
                #^check if the mse has increased and if it has then stop
                last_mse = current_mse
                iteration += 1
            return coefficients, current_mse
        except:
            raise Exception
    
    def MSE(self, coefficients):
        #MSE is the mean squared error for the coeffs and the datapoints
        poly_func = np.poly1d(coefficients[::-1])
        predicted_ys = poly_func(self.numpy_xs)
        errors = (predicted_ys - self.numpy_ys) ** 2
        return np.mean(errors)

    def general_regress(self, extra):
        best_coefficients, best_mse = self.regress(1)
        bestBIC = self.BIC(best_coefficients, best_mse)
        print(f"order 1 BIC: {round(bestBIC, 2)}")
        #^find starting baysian information criteria

        datapoints = len(self.xs)
        for i in range(2, int(np.round(np.sqrt(datapoints), 0))):
            try:
                coeffs, mse = self.regress(i)
                current_BIC = self.BIC(coeffs, mse)
                print(f"order {i} BIC: {round(current_BIC, 2)}")
                if current_BIC < bestBIC:
                    best_coefficients, best_mse = coeffs, mse
                    bestBIC = current_BIC
                    #^finding best bic
            #^regress for the specific order
            except:
                print(f"orders {i}+ cause overflow error")
                break

        
        self.coeffs = best_coefficients
        self.mse = best_mse
        print(f"\norder {len(self.coeffs) - 1} chosen, Refining: ")
        self.coeffs, self.mse = self.regress(len(self.coeffs) - 1, 0.00001, 0.00000001, 200000)
        #^make the model with the best BIC more refined
        self.plot_data(self.coeffs, 0, self.mse)
        self.plot_data(self.coeffs, extra, self.mse, 0)
        self.plot_data(self.coeffs, 0, self.mse, extra)
        self.plot_data(self.coeffs, extra, self.mse, extra)
        self.plot_data(self.coeffs, extra, self.mse, extra, True)
    
    def BIC(self, coeffs, mse):
        #cool equation to figure out the best order polynomial to model dataset
        n = len(self.xs)
        k = len(coeffs)
        return (n * np.log(mse)) + (k * np.log(n))

newtrend = polynomial_trend(r'.\Polynomial\sample_data.csv', show=True)
newtrend.general_regress(2)