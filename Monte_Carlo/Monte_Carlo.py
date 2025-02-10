import numpy as np
import matplotlib.pyplot as plt

def simulate_once():
    # our equation or probably of certain outcomes goes here, I have shown an example for particle decay
    x = np.random.rand()
    output = -0.5*np.log(x)

    return output

def run_monte_carlo(num_trials=10000):
    results = np.array([simulate_once() for _ in range(num_trials)])
    return results

def plot_results(results):
    num_trials = len(results)

    # Histogram
    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    plt.hist(results, bins=50, density=True, alpha=0.50, color='blue', edgecolor='black')
    plt.xlabel("Simulation Output")
    plt.ylabel("Probability Density")
    plt.title("Monte Carlo Histogram")
    plt.grid(True)

    # Scatter Graph
    plt.subplot(1, 2, 2)
    plt.scatter(results, range(num_trials), alpha=0.5, color='red', s=1)
    plt.xlabel("Trial Number")
    plt.ylabel("Simulation Output")
    plt.title("Monte Carlo Scatter Graph")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # This can be altered to change the number of simulated points
    num_trials = 10000
    results = run_monte_carlo(num_trials)
    
    # Calculate statistics of the results
    mean_result = np.mean(results)
    std = np.std(results)
    
    print(f"Monte Carlo Simulation Results ({num_trials} trials):")
    print(f"Mean: {mean_result:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    
    plot_results(results)

main()

