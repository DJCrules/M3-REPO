#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

typedef vector<double> Vec;
typedef vector<Vec> Mat;

class LinearRegression {
private:
    Vec theta;
    double lr;
    double dec;
    double catch_threshold;
    double min_lr;
    int epochs;

public:
    LinearRegression(double learning_rate = 0.01, int max_epochs = 10000, double decay = 0.9999, double catch_thr = 1e-6, double min_lr = 1e-8)
        : lr(learning_rate), epochs(max_epochs), dec(decay), catch_threshold(catch_thr), min_lr(min_lr) {}

    Vec fit(const Mat &X, const Vec &y) {
        int m = X.size();
        int n = X[0].size();
        theta.resize(n + 1);
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> d(0, 1);

        for (double &t : theta) t = d(gen); // Random initialization

        double mse = numeric_limits<double>::infinity();

        for (int epoch = 0; epoch < epochs; epoch++) {
            Vec predictions(m, 0.0);
            for (int i = 0; i < m; i++) {
                predictions[i] = theta[0]; // Bias term
                for (int j = 0; j < n; j++) {
                    predictions[i] += theta[j + 1] * X[i][j];
                }
            }

            Vec errors(m);
            for (int i = 0; i < m; i++) errors[i] = predictions[i] - y[i];

            Vec gradient(n + 1, 0.0);
            for (int j = 0; j < n + 1; j++) {
                for (int i = 0; i < m; i++) {
                    gradient[j] += (j == 0 ? 1.0 : X[i][j - 1]) * errors[i];
                }
                gradient[j] /= m;
            }

            for (int j = 0; j < n + 1; j++) {
                theta[j] -= lr * gradient[j];
            }

            lr = max(lr * dec, min_lr);

            double new_mse = 0.0;
            for (double err : errors) new_mse += err * err;
            new_mse /= m;

            if (epoch % 1000 == 0) {
                cout << "Epoch " << epoch << ": MSE = " << new_mse << endl;
            }

            if (abs(mse - new_mse) < catch_threshold) {
                break; // Early stopping
            }
            mse = new_mse;
        }
        return theta;
    }
};

int main() {
    Mat X = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}};
    Vec y = {2, 3, 4, 5, 6};

    LinearRegression lr;
    Vec theta = lr.fit(X, y);

    cout << "Final theta values: ";
    for (double t : theta) cout << t << " ";
    cout << endl;

    return 0;
}
