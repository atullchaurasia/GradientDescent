import numpy as np
import matplotlib.pyplot as plt

# Data generation (simple linear data)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 data points
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Add bias term to X (for the intercept term in the model)
X_b = np.c_[np.ones((100, 1)), X]  # X_b is X with an added column of ones for the bias term

# Mean Squared Error Loss Function
def compute_mse(theta, X, y):
    predictions = X.dot(theta)
    errors = predictions - y
    return (1 / len(X)) * np.sum(errors ** 2)

# Gradient Calculation
def compute_gradient(theta, X, y):
    predictions = X.dot(theta)
    errors = predictions - y
    return (2 / len(X)) * X.T.dot(errors)

# Stochastic Gradient Descent with Visualization
def stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=50, plot_interval=10):
    theta = np.random.randn(2, 1)  # Random initialization of model parameters (theta0 and theta1)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 1], y, s=10, label="Data points")
    
    for epoch in range(n_epochs):
        for i in range(len(X)):
            random_index = np.random.randint(len(X))  # Pick a random data point
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradient = compute_gradient(theta, xi, yi)
            theta = theta - learning_rate * gradient

        # Plotting the line at certain intervals to show progress
        if epoch % plot_interval == 0:
            y_pred = X.dot(theta)
            plt.plot(X[:, 1], y_pred, label=f"Epoch {epoch}", alpha=0.7)

    # Final line after all epochs
    y_final = X.dot(theta)
    plt.plot(X[:, 1], y_final, 'r-', linewidth=2, label="Final Regression Line")
    
    plt.title("Stochastic Gradient Descent - Linear Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

    return theta

# Running Stochastic Gradient Descent with visualization
theta_sgd = stochastic_gradient_descent(X_b, y, learning_rate=0.01, n_epochs=50, plot_interval=10)
print(f"Final Theta after Stochastic Gradient Descent: {theta_sgd}")
