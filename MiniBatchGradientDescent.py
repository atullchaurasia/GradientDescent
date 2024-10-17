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

# Mini-Batch Gradient Descent with Visualization
def mini_batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000, batch_size=20, plot_interval=100):
    theta = np.random.randn(2, 1)  # Random initialization of model parameters (theta0 and theta1)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 1], y, s=10, label="Data points")
    
    for iteration in range(n_iterations):
        shuffled_indices = np.random.permutation(len(X))  # Shuffle the data
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        
        for i in range(0, len(X), batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradient = compute_gradient(theta, xi, yi)
            theta = theta - learning_rate * gradient
        
        # Plotting the line at certain intervals to show progress
        if iteration % plot_interval == 0:
            y_pred = X.dot(theta)
            plt.plot(X[:, 1], y_pred, label=f"Iteration {iteration}", alpha=0.7)

    # Final line after all iterations
    y_final = X.dot(theta)
    plt.plot(X[:, 1], y_final, 'r-', linewidth=2, label="Final Regression Line")
    
    plt.title("Mini-Batch Gradient Descent - Linear Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

    return theta

# Running Mini-Batch Gradient Descent with visualization
theta_mini_batch = mini_batch_gradient_descent(X_b, y, learning_rate=0.01, n_iterations=1000, batch_size=20, plot_interval=100)
print(f"Final Theta after Mini-Batch Gradient Descent: {theta_mini_batch}")
