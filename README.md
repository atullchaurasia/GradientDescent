# Gradient Descent Visualization

This repository contains two Python scripts that demonstrate **Gradient Descent** for 2D and 3D functions using `matplotlib` to visualize the optimization process. Gradient Descent is an optimization algorithm that iteratively adjusts parameters to minimize a function. The scripts help visualize how the algorithm converges to the function's minimum.

## Files

### 1. `2DGradientDescent.py`

This script demonstrates **gradient descent on a simple quadratic function** \( y = x^2 \) in two dimensions.

#### Key Components:
- **`y_function(x)`**: The function \( y = x^2 \), which we aim to minimize.
- **`y_derivative(x)`**: The derivative of the function, \( y' = 2x \), used to determine the direction of steepest descent.
- **Gradient Descent Loop**: The algorithm starts at an initial `current_position` and iteratively updates it using the learning rate and the derivative to reduce the function's value.
  
#### Visualization:
- The **quadratic curve** is plotted using `matplotlib`, and the current position of the gradient descent step is visualized as a red dot on the curve.
- The algorithm takes **1000 steps** to reach the minimum point on the curve.

#### Code Walkthrough:
```python
current_position = (80, y_function(80))  # Starting point
learning_rate = 0.01  # Step size
for _ in range(1000):  # Iterating 1000 times
    new_x = current_position[0] - learning_rate * y_derivative(current_position[0])
    new_y = y_function(new_x)
    current_position = (new_x, new_y)
```
This loop gradually moves the position towards the minimum by following the negative of the gradient.

---

### 2. `3DGradientDescent.py`

This script extends the idea to **3D gradient descent** on the function \( z(x, y) = \frac{\sin(5x) \cdot \cos(5y)}{5} \).

#### Key Components:
- **`z_function(x, y)`**: A more complex function that represents a 3D surface.
- **`calculate_gradient(x, y)`**: The partial derivatives \( \frac{\partial z}{\partial x} \) and \( \frac{\partial z}{\partial y} \), which indicate the direction to move in to minimize the function.

#### Visualization:
- The **3D surface plot** is generated using `matplotlib`'s `plot_surface`, and the current positions of three points are visualized as magenta, red, and yellow dots on the surface.
- The algorithm updates three different starting positions over **1000 iterations**.

#### Code Walkthrough:
```python
current_position1 = (0.7, 0.4, z_function(0.7, 0.4))  # Starting positions
learning_rate = 0.01  # Step size

for _ in range(1000):
    X_derivative, Y_derivative = calculate_gradient(current_position1[0], current_position1[1])
    X_new, Y_new = current_position1[0] - learning_rate * X_derivative, current_position1[1] - learning_rate * Y_derivative
    current_position1 = (X_new, Y_new, z_function(X_new, Y_new))
```
This loop updates the positions in both x and y directions to move toward the minimum of the surface.

---

## How to Run

1. Install the required dependencies:
   ```bash
   pip install numpy matplotlib
   ```
2. Run the scripts:
   ```bash
   python 2DGradientDescent.py
   python 3DGradientDescent.py
   ```
3. You will see a real-time visualization of the gradient descent process converging to the minimum.

## Purpose

- The `2DGradientDescent.py` script provides an intuitive example of how gradient descent works in one dimension.
- The `3DGradientDescent.py` script showcases how gradient descent can be applied to functions in higher dimensions, with real-time visualizations of three different starting points converging to the function's minimum.
