import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import random as r

def chebyshev_polynomial(x, n):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        return 2 * x * chebyshev_polynomial(x, n - 1) - chebyshev_polynomial(x, n - 2)

def generate_data(cp_degree):
    """
    Generate training data for Chebyshev polynomials up to the specified degree.
    """
    X_train = np.linspace(-1, 1, 1000)
    y_train = np.stack([chebyshev_polynomial(X_train, n) for n in range(cp_degree + 1)], axis=1)
    return X_train, y_train

def plot_results(X_plot, y_true, y_pred):
    """
    Plot the original polynomials and their approximations.
    """
    plt.figure(figsize=(10, 6))
    plt.title(f"Approximation of Chebyshev Polynomials using ANN (Degree {cp_degree})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    for n in range(cp_degree + 1):
        plt.plot(X_plot, y_true[:, n], label=f"Order {n} (True)", linestyle="--", alpha=0.8)

    for n in range(cp_degree + 1):
        plt.plot(X_plot, y_pred[:, n], label=f"Order {n} (Approximation)")

    plt.legend()

def find_best_approximation(X_test, y_test, y_pred):
    """
    Find the best approximation with the smallest mean squared error.
    """
    mse = np.mean((y_test.reshape(-1, 1) - y_pred) ** 2, axis=0)
    best_idx = np.argmin(mse)
    return y_pred[:, best_idx]

# Set the maximum degree of Chebyshev polynomial to be learned
cp_degree = 10

# Generate training data
X_train, y_train = generate_data(cp_degree)

# Create the ANN model
model = Sequential()
model.add(Dense(20, input_shape=(1,), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(cp_degree + 1, activation='linear'))
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=2000, batch_size=32, verbose=1)

# Generate test data for plotting
X_plot = np.linspace(-1, 1, 1000)

# Evaluate the original polynomials for test data
y_true = np.stack([chebyshev_polynomial(X_plot, n) for n in range(cp_degree + 1)], axis=1)

# Evaluate the approximations using the trained model for training data
y_pred_train = model.predict(X_plot)

# Generate random data for testing
x_test = np.array([r.uniform(-1, 1) for _ in range(1000)])
y_test = np.array([r.uniform(-1, 1) for _ in range(1000)])

# Evaluate the approximation for testing data
y_pred_test = model.predict(x_test.reshape(-1, 1))

# Find the best approximation
best_approximation = find_best_approximation(x_test, y_test, y_pred_test)

# Plot the results for training data
plot_results(X_plot, y_true, y_pred_train)
plt.show()

# Plot the random group of dots and its best approximation
plt.figure(figsize=(10, 6))
plt.title("Approximation of Random Group of Dots using ANN")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.scatter(x_test, y_test, label="Random Group of Dots", linestyle="--", alpha=0.8)
plt.plot(x_test, best_approximation, label="Best Approximation")
plt.legend()
plt.show()
