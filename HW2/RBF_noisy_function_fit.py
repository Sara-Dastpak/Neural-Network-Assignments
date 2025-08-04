import numpy as np
import matplotlib.pyplot as plt

# Generate data
def generate_data(N=20, noise_var=0.2):
    x = np.linspace(0, 2, N)
    true_function = lambda x: 0.3 * np.cos(3 * np.pi * x) + 0.7 * np.sin(np.pi * x) + 0.5
    f_x = true_function(x)
    noise = np.random.normal(0, np.sqrt(noise_var), size=N)
    y = f_x + noise
    return x, y, f_x

# Gaussian function
def gaussian(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# RBF network function
def rbf_network(x, centers, weights, sigmas):
    return sum(w * gaussian(x, c, s) for w, c, s in zip(weights, centers, sigmas))
def mse_loss(params, X, Y, centers):
    n_centers = len(centers)
    sigmas = params[:n_centers]
    weights = params[n_centers:]
    predictions = np.array([rbf_network(x, centers, weights, sigmas) for x in X])
    return np.mean((Y - predictions) ** 2)

# Gradient of loss function
def mse_gradient(params, X, Y, centers):
    n_centers = len(centers)
    sigmas = params[:n_centers]
    weights = params[n_centers:]
    
    grad_sigmas = np.zeros(n_centers)
    grad_weights = np.zeros(n_centers)
    
    for i, x in enumerate(X):
        prediction = rbf_network(x, centers, weights, sigmas)
        error = prediction - Y[i]
        
        for j, (c, sigma) in enumerate(zip(centers, sigmas)):
            rbf_val = gaussian(x, c, sigma)
            grad_weights[j] += 2 * error * rbf_val / len(X)
            grad_sigmas[j] += 2 * error * weights[j] * rbf_val * ((x - c) ** 2) / (sigma ** 3 * len(X))
    
    return np.concatenate([grad_sigmas, grad_weights])
# Train RBF network
def train_rbf(X, Y, centers, learning_rate=0.01, max_iters=1000, tol=1e-6):
    M = len(centers)
    sigmas = np.ones(M)
    weights = np.random.rand(M)
    params = np.concatenate([sigmas, weights])
    
    for iteration in range(max_iters):
        grad = mse_gradient(params, X, Y, centers)
        params -= learning_rate * grad
        loss = mse_loss(params, X, Y, centers)
        if np.linalg.norm(grad) < tol:
            break
    return params

# Plot results
def plot_results(X, Y, f_true, centers, params, M, domain=[-2, 2]):
    sigmas = params[:M]
    weights = params[M:]
    x_dense = np.linspace(domain[0], domain[1], 100)
    y_pred = np.array([rbf_network(x, centers, weights, sigmas) for x in x_dense])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, label="Noisy data")
    plt.plot(x_dense, f_true(x_dense), label="True function", color="green")
    plt.plot(x_dense, y_pred, label="RBF Approximation", color="red")
    plt.title(f"RBF Network with M={M} centers")
    plt.legend()
    plt.show()

N = 20
M_values = [10, 50]
x, y, f_x = generate_data(N)

for M in M_values:
    centers = np.random.uniform(0, 1, M)
    params = train_rbf(x, y, centers)
    plot_results(x, y, lambda x: 0.3 * np.cos(3 * np.pi * x) + 0.7 * np.sin(np.pi * x) + 0.5, centers, params, M)
