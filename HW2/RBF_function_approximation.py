import numpy as np

#target function
def target_function(x):
    return np.sin(x) + np.cos(2 * x) - 3 * x + 1

# Gaussian function
def gaussian(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# RBF Network
def rbf_network(x, centers, weights, sigmas):
    rbf_values = np.array([gaussian(x, c, s) for c, s in zip(centers, sigmas)])
    return np.dot(weights, rbf_values)

#MSE
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

# Gradient descent
def gradient_descent(X, Y, centers, learning_rate=0.01, max_iters=1000, tol=1e-6):
    n_centers = len(centers)
    sigmas = np.ones(n_centers)
    weights = np.random.rand(n_centers)
    
    params = np.concatenate([sigmas, weights])
    
    for iteration in range(max_iters):
        grad = mse_gradient(params, X, Y, centers)
        params -= learning_rate * grad
        loss = mse_loss(params, X, Y, centers)
        
        if np.linalg.norm(grad) < tol:
            break
    
    return params

X_train = np.linspace(-5, 5, 100)
Y_train = target_function(X_train)

centers = np.linspace(-5, 5, 5)

optimal_params = gradient_descent(X_train, Y_train, centers)
optimal_sigmas = optimal_params[:len(centers)]
optimal_weights = optimal_params[len(centers):]
Y_pred = np.array([rbf_network(x, centers, optimal_weights, optimal_sigmas) for x in X_train])

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(X_train, Y_train, label='Original Function', linewidth=2)
plt.plot(X_train, Y_pred, label='RBF Approximation', linestyle='--', linewidth=2)
plt.scatter(centers, target_function(centers), color='red', label='RBF Centers')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('RBF Approximation')
plt.grid()
plt.show()
