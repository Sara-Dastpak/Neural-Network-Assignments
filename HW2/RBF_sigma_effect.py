import numpy as np

# Data
X = np.array([1, 3, 5, 7, 9])
D = np.array([1.2, 2.4, 2.9, 4.5, 5.1])
C = np.array([2, 6, 8])
W = np.array([-0.6, 0.7, 1.2])
sigmas = [1.0, 0.5, 2]

# Gaussian RBF function
def gaussian(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# RBF Network Function
def rbf_network(X, C, W, sigma):
    outputs = []
    for x in X:
        rbf_outputs = [gaussian(x, c, sigma) for c in C]
        outputs.append(np.dot(rbf_outputs, W))
    return np.array(outputs)

# Calculate errors for each sigma
results = {}
for sigma in sigmas:
    predictions = rbf_network(X, C, W, sigma)
    errors = D - predictions
    mse = np.mean(errors ** 2)
    results[sigma] = {"Predictions": predictions, "Errors": errors, "MSE": mse}

#results
for sigma, result in results.items():
    print(f"Sigma = {sigma}")
    print(f"Predictions: {result['Predictions']}")
    print(f"Errors: {result['Errors']}")
    print(f"MSE: {result['MSE']:.4f}\n")
