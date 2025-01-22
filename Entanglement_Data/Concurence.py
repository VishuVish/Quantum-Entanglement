import numpy as np
from scipy.linalg import eigh

# Define the given matrix
rho = np.array([
    [0.97, -0.04, -0.07, 0.03],
    [-0.04, 0.01, -0.02, 0.01],
    [-0.07, -0.02, 0.02, 0.01],
    [0.03, 0.01, 0.01, -0.01]
])

# Step 1: Square the matrix
matrix_squared = np.dot(rho, rho)
print("Squared Matrix:\n", matrix_squared)

# Step 2: Extract diagonal elements
diagonal_elements = np.diagonal(matrix_squared)
print("Diagonal Elements of Squared Matrix:", diagonal_elements)

# Step 3: Calculate the trace
purity = np.sum(diagonal_elements)
print("Purity (Trace of Squared Matrix):", purity)

print("Purity:", purity)

# Define the Pauli-Y matrix
sigma_y = np.array([
    [0, -1j],
    [1j, 0]
])

# Calculate sigma_y tensor sigma_y
sigma_y_tensor = np.kron(sigma_y, sigma_y)

# Calculate R
R = np.dot(np.dot(rho, sigma_y_tensor), np.dot(rho.conj(), sigma_y_tensor))

# Get the eigenvalues of R
eigenvalues, _ = eigh(R)
lambda_vals = np.sqrt(np.abs(np.sort(eigenvalues)[::-1]))

# Calculate the concurrence
concurrence = max(0, lambda_vals[0] - lambda_vals[1] - lambda_vals[2] - lambda_vals[3])

print("Concurrence:", concurrence)
