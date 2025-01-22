import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigh

# Manually input your matrix values
matrix = np.array([
    [0.25, 0, 0, 0.33],
    [0, 0.18, 0.1, 0],
    [0, 1, 0.12, 0],
    [0.17, -0.1, 0, 0.45]
])
# Test 1: Trace Test (Unit Trace)
trace = np.trace(matrix)
print(f"Trace of the matrix: {trace:.2f}")
if np.isclose(trace, 1.0):
    print("Trace Test Passed: The matrix has a unit trace.")
else:
    print("Trace Test Failed: The matrix does not have a unit trace.")

# Test 2: Hermiticity Test
is_hermitian = np.allclose(matrix, matrix.conj().T)
if is_hermitian:
    print("Hermiticity Test Passed: The matrix is Hermitian.")
else:
    print("Hermiticity Test Failed: The matrix is not Hermitian.")

# Test 3: Positive Semi-Definiteness Test
eigenvalues = np.linalg.eigvalsh(matrix)
print("Eigenvalues of the matrix:", eigenvalues)
if np.all(eigenvalues >= 0):
    print("Positive Semi-Definiteness Test Passed: All eigenvalues are non-negative.")
else:
    print("Positive Semi-Definiteness Test Failed: The matrix has negative eigenvalues.")

# Test 4: Purity Test
matrix_squared = np.dot(matrix, matrix)
purity = np.trace(matrix_squared)
print(f"Purity (Trace of Squared Matrix): {purity:.2f}")
if np.isclose(purity, 1.0):
    print("Purity Test Passed: The state is pure.")
else:
    print("Purity Test: The state is mixed (Purity < 1).")

#Concurrence Calculation
sigma_y = np.array([
    [0, -1j],
    [1j, 0]
])
sigma_y_tensor = np.kron(sigma_y, sigma_y)
R = np.dot(np.dot(matrix, sigma_y_tensor), np.dot(matrix.conj(), sigma_y_tensor))
eigenvalues, _ = eigh(R)
lambda_vals = np.sqrt(np.abs(np.sort(eigenvalues)[::-1]))
concurrence = max(0, lambda_vals[0] - lambda_vals[1] - lambda_vals[2] - lambda_vals[3])
print("Concurrence:", concurrence)

#plotting density matrix
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
x = x.flatten()
y = y.flatten()
z = np.zeros_like(x)
dz = matrix.flatten()


colors = ['Gold' if val > 0.05 else 'lightgreen' if val < 0 else 'skyblue' for val in matrix.flatten()]

# Plot the bars
ax.bar3d(x, y, z, 0.5, 0.5, dz=dz, color=colors, zsort='average')
ax.set_xticks(np.arange(matrix.shape[0]) + 0.25)
ax.set_yticks(np.arange(matrix.shape[1]) + 0.25)
ax.set_xticklabels(['|HH⟩', '|HV⟩', '|VH⟩', '|VV⟩'])  # Customize labels if needed
ax.set_yticklabels(['|HH⟩', '|HV⟩', '|VH⟩', '|VV⟩'])  # Customize labels if needed
ax.set_zlim(-0.2, 0.5)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Values')
plt.title('Density Matrix for given $|\\phi_{+}\\rangle$ State')

plt.show()

