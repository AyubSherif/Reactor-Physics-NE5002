import numpy as np
import matplotlib.pyplot as plt
from iterative_solvers import Gauss_Seidel_optimized
from time import time 

# Decorator to measure the execution time of a function
def timer_func(func): 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func

@timer_func
def diffusion_solver_1D(t, num_mesh_points, material_props, Q, tol, max_iterations):
    """
    1D diffusion solver for heterogeneous media with user-defined boundary conditions.
    """

    # Prompt for user-defined boundary conditions
    left_boundary = input("Enter the left boundary condition (v for vacuum or r for reflective): ").strip().lower()
    right_boundary = input("Enter the right boundary condition (v for vacuum or r for reflective): ").strip().lower()

    dx = t / (num_mesh_points - 1)

    # Calculate extrapolated distances based on user-defined vacuum boundaries
    D_left = (1 / 3 / material_props[0]['sigma_t']) // dx * dx if left_boundary == "v" else 0
    D_right = (1 / 3 / material_props[-1]['sigma_t']) // dx * dx if right_boundary == "v" else 0

    # Adjust slab thickness and mesh points
    new_t = t + D_left + D_right
    new_num_mesh_points = int(new_t / dx) + 1
    x = np.linspace(-D_left if left_boundary == "v" else 0, 
                    t + D_right if right_boundary == "v" else t, 
                    new_num_mesh_points)

    # Initialize arrays for cross-sections and source term
    sigma_t = np.zeros(new_num_mesh_points)
    sigma_a = np.zeros(new_num_mesh_points)
    D = np.zeros(new_num_mesh_points)
    b = np.full(new_num_mesh_points, Q)

    # Populate material properties
    for i in range(new_num_mesh_points):
        position = i * dx - D_left  # Adjust position for vacuum left side

        # Find material region
        for material in material_props:
            if material['start'] - D_left <= position <= material['end'] + (D_right if material == material_props[-1] else 0):
                sigma_t[i] = material['sigma_t']
                sigma_s = material['sigma_s_ratio'] * material['sigma_t']
                sigma_a[i] = material['sigma_t'] - sigma_s
                D[i] = 1 / (3 * material['sigma_t'])
                break

    # Initialize tridiagonal matrix components
    lower_diag = np.zeros(new_num_mesh_points - 1)
    main_diag = np.zeros(new_num_mesh_points)
    upper_diag = np.zeros(new_num_mesh_points - 1)

    # Interior points setup
    for i in range(1, new_num_mesh_points - 1):
        D_left_avg = (D[i] + D[i - 1]) / 2
        D_right_avg = (D[i] + D[i + 1]) / 2

        lower_diag[i - 1] = -D_left_avg / dx**2
        main_diag[i] = (D_left_avg + D_right_avg) / dx**2 + sigma_a[i]
        upper_diag[i] = -D_right_avg / dx**2

    # Apply boundary conditions
    if left_boundary == "r":
        main_diag[0] = 2 * D[1] / dx**2 + sigma_a[1]
        upper_diag[0] = -2 * D[1] / dx**2
    else:  # Vacuum
        main_diag[0] = 1
        b[0] = 0

    if right_boundary == "r":
        main_diag[-1] = 2 * D[-2] / dx**2 + sigma_a[-2]
        lower_diag[-1] = -2 * D[-2] / dx**2
    else:  # Vacuum
        main_diag[-1] = 1
        b[-1] = 0

    # Initial guess for neutron flux
    Phi = np.zeros(new_num_mesh_points)

    # Solve using the Gauss-Seidel method
    Phi, residuals = Gauss_Seidel_optimized(lower_diag, main_diag, upper_diag, b, Phi, tol, max_iterations)

    return x, Phi, residuals


# Example usage
t = 15.0  # Total slab thickness in cm
num_mesh_points = 100
Q = 1.0  # Fixed source
tol = 1e-6  # Convergence tolerance
max_iterations = 1000000  # Max iterations for convergence

# Material properties for regions
material_props = [
    {'start': 0, 'end': 5, 'sigma_t': 1.0, 'sigma_s_ratio': 0.9},
    {'start': 5, 'end': 10, 'sigma_t': 1.0, 'sigma_s_ratio': 0.8},
    {'start': 10, 'end': 15, 'sigma_t': 1.0, 'sigma_s_ratio': 0.99}
]

# Run the solver
x, phi, residuals = diffusion_solver_1D(t, num_mesh_points, material_props, Q, tol, max_iterations)

# Plot neutron flux
plt.figure(figsize=(10, 5))
plt.plot(x, phi, label='Neutron Flux $\phi$')
plt.xlabel('x (cm)')
plt.ylabel('$\phi$')
plt.title('1D Diffusion Solution with Extrapolated Vacuum Boundaries (Distinct Left/Right)')
plt.legend()
plt.grid(True)
plt.show()

# Plot convergence residuals
plt.figure(figsize=(10, 5))
plt.plot(range(len(residuals)), residuals, label='Residual (Error)')
plt.title('Residual vs Iteration (Gauss-Seidel) - Extrapolated Vacuum Boundaries')
plt.xlabel('Iteration')
plt.ylabel('Residual (Error)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
