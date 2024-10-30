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
def diffusion_solver_1D_vacuum_extrapolated_boundaries(t, num_mesh_points, material_props, Q, tol, max_iterations):
    """
    1D diffusion solver for heterogeneous media with vacuum boundaries extrapolated by different distances.

    Parameters:
    t               : Total thickness of the slab
    num_mesh_points : Number of mesh points
    material_props  : List of dictionaries with material properties, each with:
                      {'start': start_position, 'end': end_position, 
                       'sigma_t': total cross-section, 'sigma_s_ratio': scattering-to-total cross-section ratio}
    Q               : Fixed source in each region
    tol             : Convergence tolerance
    max_iterations  : Maximum number of iterations allowed

    Returns:
    x               : Spatial domain (mesh points)
    phi             : Neutron flux solution
    residuals       : Convergence history (residuals per iteration)
    """
    
    dx = t / (num_mesh_points - 1)
    # Calculate extrapolated distances using the edge material properties
    D_left = (1 / 3 / material_props[0]['sigma_t']) // dx * dx
    D_right = (1 / 3 / material_props[-1]['sigma_t']) // dx * dx
    
    # Extended thickness and new number of mesh points
    new_t = t + D_left + D_right
    new_num_mesh_points = int(new_t / dx) + 1
    x = np.linspace(-D_left, t + D_right, new_num_mesh_points)
    
    # Initialize cross-section arrays for material properties
    sigma_t = np.zeros(new_num_mesh_points)
    sigma_a = np.zeros(new_num_mesh_points)
    D = np.zeros(new_num_mesh_points)
    b = np.full(new_num_mesh_points, Q)
    
    # Populate cross-section arrays based on material regions
    for i in range(new_num_mesh_points):
        position = i * dx - D_left  # Adjust position relative to the slab

        # Find the material region for this position
        for material in material_props:
            if material['start'] - D_left <= position <= material['end'] + (D_right if material == material_props[-1] else 0):
                sigma_t[i] = material['sigma_t']
                sigma_s = material['sigma_s_ratio'] * material['sigma_t']
                sigma_a[i] = material['sigma_t'] - sigma_s
                D[i] = 1 / (3 * material['sigma_t'])
                break

    # Initialize tridiagonal matrix diagonals for the diffusion equation
    lower_diag = np.zeros(new_num_mesh_points - 1)
    main_diag = np.zeros(new_num_mesh_points)
    upper_diag = np.zeros(new_num_mesh_points - 1)

    # Set up tridiagonal coefficients for each internal point
    for i in range(1, new_num_mesh_points - 1):
        D_left_avg = (D[i] + D[i - 1]) / 2
        D_right_avg = (D[i] + D[i + 1]) / 2

        # Fill tridiagonal matrix coefficients
        lower_diag[i - 1] = -D_left_avg / dx**2
        main_diag[i] = (D_left_avg + D_right_avg) / dx**2 + sigma_a[i]
        upper_diag[i] = -D_right_avg / dx**2

    # Apply vacuum boundary conditions at both ends
    main_diag[0] = main_diag[-1] = 1
    b[0] = b[-1] = 0

    # Initial guess for the neutron flux
    Phi = np.zeros(new_num_mesh_points)

    # Solve using the optimized Gauss-Seidel method
    Phi, residuals = Gauss_Seidel_optimized(lower_diag, main_diag, upper_diag, b, Phi, tol, max_iterations)

    return x, Phi, residuals

# Example usage
t = 10.0  # Total slab thickness in cm
num_mesh_points = 1000
Q = 1.0  # Fixed source
tol = 1e-6  # Convergence tolerance
max_iterations = 1000000  # Max iterations for convergence

# Material properties for regions
material_props = [
    {'start': 0, 'end': 5, 'sigma_t': 1.0, 'sigma_s_ratio': 0.9},
    {'start': 5, 'end': 10, 'sigma_t': 1.0, 'sigma_s_ratio': 0.8}
]

# Run the solver
x, phi, residuals = diffusion_solver_1D_vacuum_extrapolated_boundaries(t, num_mesh_points, material_props, Q, tol, max_iterations)

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
