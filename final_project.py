import numpy as np
import matplotlib.pyplot as plt
from iterative_solvers import Gauss_Seidel_optimized
from time import time 

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
    
    # Calculate extrapolated distances and ensure they are multiples of dx
    sigma_t_first = material_props[0]['sigma_t']
    sigma_t_last = material_props[-1]['sigma_t']
    D_left = int((2 / 3 / sigma_t_first) / dx) * dx
    D_right = int((2 / 3 / sigma_t_last) / dx) * dx
    
    # Extended thickness with distinct extrapolated distances
    new_t = t + D_left + D_right
    new_num_mesh_points = int(new_t / dx) + 1
    x = np.linspace(-D_left, t + D_right, new_num_mesh_points)
    
    # Initialize arrays for cross-sections
    sigma_t = np.zeros(new_num_mesh_points)
    sigma_a = np.zeros(new_num_mesh_points)
    D = np.zeros(new_num_mesh_points)
    b = np.full(new_num_mesh_points, Q)
    
    # Assign cross-sections based on material properties
    for idx, material in enumerate(material_props):
        if idx == 0:  # First material
            start_idx = int((material['start'] - D_left) / dx)
            end_idx = int(material['end'] / dx) + 1
        elif idx == len(material_props) - 1:  # Last material
            start_idx = int(material['start'] / dx)
            end_idx = int((material['end'] + D_right) / dx) + 1
        else:  # Middle materials
            start_idx = int(material['start'] / dx)
            end_idx = int(material['end'] / dx) + 1

        # Set cross-sections for the material region
        sigma_t[start_idx:end_idx] = material['sigma_t']
        sigma_s = material['sigma_s_ratio'] * material['sigma_t']
        sigma_a[start_idx:end_idx] = material['sigma_t'] - sigma_s
        D[start_idx:end_idx] = 1 / (3 * material['sigma_t'])

    # Initialize diagonals for the tridiagonal matrix
    lower_diag = np.zeros(new_num_mesh_points - 1)
    main_diag = np.zeros(new_num_mesh_points)
    upper_diag = np.zeros(new_num_mesh_points - 1)

    # Set up the internal grid points for the diffusion equation
    for i in range(1, new_num_mesh_points - 1):
        # Calculate average diffusion coefficients across the left and right interfaces
        D_left_avg = (D[i] + D[i - 1]) / 2
        D_right_avg = (D[i] + D[i + 1]) / 2

        # Lower diagonal (A[i, i-1])
        lower_diag[i - 1] = -D_left_avg / dx**2
        
        # Main diagonal (A[i, i])
        main_diag[i] = (D_left_avg + D_right_avg) / dx**2 + sigma_a[i]
        
        # Upper diagonal (A[i, i+1])
        upper_diag[i] = -D_right_avg / dx**2

    # Apply vacuum boundary conditions at both extrapolated ends
    main_diag[0] = 1
    main_diag[-1] = 1
    b[0] = 0
    b[-1] = 0

    # Initial guess for the neutron flux
    Phi = np.zeros(new_num_mesh_points)

    # Solve using the optimized Gauss-Seidel method
    Phi, residuals = Gauss_Seidel_optimized(lower_diag, main_diag, upper_diag, b, Phi, tol, max_iterations)

    return x, Phi, residuals

# Example usage of the solver for heterogeneous media with distinct vacuum boundary extrapolations
t = 10.0  # Total slab thickness in cm
num_mesh_points = 100
Q = 1.0  # Fixed source
tol = 1e-6  # Convergence tolerance
max_iterations = 1000000  # Maximum number of iterations

# Define material properties for heterogeneous regions
material_props = [
    {'start': 0, 'end': 5, 'sigma_t': 1.0, 'sigma_s_ratio': 0.9},
    {'start': 5, 'end': 10, 'sigma_t': 2.0, 'sigma_s_ratio': 0.8}
]

# Run the solver
x, phi, residuals = diffusion_solver_1D_vacuum_extrapolated_boundaries(t, num_mesh_points, material_props, Q, tol, max_iterations)

# Plot the neutron flux distribution
plt.figure(figsize=(10, 5))
plt.plot(x, phi, label='Neutron Flux $\phi$')
plt.xlabel('x (cm)')
plt.ylabel('$\phi$')
plt.title('1D Diffusion Solution with Extrapolated Vacuum Boundaries (Distinct Left/Right)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the residuals to check convergence
plt.figure(figsize=(10, 5))
plt.plot(range(len(residuals)), residuals, label='Residual (Error)')
plt.title('Residual vs Iteration (Gauss-Seidel) - Extrapolated Vacuum Boundaries')
plt.xlabel('Iteration')
plt.ylabel('Residual (Error)')
plt.yscale('log')  # Log scale to observe convergence
plt.legend()
plt.grid(True)
plt.show()
