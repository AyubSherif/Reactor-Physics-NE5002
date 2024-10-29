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
def diffusion_solver_1D_heterogeneous(t, num_mesh_points, LB, RB, material_props, Q, tol, max_iterations):
    """
    1D diffusion solver for heterogeneous media using Gauss-Seidel with vacuum boundary conditions.

    Parameters:
    t               : Total thickness of the slab
    num_mesh_points : Number of mesh points
    LB, RB          : Boundary conditions (vacuum)
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
    x = np.linspace(0, t, num_mesh_points)
    
    # Initialize arrays for cross-sections based on material properties
    sigma_t = np.zeros(num_mesh_points)
    sigma_a = np.zeros(num_mesh_points)
    D = np.zeros(num_mesh_points)  # Diffusion coefficient
    b = np.full(num_mesh_points, Q)  # Source term for each mesh point
    
    # Assign cross-section values based on material properties for heterogeneous regions
    for material in material_props:
        start_idx = int(material['start'] / dx)
        end_idx = int(material['end'] / dx) + 1
        sigma_t[start_idx:end_idx] = material['sigma_t']
        sigma_s = material['sigma_s_ratio'] * material['sigma_t']
        sigma_a[start_idx:end_idx] = material['sigma_t'] - sigma_s
        D[start_idx:end_idx] = 1 / (3 * material['sigma_t'])

    # Initialize diagonals for the tridiagonal matrix
    lower_diag = np.zeros(num_mesh_points - 1)
    main_diag = np.zeros(num_mesh_points)
    upper_diag = np.zeros(num_mesh_points - 1)

    # Set up the internal grid points for the diffusion equation
    for i in range(1, num_mesh_points - 1):
        # Calculate average diffusion coefficients across the left and right interfaces
        D_left = (D[i] + D[i - 1]) / 2  # Between i and i-1
        D_right = (D[i] + D[i + 1]) / 2  # Between i and i+1

        # Lower diagonal (A[i, i-1]), represents -D_left / dx^2 * phi_{i-1}
        lower_diag[i - 1] = -D_left / dx**2
        
        # Main diagonal (A[i, i]), combines D_left and D_right terms and sigma_a * phi_i
        main_diag[i] = (D_left + D_right) / dx**2 + sigma_a[i]
        
        # Upper diagonal (A[i, i+1]), represents -D_right / dx^2 * phi_{i+1}
        upper_diag[i] = -D_right / dx**2

    # Apply vacuum boundary conditions
    main_diag[0] = 1
    main_diag[-1] = 1
    b[0] = LB
    b[-1] = RB

    # Initial guess for the neutron flux
    Phi = np.zeros(num_mesh_points)

    # Solve using the optimized Gauss-Seidel method
    Phi, residuals = Gauss_Seidel_optimized(lower_diag, main_diag, upper_diag, b, Phi, tol, max_iterations)

    return x, Phi, residuals

# Example usage of the solver for heterogeneous media
t = 10.0  # Total slab thickness in cm
num_mesh_points = 100
LB = 0  # Left boundary (vacuum)
RB = 0  # Right boundary (vacuum)
Q = 1.0  # Fixed source
tol = 1e-6  # Convergence tolerance
max_iterations = 1000000  # Maximum number of iterations

# Define material properties for heterogeneous regions
material_props = [
    {'start': 0, 'end': 5, 'sigma_t': 1.0, 'sigma_s_ratio': 0.9},
    {'start': 5, 'end': 10, 'sigma_t': 2.0, 'sigma_s_ratio': 0.8}
]

# Run the solver
x, phi, residuals = diffusion_solver_1D_heterogeneous(t, num_mesh_points, LB, RB, material_props, Q, tol, max_iterations)

# Plot the neutron flux distribution
plt.figure(figsize=(10, 5))
plt.plot(x, phi, label='Neutron Flux $\phi$')
plt.xlabel('x (cm)')
plt.ylabel('$\phi$')
plt.title('1D Diffusion Solution for Heterogeneous Media')
plt.legend()
plt.grid(True)
plt.show()

# Plot the residuals to check convergence
plt.figure(figsize=(10, 5))
plt.plot(range(len(residuals)), residuals, label='Residual (Error)')
plt.title('Residual vs Iteration (Gauss-Seidel) - Heterogeneous Media')
plt.xlabel('Iteration')
plt.ylabel('Residual (Error)')
plt.yscale('log')  # Log scale to observe convergence
plt.legend()
plt.grid(True)
plt.show()
