import numpy as np
import matplotlib.pyplot as plt
from iterative_solvers import Gauss_Seidel_optimized
from time import time 
  
  
def timer_func(func): 
    # This function shows the execution time of  
    # the function object passed 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func

@timer_func
def diffusion_solver_1D_optimized(t, num_mesh_points, LB, RB, sigma_t, sigma_s_ratio, Q, tol, max_iterations):
    """
    Optimized 1D diffusion solver using Gauss-Seidel method with vacuum boundary conditions.
    
    Parameters:
    t               : Thickness of the slab
    num_mesh_points : Number of mesh points
    sigma_t         : Total cross-section
    sigma_s_ratio   : Ratio of sigma_s to sigma_t
    Q               : Fixed source
    tol             : Convergence tolerance
    max_iterations  : Maximum number of iterations allowed
    
    Returns:
    x               : Spatial domain (mesh points)
    phi             : Neutron flux solution
    residuals       : Convergence history (residuals per iteration)
    """
    dx = t / (num_mesh_points - 1)  # Step size
    D = 1 / (3 * sigma_t)  # Diffusion coefficient
    new_t = t + 2*D
    new_num_mesh_points = int(new_t / dx) + 1
    
    sigma_s = sigma_s_ratio * sigma_t
    sigma_a = sigma_t - sigma_s  # Absorption cross-section

    # Initialize diagonals for the tridiagonal matrix
    lower_diag = np.zeros(new_num_mesh_points - 1)  # A[i, i-1]
    main_diag = np.zeros(new_num_mesh_points)       # A[i, i]
    upper_diag = np.zeros(new_num_mesh_points - 1)  # A[i, i+1]
    b = np.full(new_num_mesh_points, Q)  # Right-hand side vector

    # Apply vacuum boundary conditions
    main_diag[0] = 1  # Left boundary (vacuum)
    main_diag[-1] = 1  # Right boundary (vacuum)
    b[0] = LB
    b[-1] = RB

    # Set up the internal grid points for the diffusion equation
    for i in range(1, new_num_mesh_points - 1):
        lower_diag[i - 1] = -1 / dx**2 / (3 * sigma_t)  # A[i, i-1]
        main_diag[i] = (2 / dx**2 / (3 * sigma_t) + sigma_a)  # A[i, i]
        upper_diag[i] = -1 / dx**2 / (3 * sigma_t)  # A[i, i+1]

    # Initial guess for the neutron flux
    Phi = np.zeros(new_num_mesh_points)

    # Solve using the optimized Gauss-Seidel method
    Phi, residuals = Gauss_Seidel_optimized(lower_diag, main_diag, upper_diag, b, Phi, tol, max_iterations)

    # Spatial grid points
    x = np.linspace(-D, t + D, new_num_mesh_points)

    return x, Phi, residuals


# Example usage of diffusion_solver_1D_optimized
t = 10.0  # Slab thickness in cm
num_mesh_points = 1000
sigma_t = 1.0  # Total cross-section
sigma_s_ratios = [0.5, 0.8, 0.9, 0.99, 1]  # Ratio of sigma_s to sigma_t
Q = 1.0  # Fixed source
LB = 0  # Left boundary (vacuum)
RB = 0  # Right boundary (vacuum)
tol = 1e-6  # Convergence tolerance
max_iterations = 1000  # Maximum number of iterations

# Create a plot
plt.figure(figsize=(10, 5))

# Loop through each sigma_s_ratio, solve the diffusion equation, and plot the results
for sigma_s_ratio in sigma_s_ratios:
    x, phi, residuals = diffusion_solver_1D_optimized(t, num_mesh_points, LB, RB, sigma_t, sigma_s_ratio, Q, tol, max_iterations)
    plt.plot(x, phi, label=f'$\Sigma_s / \Sigma_t$ = {sigma_s_ratio}')

# Plot settings
plt.xlabel('Position (cm)')
plt.ylabel('Neutron Flux $\phi$')
plt.title('1D Diffusion Solution for Different $\Sigma_s / \Sigma_t$ Ratios')
plt.legend()
plt.grid(True)
plt.show()

'''
x, phi, residuals = diffusion_solver_1D_optimized(t, num_mesh_points, LB, RB, sigma_t, sigma_s_ratio, Q, tol, max_iterations)

# Plot the solution
plt.figure(figsize=(10, 5))
plt.plot(x, phi, label='Neutron Flux $\phi$', marker='o')
plt.xlabel('Position (cm)')
plt.ylabel('Neutron Flux $\phi$')
plt.title('1D Diffusion Solution with Vacuum Boundary (Optimized)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the residuals
plt.figure(figsize=(10, 5))
plt.plot(range(len(residuals)), residuals, label='Residual (Error)', marker='x')
plt.title('Residual vs Iteration (Gauss-Seidel) - Optimized')
plt.xlabel('Iteration')
plt.ylabel('Residual (Error)')
plt.yscale('log')  # Log scale to observe convergence
plt.legend()
plt.grid(True)
plt.show()
'''