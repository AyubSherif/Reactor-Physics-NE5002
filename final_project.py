import numpy as np
import matplotlib.pyplot as plt
from iterative_solvers import Gauss_Seidel
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
def diffusion_solver_1D(t, num_mesh_points, LB, RB, sigma_t, sigma_s_ratio, Q, tol, max_iterations):
    """
    Solves the 1D diffusion equation using Gauss-Seidel method with vacuum boundary conditions.
    
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

    # Initialize coefficient matrix A and right-hand side vector b
    A = np.zeros((new_num_mesh_points, new_num_mesh_points))
    b = np.full(new_num_mesh_points, Q)

    # Apply vacuum boundary conditions
    A[0, 0] = 1  # Left boundary (vacuum)
    A[-1, -1] = 1  # Right boundary (vacuum)
    b[0] = LB
    b[-1] = RB

    # Set up the internal grid points for the diffusion equation
    for i in range(1, new_num_mesh_points - 1):
        A[i, i - 1] = -1 / dx**2 / (3 * sigma_t)  
        A[i, i] = (2 / dx**2 / (3 * sigma_t) + sigma_a)
        A[i, i + 1] = -1 / dx**2 / (3 * sigma_t)  

    # Initial guess for the neutron flux
    Phi = np.zeros(new_num_mesh_points)

    # Solve using the Gauss-Seidel method
    Phi, residuals = Gauss_Seidel(A, b, Phi, tol, max_iterations)

    # Spatial grid points
    x = np.linspace(-D, t+D, new_num_mesh_points)

    return x, Phi, residuals

# Example usage of diffusion_solver_1D
t = 10.0  # Slab thickness in cm
num_mesh_points = 100
sigma_t = 1.0  # Total cross-section
sigma_s_ratio = 0.9  # Ratio of sigma_s to sigma_t
Q = 1.0  # Fixed source
LB = 0  # Left boundary (vacuum)
RB = 0  # Right boundary (vacuum)
tol = 1e-6  # Convergence tolerance
max_iterations = 10000  # Maximum number of iterations

x, phi, residuals = diffusion_solver_1D(t, num_mesh_points, LB, RB, sigma_t, sigma_s_ratio, Q, tol, max_iterations)

# Plot the solution

plt.figure(figsize=(10, 5))
plt.plot(x, phi, label='Neutron Flux $\phi$')
plt.xlabel('Position (cm)')
plt.ylabel('Neutron Flux $\phi$')
plt.title('1D Diffusion Solution with Vacuum Boundary')
plt.legend()
plt.grid(True)
plt.show()

# Plot the residuals
plt.figure(figsize=(10, 5))
plt.plot(range(len(residuals)), residuals, label='Residual (Error)')
plt.title('Residual vs Iteration (Gauss-Seidel)')
plt.xlabel('Iteration')
plt.ylabel('Residual (Error)')
plt.yscale('log')  # Log scale to observe convergence
plt.legend()
plt.grid(True)
plt.show()